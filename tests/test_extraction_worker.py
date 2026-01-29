"""
Tests for extraction worker
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extraction.queue import ExtractionQueue
from src.extraction.worker import ExtractionWorker, create_worker
from src.extraction.models import JobStatus


class TestExtractionWorker:
    """Tests for ExtractionWorker"""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            upload_dir = Path(tmpdir) / "uploads"
            output_dir = Path(tmpdir) / "exports"
            upload_dir.mkdir()
            output_dir.mkdir()
            yield upload_dir, output_dir

    @pytest.fixture
    def queue(self):
        """Create a temporary queue for testing"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        q = ExtractionQueue(db_path)
        yield q
        try:
            os.unlink(db_path)
        except:
            pass

    @pytest.fixture
    def worker(self, queue, temp_dirs):
        """Create worker with test configuration"""
        upload_dir, output_dir = temp_dirs
        return ExtractionWorker(
            queue=queue,
            windows_url="http://test-windows:8001",
            upload_dir=str(upload_dir),
            output_dir=str(output_dir),
            poll_interval=0.1,
            max_backoff=1.0,
            request_timeout=10.0
        )

    def test_worker_init(self, worker):
        """Test worker initialization"""
        assert worker.windows_url == "http://test-windows:8001"
        assert worker.poll_interval == 0.1
        assert worker._running is False

    def test_worker_stop(self, worker):
        """Test worker stop method"""
        worker._running = True
        worker.stop()
        assert worker._running is False

    @patch('src.extraction.worker.requests.get')
    def test_check_service_health_success(self, mock_get, worker):
        """Test health check when service is healthy"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_get.return_value = mock_response

        result = worker._check_service_health()
        assert result is True
        mock_get.assert_called_once()

    @patch('src.extraction.worker.requests.get')
    def test_check_service_health_unhealthy(self, mock_get, worker):
        """Test health check when service is unhealthy"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "unhealthy"}
        mock_get.return_value = mock_response

        result = worker._check_service_health()
        assert result is False

    @patch('src.extraction.worker.requests.get')
    def test_check_service_health_error(self, mock_get, worker):
        """Test health check when request fails"""
        mock_get.side_effect = Exception("Connection refused")

        result = worker._check_service_health()
        assert result is False

    @patch('src.extraction.worker.requests.get')
    def test_check_service_health_timeout(self, mock_get, worker):
        """Test health check with timeout"""
        import requests
        mock_get.side_effect = requests.Timeout()

        result = worker._check_service_health()
        assert result is False

    def test_process_next_job_empty_queue(self, worker):
        """Test processing when queue is empty"""
        result = worker._process_next_job()
        assert result is False

    @patch.object(ExtractionWorker, '_check_service_health')
    def test_process_next_job_service_unavailable(self, mock_health, worker, queue, temp_dirs):
        """Test processing when Windows service is unavailable"""
        upload_dir, _ = temp_dirs

        # Create test XRK file
        xrk_file = upload_dir / "test.xrk"
        xrk_file.write_bytes(b"fake xrk data")

        # Add job to queue
        job = queue.submit("test.xrk")

        # Mock unhealthy service
        mock_health.return_value = False

        result = worker._process_next_job()

        assert result is True
        updated_job = queue.get(job.id)
        assert updated_job.status == JobStatus.FAILED
        assert "unavailable" in updated_job.error_message

    @patch('src.extraction.worker.requests.post')
    @patch('src.extraction.worker.requests.get')
    def test_extract_file_success(self, mock_get, mock_post, worker, queue, temp_dirs):
        """Test successful file extraction"""
        upload_dir, output_dir = temp_dirs

        # Create test XRK file
        xrk_file = upload_dir / "test.xrk"
        xrk_file.write_bytes(b"fake xrk data")

        # Add job to queue
        job = queue.submit("test.xrk")
        job.mark_processing()
        queue._update_job(job)

        # Mock extraction response
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {
            "success": True,
            "output_filename": "test_output.parquet",
            "rows": 100,
            "columns": 10
        }
        mock_post.return_value = mock_post_response

        # Mock download response
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.iter_content.return_value = [b"parquet data"]
        mock_get.return_value = mock_get_response

        # Execute extraction
        result = worker._extract_file(job)

        assert result is not None
        assert result.name == "test_output.parquet"
        assert result.exists()

    @patch('src.extraction.worker.requests.post')
    def test_extract_file_upload_failure(self, mock_post, worker, queue, temp_dirs):
        """Test extraction when upload fails"""
        upload_dir, _ = temp_dirs

        # Create test XRK file
        xrk_file = upload_dir / "test.xrk"
        xrk_file.write_bytes(b"fake xrk data")

        job = queue.submit("test.xrk")
        job.mark_processing()

        # Mock failed response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        with pytest.raises(RuntimeError, match="Extraction failed"):
            worker._extract_file(job)

    def test_extract_file_not_found(self, worker, queue):
        """Test extraction when XRK file doesn't exist"""
        job = queue.submit("nonexistent.xrk")
        job.mark_processing()

        with pytest.raises(FileNotFoundError):
            worker._extract_file(job)

    def test_handle_failure_backoff(self, worker):
        """Test exponential backoff on failures"""
        initial_backoff = worker._current_backoff

        # Simulate failures
        with patch('src.extraction.worker.time.sleep'):
            worker._handle_failure()
            assert worker._current_backoff == initial_backoff * 2
            assert worker._consecutive_failures == 1

            worker._handle_failure()
            assert worker._current_backoff == initial_backoff * 4
            assert worker._consecutive_failures == 2

    def test_handle_failure_max_backoff(self, worker):
        """Test backoff doesn't exceed maximum"""
        worker._current_backoff = worker.max_backoff

        with patch('src.extraction.worker.time.sleep'):
            worker._handle_failure()
            assert worker._current_backoff == worker.max_backoff

    @patch.object(ExtractionWorker, '_check_service_health')
    @patch.object(ExtractionWorker, '_extract_file')
    def test_process_single_success(self, mock_extract, mock_health, worker, queue, temp_dirs):
        """Test processing a single job by ID"""
        upload_dir, output_dir = temp_dirs

        # Create test file
        xrk_file = upload_dir / "test.xrk"
        xrk_file.write_bytes(b"fake data")

        job = queue.submit("test.xrk")

        mock_health.return_value = True
        mock_extract.return_value = output_dir / "output.parquet"

        result = worker.process_single(job.id)

        assert result is True
        updated = queue.get(job.id)
        assert updated.status == JobStatus.COMPLETED

    def test_process_single_not_found(self, worker):
        """Test processing nonexistent job"""
        result = worker.process_single(9999)
        assert result is False

    def test_process_single_not_pending(self, worker, queue):
        """Test processing job that's not pending"""
        job = queue.submit("test.xrk")
        queue.mark_completed(job.id, "/output/test.parquet")

        result = worker.process_single(job.id)
        assert result is False


class TestCreateWorker:
    """Tests for create_worker factory function"""

    def test_create_worker_defaults(self):
        """Test creating worker with defaults"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            worker = create_worker(db_path=db_path)
            assert worker is not None
            assert isinstance(worker.queue, ExtractionQueue)
        finally:
            os.unlink(db_path)

    def test_create_worker_custom_url(self):
        """Test creating worker with custom URL"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            worker = create_worker(
                db_path=db_path,
                windows_url="http://custom:9000"
            )
            assert worker.windows_url == "http://custom:9000"
        finally:
            os.unlink(db_path)


class TestWorkerIntegration:
    """Integration tests for worker with queue"""

    @pytest.fixture
    def setup(self):
        """Set up test environment"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "queue.db"
            upload_dir = Path(tmpdir) / "uploads"
            output_dir = Path(tmpdir) / "exports"
            upload_dir.mkdir()
            output_dir.mkdir()

            queue = ExtractionQueue(str(db_path))
            worker = ExtractionWorker(
                queue=queue,
                windows_url="http://test:8001",
                upload_dir=str(upload_dir),
                output_dir=str(output_dir)
            )

            yield {
                "queue": queue,
                "worker": worker,
                "upload_dir": upload_dir,
                "output_dir": output_dir
            }

    @patch.object(ExtractionWorker, '_check_service_health')
    @patch.object(ExtractionWorker, '_extract_file')
    def test_full_job_lifecycle(self, mock_extract, mock_health, setup):
        """Test complete job lifecycle through worker"""
        queue = setup["queue"]
        worker = setup["worker"]
        upload_dir = setup["upload_dir"]
        output_dir = setup["output_dir"]

        # Create test file
        xrk_file = upload_dir / "session.xrk"
        xrk_file.write_bytes(b"xrk content")

        # Submit job
        job = queue.submit("session.xrk")
        assert job.status == JobStatus.PENDING

        # Mock successful extraction
        mock_health.return_value = True
        output_file = output_dir / "session.parquet"
        output_file.write_bytes(b"parquet content")
        mock_extract.return_value = output_file

        # Process job
        result = worker._process_next_job()

        assert result is True
        completed_job = queue.get(job.id)
        assert completed_job.status == JobStatus.COMPLETED
        assert "session.parquet" in completed_job.output_path

    @patch.object(ExtractionWorker, '_check_service_health')
    def test_retry_on_failure(self, mock_health, setup):
        """Test that failed jobs can be retried"""
        queue = setup["queue"]
        worker = setup["worker"]
        upload_dir = setup["upload_dir"]

        # Create test file
        xrk_file = upload_dir / "retry_test.xrk"
        xrk_file.write_bytes(b"data")

        # Submit job with retries
        job = queue.submit("retry_test.xrk", max_retries=3)

        # Mock unhealthy service
        mock_health.return_value = False

        # First attempt fails
        worker._process_next_job()
        failed_job = queue.get(job.id)
        assert failed_job.status == JobStatus.FAILED
        assert failed_job.retry_count == 1

        # Retry the job
        retried = queue.retry(job.id)
        assert retried is not None
        assert retried.status == JobStatus.PENDING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
