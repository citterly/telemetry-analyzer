"""
Tests for extraction queue module
"""

import os
import sys
import tempfile
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extraction.models import ExtractionJob, JobStatus
from src.extraction.queue import ExtractionQueue


class TestExtractionJob:
    """Tests for ExtractionJob model"""

    def test_job_creation(self):
        """Test basic job creation"""
        job = ExtractionJob(xrk_filename="test.xrk")
        assert job.xrk_filename == "test.xrk"
        assert job.status == JobStatus.PENDING
        assert job.retry_count == 0

    def test_job_to_dict(self):
        """Test job serialization to dict"""
        job = ExtractionJob(id=1, xrk_filename="test.xrk")
        data = job.to_dict()
        assert data["id"] == 1
        assert data["xrk_filename"] == "test.xrk"
        assert data["status"] == "pending"

    def test_job_from_dict(self):
        """Test job deserialization from dict"""
        data = {
            "id": 1,
            "xrk_filename": "test.xrk",
            "status": "processing",
            "retry_count": 2
        }
        job = ExtractionJob.from_dict(data)
        assert job.id == 1
        assert job.xrk_filename == "test.xrk"
        assert job.status == JobStatus.PROCESSING
        assert job.retry_count == 2

    def test_job_can_retry(self):
        """Test retry eligibility check"""
        job = ExtractionJob(xrk_filename="test.xrk", max_retries=3)
        job.status = JobStatus.FAILED
        job.retry_count = 2
        assert job.can_retry() is True

        job.retry_count = 3
        assert job.can_retry() is False

    def test_job_mark_processing(self):
        """Test marking job as processing"""
        job = ExtractionJob(xrk_filename="test.xrk")
        job.mark_processing()
        assert job.status == JobStatus.PROCESSING
        assert job.started_at is not None

    def test_job_mark_completed(self):
        """Test marking job as completed"""
        job = ExtractionJob(xrk_filename="test.xrk")
        job.mark_completed("/path/to/output.parquet")
        assert job.status == JobStatus.COMPLETED
        assert job.output_path == "/path/to/output.parquet"
        assert job.completed_at is not None

    def test_job_mark_failed(self):
        """Test marking job as failed"""
        job = ExtractionJob(xrk_filename="test.xrk")
        job.mark_failed("Connection timeout")
        assert job.status == JobStatus.FAILED
        assert job.error_message == "Connection timeout"
        assert job.retry_count == 1


class TestExtractionQueue:
    """Tests for ExtractionQueue"""

    @pytest.fixture
    def queue(self):
        """Create a temporary queue for testing"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        q = ExtractionQueue(db_path)
        yield q
        # Cleanup
        try:
            os.unlink(db_path)
        except:
            pass

    def test_queue_submit(self, queue):
        """Test submitting a job to the queue"""
        job = queue.submit("test_file.xrk")
        assert job.id is not None
        assert job.xrk_filename == "test_file.xrk"
        assert job.status == JobStatus.PENDING

    def test_queue_submit_with_priority(self, queue):
        """Test submitting jobs with different priorities"""
        job1 = queue.submit("low_priority.xrk", priority=1)
        job2 = queue.submit("high_priority.xrk", priority=10)

        # High priority job should be next
        next_job = queue.get_next_pending()
        assert next_job.id == job2.id

    def test_queue_get(self, queue):
        """Test retrieving a job by ID"""
        submitted = queue.submit("test.xrk")
        retrieved = queue.get(submitted.id)
        assert retrieved is not None
        assert retrieved.xrk_filename == "test.xrk"

    def test_queue_get_nonexistent(self, queue):
        """Test retrieving a nonexistent job"""
        job = queue.get(9999)
        assert job is None

    def test_queue_status(self, queue):
        """Test getting job status"""
        job = queue.submit("test.xrk")
        status = queue.get_status(job.id)
        assert status == JobStatus.PENDING

    def test_queue_claim_next(self, queue):
        """Test claiming next pending job"""
        queue.submit("job1.xrk")
        queue.submit("job2.xrk")

        claimed = queue.claim_next()
        assert claimed is not None
        assert claimed.status == JobStatus.PROCESSING

        # Next claim should get job2
        claimed2 = queue.claim_next()
        assert claimed2.id != claimed.id

    def test_queue_claim_empty(self, queue):
        """Test claiming from empty queue"""
        claimed = queue.claim_next()
        assert claimed is None

    def test_queue_list_jobs(self, queue):
        """Test listing all jobs"""
        queue.submit("job1.xrk")
        queue.submit("job2.xrk")
        queue.submit("job3.xrk")

        jobs = queue.list_jobs()
        assert len(jobs) == 3

    def test_queue_list_jobs_by_status(self, queue):
        """Test listing jobs filtered by status"""
        job1 = queue.submit("job1.xrk")
        queue.submit("job2.xrk")

        queue.mark_completed(job1.id, "/output/job1.parquet")

        pending = queue.list_jobs(status=JobStatus.PENDING)
        completed = queue.list_jobs(status=JobStatus.COMPLETED)

        assert len(pending) == 1
        assert len(completed) == 1

    def test_queue_count(self, queue):
        """Test counting jobs"""
        queue.submit("job1.xrk")
        queue.submit("job2.xrk")

        assert queue.count() == 2
        assert queue.count(JobStatus.PENDING) == 2
        assert queue.count(JobStatus.COMPLETED) == 0

    def test_queue_mark_completed(self, queue):
        """Test marking a job as completed"""
        job = queue.submit("test.xrk")
        updated = queue.mark_completed(job.id, "/output/test.parquet")

        assert updated.status == JobStatus.COMPLETED
        assert updated.output_path == "/output/test.parquet"

    def test_queue_mark_failed(self, queue):
        """Test marking a job as failed"""
        job = queue.submit("test.xrk")
        updated = queue.mark_failed(job.id, "Network error")

        assert updated.status == JobStatus.FAILED
        assert updated.error_message == "Network error"
        assert updated.retry_count == 1

    def test_queue_retry(self, queue):
        """Test retrying a failed job"""
        job = queue.submit("test.xrk", max_retries=3)
        queue.mark_failed(job.id, "Error 1")

        retried = queue.retry(job.id)
        assert retried is not None
        assert retried.status == JobStatus.PENDING

    def test_queue_retry_exhausted(self, queue):
        """Test that exhausted retries cannot be retried"""
        job = queue.submit("test.xrk", max_retries=1)
        queue.mark_failed(job.id, "Error")

        # Should not be able to retry (1 retry already used)
        retried = queue.retry(job.id)
        assert retried is None

    def test_queue_retry_all_failed(self, queue):
        """Test retrying all failed jobs"""
        job1 = queue.submit("job1.xrk", max_retries=3)
        job2 = queue.submit("job2.xrk", max_retries=3)
        job3 = queue.submit("job3.xrk", max_retries=1)

        queue.mark_failed(job1.id, "Error")
        queue.mark_failed(job2.id, "Error")
        queue.mark_failed(job3.id, "Error")  # Will exhaust retries

        count = queue.retry_all_failed()
        assert count == 2  # job3 can't retry

    def test_queue_delete(self, queue):
        """Test deleting a job"""
        job = queue.submit("test.xrk")
        assert queue.delete(job.id) is True
        assert queue.get(job.id) is None

    def test_queue_clear_completed(self, queue):
        """Test clearing completed jobs"""
        job1 = queue.submit("job1.xrk")
        job2 = queue.submit("job2.xrk")
        queue.submit("job3.xrk")

        queue.mark_completed(job1.id, "/out/1.parquet")
        queue.mark_completed(job2.id, "/out/2.parquet")

        cleared = queue.clear_completed()
        assert cleared == 2
        assert queue.count() == 1

    def test_queue_stats(self, queue):
        """Test getting queue statistics"""
        job1 = queue.submit("job1.xrk")
        queue.submit("job2.xrk")
        queue.submit("job3.xrk")

        queue.mark_completed(job1.id, "/out/1.parquet")

        stats = queue.get_stats()
        assert stats["pending"] == 2
        assert stats["completed"] == 1
        assert stats["total"] == 3

    def test_queue_persistence(self):
        """Test that queue survives reconnection"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Create queue and add job
            q1 = ExtractionQueue(db_path)
            job = q1.submit("persistent.xrk")
            job_id = job.id

            # Create new queue instance pointing to same DB
            q2 = ExtractionQueue(db_path)
            retrieved = q2.get(job_id)

            assert retrieved is not None
            assert retrieved.xrk_filename == "persistent.xrk"
        finally:
            os.unlink(db_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
