"""
Queue Worker for XRK Extraction
Background worker that polls the queue, calls Windows API, handles failures with backoff.

Usage:
    python -m src.extraction.worker --windows-url http://windows-machine:8001

The worker:
1. Polls the extraction queue for pending jobs
2. Claims the next job atomically
3. Uploads XRK file to Windows extraction service
4. Downloads resulting Parquet
5. Marks job complete or failed with retry
"""

import os
import sys
import time
import signal
import logging
import argparse
import tempfile
import requests
from pathlib import Path
from typing import Optional
from datetime import datetime

from .queue import ExtractionQueue
from .models import ExtractionJob, JobStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("extraction.worker")


class ExtractionWorker:
    """
    Background worker that processes XRK extraction jobs.

    Polls the queue, sends files to Windows service, handles retries.
    """

    def __init__(
        self,
        queue: ExtractionQueue,
        windows_url: str = "http://localhost:8001",
        upload_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        poll_interval: float = 5.0,
        max_backoff: float = 300.0,
        request_timeout: float = 600.0
    ):
        """
        Initialize the worker.

        Args:
            queue: ExtractionQueue instance
            windows_url: URL of Windows extraction service
            upload_dir: Directory containing XRK files to process
            output_dir: Directory to save extracted Parquet files
            poll_interval: Seconds between queue polls
            max_backoff: Maximum backoff time after failures
            request_timeout: HTTP request timeout in seconds
        """
        self.queue = queue
        self.windows_url = windows_url.rstrip('/')
        self.upload_dir = Path(upload_dir) if upload_dir else Path("data/uploads")
        self.output_dir = Path(output_dir) if output_dir else Path("data/exports/processed")
        self.poll_interval = poll_interval
        self.max_backoff = max_backoff
        self.request_timeout = request_timeout

        self._running = False
        self._current_backoff = poll_interval
        self._consecutive_failures = 0

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        """Start the worker loop"""
        self._running = True
        logger.info(f"Worker starting, polling {self.windows_url}")
        logger.info(f"Upload dir: {self.upload_dir}")
        logger.info(f"Output dir: {self.output_dir}")

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        while self._running:
            try:
                processed = self._process_next_job()

                if processed:
                    # Reset backoff on success
                    self._current_backoff = self.poll_interval
                    self._consecutive_failures = 0
                else:
                    # No jobs available, sleep for poll interval
                    time.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"Worker error: {e}")
                self._handle_failure()

        logger.info("Worker stopped")

    def stop(self):
        """Stop the worker gracefully"""
        logger.info("Stopping worker...")
        self._running = False

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def _handle_failure(self):
        """Handle a failure with exponential backoff"""
        self._consecutive_failures += 1
        self._current_backoff = min(
            self._current_backoff * 2,
            self.max_backoff
        )
        logger.warning(f"Backing off for {self._current_backoff:.1f}s (failure #{self._consecutive_failures})")
        time.sleep(self._current_backoff)

    def _process_next_job(self) -> bool:
        """
        Process the next pending job.

        Returns:
            True if a job was processed (success or failure), False if no jobs
        """
        # Claim next job atomically
        job = self.queue.claim_next()
        if not job:
            return False

        logger.info(f"Processing job {job.id}: {job.xrk_filename}")

        try:
            # Check if Windows service is healthy
            if not self._check_service_health():
                self.queue.mark_failed(job.id, "Windows service unavailable")
                return True

            # Process the job
            output_path = self._extract_file(job)

            if output_path:
                self.queue.mark_completed(job.id, str(output_path))
                logger.info(f"Job {job.id} completed: {output_path}")
            else:
                self.queue.mark_failed(job.id, "Extraction returned no output")
                logger.error(f"Job {job.id} failed: no output")

        except requests.RequestException as e:
            error_msg = f"Network error: {e}"
            logger.error(f"Job {job.id} failed: {error_msg}")
            self.queue.mark_failed(job.id, error_msg)

        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error(f"Job {job.id} failed: {error_msg}")
            self.queue.mark_failed(job.id, error_msg)

        return True

    def _check_service_health(self) -> bool:
        """Check if Windows extraction service is healthy"""
        try:
            response = requests.get(
                f"{self.windows_url}/health",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "healthy"
            return False
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    def _extract_file(self, job: ExtractionJob) -> Optional[Path]:
        """
        Extract an XRK file via the Windows service.

        Args:
            job: The extraction job

        Returns:
            Path to the extracted Parquet file, or None on failure
        """
        xrk_path = self.upload_dir / job.xrk_filename

        if not xrk_path.exists():
            raise FileNotFoundError(f"XRK file not found: {xrk_path}")

        # Upload file to Windows service
        logger.info(f"Uploading {xrk_path.name} to extraction service...")

        with open(xrk_path, 'rb') as f:
            files = {'file': (xrk_path.name, f, 'application/octet-stream')}
            response = requests.post(
                f"{self.windows_url}/extract",
                files=files,
                timeout=self.request_timeout
            )

        if response.status_code != 200:
            raise RuntimeError(f"Extraction failed: {response.text}")

        result = response.json()
        if not result.get('success'):
            raise RuntimeError(f"Extraction failed: {result.get('message')}")

        output_filename = result.get('output_filename')
        if not output_filename:
            raise RuntimeError("No output filename in response")

        # Download the Parquet file
        logger.info(f"Downloading {output_filename}...")

        download_response = requests.get(
            f"{self.windows_url}/download/{output_filename}",
            timeout=self.request_timeout,
            stream=True
        )

        if download_response.status_code != 200:
            raise RuntimeError(f"Download failed: {download_response.text}")

        # Save to output directory
        output_path = self.output_dir / output_filename

        with open(output_path, 'wb') as f:
            for chunk in download_response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Saved to {output_path}")
        return output_path

    def process_single(self, job_id: int) -> bool:
        """
        Process a single job by ID (for testing/manual processing).

        Args:
            job_id: The job ID to process

        Returns:
            True if successful, False otherwise
        """
        job = self.queue.get(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return False

        if job.status != JobStatus.PENDING:
            logger.warning(f"Job {job_id} not pending (status: {job.status})")
            return False

        # Mark as processing
        job.mark_processing()
        self.queue._update_job(job)

        try:
            output_path = self._extract_file(job)
            if output_path:
                self.queue.mark_completed(job_id, str(output_path))
                return True
        except Exception as e:
            self.queue.mark_failed(job_id, str(e))

        return False


def create_worker(
    db_path: str = "extraction_queue.db",
    windows_url: str = "http://localhost:8001",
    **kwargs
) -> ExtractionWorker:
    """
    Factory function to create a worker with queue.

    Args:
        db_path: Path to queue database
        windows_url: Windows service URL
        **kwargs: Additional arguments for ExtractionWorker

    Returns:
        Configured ExtractionWorker instance
    """
    queue = ExtractionQueue(db_path)
    return ExtractionWorker(queue, windows_url, **kwargs)


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="XRK Extraction Worker")
    parser.add_argument(
        "--windows-url",
        default="http://localhost:8001",
        help="URL of Windows extraction service"
    )
    parser.add_argument(
        "--db-path",
        default="data/extraction_queue.db",
        help="Path to queue database"
    )
    parser.add_argument(
        "--upload-dir",
        default="data/uploads",
        help="Directory containing XRK files"
    )
    parser.add_argument(
        "--output-dir",
        default="data/exports/processed",
        help="Directory for extracted Parquet files"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between queue polls"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="HTTP request timeout"
    )

    args = parser.parse_args()

    worker = create_worker(
        db_path=args.db_path,
        windows_url=args.windows_url,
        upload_dir=args.upload_dir,
        output_dir=args.output_dir,
        poll_interval=args.poll_interval,
        request_timeout=args.timeout
    )

    logger.info("Starting extraction worker...")
    worker.start()


if __name__ == "__main__":
    main()
