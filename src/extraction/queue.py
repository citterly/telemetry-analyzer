"""
SQLite-backed extraction queue for XRK processing jobs
Survives restarts and supports retry logic
"""

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from contextlib import contextmanager

from .models import ExtractionJob, JobStatus


class ExtractionQueue:
    """
    SQLite-backed job queue for XRK extraction.

    Thread-safe and persistent across restarts.
    """

    def __init__(self, db_path: str = "extraction_queue.db"):
        """
        Initialize the queue with SQLite backend.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Create database tables if they don't exist"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS extraction_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    xrk_filename TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    error_message TEXT,
                    output_path TEXT,
                    priority INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON extraction_jobs(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_priority_created
                ON extraction_jobs(priority DESC, created_at ASC)
            """)
            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper cleanup"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _row_to_job(self, row: sqlite3.Row) -> ExtractionJob:
        """Convert database row to ExtractionJob"""
        return ExtractionJob.from_dict(dict(row))

    def submit(self, xrk_filename: str, priority: int = 0, max_retries: int = 3) -> ExtractionJob:
        """
        Submit a new extraction job to the queue.

        Args:
            xrk_filename: Path to the XRK file
            priority: Job priority (higher = more urgent)
            max_retries: Maximum retry attempts on failure

        Returns:
            The created ExtractionJob with assigned ID
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO extraction_jobs
                    (xrk_filename, status, created_at, updated_at, priority, max_retries)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (xrk_filename, JobStatus.PENDING.value, now, now, priority, max_retries))
                conn.commit()
                job_id = cursor.lastrowid

        return self.get(job_id)

    def get(self, job_id: int) -> Optional[ExtractionJob]:
        """
        Get a job by ID.

        Args:
            job_id: The job ID

        Returns:
            ExtractionJob or None if not found
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM extraction_jobs WHERE id = ?",
                (job_id,)
            ).fetchone()

            if row:
                return self._row_to_job(row)
            return None

    def get_status(self, job_id: int) -> Optional[JobStatus]:
        """
        Get just the status of a job.

        Args:
            job_id: The job ID

        Returns:
            JobStatus or None if not found
        """
        job = self.get(job_id)
        return job.status if job else None

    def get_next_pending(self) -> Optional[ExtractionJob]:
        """
        Get the next pending job (highest priority, oldest first).

        Returns:
            Next pending ExtractionJob or None if queue is empty
        """
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM extraction_jobs
                WHERE status = ?
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
            """, (JobStatus.PENDING.value,)).fetchone()

            if row:
                return self._row_to_job(row)
            return None

    def claim_next(self) -> Optional[ExtractionJob]:
        """
        Atomically claim the next pending job for processing.

        Returns:
            The claimed job (now in PROCESSING state) or None
        """
        with self._lock:
            job = self.get_next_pending()
            if job:
                job.mark_processing()
                self._update_job(job)
                return job
            return None

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ExtractionJob]:
        """
        List jobs, optionally filtered by status.

        Args:
            status: Filter by job status (None for all)
            limit: Maximum jobs to return
            offset: Number of jobs to skip

        Returns:
            List of ExtractionJob objects
        """
        with self._get_connection() as conn:
            if status:
                rows = conn.execute("""
                    SELECT * FROM extraction_jobs
                    WHERE status = ?
                    ORDER BY priority DESC, created_at DESC
                    LIMIT ? OFFSET ?
                """, (status.value, limit, offset)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM extraction_jobs
                    ORDER BY priority DESC, created_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset)).fetchall()

            return [self._row_to_job(row) for row in rows]

    def count(self, status: Optional[JobStatus] = None) -> int:
        """
        Count jobs, optionally filtered by status.

        Args:
            status: Filter by job status (None for all)

        Returns:
            Number of matching jobs
        """
        with self._get_connection() as conn:
            if status:
                result = conn.execute(
                    "SELECT COUNT(*) FROM extraction_jobs WHERE status = ?",
                    (status.value,)
                ).fetchone()
            else:
                result = conn.execute(
                    "SELECT COUNT(*) FROM extraction_jobs"
                ).fetchone()

            return result[0]

    def mark_completed(self, job_id: int, output_path: str) -> Optional[ExtractionJob]:
        """
        Mark a job as completed.

        Args:
            job_id: The job ID
            output_path: Path to the output Parquet file

        Returns:
            Updated ExtractionJob or None if not found
        """
        job = self.get(job_id)
        if job:
            job.mark_completed(output_path)
            self._update_job(job)
            return job
        return None

    def mark_failed(self, job_id: int, error: str) -> Optional[ExtractionJob]:
        """
        Mark a job as failed.

        Args:
            job_id: The job ID
            error: Error message describing the failure

        Returns:
            Updated ExtractionJob or None if not found
        """
        job = self.get(job_id)
        if job:
            job.mark_failed(error)
            self._update_job(job)
            return job
        return None

    def retry(self, job_id: int) -> Optional[ExtractionJob]:
        """
        Reset a failed job for retry.

        Args:
            job_id: The job ID

        Returns:
            Updated ExtractionJob or None if not found or can't retry
        """
        job = self.get(job_id)
        if job and job.can_retry():
            job.reset_for_retry()
            self._update_job(job)
            return job
        return None

    def retry_all_failed(self) -> int:
        """
        Reset all failed jobs that can be retried.

        Returns:
            Number of jobs reset for retry
        """
        failed_jobs = self.list_jobs(status=JobStatus.FAILED)
        count = 0
        for job in failed_jobs:
            if job.can_retry():
                job.reset_for_retry()
                self._update_job(job)
                count += 1
        return count

    def _update_job(self, job: ExtractionJob) -> None:
        """Update a job in the database"""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE extraction_jobs SET
                    status = ?,
                    updated_at = ?,
                    started_at = ?,
                    completed_at = ?,
                    retry_count = ?,
                    error_message = ?,
                    output_path = ?
                WHERE id = ?
            """, (
                job.status.value,
                job.updated_at.isoformat(),
                job.started_at.isoformat() if job.started_at else None,
                job.completed_at.isoformat() if job.completed_at else None,
                job.retry_count,
                job.error_message,
                job.output_path,
                job.id
            ))
            conn.commit()

    def delete(self, job_id: int) -> bool:
        """
        Delete a job from the queue.

        Args:
            job_id: The job ID

        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM extraction_jobs WHERE id = ?",
                (job_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def clear_completed(self) -> int:
        """
        Remove all completed jobs from the queue.

        Returns:
            Number of jobs removed
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM extraction_jobs WHERE status = ?",
                (JobStatus.COMPLETED.value,)
            )
            conn.commit()
            return cursor.rowcount

    def get_stats(self) -> dict:
        """
        Get queue statistics.

        Returns:
            Dictionary with counts by status
        """
        return {
            "pending": self.count(JobStatus.PENDING),
            "processing": self.count(JobStatus.PROCESSING),
            "completed": self.count(JobStatus.COMPLETED),
            "failed": self.count(JobStatus.FAILED),
            "total": self.count(),
        }
