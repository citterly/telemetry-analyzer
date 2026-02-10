"""
Queue management API router.

Extraction queue dashboard and job management endpoints.
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

from src.extraction.queue import ExtractionQueue
from src.extraction.models import JobStatus
from ..deps import config

router = APIRouter()

# Queue singleton
_extraction_queue = None


def get_queue() -> ExtractionQueue:
    """Get or create the extraction queue singleton."""
    global _extraction_queue
    if _extraction_queue is None:
        db_path = Path(config.DATA_DIR) / "extraction_queue.db"
        _extraction_queue = ExtractionQueue(str(db_path))
    return _extraction_queue


@router.get("/api/queue/stats")
async def get_queue_stats():
    """Get queue statistics"""
    queue = get_queue()
    return queue.get_stats()


@router.get("/api/queue/jobs")
async def list_queue_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """List jobs in the queue"""
    queue = get_queue()

    job_status = None
    if status and status != "all":
        try:
            job_status = JobStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    jobs = queue.list_jobs(status=job_status, limit=limit, offset=offset)
    return {
        "jobs": [job.to_dict() for job in jobs],
        "total": queue.count(job_status),
        "status_filter": status
    }


@router.get("/api/queue/jobs/{job_id}")
async def get_queue_job(job_id: int):
    """Get a specific job"""
    queue = get_queue()
    job = queue.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job.to_dict()


@router.post("/api/queue/jobs/{job_id}/retry")
async def retry_queue_job(job_id: int):
    """Retry a failed job"""
    queue = get_queue()
    job = queue.retry(job_id)
    if not job:
        raise HTTPException(status_code=400, detail=f"Cannot retry job {job_id} - not found or not eligible")
    return {"success": True, "job": job.to_dict()}


@router.post("/api/queue/retry-all")
async def retry_all_failed_jobs():
    """Retry all failed jobs that are eligible"""
    queue = get_queue()
    count = queue.retry_all_failed()
    return {"success": True, "retried_count": count}


@router.delete("/api/queue/jobs/{job_id}")
async def delete_queue_job(job_id: int):
    """Delete a job from the queue"""
    queue = get_queue()
    success = queue.delete(job_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return {"success": True}


@router.post("/api/queue/clear-completed")
async def clear_completed_jobs():
    """Remove all completed jobs from the queue"""
    queue = get_queue()
    count = queue.clear_completed()
    return {"success": True, "cleared_count": count}
