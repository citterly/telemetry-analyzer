"""
Data models for extraction queue
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import json


class JobStatus(str, Enum):
    """Status of an extraction job"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExtractionJob:
    """Represents an XRK extraction job in the queue"""

    id: Optional[int] = None
    xrk_filename: str = ""
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    output_path: Optional[str] = None
    priority: int = 0  # Higher = more urgent

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "xrk_filename": self.xrk_filename,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "error_message": self.error_message,
            "output_path": self.output_path,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractionJob":
        """Create from dictionary"""
        def parse_dt(val):
            if val is None:
                return None
            if isinstance(val, datetime):
                return val
            return datetime.fromisoformat(val)

        return cls(
            id=data.get("id"),
            xrk_filename=data.get("xrk_filename", ""),
            status=JobStatus(data.get("status", "pending")),
            created_at=parse_dt(data.get("created_at")) or datetime.utcnow(),
            updated_at=parse_dt(data.get("updated_at")) or datetime.utcnow(),
            started_at=parse_dt(data.get("started_at")),
            completed_at=parse_dt(data.get("completed_at")),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            error_message=data.get("error_message"),
            output_path=data.get("output_path"),
            priority=data.get("priority", 0),
        )

    def can_retry(self) -> bool:
        """Check if job can be retried"""
        return self.retry_count < self.max_retries and self.status == JobStatus.FAILED

    def mark_processing(self) -> None:
        """Mark job as processing"""
        self.status = JobStatus.PROCESSING
        self.started_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def mark_completed(self, output_path: str) -> None:
        """Mark job as completed"""
        self.status = JobStatus.COMPLETED
        self.output_path = output_path
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.error_message = None

    def mark_failed(self, error: str) -> None:
        """Mark job as failed"""
        self.status = JobStatus.FAILED
        self.error_message = error
        self.retry_count += 1
        self.updated_at = datetime.utcnow()

    def reset_for_retry(self) -> None:
        """Reset job for retry attempt"""
        self.status = JobStatus.PENDING
        self.started_at = None
        self.updated_at = datetime.utcnow()
