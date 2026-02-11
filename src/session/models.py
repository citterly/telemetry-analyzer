"""
Data models for session management.

Dataclasses for sessions, laps, stints, and setups stored in the session database.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import json


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SessionType(str, Enum):
    PRACTICE = "practice"
    QUALIFYING = "qualifying"
    RACE = "race"
    TEST = "test"
    UNKNOWN = "unknown"


class ImportStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"


class LapClassification(str, Enum):
    OUT_LAP = "out_lap"
    IN_LAP = "in_lap"
    WARM_UP = "warm_up"
    COOL_DOWN = "cool_down"
    HOT_LAP = "hot_lap"
    NORMAL = "normal"
    RACE_PACE = "race_pace"
    INCOMPLETE = "incomplete"


@dataclass
class Session:
    """A telemetry session imported from a Parquet file."""

    id: Optional[int] = None
    parquet_path: str = ""
    file_hash: str = ""
    track_id: Optional[str] = None
    track_name: Optional[str] = None
    track_confidence: float = 0.0
    vehicle_id: Optional[str] = None
    session_date: Optional[str] = None
    session_type: SessionType = SessionType.UNKNOWN
    import_status: ImportStatus = ImportStatus.PENDING
    total_laps: int = 0
    best_lap_time: Optional[float] = None
    total_duration: Optional[float] = None
    notes: str = ""
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)

    # Enhanced metadata fields (arch-007)
    driver_name: Optional[str] = None
    run_number: Optional[int] = None
    weather_conditions: Optional[str] = None
    track_conditions: Optional[str] = None
    setup_snapshot: Optional[Dict[str, Any]] = None
    tire_pressures: Optional[Dict[str, float]] = None
    tags: List[str] = field(default_factory=list)
    last_accessed: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "parquet_path": self.parquet_path,
            "file_hash": self.file_hash,
            "track_id": self.track_id,
            "track_name": self.track_name,
            "track_confidence": self.track_confidence,
            "vehicle_id": self.vehicle_id,
            "session_date": self.session_date,
            "session_type": self.session_type.value,
            "import_status": self.import_status.value,
            "total_laps": self.total_laps,
            "best_lap_time": self.best_lap_time,
            "total_duration": self.total_duration,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            # Enhanced metadata
            "driver_name": self.driver_name,
            "run_number": self.run_number,
            "weather_conditions": self.weather_conditions,
            "track_conditions": self.track_conditions,
            "setup_snapshot": self.setup_snapshot,
            "tire_pressures": self.tire_pressures,
            "tags": self.tags,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        def parse_dt(val):
            if val is None:
                return None
            if isinstance(val, datetime):
                return val
            return datetime.fromisoformat(val)

        def parse_json_field(val):
            """Parse JSON fields that may be stored as strings in the database."""
            if val is None:
                return None
            if isinstance(val, str):
                return json.loads(val)
            return val

        def parse_tags(val):
            """Parse tags field (may be JSON array string or list)."""
            if val is None:
                return []
            if isinstance(val, str):
                return json.loads(val)
            if isinstance(val, list):
                return val
            return []

        return cls(
            id=data.get("id"),
            parquet_path=data.get("parquet_path", ""),
            file_hash=data.get("file_hash", ""),
            track_id=data.get("track_id"),
            track_name=data.get("track_name"),
            track_confidence=data.get("track_confidence", 0.0),
            vehicle_id=data.get("vehicle_id"),
            session_date=data.get("session_date"),
            session_type=SessionType(data.get("session_type", "unknown")),
            import_status=ImportStatus(data.get("import_status", "pending")),
            total_laps=data.get("total_laps", 0),
            best_lap_time=data.get("best_lap_time"),
            total_duration=data.get("total_duration"),
            notes=data.get("notes", ""),
            created_at=parse_dt(data.get("created_at")) or _now(),
            updated_at=parse_dt(data.get("updated_at")) or _now(),
            # Enhanced metadata
            driver_name=data.get("driver_name"),
            run_number=data.get("run_number"),
            weather_conditions=data.get("weather_conditions"),
            track_conditions=data.get("track_conditions"),
            setup_snapshot=parse_json_field(data.get("setup_snapshot")),
            tire_pressures=parse_json_field(data.get("tire_pressures")),
            tags=parse_tags(data.get("tags")),
            last_accessed=parse_dt(data.get("last_accessed")),
        )


@dataclass
class Lap:
    """A single lap within a session."""

    id: Optional[int] = None
    session_id: int = 0
    lap_number: int = 0
    stint_number: int = 0
    lap_time: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    start_index: int = 0
    end_index: int = 0
    classification: LapClassification = LapClassification.NORMAL
    classification_confidence: float = 0.0
    user_override: bool = False
    max_speed_mph: float = 0.0
    max_rpm: float = 0.0
    avg_rpm: float = 0.0
    sample_count: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "lap_number": self.lap_number,
            "stint_number": self.stint_number,
            "lap_time": self.lap_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "classification": self.classification.value,
            "classification_confidence": self.classification_confidence,
            "user_override": self.user_override,
            "max_speed_mph": self.max_speed_mph,
            "max_rpm": self.max_rpm,
            "avg_rpm": self.avg_rpm,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Lap":
        return cls(
            id=data.get("id"),
            session_id=data.get("session_id", 0),
            lap_number=data.get("lap_number", 0),
            stint_number=data.get("stint_number", 0),
            lap_time=data.get("lap_time", 0.0),
            start_time=data.get("start_time", 0.0),
            end_time=data.get("end_time", 0.0),
            start_index=data.get("start_index", 0),
            end_index=data.get("end_index", 0),
            classification=LapClassification(data.get("classification", "normal")),
            classification_confidence=data.get("classification_confidence", 0.0),
            user_override=bool(data.get("user_override", False)),
            max_speed_mph=data.get("max_speed_mph", 0.0),
            max_rpm=data.get("max_rpm", 0.0),
            avg_rpm=data.get("avg_rpm", 0.0),
            sample_count=data.get("sample_count", 0),
        )


@dataclass
class Stint:
    """A group of consecutive laps between pit stops."""

    id: Optional[int] = None
    session_id: int = 0
    stint_number: int = 0
    start_lap: int = 0
    end_lap: int = 0
    lap_count: int = 0
    best_lap_time: Optional[float] = None
    avg_lap_time: Optional[float] = None
    start_time: float = 0.0
    end_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "stint_number": self.stint_number,
            "start_lap": self.start_lap,
            "end_lap": self.end_lap,
            "lap_count": self.lap_count,
            "best_lap_time": self.best_lap_time,
            "avg_lap_time": self.avg_lap_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Stint":
        return cls(
            id=data.get("id"),
            session_id=data.get("session_id", 0),
            stint_number=data.get("stint_number", 0),
            start_lap=data.get("start_lap", 0),
            end_lap=data.get("end_lap", 0),
            lap_count=data.get("lap_count", 0),
            best_lap_time=data.get("best_lap_time"),
            avg_lap_time=data.get("avg_lap_time"),
            start_time=data.get("start_time", 0.0),
            end_time=data.get("end_time", 0.0),
        )


@dataclass
class Setup:
    """Setup data for a session (pre or post session)."""

    id: Optional[int] = None
    session_id: int = 0
    setup_point: str = "pre"  # "pre" or "post"
    setup_data: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    created_at: datetime = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "setup_point": self.setup_point,
            "setup_data": self.setup_data,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Setup":
        def parse_dt(val):
            if val is None:
                return None
            if isinstance(val, datetime):
                return val
            return datetime.fromisoformat(val)

        setup_data = data.get("setup_data", {})
        if isinstance(setup_data, str):
            setup_data = json.loads(setup_data)

        return cls(
            id=data.get("id"),
            session_id=data.get("session_id", 0),
            setup_point=data.get("setup_point", "pre"),
            setup_data=setup_data,
            notes=data.get("notes", ""),
            created_at=parse_dt(data.get("created_at")) or _now(),
        )


@dataclass
class LapClassificationResult:
    """Result of auto-classifying a single lap."""

    lap_number: int
    classification: LapClassification
    confidence: float
    stint_number: int
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "lap_number": self.lap_number,
            "classification": self.classification.value,
            "confidence": self.confidence,
            "stint_number": self.stint_number,
            "flags": self.flags,
        }
