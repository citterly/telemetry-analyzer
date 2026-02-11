"""
Data models for analysis context.

Defines what data the user is currently analyzing.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ScopeMode(str, Enum):
    """Analysis scope mode."""
    SINGLE = "single"  # Single session analysis
    MULTI = "multi"  # Multiple sessions (comparison)
    FILTERED = "filtered"  # Cross-session query (e.g., all Road America Turn 3)


@dataclass
class AnalysisScope:
    """
    Defines what data to analyze.

    Three modes:
    1. Single: analyze one session (most common)
    2. Multi: compare multiple sessions (baseline vs current)
    3. Filtered: cross-session query (all data matching criteria)
    """
    mode: ScopeMode = ScopeMode.SINGLE
    session_ids: List[str] = field(default_factory=list)
    baseline_session_id: Optional[str] = None
    filters: Optional[Dict] = None  # For filtered mode (track, corner, date range, tags)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "session_ids": self.session_ids,
            "baseline_session_id": self.baseline_session_id,
            "filters": self.filters,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisScope":
        return cls(
            mode=ScopeMode(data.get("mode", "single")),
            session_ids=data.get("session_ids", []),
            baseline_session_id=data.get("baseline_session_id"),
            filters=data.get("filters"),
        )


@dataclass
class AnalysisContext:
    """
    User's current analysis state.

    Persists across page navigation within a browser session.
    """
    scope: AnalysisScope
    active_session_id: str  # Which session is "primary"
    created_at: datetime = field(default_factory=_now)
    last_accessed: datetime = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "scope": self.scope.to_dict(),
            "active_session_id": self.active_session_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisContext":
        def parse_dt(val):
            if val is None:
                return None
            if isinstance(val, datetime):
                return val
            return datetime.fromisoformat(val)

        return cls(
            scope=AnalysisScope.from_dict(data.get("scope", {})),
            active_session_id=data.get("active_session_id", ""),
            created_at=parse_dt(data.get("created_at")) or _now(),
            last_accessed=parse_dt(data.get("last_accessed")) or _now(),
        )
