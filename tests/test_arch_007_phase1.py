"""
Tests for arch-007 Phase 1: Backend context system and enhanced session metadata.
"""

import json
import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from src.context import AnalysisContext, AnalysisScope, ScopeMode, ContextStorage
from src.session.models import Session, SessionType, ImportStatus
from src.session.session_database import SessionDatabase


class TestAnalysisScope:
    """Test AnalysisScope model."""

    def test_single_mode(self):
        scope = AnalysisScope(
            mode=ScopeMode.SINGLE,
            session_ids=["session-123"],
        )
        assert scope.mode == ScopeMode.SINGLE
        assert scope.session_ids == ["session-123"]
        assert scope.baseline_session_id is None

    def test_multi_mode_with_baseline(self):
        scope = AnalysisScope(
            mode=ScopeMode.MULTI,
            session_ids=["session-123", "session-456"],
            baseline_session_id="session-789",
        )
        assert scope.mode == ScopeMode.MULTI
        assert len(scope.session_ids) == 2
        assert scope.baseline_session_id == "session-789"

    def test_to_dict(self):
        scope = AnalysisScope(
            mode=ScopeMode.SINGLE,
            session_ids=["session-123"],
        )
        data = scope.to_dict()
        assert data["mode"] == "single"
        assert data["session_ids"] == ["session-123"]

    def test_from_dict(self):
        data = {
            "mode": "multi",
            "session_ids": ["s1", "s2"],
            "baseline_session_id": "s3",
        }
        scope = AnalysisScope.from_dict(data)
        assert scope.mode == ScopeMode.MULTI
        assert scope.session_ids == ["s1", "s2"]
        assert scope.baseline_session_id == "s3"


class TestAnalysisContext:
    """Test AnalysisContext model."""

    def test_creation(self):
        scope = AnalysisScope(mode=ScopeMode.SINGLE, session_ids=["s1"])
        context = AnalysisContext(scope=scope, active_session_id="s1")

        assert context.scope == scope
        assert context.active_session_id == "s1"
        assert context.created_at is not None
        assert context.last_accessed is not None

    def test_to_dict(self):
        scope = AnalysisScope(mode=ScopeMode.SINGLE, session_ids=["s1"])
        context = AnalysisContext(scope=scope, active_session_id="s1")

        data = context.to_dict()
        assert data["active_session_id"] == "s1"
        assert data["scope"]["mode"] == "single"
        assert "created_at" in data
        assert "last_accessed" in data

    def test_from_dict(self):
        data = {
            "scope": {
                "mode": "single",
                "session_ids": ["s1"],
                "baseline_session_id": None,
                "filters": None,
            },
            "active_session_id": "s1",
            "created_at": "2025-01-15T10:00:00+00:00",
            "last_accessed": "2025-01-15T10:30:00+00:00",
        }
        context = AnalysisContext.from_dict(data)
        assert context.active_session_id == "s1"
        assert context.scope.mode == ScopeMode.SINGLE


class TestEnhancedSessionMetadata:
    """Test Session model with enhanced metadata fields."""

    def test_session_with_enhanced_fields(self):
        session = Session(
            parquet_path="/path/to/session.parquet",
            track_id="road-america",
            vehicle_id="mustang-gt-2019",
            driver_name="Chris",
            run_number=3,
            weather_conditions="Dry, 75F, Partly Cloudy",
            track_conditions="Rubbered In",
            setup_snapshot={"name": "2024 Gearing", "weight_lbs": 3450},
            tire_pressures={"FL": 32, "FR": 32, "RL": 28, "RR": 28},
            tags=["baseline", "new-dampers"],
        )

        assert session.driver_name == "Chris"
        assert session.run_number == 3
        assert session.weather_conditions == "Dry, 75F, Partly Cloudy"
        assert session.track_conditions == "Rubbered In"
        assert session.setup_snapshot["name"] == "2024 Gearing"
        assert session.tire_pressures["FL"] == 32
        assert session.tags == ["baseline", "new-dampers"]

    def test_session_to_dict_with_metadata(self):
        session = Session(
            parquet_path="/path/to/session.parquet",
            driver_name="Chris",
            run_number=2,
            tags=["test", "baseline"],
        )

        data = session.to_dict()
        assert data["driver_name"] == "Chris"
        assert data["run_number"] == 2
        assert data["tags"] == ["test", "baseline"]

    def test_session_from_dict_with_json_fields(self):
        """Test parsing JSON fields stored as strings (from database)."""
        data = {
            "id": 1,
            "parquet_path": "/path/to/session.parquet",
            "setup_snapshot": '{"name": "Test Setup", "weight_lbs": 3400}',
            "tire_pressures": '{"FL": 30, "FR": 30}',
            "tags": '["tag1", "tag2"]',
        }

        session = Session.from_dict(data)
        assert session.setup_snapshot["name"] == "Test Setup"
        assert session.tire_pressures["FL"] == 30
        assert session.tags == ["tag1", "tag2"]

    def test_session_from_dict_with_dict_fields(self):
        """Test parsing JSON fields that are already dicts/lists."""
        data = {
            "id": 1,
            "parquet_path": "/path/to/session.parquet",
            "setup_snapshot": {"name": "Test Setup"},
            "tire_pressures": {"FL": 30},
            "tags": ["tag1", "tag2"],
        }

        session = Session.from_dict(data)
        assert session.setup_snapshot["name"] == "Test Setup"
        assert session.tire_pressures["FL"] == 30
        assert session.tags == ["tag1", "tag2"]


class TestSessionDatabaseWithMetadata:
    """Test SessionDatabase with enhanced metadata fields."""

    def test_create_and_retrieve_session_with_metadata(self):
        # Use temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            db = SessionDatabase(db_path)

            # Create session with enhanced metadata
            session = Session(
                parquet_path="/test/session.parquet",
                file_hash="abc123",
                track_id="road-america",
                vehicle_id="mustang-gt-2019",
                session_date="2025-06-15T14:30:00",
                session_type=SessionType.PRACTICE,
                import_status=ImportStatus.CONFIRMED,
                driver_name="Chris",
                run_number=3,
                weather_conditions="Dry, 75F",
                track_conditions="Rubbered In",
                setup_snapshot={"name": "2024 Gearing", "weight_lbs": 3450},
                tire_pressures={"FL": 32, "FR": 32, "RL": 28, "RR": 28},
                tags=["baseline", "new-dampers"],
            )

            created = db.create_session(session)
            assert created.id is not None

            # Retrieve and verify
            retrieved = db.get_session(created.id)
            assert retrieved is not None
            assert retrieved.driver_name == "Chris"
            assert retrieved.run_number == 3
            assert retrieved.weather_conditions == "Dry, 75F"
            assert retrieved.track_conditions == "Rubbered In"
            assert retrieved.setup_snapshot["name"] == "2024 Gearing"
            assert retrieved.tire_pressures["FL"] == 32
            assert retrieved.tags == ["baseline", "new-dampers"]

        finally:
            Path(db_path).unlink()

    def test_update_session_with_metadata(self):
        # Use temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            db = SessionDatabase(db_path)

            # Create session
            session = Session(
                parquet_path="/test/session.parquet",
                file_hash="abc123",
                driver_name="Driver A",
                tags=["tag1"],
            )
            created = db.create_session(session)

            # Update with new metadata
            created.driver_name = "Driver B"
            created.run_number = 5
            created.tags = ["tag1", "tag2", "tag3"]

            updated = db.update_session(created)

            # Retrieve and verify
            retrieved = db.get_session(updated.id)
            assert retrieved.driver_name == "Driver B"
            assert retrieved.run_number == 5
            assert retrieved.tags == ["tag1", "tag2", "tag3"]

        finally:
            Path(db_path).unlink()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
