"""
Tests for arch-007 Phase 2: Enhanced session import wizard.
"""

import json
import pytest
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient

from src.main.app import app
from src.session.session_database import SessionDatabase
from src.session.models import Session, SessionType, ImportStatus


class TestEnhancedSessionImport:
    """Test enhanced session import with metadata."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        db = SessionDatabase(db_path)
        yield db

        Path(db_path).unlink()

    def test_confirm_with_enhanced_metadata(self, temp_db):
        """Test confirming a session with all enhanced metadata fields."""
        # Create a session
        session = Session(
            parquet_path="/test/session.parquet",
            file_hash="test123",
            track_id="road-america",
            vehicle_id="mustang-gt-2019",
        )
        created = temp_db.create_session(session)

        # Update with enhanced metadata (simulating confirm endpoint)
        created.import_status = ImportStatus.CONFIRMED
        created.session_type = SessionType.PRACTICE
        created.session_date = "2025-06-15T14:30:00"
        created.driver_name = "Chris"
        created.run_number = 3
        created.weather_conditions = "Dry, 75F, Partly Cloudy"
        created.track_conditions = "Rubbered In"
        created.tire_pressures = {"FL": 32, "FR": 32, "RL": 28, "RR": 28}
        created.tags = ["baseline", "new-dampers"]
        created.notes = "First session with new Penske shocks"

        updated = temp_db.update_session(created)

        # Verify all fields persisted
        retrieved = temp_db.get_session(updated.id)
        assert retrieved.import_status == ImportStatus.CONFIRMED
        assert retrieved.driver_name == "Chris"
        assert retrieved.run_number == 3
        assert retrieved.weather_conditions == "Dry, 75F, Partly Cloudy"
        assert retrieved.track_conditions == "Rubbered In"
        assert retrieved.tire_pressures["FL"] == 32
        assert retrieved.tags == ["baseline", "new-dampers"]
        assert retrieved.notes == "First session with new Penske shocks"

    def test_confirm_endpoint_accepts_enhanced_fields(self, client, monkeypatch):
        """Test that confirm endpoint accepts and stores enhanced metadata."""
        # Create a test session
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            db = SessionDatabase(db_path)

            # Create session
            session = Session(
                parquet_path="/test/session.parquet",
                file_hash="test456",
                vehicle_id="mustang-gt-2019",
            )
            created = db.create_session(session)

            # Mock the session database
            def mock_get_session_db():
                return db

            import src.main.routers.sessions as sessions_module
            monkeypatch.setattr(sessions_module, "get_session_db", mock_get_session_db)

            # Confirm with enhanced metadata
            response = client.put(
                f"/api/v2/sessions/{created.id}/confirm",
                json={
                    "session_type": "practice",
                    "session_date": "2025-06-15T14:30:00",
                    "driver_name": "Chris",
                    "run_number": 3,
                    "weather_conditions": "Dry, 75F",
                    "track_conditions": "Rubbered In",
                    "tire_pressures": {"FL": 32, "FR": 32, "RL": 28, "RR": 28},
                    "tags": ["baseline", "test"],
                    "notes": "Test session",
                }
            )

            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "ok"
            assert result["session"]["driver_name"] == "Chris"
            assert result["session"]["run_number"] == 3
            assert result["session"]["tags"] == ["baseline", "test"]

        finally:
            Path(db_path).unlink()

    def test_drivers_recent_endpoint(self, client, monkeypatch):
        """Test the recent drivers autocomplete endpoint."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            db = SessionDatabase(db_path)

            # Create sessions with different drivers
            for i, driver in enumerate(["Chris", "John", "Sarah", "Chris"]):
                session = Session(
                    parquet_path=f"/test/session{i}.parquet",
                    file_hash=f"hash{i}",
                    driver_name=driver,
                )
                db.create_session(session)

            # Mock the session database
            def mock_get_session_db():
                return db

            import src.main.routers.sessions as sessions_module
            monkeypatch.setattr(sessions_module, "get_session_db", mock_get_session_db)

            # Get recent drivers
            response = client.get("/api/v2/drivers/recent")

            assert response.status_code == 200
            result = response.json()
            drivers = result["drivers"]

            # Should have 3 unique drivers (Chris appears twice but only listed once)
            assert len(drivers) == 3
            assert "Chris" in drivers
            assert "John" in drivers
            assert "Sarah" in drivers

        finally:
            Path(db_path).unlink()

    def test_setup_snapshot_captured(self, temp_db):
        """Test that setup snapshot can be stored and retrieved."""
        session = Session(
            parquet_path="/test/session.parquet",
            file_hash="test789",
            vehicle_id="mustang-gt-2019",
            setup_snapshot={
                "name": "2024 Gearing",
                "transmission_ratios": [2.66, 1.78, 1.30, 1.00, 0.80, 0.63],
                "final_drive": 3.73,
                "weight_lbs": 3450,
            },
        )

        created = temp_db.create_session(session)
        retrieved = temp_db.get_session(created.id)

        assert retrieved.setup_snapshot is not None
        assert retrieved.setup_snapshot["name"] == "2024 Gearing"
        assert retrieved.setup_snapshot["final_drive"] == 3.73


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
