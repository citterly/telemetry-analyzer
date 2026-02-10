"""
Tests for session database module
"""

import os
import sys
import tempfile
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.session.models import (
    Session,
    Lap,
    Stint,
    Setup,
    SessionType,
    ImportStatus,
    LapClassification,
)
from src.session.session_database import SessionDatabase


class TestSessionModels:
    """Tests for Session model data classes"""

    def test_session_to_dict(self):
        """Test Session serialization to dict"""
        session = Session(
            id=1,
            parquet_path="/data/session1.parquet",
            file_hash="abc123",
            track_id="track_001",
            track_name="Laguna Seca",
            track_confidence=0.95,
            vehicle_id="veh_001",
            session_date="2024-01-15",
            session_type=SessionType.RACE,
            import_status=ImportStatus.CONFIRMED,
            total_laps=25,
            best_lap_time=92.5,
            total_duration=2500.0,
            notes="Test session",
        )
        data = session.to_dict()

        assert data["id"] == 1
        assert data["parquet_path"] == "/data/session1.parquet"
        assert data["file_hash"] == "abc123"
        assert data["track_id"] == "track_001"
        assert data["track_name"] == "Laguna Seca"
        assert data["track_confidence"] == 0.95
        assert data["vehicle_id"] == "veh_001"
        assert data["session_date"] == "2024-01-15"
        assert data["session_type"] == "race"
        assert data["import_status"] == "confirmed"
        assert data["total_laps"] == 25
        assert data["best_lap_time"] == 92.5
        assert data["total_duration"] == 2500.0
        assert data["notes"] == "Test session"

    def test_session_from_dict(self):
        """Test Session deserialization from dict"""
        data = {
            "id": 1,
            "parquet_path": "/data/session1.parquet",
            "file_hash": "abc123",
            "track_id": "track_001",
            "track_name": "Laguna Seca",
            "track_confidence": 0.95,
            "vehicle_id": "veh_001",
            "session_date": "2024-01-15",
            "session_type": "race",
            "import_status": "confirmed",
            "total_laps": 25,
            "best_lap_time": 92.5,
            "total_duration": 2500.0,
            "notes": "Test session",
            "created_at": "2024-01-15T10:00:00+00:00",
            "updated_at": "2024-01-15T10:00:00+00:00",
        }
        session = Session.from_dict(data)

        assert session.id == 1
        assert session.parquet_path == "/data/session1.parquet"
        assert session.file_hash == "abc123"
        assert session.track_id == "track_001"
        assert session.track_name == "Laguna Seca"
        assert session.track_confidence == 0.95
        assert session.vehicle_id == "veh_001"
        assert session.session_date == "2024-01-15"
        assert session.session_type == SessionType.RACE
        assert session.import_status == ImportStatus.CONFIRMED
        assert session.total_laps == 25
        assert session.best_lap_time == 92.5
        assert session.total_duration == 2500.0
        assert session.notes == "Test session"

    def test_lap_to_dict(self):
        """Test Lap serialization to dict"""
        lap = Lap(
            id=1,
            session_id=5,
            lap_number=3,
            stint_number=1,
            lap_time=92.5,
            start_time=150.0,
            end_time=242.5,
            start_index=1500,
            end_index=2425,
            classification=LapClassification.HOT_LAP,
            classification_confidence=0.9,
            user_override=True,
            max_speed_mph=145.5,
            max_rpm=7500.0,
            avg_rpm=6200.0,
            sample_count=925,
        )
        data = lap.to_dict()

        assert data["id"] == 1
        assert data["session_id"] == 5
        assert data["lap_number"] == 3
        assert data["stint_number"] == 1
        assert data["lap_time"] == 92.5
        assert data["start_time"] == 150.0
        assert data["end_time"] == 242.5
        assert data["start_index"] == 1500
        assert data["end_index"] == 2425
        assert data["classification"] == "hot_lap"
        assert data["classification_confidence"] == 0.9
        assert data["user_override"] is True
        assert data["max_speed_mph"] == 145.5
        assert data["max_rpm"] == 7500.0
        assert data["avg_rpm"] == 6200.0
        assert data["sample_count"] == 925

    def test_lap_from_dict(self):
        """Test Lap deserialization from dict"""
        data = {
            "id": 1,
            "session_id": 5,
            "lap_number": 3,
            "stint_number": 1,
            "lap_time": 92.5,
            "start_time": 150.0,
            "end_time": 242.5,
            "start_index": 1500,
            "end_index": 2425,
            "classification": "hot_lap",
            "classification_confidence": 0.9,
            "user_override": 1,
            "max_speed_mph": 145.5,
            "max_rpm": 7500.0,
            "avg_rpm": 6200.0,
            "sample_count": 925,
        }
        lap = Lap.from_dict(data)

        assert lap.id == 1
        assert lap.session_id == 5
        assert lap.lap_number == 3
        assert lap.stint_number == 1
        assert lap.lap_time == 92.5
        assert lap.start_time == 150.0
        assert lap.end_time == 242.5
        assert lap.start_index == 1500
        assert lap.end_index == 2425
        assert lap.classification == LapClassification.HOT_LAP
        assert lap.classification_confidence == 0.9
        assert lap.user_override is True
        assert lap.max_speed_mph == 145.5
        assert lap.max_rpm == 7500.0
        assert lap.avg_rpm == 6200.0
        assert lap.sample_count == 925

    def test_stint_to_dict(self):
        """Test Stint serialization to dict"""
        stint = Stint(
            id=1,
            session_id=5,
            stint_number=2,
            start_lap=6,
            end_lap=12,
            lap_count=7,
            best_lap_time=91.2,
            avg_lap_time=93.5,
            start_time=500.0,
            end_time=1150.0,
        )
        data = stint.to_dict()

        assert data["id"] == 1
        assert data["session_id"] == 5
        assert data["stint_number"] == 2
        assert data["start_lap"] == 6
        assert data["end_lap"] == 12
        assert data["lap_count"] == 7
        assert data["best_lap_time"] == 91.2
        assert data["avg_lap_time"] == 93.5
        assert data["start_time"] == 500.0
        assert data["end_time"] == 1150.0

    def test_stint_from_dict(self):
        """Test Stint deserialization from dict"""
        data = {
            "id": 1,
            "session_id": 5,
            "stint_number": 2,
            "start_lap": 6,
            "end_lap": 12,
            "lap_count": 7,
            "best_lap_time": 91.2,
            "avg_lap_time": 93.5,
            "start_time": 500.0,
            "end_time": 1150.0,
        }
        stint = Stint.from_dict(data)

        assert stint.id == 1
        assert stint.session_id == 5
        assert stint.stint_number == 2
        assert stint.start_lap == 6
        assert stint.end_lap == 12
        assert stint.lap_count == 7
        assert stint.best_lap_time == 91.2
        assert stint.avg_lap_time == 93.5
        assert stint.start_time == 500.0
        assert stint.end_time == 1150.0

    def test_setup_to_dict(self):
        """Test Setup serialization to dict"""
        setup = Setup(
            id=1,
            session_id=5,
            setup_point="pre",
            setup_data={"front_pressure": 32.0, "rear_pressure": 30.0},
            notes="Cold tire pressures",
        )
        data = setup.to_dict()

        assert data["id"] == 1
        assert data["session_id"] == 5
        assert data["setup_point"] == "pre"
        assert data["setup_data"]["front_pressure"] == 32.0
        assert data["setup_data"]["rear_pressure"] == 30.0
        assert data["notes"] == "Cold tire pressures"

    def test_setup_from_dict(self):
        """Test Setup deserialization from dict"""
        data = {
            "id": 1,
            "session_id": 5,
            "setup_point": "pre",
            "setup_data": {"front_pressure": 32.0, "rear_pressure": 30.0},
            "notes": "Cold tire pressures",
            "created_at": "2024-01-15T10:00:00+00:00",
        }
        setup = Setup.from_dict(data)

        assert setup.id == 1
        assert setup.session_id == 5
        assert setup.setup_point == "pre"
        assert setup.setup_data["front_pressure"] == 32.0
        assert setup.setup_data["rear_pressure"] == 30.0
        assert setup.notes == "Cold tire pressures"

    def test_setup_from_dict_json_string(self):
        """Test Setup deserialization from dict with JSON string setup_data"""
        data = {
            "id": 1,
            "session_id": 5,
            "setup_point": "pre",
            "setup_data": '{"front_pressure": 32.0, "rear_pressure": 30.0}',
            "notes": "Cold tire pressures",
            "created_at": "2024-01-15T10:00:00+00:00",
        }
        setup = Setup.from_dict(data)

        assert setup.setup_data["front_pressure"] == 32.0
        assert setup.setup_data["rear_pressure"] == 30.0


class TestSessionDatabase:
    """Tests for SessionDatabase CRUD operations"""

    @pytest.fixture
    def db(self):
        """Create a temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        database = SessionDatabase(db_path)
        yield database
        # Cleanup
        try:
            os.unlink(db_path)
        except:
            pass

    # ------------------------------------------------------------------
    # Session tests
    # ------------------------------------------------------------------

    def test_create_session(self, db):
        """Test creating a session"""
        session = Session(
            parquet_path="/data/session1.parquet",
            file_hash="abc123",
            track_id="track_001",
            track_name="Laguna Seca",
            vehicle_id="veh_001",
            session_type=SessionType.RACE,
        )
        created = db.create_session(session)

        assert created.id is not None
        assert created.parquet_path == "/data/session1.parquet"
        assert created.file_hash == "abc123"
        assert created.track_id == "track_001"
        assert created.track_name == "Laguna Seca"
        assert created.vehicle_id == "veh_001"
        assert created.session_type == SessionType.RACE

    def test_get_session(self, db):
        """Test retrieving a session by ID"""
        session = Session(
            parquet_path="/data/session1.parquet",
            track_name="Laguna Seca",
        )
        created = db.create_session(session)

        retrieved = db.get_session(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.parquet_path == "/data/session1.parquet"
        assert retrieved.track_name == "Laguna Seca"

    def test_get_session_not_found(self, db):
        """Test retrieving a nonexistent session"""
        session = db.get_session(9999)
        assert session is None

    def test_get_session_by_parquet_path(self, db):
        """Test retrieving a session by parquet path"""
        session = Session(
            parquet_path="/data/unique_session.parquet",
            track_name="Sonoma",
        )
        db.create_session(session)

        retrieved = db.get_session_by_parquet_path("/data/unique_session.parquet")
        assert retrieved is not None
        assert retrieved.parquet_path == "/data/unique_session.parquet"
        assert retrieved.track_name == "Sonoma"

    def test_get_session_by_parquet_path_not_found(self, db):
        """Test retrieving a session by nonexistent parquet path"""
        session = db.get_session_by_parquet_path("/nonexistent.parquet")
        assert session is None

    def test_get_session_by_hash(self, db):
        """Test retrieving a session by file hash"""
        session = Session(
            parquet_path="/data/session1.parquet",
            file_hash="unique_hash_123",
        )
        db.create_session(session)

        retrieved = db.get_session_by_hash("unique_hash_123")
        assert retrieved is not None
        assert retrieved.file_hash == "unique_hash_123"

    def test_get_session_by_hash_not_found(self, db):
        """Test retrieving a session by nonexistent hash"""
        session = db.get_session_by_hash("nonexistent_hash")
        assert session is None

    def test_list_sessions(self, db):
        """Test listing all sessions"""
        db.create_session(Session(parquet_path="/data/s1.parquet"))
        db.create_session(Session(parquet_path="/data/s2.parquet"))
        db.create_session(Session(parquet_path="/data/s3.parquet"))

        sessions = db.list_sessions()
        assert len(sessions) == 3

    def test_list_sessions_by_track(self, db):
        """Test listing sessions filtered by track"""
        db.create_session(Session(
            parquet_path="/data/s1.parquet",
            track_id="track_001",
        ))
        db.create_session(Session(
            parquet_path="/data/s2.parquet",
            track_id="track_001",
        ))
        db.create_session(Session(
            parquet_path="/data/s3.parquet",
            track_id="track_002",
        ))

        sessions = db.list_sessions(track_id="track_001")
        assert len(sessions) == 2
        assert all(s.track_id == "track_001" for s in sessions)

    def test_list_sessions_by_vehicle(self, db):
        """Test listing sessions filtered by vehicle"""
        db.create_session(Session(
            parquet_path="/data/s1.parquet",
            vehicle_id="veh_001",
        ))
        db.create_session(Session(
            parquet_path="/data/s2.parquet",
            vehicle_id="veh_002",
        ))

        sessions = db.list_sessions(vehicle_id="veh_001")
        assert len(sessions) == 1
        assert sessions[0].vehicle_id == "veh_001"

    def test_list_sessions_by_session_type(self, db):
        """Test listing sessions filtered by session type"""
        db.create_session(Session(
            parquet_path="/data/s1.parquet",
            session_type=SessionType.RACE,
        ))
        db.create_session(Session(
            parquet_path="/data/s2.parquet",
            session_type=SessionType.PRACTICE,
        ))
        db.create_session(Session(
            parquet_path="/data/s3.parquet",
            session_type=SessionType.RACE,
        ))

        sessions = db.list_sessions(session_type=SessionType.RACE)
        assert len(sessions) == 2
        assert all(s.session_type == SessionType.RACE for s in sessions)

    def test_list_sessions_by_import_status(self, db):
        """Test listing sessions filtered by import status"""
        db.create_session(Session(
            parquet_path="/data/s1.parquet",
            import_status=ImportStatus.CONFIRMED,
        ))
        db.create_session(Session(
            parquet_path="/data/s2.parquet",
            import_status=ImportStatus.PENDING,
        ))
        db.create_session(Session(
            parquet_path="/data/s3.parquet",
            import_status=ImportStatus.CONFIRMED,
        ))

        sessions = db.list_sessions(import_status=ImportStatus.CONFIRMED)
        assert len(sessions) == 2
        assert all(s.import_status == ImportStatus.CONFIRMED for s in sessions)

    def test_list_sessions_with_limit(self, db):
        """Test listing sessions with limit"""
        for i in range(10):
            db.create_session(Session(parquet_path=f"/data/s{i}.parquet"))

        sessions = db.list_sessions(limit=5)
        assert len(sessions) == 5

    def test_list_sessions_with_offset(self, db):
        """Test listing sessions with offset"""
        for i in range(10):
            db.create_session(Session(parquet_path=f"/data/s{i}.parquet"))

        first_page = db.list_sessions(limit=3, offset=0)
        second_page = db.list_sessions(limit=3, offset=3)

        assert len(first_page) == 3
        assert len(second_page) == 3
        assert first_page[0].id != second_page[0].id

    def test_update_session(self, db):
        """Test updating a session"""
        session = Session(
            parquet_path="/data/session1.parquet",
            track_name="Unknown",
            total_laps=0,
        )
        created = db.create_session(session)

        # Update fields
        created.track_name = "Laguna Seca"
        created.track_id = "track_001"
        created.total_laps = 20
        created.best_lap_time = 92.5

        updated = db.update_session(created)
        assert updated.track_name == "Laguna Seca"
        assert updated.track_id == "track_001"
        assert updated.total_laps == 20
        assert updated.best_lap_time == 92.5

        # Verify changes persisted
        retrieved = db.get_session(created.id)
        assert retrieved.track_name == "Laguna Seca"
        assert retrieved.total_laps == 20

    def test_delete_session(self, db):
        """Test deleting a session"""
        session = Session(parquet_path="/data/session1.parquet")
        created = db.create_session(session)

        assert db.delete_session(created.id) is True
        assert db.get_session(created.id) is None

    def test_delete_session_not_found(self, db):
        """Test deleting a nonexistent session"""
        assert db.delete_session(9999) is False

    # ------------------------------------------------------------------
    # Lap tests
    # ------------------------------------------------------------------

    def test_create_lap(self, db):
        """Test creating a lap"""
        session = db.create_session(Session(parquet_path="/data/s1.parquet"))

        lap = Lap(
            session_id=session.id,
            lap_number=1,
            lap_time=95.5,
            start_time=0.0,
            end_time=95.5,
            max_speed_mph=140.0,
        )
        created = db.create_lap(lap)

        assert created.id is not None
        assert created.session_id == session.id
        assert created.lap_number == 1
        assert created.lap_time == 95.5
        assert created.max_speed_mph == 140.0

    def test_bulk_create_laps(self, db):
        """Test creating multiple laps at once"""
        session = db.create_session(Session(parquet_path="/data/s1.parquet"))

        laps = [
            Lap(session_id=session.id, lap_number=1, lap_time=95.5),
            Lap(session_id=session.id, lap_number=2, lap_time=93.2),
            Lap(session_id=session.id, lap_number=3, lap_time=92.8),
        ]
        created = db.bulk_create_laps(laps)

        assert len(created) == 3
        assert all(lap.id is not None for lap in created)
        assert created[0].lap_number == 1
        assert created[1].lap_number == 2
        assert created[2].lap_number == 3

    def test_get_laps(self, db):
        """Test retrieving all laps for a session"""
        session = db.create_session(Session(parquet_path="/data/s1.parquet"))

        db.create_lap(Lap(session_id=session.id, lap_number=1, lap_time=95.5))
        db.create_lap(Lap(session_id=session.id, lap_number=2, lap_time=93.2))
        db.create_lap(Lap(session_id=session.id, lap_number=3, lap_time=92.8))

        laps = db.get_laps(session.id)
        assert len(laps) == 3
        assert laps[0].lap_number == 1
        assert laps[1].lap_number == 2
        assert laps[2].lap_number == 3

    def test_get_laps_empty(self, db):
        """Test retrieving laps for a session with no laps"""
        session = db.create_session(Session(parquet_path="/data/s1.parquet"))
        laps = db.get_laps(session.id)
        assert len(laps) == 0

    def test_update_lap_classification(self, db):
        """Test updating lap classification"""
        session = db.create_session(Session(parquet_path="/data/s1.parquet"))
        lap = db.create_lap(Lap(
            session_id=session.id,
            lap_number=1,
            lap_time=95.5,
            classification=LapClassification.NORMAL,
        ))

        success = db.update_lap_classification(
            lap.id,
            LapClassification.HOT_LAP,
            confidence=0.95,
            user_override=True,
        )

        assert success is True

        # Verify changes
        laps = db.get_laps(session.id)
        assert laps[0].classification == LapClassification.HOT_LAP
        assert laps[0].classification_confidence == 0.95
        assert laps[0].user_override is True

    def test_update_lap_classification_not_found(self, db):
        """Test updating classification for nonexistent lap"""
        success = db.update_lap_classification(
            9999,
            LapClassification.HOT_LAP,
        )
        assert success is False

    # ------------------------------------------------------------------
    # Stint tests
    # ------------------------------------------------------------------

    def test_create_stint(self, db):
        """Test creating a stint"""
        session = db.create_session(Session(parquet_path="/data/s1.parquet"))

        stint = Stint(
            session_id=session.id,
            stint_number=1,
            start_lap=1,
            end_lap=5,
            lap_count=5,
            best_lap_time=92.5,
            avg_lap_time=94.2,
        )
        created = db.create_stint(stint)

        assert created.id is not None
        assert created.session_id == session.id
        assert created.stint_number == 1
        assert created.start_lap == 1
        assert created.end_lap == 5
        assert created.lap_count == 5
        assert created.best_lap_time == 92.5
        assert created.avg_lap_time == 94.2

    def test_get_stints(self, db):
        """Test retrieving all stints for a session"""
        session = db.create_session(Session(parquet_path="/data/s1.parquet"))

        db.create_stint(Stint(
            session_id=session.id,
            stint_number=1,
            start_lap=1,
            end_lap=5,
            lap_count=5,
        ))
        db.create_stint(Stint(
            session_id=session.id,
            stint_number=2,
            start_lap=6,
            end_lap=10,
            lap_count=5,
        ))

        stints = db.get_stints(session.id)
        assert len(stints) == 2
        assert stints[0].stint_number == 1
        assert stints[1].stint_number == 2

    def test_get_stints_empty(self, db):
        """Test retrieving stints for a session with no stints"""
        session = db.create_session(Session(parquet_path="/data/s1.parquet"))
        stints = db.get_stints(session.id)
        assert len(stints) == 0

    # ------------------------------------------------------------------
    # Setup tests
    # ------------------------------------------------------------------

    def test_save_setup(self, db):
        """Test saving a setup"""
        session = db.create_session(Session(parquet_path="/data/s1.parquet"))

        setup = Setup(
            session_id=session.id,
            setup_point="pre",
            setup_data={"front_pressure": 32.0, "rear_pressure": 30.0},
            notes="Cold pressures",
        )
        saved = db.save_setup(setup)

        assert saved.id is not None
        assert saved.session_id == session.id
        assert saved.setup_point == "pre"
        assert saved.setup_data["front_pressure"] == 32.0
        assert saved.notes == "Cold pressures"

    def test_get_setup(self, db):
        """Test retrieving a setup"""
        session = db.create_session(Session(parquet_path="/data/s1.parquet"))

        setup = Setup(
            session_id=session.id,
            setup_point="pre",
            setup_data={"front_pressure": 32.0},
        )
        db.save_setup(setup)

        retrieved = db.get_setup(session.id, "pre")
        assert retrieved is not None
        assert retrieved.setup_point == "pre"
        assert retrieved.setup_data["front_pressure"] == 32.0

    def test_get_setup_not_found(self, db):
        """Test retrieving a nonexistent setup"""
        session = db.create_session(Session(parquet_path="/data/s1.parquet"))
        setup = db.get_setup(session.id, "pre")
        assert setup is None

    def test_save_setup_upsert_behavior(self, db):
        """Test that saving a setup with same session_id and setup_point updates existing"""
        session = db.create_session(Session(parquet_path="/data/s1.parquet"))

        # Save initial setup
        setup1 = Setup(
            session_id=session.id,
            setup_point="pre",
            setup_data={"front_pressure": 32.0},
            notes="Initial",
        )
        saved1 = db.save_setup(setup1)

        # Save again with same session_id and setup_point
        setup2 = Setup(
            session_id=session.id,
            setup_point="pre",
            setup_data={"front_pressure": 34.0},
            notes="Updated",
        )
        saved2 = db.save_setup(setup2)

        # Should have same ID (upsert)
        assert saved2.id == saved1.id

        # Verify only one setup exists
        all_setups = db.get_setups(session.id)
        assert len(all_setups) == 1
        assert all_setups[0].setup_data["front_pressure"] == 34.0
        assert all_setups[0].notes == "Updated"

    def test_get_setups_multiple(self, db):
        """Test retrieving multiple setups for a session"""
        session = db.create_session(Session(parquet_path="/data/s1.parquet"))

        db.save_setup(Setup(
            session_id=session.id,
            setup_point="pre",
            setup_data={"front_pressure": 32.0},
        ))
        db.save_setup(Setup(
            session_id=session.id,
            setup_point="post",
            setup_data={"front_pressure": 36.0},
        ))

        setups = db.get_setups(session.id)
        assert len(setups) == 2

    # ------------------------------------------------------------------
    # Stats tests
    # ------------------------------------------------------------------

    def test_get_stats(self, db):
        """Test getting database statistics"""
        # Create sessions with different statuses
        db.create_session(Session(
            parquet_path="/data/s1.parquet",
            import_status=ImportStatus.CONFIRMED,
        ))
        db.create_session(Session(
            parquet_path="/data/s2.parquet",
            import_status=ImportStatus.CONFIRMED,
        ))
        db.create_session(Session(
            parquet_path="/data/s3.parquet",
            import_status=ImportStatus.PENDING,
        ))

        # Create laps
        session1 = db.get_session_by_parquet_path("/data/s1.parquet")
        db.create_lap(Lap(session_id=session1.id, lap_number=1, lap_time=95.0))
        db.create_lap(Lap(session_id=session1.id, lap_number=2, lap_time=94.0))

        stats = db.get_stats()
        assert stats["total_sessions"] == 3
        assert stats["confirmed_sessions"] == 2
        assert stats["pending_sessions"] == 1
        assert stats["total_laps"] == 2

    def test_get_stats_empty(self, db):
        """Test getting stats for empty database"""
        stats = db.get_stats()
        assert stats["total_sessions"] == 0
        assert stats["confirmed_sessions"] == 0
        assert stats["pending_sessions"] == 0
        assert stats["total_laps"] == 0

    # ------------------------------------------------------------------
    # Persistence tests
    # ------------------------------------------------------------------

    def test_persistence(self):
        """Test that database survives reconnection"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Create database and add session
            db1 = SessionDatabase(db_path)
            session = Session(
                parquet_path="/data/persistent.parquet",
                track_name="Test Track",
                total_laps=10,
            )
            created = db1.create_session(session)
            session_id = created.id

            # Create new database instance pointing at same file
            db2 = SessionDatabase(db_path)
            retrieved = db2.get_session(session_id)

            assert retrieved is not None
            assert retrieved.parquet_path == "/data/persistent.parquet"
            assert retrieved.track_name == "Test Track"
            assert retrieved.total_laps == 10
        finally:
            os.unlink(db_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
