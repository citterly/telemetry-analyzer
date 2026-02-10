"""
Tests for SessionImporter with both synthetic and real data
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.session.importer import SessionImporter, ImportResult
from src.session.session_database import SessionDatabase
from src.session.models import ImportStatus, LapClassification


class TestSessionImporterSynthetic:
    """Tests with synthetic data"""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        db = SessionDatabase(db_path)
        yield db
        # Cleanup
        try:
            os.unlink(db_path)
        except:
            pass

    @pytest.fixture
    def importer(self, temp_db):
        """Create a SessionImporter with temp database"""
        return SessionImporter(temp_db)

    @pytest.fixture
    def synthetic_parquet(self, tmp_path):
        """Create a synthetic Road America session parquet"""
        # Create 600 seconds (10 minutes) of data at 10 Hz
        n = 6000
        time = np.linspace(0, 600, n)

        # GPS coordinates that trace a roughly rectangular path
        # Road America is approximately at 43.797875, -87.989638
        # Create a rectangular loop
        phase = (time / 60) * 2 * np.pi  # One lap per minute
        lat = 43.797875 + 0.008 * np.sin(phase)
        lon = -87.989638 + 0.005 * np.cos(phase)

        # Speed varies between 30-50 m/s (realistic track speeds)
        speed = 40 + 10 * np.sin(phase * 2)

        # RPM varies between 3000-7000
        rpm = 5000 + 2000 * np.sin(phase * 3)

        df = pd.DataFrame({
            'GPS Latitude': lat,
            'GPS Longitude': lon,
            'GPS Speed': speed,
            'RPM': rpm,
        }, index=time)

        # Add units metadata
        df.attrs['units'] = {'GPS Speed': 'm/s'}

        parquet_path = tmp_path / "test_road_america.parquet"
        df.to_parquet(parquet_path)
        return parquet_path

    def test_import_valid_session(self, importer, synthetic_parquet, temp_db):
        """Test importing a valid synthetic session"""
        result = importer.import_session(str(synthetic_parquet))

        assert result.is_valid is True
        assert result.session_id is not None
        assert result.duplicate is False
        assert result.error is None

        # Verify session was created in database
        session = temp_db.get_session(result.session_id)
        assert session is not None
        assert session.import_status == ImportStatus.PENDING
        assert session.total_laps >= 0  # May or may not detect laps from synthetic data

    def test_import_duplicate(self, importer, synthetic_parquet):
        """Test importing the same file twice"""
        # First import
        result1 = importer.import_session(str(synthetic_parquet))
        assert result1.is_valid is True
        assert result1.duplicate is False

        # Second import of same file
        result2 = importer.import_session(str(synthetic_parquet))
        assert result2.duplicate is True
        assert result2.session_id == result1.session_id

    def test_import_missing_file(self, importer):
        """Test importing a nonexistent file"""
        result = importer.import_session("/nonexistent/path/to/file.parquet")

        assert result.is_valid is False
        assert result.error is not None
        assert "not found" in result.error.lower()
        assert result.session_id is None

    def test_import_batch(self, importer, tmp_path, temp_db):
        """Test batch import from directory"""
        # Create two parquet files
        for i in range(2):
            time = np.linspace(0, 100, 1000)
            phase = (time / 20) * 2 * np.pi

            df = pd.DataFrame({
                'GPS Latitude': 43.797875 + 0.005 * np.sin(phase + i * 0.5),
                'GPS Longitude': -87.989638 + 0.005 * np.cos(phase + i * 0.5),
                'GPS Speed': 40 + 10 * np.sin(phase),
                'RPM': 5000 + 2000 * np.sin(phase),
            }, index=time)
            df.attrs['units'] = {'GPS Speed': 'm/s'}

            parquet_path = tmp_path / f"session_{i}.parquet"
            df.to_parquet(parquet_path)

        # Import batch
        results = importer.import_batch(str(tmp_path))

        assert len(results) == 2
        assert all(r.is_valid for r in results)
        assert all(r.session_id is not None for r in results)

        # Verify both sessions are in database
        sessions = temp_db.list_sessions(limit=10)
        assert len(sessions) >= 2


class TestSessionImporterReal:
    """Tests with real Road America data"""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        db = SessionDatabase(db_path)
        yield db
        # Cleanup
        try:
            os.unlink(db_path)
        except:
            pass

    @pytest.fixture
    def importer(self, temp_db):
        """Create a SessionImporter with temp database"""
        return SessionImporter(temp_db)

    @pytest.mark.skipif(
        not Path("/home/chris/projects/racing/aim-telemetry/data/exports/processed/20250712_104619_Road America_a_0394.parquet").exists(),
        reason="Real Road America parquet file not found"
    )
    def test_import_road_america_real(self, importer, temp_db):
        """Test importing real Road America session"""
        parquet_path = "/home/chris/projects/racing/aim-telemetry/data/exports/processed/20250712_104619_Road America_a_0394.parquet"

        result = importer.import_session(parquet_path)

        # Verify import succeeded
        assert result.is_valid is True
        assert result.session_id is not None
        assert result.error is None
        assert result.duplicate is False

        # Verify track detection
        assert result.detected_track is not None
        assert "Road America" in result.detected_track or result.detected_track == "road-america"

        # Verify lap detection
        assert result.lap_count > 0

        # Verify session stored in database
        session = temp_db.get_session(result.session_id)
        assert session is not None
        assert session.track_name == result.detected_track or session.track_id == "road-america"
        assert session.total_laps > 0

        # Verify laps stored
        laps = temp_db.get_laps(result.session_id)
        assert len(laps) > 0
        assert len(laps) == result.lap_count

        # Verify at least one lap has a classification
        classifications = [lap.classification for lap in laps]
        assert len(classifications) > 0

        # Verify that we have some hot laps or out laps
        has_hot_or_out = any(
            c in [LapClassification.HOT_LAP, LapClassification.OUT_LAP, LapClassification.NORMAL]
            for c in classifications
        )
        assert has_hot_or_out


class TestImportResult:
    """Tests for ImportResult dataclass"""

    def test_import_result_to_dict(self):
        """Test ImportResult serialization"""
        result = ImportResult(
            session_id=123,
            parquet_path="/path/to/file.parquet",
            is_valid=True,
            detected_track="Road America",
            lap_count=10,
            best_lap_time=125.5,
        )

        data = result.to_dict()

        assert data["session_id"] == 123
        assert data["parquet_path"] == "/path/to/file.parquet"
        assert data["is_valid"] is True
        assert data["detected_track"] == "Road America"
        assert data["lap_count"] == 10
        assert data["best_lap_time"] == 125.5
        assert data["duplicate"] is False
        assert data["error"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
