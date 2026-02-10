"""
Tests for Session Management API (v2) endpoints
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSessionAPIBasics:
    """Basic API tests without requiring data"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        from src.main.app import app
        return TestClient(app)

    def test_sessions_html_page(self, client):
        """Test that /sessions page loads"""
        response = client.get("/sessions")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Sessions" in response.text

    def test_session_import_page(self, client):
        """Test that /sessions/import page loads"""
        response = client.get("/sessions/import")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Import Session" in response.text or "Import" in response.text


class TestSessionAPIWithData:
    """API tests with synthetic session data"""

    @pytest.fixture
    def client_with_temp_db(self, tmp_path, monkeypatch):
        """Create test client with temporary database"""
        # Create a temporary database file
        db_path = tmp_path / "test_sessions.db"

        # Patch the session database to use temp path
        def mock_get_session_db():
            from src.session.session_database import SessionDatabase
            return SessionDatabase(str(db_path))

        # Import app and patch the sessions router
        from src.main.routers import sessions as sessions_router
        monkeypatch.setattr(sessions_router, "get_session_db", mock_get_session_db)
        monkeypatch.setattr(sessions_router, "_session_db", None)

        from src.main.app import app
        return TestClient(app)

    @pytest.fixture
    def synthetic_parquet(self, tmp_path):
        """Create a synthetic parquet for testing"""
        n = 3000
        time = np.linspace(0, 300, n)
        phase = (time / 60) * 2 * np.pi

        df = pd.DataFrame({
            'GPS Latitude': 43.797875 + 0.008 * np.sin(phase),
            'GPS Longitude': -87.989638 + 0.005 * np.cos(phase),
            'GPS Speed': 40 + 10 * np.sin(phase * 2),
            'RPM': 5000 + 2000 * np.sin(phase * 3),
        }, index=time)

        df.attrs['units'] = {'GPS Speed': 'm/s'}

        parquet_path = tmp_path / "test_session.parquet"
        df.to_parquet(parquet_path)
        return parquet_path

    def test_list_sessions_empty(self, client_with_temp_db):
        """Test listing sessions when database is empty"""
        response = client_with_temp_db.get("/api/v2/sessions")
        assert response.status_code == 200

        data = response.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)
        assert len(data["sessions"]) == 0
        assert "stats" in data

    def test_import_and_list(self, client_with_temp_db, synthetic_parquet):
        """Test importing a session and listing it"""
        # Import the session
        import_data = {
            "parquet_path": str(synthetic_parquet),
            "vehicle_id": "test-vehicle",
            "notes": "Test session"
        }
        import_response = client_with_temp_db.post(
            "/api/v2/sessions/import",
            json=import_data
        )
        assert import_response.status_code == 200
        import_result = import_response.json()
        assert import_result["is_valid"] is True
        assert import_result["session_id"] is not None

        # List sessions
        list_response = client_with_temp_db.get("/api/v2/sessions")
        assert list_response.status_code == 200

        data = list_response.json()
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["id"] == import_result["session_id"]
        assert data["sessions"][0]["vehicle_id"] == "test-vehicle"

    def test_get_session(self, client_with_temp_db, synthetic_parquet):
        """Test getting a single session by ID"""
        # Import a session first
        import_data = {"parquet_path": str(synthetic_parquet)}
        import_response = client_with_temp_db.post(
            "/api/v2/sessions/import",
            json=import_data
        )
        session_id = import_response.json()["session_id"]

        # Get the session
        response = client_with_temp_db.get(f"/api/v2/sessions/{session_id}")
        assert response.status_code == 200

        data = response.json()
        assert "session" in data
        assert data["session"]["id"] == session_id
        assert "stints" in data
        assert "setups" in data

    def test_get_session_not_found(self, client_with_temp_db):
        """Test getting a nonexistent session"""
        response = client_with_temp_db.get("/api/v2/sessions/99999")
        assert response.status_code == 404

    def test_get_session_laps(self, client_with_temp_db, synthetic_parquet):
        """Test getting laps for a session"""
        # Import a session
        import_data = {"parquet_path": str(synthetic_parquet)}
        import_response = client_with_temp_db.post(
            "/api/v2/sessions/import",
            json=import_data
        )
        session_id = import_response.json()["session_id"]

        # Get laps
        response = client_with_temp_db.get(f"/api/v2/sessions/{session_id}/laps")
        assert response.status_code == 200

        data = response.json()
        assert "laps" in data
        assert isinstance(data["laps"], list)
        # May or may not have laps depending on detection
        if len(data["laps"]) > 0:
            lap = data["laps"][0]
            assert "lap_number" in lap
            assert "lap_time" in lap
            assert "classification" in lap

    def test_classify_lap(self, client_with_temp_db, synthetic_parquet):
        """Test updating lap classification"""
        # Import a session with laps
        import_data = {"parquet_path": str(synthetic_parquet)}
        import_response = client_with_temp_db.post(
            "/api/v2/sessions/import",
            json=import_data
        )
        session_id = import_response.json()["session_id"]

        # Get laps to find one
        laps_response = client_with_temp_db.get(f"/api/v2/sessions/{session_id}/laps")
        laps = laps_response.json()["laps"]

        if len(laps) > 0:
            lap_number = laps[0]["lap_number"]

            # Update classification
            update_data = {"classification": "hot_lap"}
            response = client_with_temp_db.put(
                f"/api/v2/sessions/{session_id}/laps/{lap_number}/classify",
                json=update_data
            )
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "ok"
            assert data["lap_number"] == lap_number
            assert data["classification"] == "hot_lap"

    def test_confirm_session(self, client_with_temp_db, synthetic_parquet):
        """Test confirming a session"""
        # Import a session
        import_data = {"parquet_path": str(synthetic_parquet)}
        import_response = client_with_temp_db.post(
            "/api/v2/sessions/import",
            json=import_data
        )
        session_id = import_response.json()["session_id"]

        # Confirm the session
        confirm_data = {
            "session_type": "test",
            "notes": "Confirmed test session"
        }
        response = client_with_temp_db.put(
            f"/api/v2/sessions/{session_id}/confirm",
            json=confirm_data
        )
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["session"]["import_status"] == "confirmed"
        assert data["session"]["session_type"] == "test"

    def test_setup_crud(self, client_with_temp_db, synthetic_parquet):
        """Test saving and getting setup data"""
        # Import a session
        import_data = {"parquet_path": str(synthetic_parquet)}
        import_response = client_with_temp_db.post(
            "/api/v2/sessions/import",
            json=import_data
        )
        session_id = import_response.json()["session_id"]

        # Save setup
        setup_data = {
            "setup_point": "pre",
            "setup_data": {
                "front_spring": "500 lb/in",
                "rear_spring": "550 lb/in",
                "front_damper": "3 clicks",
                "rear_damper": "4 clicks"
            },
            "notes": "Baseline setup"
        }
        put_response = client_with_temp_db.put(
            f"/api/v2/sessions/{session_id}/setup",
            json=setup_data
        )
        assert put_response.status_code == 200

        put_data = put_response.json()
        assert put_data["status"] == "ok"

        # Get setup
        get_response = client_with_temp_db.get(
            f"/api/v2/sessions/{session_id}/setup?setup_point=pre"
        )
        assert get_response.status_code == 200

        get_data = get_response.json()
        assert get_data["setup"] is not None
        assert get_data["setup"]["setup_data"]["front_spring"] == "500 lb/in"
        assert get_data["setup"]["notes"] == "Baseline setup"

    def test_session_detail_page(self, client_with_temp_db, synthetic_parquet):
        """Test that session detail page loads"""
        # Import a session
        import_data = {"parquet_path": str(synthetic_parquet)}
        import_response = client_with_temp_db.post(
            "/api/v2/sessions/import",
            json=import_data
        )
        session_id = import_response.json()["session_id"]

        # Load detail page
        response = client_with_temp_db.get(f"/sessions/{session_id}")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_import_missing_file(self, client_with_temp_db):
        """Test importing a nonexistent file"""
        import_data = {"parquet_path": "/nonexistent/file.parquet"}
        response = client_with_temp_db.post(
            "/api/v2/sessions/import",
            json=import_data
        )
        assert response.status_code == 404

    def test_import_duplicate(self, client_with_temp_db, synthetic_parquet):
        """Test importing the same file twice"""
        import_data = {"parquet_path": str(synthetic_parquet)}

        # First import
        response1 = client_with_temp_db.post(
            "/api/v2/sessions/import",
            json=import_data
        )
        assert response1.status_code == 200
        result1 = response1.json()
        assert result1["duplicate"] is False

        # Second import
        response2 = client_with_temp_db.post(
            "/api/v2/sessions/import",
            json=import_data
        )
        assert response2.status_code == 200
        result2 = response2.json()
        assert result2["duplicate"] is True
        assert result2["session_id"] == result1["session_id"]

    def test_list_sessions_with_filters(self, client_with_temp_db, synthetic_parquet, tmp_path):
        """Test listing sessions with filters"""
        # Import two sessions with different vehicles
        for i, vehicle in enumerate(["vehicle-1", "vehicle-2"]):
            # Create parquet
            time = np.linspace(0, 100, 1000)
            phase = (time / 20) * 2 * np.pi + i * 0.5
            df = pd.DataFrame({
                'GPS Latitude': 43.797875 + 0.005 * np.sin(phase),
                'GPS Longitude': -87.989638 + 0.005 * np.cos(phase),
                'GPS Speed': 40 + 10 * np.sin(phase),
                'RPM': 5000 + 2000 * np.sin(phase),
            }, index=time)
            df.attrs['units'] = {'GPS Speed': 'm/s'}

            parquet_path = tmp_path / f"session_{i}.parquet"
            df.to_parquet(parquet_path)

            # Import
            import_data = {
                "parquet_path": str(parquet_path),
                "vehicle_id": vehicle
            }
            client_with_temp_db.post("/api/v2/sessions/import", json=import_data)

        # Filter by vehicle
        response = client_with_temp_db.get("/api/v2/sessions?vehicle_id=vehicle-1")
        assert response.status_code == 200

        data = response.json()
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["vehicle_id"] == "vehicle-1"

    def test_classify_lap_invalid_classification(self, client_with_temp_db, synthetic_parquet):
        """Test updating lap with invalid classification"""
        # Import a session
        import_data = {"parquet_path": str(synthetic_parquet)}
        import_response = client_with_temp_db.post(
            "/api/v2/sessions/import",
            json=import_data
        )
        session_id = import_response.json()["session_id"]

        # Try to update with invalid classification
        update_data = {"classification": "invalid_type"}
        response = client_with_temp_db.put(
            f"/api/v2/sessions/{session_id}/laps/1/classify",
            json=update_data
        )
        # Should either be 400 (bad request) or 404 (lap not found)
        assert response.status_code in [400, 404]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
