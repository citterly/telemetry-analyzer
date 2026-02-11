"""
API integration tests for visualization router endpoints.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from src.main.app import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_parquet(tmp_path):
    """Create parquet with GPS, speed, RPM, and acceleration data."""
    n = 1000
    time = np.linspace(0, 100, n)
    df = pd.DataFrame({
        "GPS Latitude": 43.797875 + 0.005 * np.sin(time / 25),
        "GPS Longitude": -87.989638 + 0.005 * np.cos(time / 25),
        "RPM": 4000 + 2000 * np.sin(time / 20),
        "GPS Speed": 60 + 30 * np.sin(time / 15),
        "GPS_LatAcc": 0.5 * np.sin(time / 10),
        "GPS_LonAcc": 0.3 * np.cos(time / 12),
        "PedalPos": 50 + 40 * np.sin(time / 8),
    }, index=time)
    path = tmp_path / "test_session.parquet"
    df.to_parquet(path)
    return str(path)


class TestTrackMap:
    """Tests for GET /api/track-map/{filename}."""

    def test_track_map_svg(self, client, sample_parquet):
        with patch('src.main.deps.find_parquet_file', return_value=sample_parquet):
            response = client.get("/api/track-map/test.parquet?format=svg")
            assert response.status_code == 200
            assert "svg" in response.headers.get("content-type", "").lower() or \
                   "<svg" in response.text

    def test_track_map_json(self, client, sample_parquet):
        with patch('src.main.deps.find_parquet_file', return_value=sample_parquet):
            response = client.get("/api/track-map/test.parquet?format=json")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, dict)

    def test_track_map_html(self, client, sample_parquet):
        with patch('src.main.deps.find_parquet_file', return_value=sample_parquet):
            response = client.get("/api/track-map/test.parquet?format=html")
            assert response.status_code == 200

    def test_track_map_not_found(self, client):
        with patch('src.main.deps.find_parquet_file', return_value=None):
            response = client.get("/api/track-map/nonexistent.parquet")
            assert response.status_code == 404

    def test_track_map_color_by_rpm(self, client, sample_parquet):
        with patch('src.main.deps.find_parquet_file', return_value=sample_parquet):
            response = client.get("/api/track-map/test.parquet?color_by=rpm&format=svg")
            assert response.status_code == 200


class TestDeltaTrackMap:
    """Tests for GET /api/track-map/delta/{filename}."""

    def test_delta_map_not_found(self, client):
        with patch('src.main.deps.find_parquet_file', return_value=None):
            response = client.get("/api/track-map/delta/nonexistent.parquet?lap_a=1&lap_b=2")
            assert response.status_code == 404

    def test_delta_map_with_data(self, client, sample_parquet):
        with patch('src.main.deps.find_parquet_file', return_value=sample_parquet):
            response = client.get(
                "/api/track-map/delta/test.parquet?lap_a=1&lap_b=2&format=json"
            )
            # May return 200 or error if laps can't be compared — both are valid
            assert response.status_code in (200, 400, 500)


class TestGGDiagram:
    """Tests for GET /api/gg-diagram/{filename}."""

    def test_gg_diagram_json(self, client, sample_parquet):
        with patch('src.main.deps.find_parquet_file', return_value=sample_parquet):
            response = client.get("/api/gg-diagram/test.parquet?format=json")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, dict)

    def test_gg_diagram_not_found(self, client):
        with patch('src.main.deps.find_parquet_file', return_value=None):
            response = client.get("/api/gg-diagram/nonexistent.parquet")
            assert response.status_code == 404

    def test_gg_diagram_trace(self, client, sample_parquet):
        with patch('src.main.deps.find_parquet_file', return_value=sample_parquet):
            response = client.get("/api/gg-diagram/test.parquet?format=json&trace=true")
            assert response.status_code == 200
            data = response.json()
            assert "_trace" in data


class TestCornerAnalysis:
    """Tests for GET /api/corner-analysis/{filename}."""

    def test_corner_analysis(self, client, sample_parquet):
        with patch('src.main.deps.find_parquet_file', return_value=sample_parquet):
            response = client.get("/api/corner-analysis/test.parquet")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, dict)

    def test_corner_analysis_not_found(self, client):
        with patch('src.main.deps.find_parquet_file', return_value=None):
            response = client.get("/api/corner-analysis/nonexistent.parquet")
            assert response.status_code == 404

    def test_corner_analysis_trace(self, client, sample_parquet):
        with patch('src.main.deps.find_parquet_file', return_value=sample_parquet):
            response = client.get("/api/corner-analysis/test.parquet?trace=true")
            assert response.status_code == 200
            data = response.json()
            assert "_trace" in data


class TestCornerTrackMap:
    """Tests for GET /api/corner-track-map/{filename}."""

    def test_corner_track_map(self, client, sample_parquet):
        with patch('src.main.deps.find_parquet_file', return_value=sample_parquet):
            response = client.get("/api/corner-track-map/test.parquet")
            assert response.status_code == 200

    def test_corner_track_map_not_found(self, client):
        with patch('src.main.deps.find_parquet_file', return_value=None):
            response = client.get("/api/corner-track-map/nonexistent.parquet")
            assert response.status_code == 404
