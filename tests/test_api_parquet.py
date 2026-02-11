"""
API integration tests for parquet router endpoints.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, PropertyMock
from pathlib import Path

from src.main.app import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_parquet(tmp_path):
    """Create a sample parquet file in a mock data directory structure."""
    n = 100
    time = np.linspace(0, 10, n)
    df = pd.DataFrame({
        "RPM": 4000 + 2000 * np.sin(time),
        "GPS Speed": 60 + 30 * np.sin(time),
        "GPS Latitude": 43.797 + 0.001 * np.sin(time),
    }, index=time)
    path = tmp_path / "test_session.parquet"
    df.to_parquet(path)
    return tmp_path, str(path)


class TestParquetList:
    """Tests for GET /api/parquet/list."""

    def test_list_returns_200(self, client):
        response = client.get("/api/parquet/list")
        assert response.status_code == 200

    def test_list_returns_parquet_files_key(self, client):
        data = client.get("/api/parquet/list").json()
        assert "parquet_files" in data
        assert isinstance(data["parquet_files"], list)


class TestParquetView:
    """Tests for GET /api/parquet/view/{filename}."""

    def test_view_valid_file(self, client, sample_parquet):
        tmp_path, parquet_path = sample_parquet
        with patch('src.main.routers.parquet.config') as mock_config:
            mock_config.DATA_DIR = str(tmp_path)
            response = client.get("/api/parquet/view/test_session.parquet")
            assert response.status_code == 200
            data = response.json()
            assert "columns" in data
            assert "data" in data

    def test_view_pagination(self, client, sample_parquet):
        tmp_path, parquet_path = sample_parquet
        with patch('src.main.routers.parquet.config') as mock_config:
            mock_config.DATA_DIR = str(tmp_path)
            response = client.get("/api/parquet/view/test_session.parquet?limit=10&offset=5")
            assert response.status_code == 200
            data = response.json()
            assert len(data["data"]) <= 10

    def test_view_not_found(self, client, sample_parquet):
        tmp_path, _ = sample_parquet
        with patch('src.main.routers.parquet.config') as mock_config:
            mock_config.DATA_DIR = str(tmp_path)
            response = client.get("/api/parquet/view/nonexistent.parquet")
            assert response.status_code == 404


class TestParquetSummary:
    """Tests for GET /api/parquet/summary/{filename}."""

    def test_summary_valid_file(self, client, sample_parquet):
        tmp_path, parquet_path = sample_parquet
        with patch('src.main.routers.parquet.config') as mock_config:
            mock_config.DATA_DIR = str(tmp_path)
            response = client.get("/api/parquet/summary/test_session.parquet")
            assert response.status_code == 200
            data = response.json()
            assert "rows" in data or "row_count" in data or "columns" in data

    def test_summary_not_found(self, client, sample_parquet):
        tmp_path, _ = sample_parquet
        with patch('src.main.routers.parquet.config') as mock_config:
            mock_config.DATA_DIR = str(tmp_path)
            response = client.get("/api/parquet/summary/nonexistent.parquet")
            assert response.status_code == 404
