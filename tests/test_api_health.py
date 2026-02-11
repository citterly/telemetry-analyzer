"""
API integration tests for health and stats endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.main.app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_has_version(self, client):
        data = client.get("/health").json()
        assert "version" in data


class TestStatsEndpoint:
    """Tests for GET /api/stats."""

    def test_stats_returns_200(self, client):
        response = client.get("/api/stats")
        assert response.status_code == 200

    def test_stats_has_stats_key(self, client):
        data = client.get("/api/stats").json()
        assert "stats" in data
