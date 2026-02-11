"""
API integration tests for queue router endpoints.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from src.main.app import app
from src.extraction.queue import ExtractionQueue


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def isolated_queue(tmp_path):
    """Use a temporary database for queue tests to avoid polluting real data."""
    db_path = tmp_path / "test_queue.db"
    test_queue = ExtractionQueue(str(db_path))
    with patch('src.main.routers.queue.get_queue', return_value=test_queue):
        yield test_queue


class TestQueueStats:
    """Tests for GET /api/queue/stats."""

    def test_stats_returns_200(self, client):
        response = client.get("/api/queue/stats")
        assert response.status_code == 200

    def test_stats_has_counts(self, client):
        data = client.get("/api/queue/stats").json()
        assert isinstance(data, dict)


class TestQueueJobsList:
    """Tests for GET /api/queue/jobs."""

    def test_list_returns_200(self, client):
        response = client.get("/api/queue/jobs")
        assert response.status_code == 200

    def test_list_has_jobs_array(self, client):
        data = client.get("/api/queue/jobs").json()
        assert "jobs" in data
        assert isinstance(data["jobs"], list)

    def test_list_empty_initially(self, client):
        data = client.get("/api/queue/jobs").json()
        assert len(data["jobs"]) == 0

    def test_list_with_status_filter(self, client):
        response = client.get("/api/queue/jobs?status=pending")
        assert response.status_code == 200

    def test_list_invalid_status(self, client):
        response = client.get("/api/queue/jobs?status=invalid_status")
        assert response.status_code == 400


class TestQueueJobDetail:
    """Tests for GET /api/queue/jobs/{job_id}."""

    def test_job_not_found(self, client):
        response = client.get("/api/queue/jobs/9999")
        assert response.status_code == 404

    def test_job_found(self, client, isolated_queue):
        # Submit a job first
        job = isolated_queue.submit("test.xrk", priority=1)
        response = client.get(f"/api/queue/jobs/{job.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["xrk_filename"] == "test.xrk"


class TestQueueRetry:
    """Tests for POST /api/queue/jobs/{job_id}/retry."""

    def test_retry_nonexistent(self, client):
        response = client.post("/api/queue/jobs/9999/retry")
        assert response.status_code == 400


class TestQueueRetryAll:
    """Tests for POST /api/queue/retry-all."""

    def test_retry_all_empty(self, client):
        response = client.post("/api/queue/retry-all")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["retried_count"] == 0


class TestQueueDelete:
    """Tests for DELETE /api/queue/jobs/{job_id}."""

    def test_delete_nonexistent(self, client):
        response = client.delete("/api/queue/jobs/9999")
        assert response.status_code == 404

    def test_delete_existing(self, client, isolated_queue):
        job = isolated_queue.submit("test.xrk", priority=1)
        response = client.delete(f"/api/queue/jobs/{job.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestQueueClearCompleted:
    """Tests for POST /api/queue/clear-completed."""

    def test_clear_empty(self, client):
        response = client.post("/api/queue/clear-completed")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["cleared_count"] == 0
