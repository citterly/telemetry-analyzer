"""
Tests for the queue status dashboard (feat-033)
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import os


class TestQueueDashboardPage:
    """Tests for the /queue page"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        from src.main.app import app
        return TestClient(app)

    def test_queue_page_loads(self, client):
        """Test that /queue page returns 200 and contains expected elements"""
        response = client.get("/queue")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

        html = response.text
        assert "Extraction Queue" in html

    def test_queue_page_has_stats_cards(self, client):
        """Test that page has stats cards for each status"""
        response = client.get("/queue")
        html = response.text

        assert "stat-pending" in html
        assert "stat-processing" in html
        assert "stat-completed" in html
        assert "stat-failed" in html

    def test_queue_page_has_filter_buttons(self, client):
        """Test that page has filter buttons"""
        response = client.get("/queue")
        html = response.text

        assert 'data-status="all"' in html
        assert 'data-status="pending"' in html
        assert 'data-status="processing"' in html
        assert 'data-status="completed"' in html
        assert 'data-status="failed"' in html

    def test_queue_page_has_bulk_actions(self, client):
        """Test that page has bulk action buttons"""
        response = client.get("/queue")
        html = response.text

        assert "retryAllFailed()" in html
        assert "clearCompleted()" in html

    def test_queue_page_in_navigation(self, client):
        """Test that Queue link appears in navigation"""
        response = client.get("/queue")
        assert 'href="/queue"' in response.text

    def test_queue_page_has_modal(self, client):
        """Test that page has job detail modal"""
        response = client.get("/queue")
        html = response.text

        assert 'id="jobModal"' in html
        assert "Job Details" in html


class TestQueueApiEndpoints:
    """Tests for queue API endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        from src.main.app import app
        return TestClient(app)

    def test_queue_stats_endpoint(self, client):
        """Test /api/queue/stats endpoint"""
        response = client.get("/api/queue/stats")
        assert response.status_code == 200

        data = response.json()
        assert "pending" in data
        assert "processing" in data
        assert "completed" in data
        assert "failed" in data
        assert "total" in data

    def test_queue_jobs_list_endpoint(self, client):
        """Test /api/queue/jobs endpoint"""
        response = client.get("/api/queue/jobs")
        assert response.status_code == 200

        data = response.json()
        assert "jobs" in data
        assert "total" in data
        assert isinstance(data["jobs"], list)

    def test_queue_jobs_filter_by_status(self, client):
        """Test filtering jobs by status"""
        response = client.get("/api/queue/jobs?status=pending")
        assert response.status_code == 200

        data = response.json()
        assert data["status_filter"] == "pending"

    def test_queue_jobs_invalid_status(self, client):
        """Test that invalid status returns 400"""
        response = client.get("/api/queue/jobs?status=invalid")
        assert response.status_code == 400

    def test_queue_jobs_pagination(self, client):
        """Test pagination parameters"""
        response = client.get("/api/queue/jobs?limit=10&offset=0")
        assert response.status_code == 200

    def test_get_nonexistent_job(self, client):
        """Test getting a job that doesn't exist"""
        response = client.get("/api/queue/jobs/999999")
        assert response.status_code == 404

    def test_retry_nonexistent_job(self, client):
        """Test retrying a job that doesn't exist"""
        response = client.post("/api/queue/jobs/999999/retry")
        assert response.status_code == 400

    def test_delete_nonexistent_job(self, client):
        """Test deleting a job that doesn't exist"""
        response = client.delete("/api/queue/jobs/999999")
        assert response.status_code == 404

    def test_retry_all_endpoint(self, client):
        """Test retry-all endpoint"""
        response = client.post("/api/queue/retry-all")
        assert response.status_code == 200

        data = response.json()
        assert "success" in data
        assert "retried_count" in data

    def test_clear_completed_endpoint(self, client):
        """Test clear-completed endpoint"""
        response = client.post("/api/queue/clear-completed")
        assert response.status_code == 200

        data = response.json()
        assert "success" in data
        assert "cleared_count" in data


class TestQueueDashboardJavaScript:
    """Tests for JavaScript functionality"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        from src.main.app import app
        return TestClient(app)

    def test_page_has_refresh_function(self, client):
        """Test that refreshData function exists"""
        response = client.get("/queue")
        assert "async function refreshData()" in response.text

    def test_page_has_load_functions(self, client):
        """Test that load functions exist"""
        response = client.get("/queue")
        html = response.text

        assert "async function loadStats()" in html
        assert "async function loadJobs()" in html

    def test_page_has_filter_function(self, client):
        """Test that filterJobs function exists"""
        response = client.get("/queue")
        assert "function filterJobs(status)" in response.text

    def test_page_has_job_detail_function(self, client):
        """Test that showJobDetail function exists"""
        response = client.get("/queue")
        assert "async function showJobDetail(jobId)" in response.text

    def test_page_has_action_functions(self, client):
        """Test that action functions exist"""
        response = client.get("/queue")
        html = response.text

        assert "async function quickRetry(jobId)" in html
        assert "async function retryJob()" in html
        assert "async function deleteJob()" in html

    def test_page_has_auto_refresh(self, client):
        """Test that auto-refresh is set up"""
        response = client.get("/queue")
        assert "setInterval(refreshData" in response.text


class TestQueueDashboardStyling:
    """Tests for styling in queue dashboard"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        from src.main.app import app
        return TestClient(app)

    def test_page_has_status_badge_styles(self, client):
        """Test that status badge CSS classes are defined"""
        response = client.get("/queue")
        html = response.text

        assert ".status-pending" in html
        assert ".status-processing" in html
        assert ".status-completed" in html
        assert ".status-failed" in html

    def test_page_has_stat_card_styles(self, client):
        """Test that stat card CSS classes are defined"""
        response = client.get("/queue")
        html = response.text

        assert ".stat-card" in html
        assert ".stat-value" in html
        assert ".stat-label" in html
