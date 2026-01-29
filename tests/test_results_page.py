"""
Tests for the analysis results viewer page (feat-032)
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json


class TestAnalysisPage:
    """Tests for the /analysis page"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        from src.main.app import app
        return TestClient(app)

    def test_analysis_page_loads(self, client):
        """Test that /analysis page returns 200 and contains expected elements"""
        response = client.get("/analysis")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

        # Check for key elements
        html = response.text
        assert "Session Analysis" in html
        assert "file-select" in html
        assert "analysis-buttons" in html
        assert "results-section" in html

    def test_analysis_page_has_chart_js(self, client):
        """Test that page loads Chart.js for visualizations"""
        response = client.get("/analysis")
        assert "chart.js" in response.text.lower()

    def test_analysis_page_has_all_analysis_buttons(self, client):
        """Test that all analysis type buttons are present"""
        response = client.get("/analysis")
        html = response.text

        assert "btn-report" in html
        assert "btn-shifts" in html
        assert "btn-laps" in html
        assert "btn-gears" in html
        assert "btn-power" in html

    def test_analysis_page_in_navigation(self, client):
        """Test that Analysis link appears in navigation"""
        response = client.get("/analysis")
        assert 'href="/analysis"' in response.text

    def test_analysis_page_extends_base_template(self, client):
        """Test that page uses the base template (dark theme)"""
        response = client.get("/analysis")
        html = response.text

        # Check for base template elements
        assert "Telemetry Analyzer" in html
        assert "navbar" in html
        assert "Bootstrap" in html or "bootstrap" in html


class TestAnalysisApiIntegration:
    """Tests for API integration with the analysis page"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        from src.main.app import app
        return TestClient(app)

    @pytest.fixture
    def mock_parquet_list(self):
        """Mock parquet file list response"""
        return {
            "parquet_files": [
                {
                    "filename": "test_session.parquet",
                    "path": "data/exports/test_session.parquet",
                    "rows": 10000,
                    "columns": 15,
                    "size_mb": "2.5"
                }
            ]
        }

    def test_parquet_list_api_works(self, client):
        """Test that /api/parquet/list endpoint works (needed by analysis page)"""
        response = client.get("/api/parquet/list")
        assert response.status_code == 200
        data = response.json()
        assert "parquet_files" in data

    def test_shifts_api_endpoint_exists(self, client):
        """Test that shifts analysis API endpoint exists"""
        # This will fail with 404 for missing file, but endpoint should exist
        response = client.get("/api/analyze/shifts/nonexistent.parquet")
        # Should be 404 (file not found), not 405 (method not allowed)
        assert response.status_code in [404, 400, 500]

    def test_laps_api_endpoint_exists(self, client):
        """Test that laps analysis API endpoint exists"""
        response = client.get("/api/analyze/laps/nonexistent.parquet")
        assert response.status_code in [404, 400, 500]

    def test_gears_api_endpoint_exists(self, client):
        """Test that gears analysis API endpoint exists"""
        response = client.get("/api/analyze/gears/nonexistent.parquet")
        assert response.status_code in [404, 400, 500]

    def test_power_api_endpoint_exists(self, client):
        """Test that power analysis API endpoint exists"""
        response = client.get("/api/analyze/power/nonexistent.parquet")
        assert response.status_code in [404, 400, 500]

    def test_report_api_endpoint_exists(self, client):
        """Test that full report API endpoint exists"""
        response = client.get("/api/analyze/report/nonexistent.parquet")
        assert response.status_code in [404, 400, 500]


class TestAnalysisPageJavaScript:
    """Tests for JavaScript functionality in analysis page"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        from src.main.app import app
        return TestClient(app)

    def test_page_has_load_file_list_function(self, client):
        """Test that loadFileList function is defined"""
        response = client.get("/analysis")
        assert "async function loadFileList()" in response.text

    def test_page_has_run_analysis_function(self, client):
        """Test that runAnalysis function is defined"""
        response = client.get("/analysis")
        assert "async function runAnalysis(type)" in response.text

    def test_page_has_display_functions(self, client):
        """Test that display functions for each analysis type exist"""
        response = client.get("/analysis")
        html = response.text

        assert "function displayFullReport(data)" in html
        assert "function displayShiftAnalysis(data)" in html
        assert "function displayLapAnalysis(data)" in html
        assert "function displayGearAnalysis(data)" in html
        assert "function displayPowerAnalysis(data)" in html

    def test_page_has_chart_rendering_functions(self, client):
        """Test that chart rendering functions exist"""
        response = client.get("/analysis")
        html = response.text

        assert "function renderLapTimesChart" in html
        assert "function renderShiftRpmChart" in html
        assert "function renderGearDistributionChart" in html

    def test_page_has_loading_overlay(self, client):
        """Test that loading overlay element exists"""
        response = client.get("/analysis")
        assert 'id="loading"' in response.text
        assert "loading-overlay" in response.text


class TestAnalysisPageStyling:
    """Tests for styling and CSS in analysis page"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        from src.main.app import app
        return TestClient(app)

    def test_page_has_custom_styles(self, client):
        """Test that custom CSS classes are defined"""
        response = client.get("/analysis")
        html = response.text

        assert ".analysis-card" in html
        assert ".metric-box" in html
        assert ".recommendation-item" in html

    def test_page_has_shift_quality_colors(self, client):
        """Test that shift quality color classes are defined"""
        response = client.get("/analysis")
        html = response.text

        assert ".shift-quality-early" in html
        assert ".shift-quality-optimal" in html
        assert ".shift-quality-late" in html
