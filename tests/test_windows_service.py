"""
Tests for Windows extraction service
These tests verify the API structure without requiring Windows/DLL
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient


class TestWindowsServiceAPI:
    """Tests for Windows service API endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client with mocked Windows check"""
        # Import with mocked platform
        from src.extraction import windows_service
        return TestClient(windows_service.app)

    def test_health_check(self, client):
        """Test health endpoint returns expected fields"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "platform" in data
        assert "dll_available" in data
        assert "temp_dir" in data
        assert "timestamp" in data

    def test_health_check_timestamp_format(self, client):
        """Test health timestamp is valid ISO format"""
        response = client.get("/health")
        data = response.json()

        # Should parse without error
        datetime.fromisoformat(data["timestamp"])

    def test_list_files_empty(self, client):
        """Test listing files when temp dir is empty"""
        response = client.get("/files")
        assert response.status_code == 200

        data = response.json()
        assert "files" in data
        assert isinstance(data["files"], list)

    def test_download_nonexistent_file(self, client):
        """Test downloading file that doesn't exist"""
        response = client.get("/download/nonexistent.parquet")
        assert response.status_code == 404

    def test_download_non_parquet_rejected(self, client):
        """Test that non-parquet downloads are rejected"""
        # Create a temp file that's not parquet
        from src.extraction.windows_service import TEMP_DIR
        test_file = TEMP_DIR / "test.txt"
        test_file.write_text("test")

        try:
            response = client.get("/download/test.txt")
            assert response.status_code == 400
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_delete_nonexistent_file(self, client):
        """Test deleting file that doesn't exist"""
        response = client.delete("/files/nonexistent.parquet")
        assert response.status_code == 404

    def test_clear_files(self, client):
        """Test clearing all temp files"""
        response = client.delete("/files")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "cleared"
        assert "files_deleted" in data

    def test_extract_requires_xrk_extension(self, client):
        """Test that extract endpoint requires .xrk file"""
        # Create a fake file with wrong extension
        files = {"file": ("test.txt", b"fake content", "application/octet-stream")}
        response = client.post("/extract", files=files)

        # Should fail on Linux (503) or fail validation (400)
        assert response.status_code in [400, 503]

    def test_extract_on_linux_returns_503(self, client):
        """Test that extraction returns 503 on non-Windows"""
        if sys.platform == "win32":
            pytest.skip("Test only runs on non-Windows")

        files = {"file": ("test.xrk", b"fake xrk content", "application/octet-stream")}
        response = client.post("/extract", files=files)

        assert response.status_code == 503
        assert "Windows" in response.json()["detail"]


class TestExtractionResponse:
    """Tests for response models"""

    def test_extraction_response_model(self):
        """Test ExtractionResponse model fields"""
        from src.extraction.windows_service import ExtractionResponse

        response = ExtractionResponse(
            success=True,
            message="Test",
            output_filename="test.parquet",
            rows=100,
            columns=10,
            duration_seconds=60.5,
            channels=["RPM", "Speed"]
        )

        assert response.success is True
        assert response.rows == 100
        assert len(response.channels) == 2

    def test_extraction_response_minimal(self):
        """Test ExtractionResponse with minimal fields"""
        from src.extraction.windows_service import ExtractionResponse

        response = ExtractionResponse(
            success=False,
            message="Failed"
        )

        assert response.success is False
        assert response.output_filename is None


class TestHealthResponse:
    """Tests for health response model"""

    def test_health_response_model(self):
        """Test HealthResponse model"""
        from src.extraction.windows_service import HealthResponse

        response = HealthResponse(
            status="healthy",
            platform="linux",
            dll_available=False,
            temp_dir="/tmp/test",
            timestamp="2026-01-29T01:00:00"
        )

        assert response.status == "healthy"
        assert response.dll_available is False


class TestTempDirectory:
    """Tests for temp directory handling"""

    def test_temp_dir_exists(self):
        """Test that temp directory is created"""
        from src.extraction.windows_service import TEMP_DIR
        assert TEMP_DIR.exists()
        assert TEMP_DIR.is_dir()

    def test_temp_dir_writable(self):
        """Test that temp directory is writable"""
        from src.extraction.windows_service import TEMP_DIR

        test_file = TEMP_DIR / "write_test.tmp"
        try:
            test_file.write_text("test")
            assert test_file.exists()
        finally:
            if test_file.exists():
                test_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
