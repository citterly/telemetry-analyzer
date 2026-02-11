"""
API integration tests for vehicles router endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.main.app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestVehiclesList:
    """Tests for GET /api/vehicles."""

    def test_list_returns_200(self, client):
        response = client.get("/api/vehicles")
        assert response.status_code == 200

    def test_list_has_vehicles(self, client):
        data = client.get("/api/vehicles").json()
        assert "vehicles" in data
        assert isinstance(data["vehicles"], list)
        assert len(data["vehicles"]) > 0

    def test_list_has_active_vehicle(self, client):
        data = client.get("/api/vehicles").json()
        assert "active_vehicle" in data


class TestVehicleGet:
    """Tests for GET /api/vehicles/{vehicle_id}."""

    def test_get_valid_vehicle(self, client):
        # First get the list to find a valid ID
        data = client.get("/api/vehicles").json()
        vehicle_id = data["vehicles"][0]["id"]
        response = client.get(f"/api/vehicles/{vehicle_id}")
        assert response.status_code == 200

    def test_get_nonexistent_vehicle(self, client):
        response = client.get("/api/vehicles/nonexistent_vehicle_id")
        assert response.status_code == 404


class TestVehicleSetActive:
    """Tests for PUT /api/vehicles/active."""

    def test_set_active_valid(self, client):
        # Get current active vehicle to restore later
        data = client.get("/api/vehicles").json()
        original_active = data["active_vehicle"]
        vehicle_id = data["vehicles"][0]["id"]

        try:
            response = client.put("/api/vehicles/active",
                                  json={"vehicle_id": vehicle_id})
            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "ok"
        finally:
            # Restore original active vehicle
            client.put("/api/vehicles/active",
                       json={"vehicle_id": original_active})

    def test_set_active_invalid_id(self, client):
        response = client.put("/api/vehicles/active",
                              json={"vehicle_id": "nonexistent"})
        assert response.status_code in (400, 404, 500)

    def test_set_active_missing_body(self, client):
        response = client.put("/api/vehicles/active", json={})
        assert response.status_code in (400, 422)
