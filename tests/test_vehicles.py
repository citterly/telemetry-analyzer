"""
Tests for vehicle profiles (feat-041)
"""

import pytest
import json
import tempfile
from pathlib import Path


class TestVehicleModel:
    """Tests for Vehicle dataclass"""

    def test_vehicle_creation(self):
        """Test creating a Vehicle instance"""
        from src.config.vehicles import Vehicle, EngineSpec, TransmissionSetup

        vehicle = Vehicle(
            id="test-car",
            name="Test Car",
            make="Test",
            model="Model",
            year=2020,
            tire_size="255/40/18",
            tire_circumference_meters=2.0,
            weight_lbs=3000,
            engine=EngineSpec(),
            current_setup=TransmissionSetup(
                name="Test Setup",
                transmission_ratios=[2.5, 1.5, 1.0],
                final_drive=3.5,
                weight_lbs=3000
            )
        )

        assert vehicle.id == "test-car"
        assert vehicle.name == "Test Car"
        assert vehicle.weight_lbs == 3000

    def test_vehicle_weight_kg_property(self):
        """Test weight_kg conversion property"""
        from src.config.vehicles import Vehicle, EngineSpec, TransmissionSetup

        vehicle = Vehicle(
            id="test", name="Test", make="", model="", year=0,
            tire_size="", tire_circumference_meters=2.0,
            weight_lbs=2200,  # 1000 kg
            engine=EngineSpec(),
            current_setup=TransmissionSetup("Test", [2.0, 1.0], 3.5, 2200)
        )

        assert abs(vehicle.weight_kg - 997.9) < 1.0

    def test_transmission_ratios_property(self):
        """Test transmission_ratios convenience property"""
        from src.config.vehicles import Vehicle, EngineSpec, TransmissionSetup

        vehicle = Vehicle(
            id="test", name="Test", make="", model="", year=0,
            tire_size="", tire_circumference_meters=2.0, weight_lbs=3000,
            engine=EngineSpec(),
            current_setup=TransmissionSetup("Test", [2.5, 1.5, 1.0], 3.5, 3000)
        )

        assert vehicle.transmission_ratios == [2.5, 1.5, 1.0]

    def test_final_drive_property(self):
        """Test final_drive convenience property"""
        from src.config.vehicles import Vehicle, EngineSpec, TransmissionSetup

        vehicle = Vehicle(
            id="test", name="Test", make="", model="", year=0,
            tire_size="", tire_circumference_meters=2.0, weight_lbs=3000,
            engine=EngineSpec(),
            current_setup=TransmissionSetup("Test", [2.0, 1.0], 4.10, 3000)
        )

        assert vehicle.final_drive == 4.10

    def test_all_setups_property(self):
        """Test all_setups includes current and alternatives"""
        from src.config.vehicles import Vehicle, EngineSpec, TransmissionSetup

        vehicle = Vehicle(
            id="test", name="Test", make="", model="", year=0,
            tire_size="", tire_circumference_meters=2.0, weight_lbs=3000,
            engine=EngineSpec(),
            current_setup=TransmissionSetup("Current", [2.0, 1.0], 3.5, 3000),
            alternative_setups=[
                TransmissionSetup("Alt 1", [2.5, 1.5], 3.5, 3000),
                TransmissionSetup("Alt 2", [2.0, 1.0], 4.0, 3000)
            ]
        )

        assert len(vehicle.all_setups) == 3
        assert vehicle.all_setups[0].name == "Current"

    def test_get_setup_by_name(self):
        """Test getting setup by name"""
        from src.config.vehicles import Vehicle, EngineSpec, TransmissionSetup

        vehicle = Vehicle(
            id="test", name="Test", make="", model="", year=0,
            tire_size="", tire_circumference_meters=2.0, weight_lbs=3000,
            engine=EngineSpec(),
            current_setup=TransmissionSetup("Current", [2.0, 1.0], 3.5, 3000),
            alternative_setups=[
                TransmissionSetup("Racing Setup", [2.5, 1.5], 3.5, 3000)
            ]
        )

        setup = vehicle.get_setup_by_name("Racing Setup")
        assert setup is not None
        assert setup.transmission_ratios == [2.5, 1.5]

    def test_calculate_speed_at_rpm(self):
        """Test speed calculation"""
        from src.config.vehicles import Vehicle, EngineSpec, TransmissionSetup

        vehicle = Vehicle(
            id="test", name="Test", make="", model="", year=0,
            tire_size="", tire_circumference_meters=2.0, weight_lbs=3000,
            engine=EngineSpec(),
            current_setup=TransmissionSetup("Test", [2.5, 1.5, 1.0], 3.5, 3000)
        )

        # Test top gear (ratio 1.0)
        speed = vehicle.calculate_speed_at_rpm(6000, 3)
        assert speed > 0
        # At 6000 RPM with 1.0 gear ratio and 3.5 final drive
        # wheel_rpm = 6000 / 3.5 = 1714
        # wheel_speed = 1714 * 2.0 m/min = 3428 m/min
        # speed_mph = 3428 * 60 / 1609.34 = 127.8 mph
        assert 125 < speed < 130

    def test_calculate_rpm_at_speed(self):
        """Test RPM calculation"""
        from src.config.vehicles import Vehicle, EngineSpec, TransmissionSetup

        vehicle = Vehicle(
            id="test", name="Test", make="", model="", year=0,
            tire_size="", tire_circumference_meters=2.0, weight_lbs=3000,
            engine=EngineSpec(),
            current_setup=TransmissionSetup("Test", [2.5, 1.5, 1.0], 3.5, 3000)
        )

        rpm = vehicle.calculate_rpm_at_speed(100, 3)
        assert rpm > 0
        # At 100 mph in gear 3 (ratio 1.0) with 3.5 final drive
        assert 4000 < rpm < 5000

    def test_vehicle_to_dict(self):
        """Test converting vehicle to dictionary"""
        from src.config.vehicles import Vehicle, EngineSpec, TransmissionSetup

        vehicle = Vehicle(
            id="test", name="Test Car", make="Test", model="Model", year=2020,
            tire_size="255/40/18", tire_circumference_meters=2.0, weight_lbs=3000,
            engine=EngineSpec(max_rpm=7500),
            current_setup=TransmissionSetup("Test", [2.0, 1.0], 3.5, 3000)
        )

        d = vehicle.to_dict()
        assert d["id"] == "test"
        assert d["name"] == "Test Car"
        assert d["engine"]["max_rpm"] == 7500

    def test_vehicle_from_dict(self):
        """Test creating vehicle from dictionary"""
        from src.config.vehicles import Vehicle

        data = {
            "id": "test",
            "name": "Test Car",
            "make": "Test",
            "model": "Model",
            "year": 2020,
            "tire_size": "255/40/18",
            "tire_circumference_meters": 2.0,
            "weight_lbs": 3000,
            "engine": {"max_rpm": 7500},
            "current_setup": {
                "name": "Test",
                "transmission_ratios": [2.0, 1.0],
                "final_drive": 3.5,
                "weight_lbs": 3000
            }
        }

        vehicle = Vehicle.from_dict(data)
        assert vehicle.id == "test"
        assert vehicle.engine.max_rpm == 7500
        assert vehicle.current_setup.final_drive == 3.5


class TestVehicleDatabase:
    """Tests for VehicleDatabase class"""

    def test_database_loads_default(self):
        """Test database loads vehicles from default location"""
        from src.config.vehicles import VehicleDatabase

        db = VehicleDatabase()
        assert len(db.vehicles) > 0

    def test_database_has_bmw_m3(self):
        """Test database includes BMW M3"""
        from src.config.vehicles import VehicleDatabase

        db = VehicleDatabase()
        m3 = db.get("bmw-e46-m3")

        assert m3 is not None
        assert "M3" in m3.name or "BMW" in m3.make

    def test_get_by_id(self):
        """Test getting vehicle by ID"""
        from src.config.vehicles import VehicleDatabase

        db = VehicleDatabase()
        vehicle = db.get("bmw-e46-m3")
        assert vehicle is not None
        assert vehicle.id == "bmw-e46-m3"

    def test_get_by_name(self):
        """Test getting vehicle by name"""
        from src.config.vehicles import VehicleDatabase

        db = VehicleDatabase()
        vehicle = db.get_by_name("Andy's M3")
        assert vehicle is not None
        assert vehicle.id == "bmw-e46-m3"

    def test_list_vehicles(self):
        """Test listing all vehicles"""
        from src.config.vehicles import VehicleDatabase

        db = VehicleDatabase()
        vehicles = db.list_vehicles()
        assert len(vehicles) >= 1
        assert any(v.id == "bmw-e46-m3" for v in vehicles)

    def test_active_vehicle(self):
        """Test getting active vehicle"""
        from src.config.vehicles import VehicleDatabase

        db = VehicleDatabase()
        active = db.active_vehicle
        assert active is not None

    def test_set_active_vehicle(self):
        """Test switching active vehicle"""
        from src.config.vehicles import VehicleDatabase

        db = VehicleDatabase()

        # Add a second vehicle if needed
        if len(db.vehicles) < 2:
            pytest.skip("Need multiple vehicles to test switching")

        vehicle_ids = list(db.vehicles.keys())
        other_id = vehicle_ids[1] if vehicle_ids[0] == db._active_vehicle_id else vehicle_ids[0]

        result = db.set_active_vehicle(other_id)
        assert result is True
        assert db.active_vehicle.id == other_id

    def test_set_invalid_active_vehicle(self):
        """Test setting nonexistent vehicle as active"""
        from src.config.vehicles import VehicleDatabase

        db = VehicleDatabase()
        result = db.set_active_vehicle("nonexistent")
        assert result is False

    def test_save_to_json(self):
        """Test saving vehicles to JSON file"""
        from src.config.vehicles import VehicleDatabase

        db = VehicleDatabase()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            db.save_to_json(f.name)

            with open(f.name, 'r') as rf:
                data = json.load(rf)
                assert "vehicles" in data
                assert "active_vehicle" in data


class TestConvenienceFunctions:
    """Tests for module-level convenience functions"""

    def test_get_vehicle_database(self):
        """Test get_vehicle_database returns singleton"""
        from src.config.vehicles import get_vehicle_database

        db1 = get_vehicle_database()
        db2 = get_vehicle_database()
        assert db1 is db2

    def test_get_vehicle(self):
        """Test get_vehicle convenience function"""
        from src.config.vehicles import get_vehicle

        vehicle = get_vehicle("bmw-e46-m3")
        assert vehicle is not None
        assert vehicle.make == "BMW"

    def test_get_vehicle_by_name(self):
        """Test get_vehicle_by_name convenience function"""
        from src.config.vehicles import get_vehicle_by_name

        vehicle = get_vehicle_by_name("Andy's M3")
        assert vehicle is not None
        assert vehicle.id == "bmw-e46-m3"

    def test_get_active_vehicle(self):
        """Test get_active_vehicle convenience function"""
        from src.config.vehicles import get_active_vehicle

        vehicle = get_active_vehicle()
        assert vehicle is not None

    def test_set_active_vehicle(self):
        """Test set_active_vehicle convenience function"""
        from src.config.vehicles import set_active_vehicle, get_active_vehicle, get_vehicle_database

        db = get_vehicle_database()
        original_id = db._active_vehicle_id

        # Reset to original after test
        try:
            if len(db.vehicles) >= 2:
                other_id = [k for k in db.vehicles.keys() if k != original_id][0]
                result = set_active_vehicle(other_id)
                assert result is True
                assert get_active_vehicle().id == other_id
        finally:
            set_active_vehicle(original_id)


class TestBackwardCompatibility:
    """Tests for backward compatibility functions"""

    def test_get_current_setup(self):
        """Test get_current_setup returns dict"""
        from src.config.vehicles import get_current_setup

        setup = get_current_setup()
        assert isinstance(setup, dict)
        assert "name" in setup
        assert "transmission_ratios" in setup
        assert "final_drive" in setup

    def test_get_transmission_scenarios(self):
        """Test get_transmission_scenarios returns list"""
        from src.config.vehicles import get_transmission_scenarios

        scenarios = get_transmission_scenarios()
        assert isinstance(scenarios, list)
        assert len(scenarios) >= 1
        assert "name" in scenarios[0]

    def test_get_engine_specs(self):
        """Test get_engine_specs returns dict"""
        from src.config.vehicles import get_engine_specs

        specs = get_engine_specs()
        assert isinstance(specs, dict)
        assert "max_rpm" in specs
        assert "shift_rpm" in specs


class TestVehiclesJsonFile:
    """Tests for the vehicles.json data file"""

    def test_json_file_exists(self):
        """Test that vehicles.json exists"""
        json_path = Path(__file__).parent.parent / "data" / "vehicles.json"
        assert json_path.exists()

    def test_json_file_valid(self):
        """Test that vehicles.json is valid JSON"""
        json_path = Path(__file__).parent.parent / "data" / "vehicles.json"
        with open(json_path) as f:
            data = json.load(f)
        assert "vehicles" in data

    def test_json_has_multiple_vehicles(self):
        """Test that vehicles.json has multiple vehicles"""
        json_path = Path(__file__).parent.parent / "data" / "vehicles.json"
        with open(json_path) as f:
            data = json.load(f)
        assert len(data["vehicles"]) >= 2

    def test_all_vehicles_have_required_fields(self):
        """Test that all vehicles have required fields"""
        json_path = Path(__file__).parent.parent / "data" / "vehicles.json"
        with open(json_path) as f:
            data = json.load(f)

        required_fields = [
            "id", "name", "tire_size", "tire_circumference_meters",
            "weight_lbs", "current_setup"
        ]

        for vehicle in data["vehicles"]:
            for field in required_fields:
                assert field in vehicle, f"Vehicle {vehicle.get('id', 'unknown')} missing {field}"

    def test_all_vehicles_have_engine_specs(self):
        """Test that all vehicles have engine specifications"""
        json_path = Path(__file__).parent.parent / "data" / "vehicles.json"
        with open(json_path) as f:
            data = json.load(f)

        for vehicle in data["vehicles"]:
            assert "engine" in vehicle, f"Vehicle {vehicle['id']} missing engine"
            engine = vehicle["engine"]
            assert "max_rpm" in engine
            assert "shift_rpm" in engine
