"""
Vehicle profiles for multi-vehicle support.

Provides vehicle configurations with tire sizes, gear ratios, weights,
and engine specifications for different cars and setups.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import math


@dataclass
class EngineSpec:
    """Engine specifications"""
    max_rpm: int = 8000
    safe_rpm_limit: int = 7000
    shift_rpm: int = 6800
    power_band_min: int = 5500
    power_band_max: int = 7000
    idle_rpm: int = 800

    def to_dict(self) -> dict:
        return {
            "max_rpm": self.max_rpm,
            "safe_rpm_limit": self.safe_rpm_limit,
            "shift_rpm": self.shift_rpm,
            "power_band_min": self.power_band_min,
            "power_band_max": self.power_band_max,
            "idle_rpm": self.idle_rpm
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EngineSpec":
        return cls(
            max_rpm=data.get("max_rpm", 8000),
            safe_rpm_limit=data.get("safe_rpm_limit", 7000),
            shift_rpm=data.get("shift_rpm", 6800),
            power_band_min=data.get("power_band_min", 5500),
            power_band_max=data.get("power_band_max", 7000),
            idle_rpm=data.get("idle_rpm", 800)
        )


@dataclass
class TransmissionSetup:
    """Transmission configuration"""
    name: str
    transmission_ratios: List[float]
    final_drive: float
    weight_lbs: float

    @property
    def gear_count(self) -> int:
        return len(self.transmission_ratios)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "transmission_ratios": self.transmission_ratios,
            "final_drive": self.final_drive,
            "weight_lbs": self.weight_lbs
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TransmissionSetup":
        return cls(
            name=data["name"],
            transmission_ratios=data["transmission_ratios"],
            final_drive=data["final_drive"],
            weight_lbs=data.get("weight_lbs", 3000)
        )


@dataclass
class Vehicle:
    """Complete vehicle configuration"""
    id: str
    name: str
    make: str
    model: str
    year: int
    tire_size: str
    tire_circumference_meters: float
    weight_lbs: float
    engine: EngineSpec
    current_setup: TransmissionSetup
    alternative_setups: List[TransmissionSetup] = field(default_factory=list)
    notes: str = ""
    # G-force limits for analysis
    max_lateral_g: float = 1.3
    max_braking_g: float = 1.4
    power_limited_accel_g: float = 0.5

    @property
    def weight_kg(self) -> float:
        return self.weight_lbs * 0.453592

    @property
    def transmission_ratios(self) -> List[float]:
        """Convenience property for current setup ratios"""
        return self.current_setup.transmission_ratios

    @property
    def final_drive(self) -> float:
        """Convenience property for current setup final drive"""
        return self.current_setup.final_drive

    @property
    def all_setups(self) -> List[TransmissionSetup]:
        """Get current setup plus all alternatives"""
        return [self.current_setup] + self.alternative_setups

    def get_setup_by_name(self, name: str) -> Optional[TransmissionSetup]:
        """Get a transmission setup by name"""
        for setup in self.all_setups:
            if setup.name.lower() == name.lower():
                return setup
        return None

    def calculate_speed_at_rpm(self, rpm: float, gear: int) -> float:
        """
        Calculate vehicle speed in mph at given RPM and gear.

        Args:
            rpm: Engine RPM
            gear: Gear number (1-indexed)

        Returns:
            Speed in mph
        """
        if gear < 1 or gear > len(self.transmission_ratios):
            return 0.0

        gear_ratio = self.transmission_ratios[gear - 1]
        overall_ratio = gear_ratio * self.final_drive

        wheel_rpm = rpm / overall_ratio
        wheel_speed_meters_per_min = wheel_rpm * self.tire_circumference_meters
        speed_mph = wheel_speed_meters_per_min * 60 / 1609.34

        return speed_mph

    def calculate_rpm_at_speed(self, speed_mph: float, gear: int) -> float:
        """
        Calculate engine RPM at given speed and gear.

        Args:
            speed_mph: Vehicle speed in mph
            gear: Gear number (1-indexed)

        Returns:
            Engine RPM
        """
        if gear < 1 or gear > len(self.transmission_ratios):
            return 0.0

        gear_ratio = self.transmission_ratios[gear - 1]
        overall_ratio = gear_ratio * self.final_drive

        speed_meters_per_min = speed_mph * 1609.34 / 60
        wheel_rpm = speed_meters_per_min / self.tire_circumference_meters
        engine_rpm = wheel_rpm * overall_ratio

        return engine_rpm

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "make": self.make,
            "model": self.model,
            "year": self.year,
            "tire_size": self.tire_size,
            "tire_circumference_meters": self.tire_circumference_meters,
            "weight_lbs": self.weight_lbs,
            "max_lateral_g": self.max_lateral_g,
            "max_braking_g": self.max_braking_g,
            "power_limited_accel_g": self.power_limited_accel_g,
            "engine": self.engine.to_dict(),
            "current_setup": self.current_setup.to_dict(),
            "alternative_setups": [s.to_dict() for s in self.alternative_setups],
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Vehicle":
        return cls(
            id=data["id"],
            name=data["name"],
            make=data.get("make", ""),
            model=data.get("model", ""),
            year=data.get("year", 0),
            tire_size=data["tire_size"],
            tire_circumference_meters=data["tire_circumference_meters"],
            weight_lbs=data["weight_lbs"],
            engine=EngineSpec.from_dict(data.get("engine", {})),
            current_setup=TransmissionSetup.from_dict(data["current_setup"]),
            alternative_setups=[
                TransmissionSetup.from_dict(s)
                for s in data.get("alternative_setups", [])
            ],
            notes=data.get("notes", ""),
            max_lateral_g=data.get("max_lateral_g", 1.3),
            max_braking_g=data.get("max_braking_g", 1.4),
            power_limited_accel_g=data.get("power_limited_accel_g", 0.5)
        )


class VehicleDatabase:
    """
    Database of vehicle configurations.

    Loads vehicles from JSON file and provides lookup and switching.
    """

    def __init__(self, json_path: Optional[str] = None):
        """
        Initialize vehicle database.

        Args:
            json_path: Path to vehicles.json file. If None, uses default location.
        """
        self.vehicles: Dict[str, Vehicle] = {}
        self._active_vehicle_id: Optional[str] = None
        self._json_path = json_path

        if json_path:
            self.load_from_json(json_path)
        else:
            self._load_default()

    def _load_default(self) -> None:
        """Load from default location relative to this file"""
        default_path = Path(__file__).parent.parent.parent / "data" / "vehicles.json"
        if default_path.exists():
            self.load_from_json(str(default_path))
        else:
            self._add_builtin_vehicles()

    def _add_builtin_vehicles(self) -> None:
        """Add built-in vehicle configurations"""
        # Andy's BMW
        bmw = Vehicle(
            id="bmw-e46-m3",
            name="Andy's M3",
            make="BMW",
            model="E46 M3",
            year=2003,
            tire_size="275/35/18",
            tire_circumference_meters=2.026,
            weight_lbs=3450,
            engine=EngineSpec(
                max_rpm=8000,
                safe_rpm_limit=7000,
                shift_rpm=6800,
                power_band_min=5500,
                power_band_max=7000,
                idle_rpm=800
            ),
            current_setup=TransmissionSetup(
                name="Current Setup",
                transmission_ratios=[2.20, 1.64, 1.28, 1.00],
                final_drive=3.55,
                weight_lbs=3450
            ),
            alternative_setups=[
                TransmissionSetup(
                    name="New Trans + Current Final",
                    transmission_ratios=[2.88, 1.91, 1.33, 1.00],
                    final_drive=3.55,
                    weight_lbs=3400
                ),
                TransmissionSetup(
                    name="New Trans + Shorter Final",
                    transmission_ratios=[2.88, 1.91, 1.33, 1.00],
                    final_drive=3.73,
                    weight_lbs=3400
                ),
                TransmissionSetup(
                    name="New Trans + Taller Final",
                    transmission_ratios=[2.88, 1.91, 1.33, 1.00],
                    final_drive=3.31,
                    weight_lbs=3400
                )
            ],
            notes="Primary vehicle for telemetry analysis"
        )
        self.vehicles[bmw.id] = bmw
        self._active_vehicle_id = bmw.id

    def load_from_json(self, path: str) -> None:
        """Load vehicles from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)

        self.vehicles = {}
        for vehicle_data in data.get("vehicles", []):
            vehicle = Vehicle.from_dict(vehicle_data)
            self.vehicles[vehicle.id] = vehicle

        # Set active vehicle
        active_id = data.get("active_vehicle")
        if active_id and active_id in self.vehicles:
            self._active_vehicle_id = active_id
        elif self.vehicles:
            self._active_vehicle_id = list(self.vehicles.keys())[0]

    def save_to_json(self, path: str) -> None:
        """Save vehicles to JSON file"""
        data = {
            "active_vehicle": self._active_vehicle_id,
            "vehicles": [v.to_dict() for v in self.vehicles.values()]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def get(self, vehicle_id: str) -> Optional[Vehicle]:
        """Get vehicle by ID"""
        return self.vehicles.get(vehicle_id)

    def get_by_name(self, name: str) -> Optional[Vehicle]:
        """Get vehicle by name (case-insensitive)"""
        name_lower = name.lower()
        for vehicle in self.vehicles.values():
            if vehicle.name.lower() == name_lower:
                return vehicle
        return None

    def list_vehicles(self) -> List[Vehicle]:
        """Get list of all vehicles"""
        return list(self.vehicles.values())

    @property
    def active_vehicle(self) -> Optional[Vehicle]:
        """Get the currently active vehicle"""
        if self._active_vehicle_id:
            return self.vehicles.get(self._active_vehicle_id)
        return None

    def add_vehicle(self, vehicle: Vehicle) -> None:
        """Add a vehicle to the database"""
        self.vehicles[vehicle.id] = vehicle
        if self._active_vehicle_id is None:
            self._active_vehicle_id = vehicle.id

    def get_active_vehicle_id(self) -> Optional[str]:
        """Get the ID of the active vehicle"""
        return self._active_vehicle_id

    def update_vehicle(self, vehicle_id: str, data: dict) -> bool:
        """
        Update a vehicle's parameters and persist to JSON.

        Args:
            vehicle_id: ID of vehicle to update
            data: Dictionary of updated fields

        Returns:
            True if successful
        """
        if vehicle_id not in self.vehicles:
            return False

        # Create updated vehicle from data
        updated = Vehicle.from_dict(data)
        self.vehicles[vehicle_id] = updated

        # Persist to JSON
        self._persist()
        return True

    def _persist(self) -> None:
        """Save current state to JSON file"""
        json_path = self._json_path
        if json_path is None:
            json_path = str(Path(__file__).parent.parent.parent / "data" / "vehicles.json")
        self.save_to_json(json_path)

    def set_active_vehicle(self, vehicle_id: str) -> bool:
        """
        Set the active vehicle and persist to JSON.

        Args:
            vehicle_id: ID of vehicle to activate

        Returns:
            True if successful, False if vehicle not found
        """
        if vehicle_id in self.vehicles:
            self._active_vehicle_id = vehicle_id
            self._persist()
            return True
        return False


# Singleton instance
_vehicle_db: Optional[VehicleDatabase] = None


def get_vehicle_database() -> VehicleDatabase:
    """Get the singleton VehicleDatabase instance"""
    global _vehicle_db
    if _vehicle_db is None:
        _vehicle_db = VehicleDatabase()
    return _vehicle_db


def get_vehicle(vehicle_id: str) -> Optional[Vehicle]:
    """Convenience function to get a vehicle by ID"""
    return get_vehicle_database().get(vehicle_id)


def get_vehicle_by_name(name: str) -> Optional[Vehicle]:
    """Convenience function to get a vehicle by name"""
    return get_vehicle_database().get_by_name(name)


def get_active_vehicle() -> Optional[Vehicle]:
    """Convenience function to get the active vehicle"""
    return get_vehicle_database().active_vehicle


def set_active_vehicle(vehicle_id: str) -> bool:
    """Convenience function to set the active vehicle"""
    return get_vehicle_database().set_active_vehicle(vehicle_id)


# Backward compatibility exports
def get_current_setup() -> dict:
    """Get current setup in legacy dict format"""
    vehicle = get_active_vehicle()
    if vehicle:
        return vehicle.current_setup.to_dict()
    return {}


def get_transmission_scenarios() -> List[dict]:
    """Get all transmission scenarios in legacy list format"""
    vehicle = get_active_vehicle()
    if vehicle:
        return [s.to_dict() for s in vehicle.all_setups]
    return []


def get_engine_specs() -> dict:
    """Get engine specs in legacy dict format"""
    vehicle = get_active_vehicle()
    if vehicle:
        return vehicle.engine.to_dict()
    return {}


def get_tire_circumference() -> float:
    """Get tire circumference of active vehicle in meters."""
    vehicle = get_active_vehicle()
    if vehicle:
        return vehicle.tire_circumference_meters
    return 2.026  # Default BMW M3 275/35/18


def get_processing_config() -> dict:
    """Get data processing config (lap time bounds, thresholds, smoothing)."""
    from .tracks import get_track_database
    db = get_track_database()
    track = db.get("road-america")  # default
    min_lap = track.min_lap_time_seconds if track else 90
    max_lap = track.max_lap_time_seconds if track else 300
    return {
        'min_lap_time_seconds': min_lap,
        'max_lap_time_seconds': max_lap,
        'start_finish_threshold': 0.0001,
        'rpm_interpolation_method': 'linear',
        'smoothing_window': 5,
    }


# Speed/RPM utility functions (legacy interface, m/s based)

def theoretical_speed_at_rpm(
    rpm: float,
    gear_ratio: float,
    final_drive: float,
    tire_circumference: float = None,
) -> float:
    """Calculate theoretical speed in m/s at given RPM and gear ratios."""
    if rpm <= 0 or gear_ratio <= 0 or final_drive <= 0:
        return 0.0
    if tire_circumference is None:
        tire_circumference = get_tire_circumference()
    return (rpm / 60) * tire_circumference / (gear_ratio * final_drive)


def theoretical_rpm_at_speed(
    speed_ms: float,
    gear_ratio: float,
    final_drive: float,
    tire_circumference: float = None,
) -> float:
    """Calculate theoretical RPM at given speed (m/s) and gear ratios."""
    if speed_ms <= 0 or gear_ratio <= 0 or final_drive <= 0:
        return 0.0
    if tire_circumference is None:
        tire_circumference = get_tire_circumference()
    return (speed_ms * 60 * gear_ratio * final_drive) / tire_circumference


def calculate_tire_circumference(tire_size: str) -> float:
    """Calculate tire circumference from tire size string (e.g. '275/35/18')."""
    try:
        parts = tire_size.split('/')
        width_mm = int(parts[0])
        aspect_ratio = int(parts[1])
        wheel_diameter_inches = int(parts[2])
        sidewall_mm = width_mm * (aspect_ratio / 100)
        wheel_diameter_mm = wheel_diameter_inches * 25.4
        tire_diameter_mm = wheel_diameter_mm + (2 * sidewall_mm)
        return (tire_diameter_mm / 1000) * 3.14159
    except (ValueError, IndexError):
        return 2.026


# Default session constant (not vehicle-specific)
DEFAULT_SESSION = "20250712_104619_Road America_a_0394.xrk"
