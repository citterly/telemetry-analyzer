"""
Track database for multi-track support.

Provides GPS coordinates for start/finish lines, corners, and track detection
based on GPS data from telemetry sessions.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import math


@dataclass
class Corner:
    """A corner/turn on a track"""
    name: str
    lat: float
    lon: float
    apex_speed_mph: Optional[float] = None
    gear: Optional[int] = None


@dataclass
class Track:
    """Complete track configuration"""
    id: str
    name: str
    location: str
    country: str
    length_meters: float
    length_miles: float
    corner_count: int
    start_finish_lat: float
    start_finish_lon: float
    corners: Dict[str, Corner] = field(default_factory=dict)
    gps_bounds: Dict[str, float] = field(default_factory=dict)  # min_lat, max_lat, min_lon, max_lon
    main_straight_length: float = 0.0
    elevation_change_meters: float = 0.0
    lap_record_seconds: Optional[float] = None
    min_lap_time_seconds: float = 60.0
    max_lap_time_seconds: float = 600.0

    @property
    def start_finish_gps(self) -> Tuple[float, float]:
        """Return start/finish as (lat, lon) tuple for compatibility"""
        return (self.start_finish_lat, self.start_finish_lon)

    @property
    def turn_coordinates(self) -> Dict[str, Tuple[float, float]]:
        """Return corner coordinates as dict for compatibility"""
        return {name: (corner.lat, corner.lon) for name, corner in self.corners.items()}

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "location": self.location,
            "country": self.country,
            "length_meters": self.length_meters,
            "length_miles": self.length_miles,
            "corner_count": self.corner_count,
            "start_finish_lat": self.start_finish_lat,
            "start_finish_lon": self.start_finish_lon,
            "corners": {
                name: {
                    "lat": c.lat,
                    "lon": c.lon,
                    "apex_speed_mph": c.apex_speed_mph,
                    "gear": c.gear
                }
                for name, c in self.corners.items()
            },
            "gps_bounds": self.gps_bounds,
            "main_straight_length": self.main_straight_length,
            "elevation_change_meters": self.elevation_change_meters,
            "lap_record_seconds": self.lap_record_seconds,
            "min_lap_time_seconds": self.min_lap_time_seconds,
            "max_lap_time_seconds": self.max_lap_time_seconds
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Track":
        """Create Track from dictionary"""
        corners = {}
        for name, cdata in data.get("corners", {}).items():
            corners[name] = Corner(
                name=name,
                lat=cdata["lat"],
                lon=cdata["lon"],
                apex_speed_mph=cdata.get("apex_speed_mph"),
                gear=cdata.get("gear")
            )

        return cls(
            id=data["id"],
            name=data["name"],
            location=data.get("location", ""),
            country=data.get("country", "USA"),
            length_meters=data["length_meters"],
            length_miles=data.get("length_miles", data["length_meters"] / 1609.34),
            corner_count=data["corner_count"],
            start_finish_lat=data["start_finish_lat"],
            start_finish_lon=data["start_finish_lon"],
            corners=corners,
            gps_bounds=data.get("gps_bounds", {}),
            main_straight_length=data.get("main_straight_length", 0),
            elevation_change_meters=data.get("elevation_change_meters", 0),
            lap_record_seconds=data.get("lap_record_seconds"),
            min_lap_time_seconds=data.get("min_lap_time_seconds", 60),
            max_lap_time_seconds=data.get("max_lap_time_seconds", 600)
        )


class TrackDatabase:
    """
    Database of track configurations.

    Loads tracks from JSON file and provides lookup and auto-detection.
    """

    def __init__(self, json_path: Optional[str] = None):
        """
        Initialize track database.

        Args:
            json_path: Path to tracks.json file. If None, uses default location.
        """
        self.tracks: Dict[str, Track] = {}
        self._json_path = json_path

        if json_path:
            self.load_from_json(json_path)
        else:
            self._load_default()

    def _load_default(self) -> None:
        """Load from default location relative to this file"""
        default_path = Path(__file__).parent.parent.parent / "data" / "tracks.json"
        if default_path.exists():
            self.load_from_json(str(default_path))
        else:
            # Create with built-in Road America config
            self._add_builtin_tracks()

    def _add_builtin_tracks(self) -> None:
        """Add built-in track configurations"""
        # Road America
        road_america = Track(
            id="road-america",
            name="Road America",
            location="Elkhart Lake, Wisconsin",
            country="USA",
            length_meters=4048,
            length_miles=2.52,
            corner_count=14,
            start_finish_lat=43.797875,
            start_finish_lon=-87.989638,
            main_straight_length=800,
            elevation_change_meters=30,
            min_lap_time_seconds=90,
            max_lap_time_seconds=300,
            gps_bounds={
                "min_lat": 43.790,
                "max_lat": 43.806,
                "min_lon": -88.005,
                "max_lon": -87.988
            },
            corners={
                "T1": Corner("T1", 43.792069, -87.989800),
                "T3": Corner("T3", 43.791595, -87.995327),
                "T5": Corner("T5", 43.801770, -87.992596),
                "T6": Corner("T6", 43.801592, -87.996089),
                "T7": Corner("T7", 43.799578, -87.996243),
                "T8": Corner("T8", 43.797101, -87.999953),
                "Carousel": Corner("Carousel", 43.794216, -87.999916),
                "Kink": Corner("Kink", 43.798512, -88.002654),
                "T12": Corner("T12", 43.804928, -87.997402),
                "T13": Corner("T13", 43.803806, -87.994009),
                "T14": Corner("T14", 43.804001, -87.990058)
            }
        )
        self.tracks[road_america.id] = road_america

    def load_from_json(self, path: str) -> None:
        """
        Load tracks from JSON file.

        Args:
            path: Path to JSON file
        """
        with open(path, 'r') as f:
            data = json.load(f)

        self.tracks = {}
        for track_data in data.get("tracks", []):
            track = Track.from_dict(track_data)
            self.tracks[track.id] = track

    def save_to_json(self, path: str) -> None:
        """
        Save tracks to JSON file.

        Args:
            path: Path to JSON file
        """
        data = {
            "tracks": [track.to_dict() for track in self.tracks.values()]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def get(self, track_id: str) -> Optional[Track]:
        """
        Get track by ID.

        Args:
            track_id: Track identifier

        Returns:
            Track or None if not found
        """
        return self.tracks.get(track_id)

    def get_by_name(self, name: str) -> Optional[Track]:
        """
        Get track by name (case-insensitive).

        Args:
            name: Track name

        Returns:
            Track or None if not found
        """
        name_lower = name.lower()
        for track in self.tracks.values():
            if track.name.lower() == name_lower:
                return track
        return None

    def list_tracks(self) -> List[Track]:
        """Get list of all tracks"""
        return list(self.tracks.values())

    def detect_track(
        self,
        lat_data,
        lon_data,
        threshold: float = 0.8
    ) -> Optional[Track]:
        """
        Detect which track GPS data is from based on bounding box overlap.

        Args:
            lat_data: Array of latitude values
            lon_data: Array of longitude values
            threshold: Minimum overlap ratio required (0-1)

        Returns:
            Best matching Track or None if no match
        """
        import numpy as np

        # Get data bounds
        min_lat = np.nanmin(lat_data)
        max_lat = np.nanmax(lat_data)
        min_lon = np.nanmin(lon_data)
        max_lon = np.nanmax(lon_data)

        best_match = None
        best_score = 0.0

        for track in self.tracks.values():
            bounds = track.gps_bounds
            if not bounds:
                continue

            # Calculate overlap
            overlap_lat = max(0, min(max_lat, bounds["max_lat"]) - max(min_lat, bounds["min_lat"]))
            overlap_lon = max(0, min(max_lon, bounds["max_lon"]) - max(min_lon, bounds["min_lon"]))

            data_area = (max_lat - min_lat) * (max_lon - min_lon)
            if data_area <= 0:
                continue

            overlap_area = overlap_lat * overlap_lon
            score = overlap_area / data_area

            if score > best_score:
                best_score = score
                best_match = track

        if best_score >= threshold:
            return best_match

        # Try distance-based detection if bounding box fails
        return self._detect_by_start_finish(lat_data, lon_data)

    def _detect_by_start_finish(
        self,
        lat_data,
        lon_data,
        max_distance_meters: float = 500
    ) -> Optional[Track]:
        """
        Detect track by proximity to start/finish line.

        Args:
            lat_data: Array of latitude values
            lon_data: Array of longitude values
            max_distance_meters: Maximum distance to consider a match

        Returns:
            Best matching Track or None
        """
        import numpy as np

        best_match = None
        best_distance = float('inf')

        for track in self.tracks.values():
            # Find minimum distance to start/finish
            distances = self._haversine_distance(
                lat_data, lon_data,
                track.start_finish_lat, track.start_finish_lon
            )
            min_dist = np.nanmin(distances)

            if min_dist < best_distance:
                best_distance = min_dist
                best_match = track

        if best_distance <= max_distance_meters:
            return best_match
        return None

    @staticmethod
    def _haversine_distance(lat1, lon1, lat2: float, lon2: float) -> float:
        """
        Calculate distance in meters using Haversine formula.

        Works with scalar or array inputs for lat1/lon1.
        """
        import numpy as np

        R = 6371000  # Earth radius in meters

        lat1_rad = np.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)

        a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * math.cos(lat2_rad) * np.sin(delta_lon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        return R * c


# Singleton instance
_track_db: Optional[TrackDatabase] = None


def get_track_database() -> TrackDatabase:
    """Get the singleton TrackDatabase instance"""
    global _track_db
    if _track_db is None:
        _track_db = TrackDatabase()
    return _track_db


def get_track(track_id: str) -> Optional[Track]:
    """Convenience function to get a track by ID"""
    return get_track_database().get(track_id)


def get_track_by_name(name: str) -> Optional[Track]:
    """Convenience function to get a track by name"""
    return get_track_database().get_by_name(name)


def detect_track(lat_data, lon_data) -> Optional[Track]:
    """Convenience function to auto-detect track from GPS data"""
    return get_track_database().detect_track(lat_data, lon_data)


# Backward compatibility: export a TRACK_CONFIG-like dict for Road America
def get_default_track_config() -> dict:
    """Get Road America config in legacy format"""
    db = get_track_database()
    track = db.get("road-america")
    if track:
        return {
            'name': track.name,
            'length_meters': track.length_meters,
            'main_straight_length': track.main_straight_length,
            'corner_count': track.corner_count,
            'start_finish_gps': track.start_finish_gps,
            'turn_coordinates': track.turn_coordinates
        }
    return {}
