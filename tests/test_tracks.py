"""
Tests for track database (feat-040)
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json


class TestTrackModel:
    """Tests for Track dataclass"""

    def test_track_creation(self):
        """Test creating a Track instance"""
        from src.config.tracks import Track, Corner

        track = Track(
            id="test-track",
            name="Test Track",
            location="Test City",
            country="USA",
            length_meters=3000,
            length_miles=1.86,
            corner_count=10,
            start_finish_lat=40.0,
            start_finish_lon=-80.0
        )

        assert track.id == "test-track"
        assert track.name == "Test Track"
        assert track.length_meters == 3000
        assert track.corner_count == 10

    def test_track_start_finish_gps_property(self):
        """Test start_finish_gps tuple property"""
        from src.config.tracks import Track

        track = Track(
            id="test", name="Test", location="", country="",
            length_meters=1000, length_miles=0.62,
            corner_count=5,
            start_finish_lat=43.123,
            start_finish_lon=-87.456
        )

        assert track.start_finish_gps == (43.123, -87.456)

    def test_track_turn_coordinates_property(self):
        """Test turn_coordinates dict property"""
        from src.config.tracks import Track, Corner

        track = Track(
            id="test", name="Test", location="", country="",
            length_meters=1000, length_miles=0.62,
            corner_count=2,
            start_finish_lat=43.0, start_finish_lon=-87.0,
            corners={
                "T1": Corner("T1", 43.1, -87.1),
                "T2": Corner("T2", 43.2, -87.2)
            }
        )

        turns = track.turn_coordinates
        assert "T1" in turns
        assert turns["T1"] == (43.1, -87.1)
        assert turns["T2"] == (43.2, -87.2)

    def test_track_to_dict(self):
        """Test converting track to dictionary"""
        from src.config.tracks import Track

        track = Track(
            id="test", name="Test Track", location="City", country="USA",
            length_meters=3000, length_miles=1.86,
            corner_count=10,
            start_finish_lat=40.0, start_finish_lon=-80.0
        )

        d = track.to_dict()
        assert d["id"] == "test"
        assert d["name"] == "Test Track"
        assert d["length_meters"] == 3000

    def test_track_from_dict(self):
        """Test creating track from dictionary"""
        from src.config.tracks import Track

        data = {
            "id": "test",
            "name": "Test Track",
            "location": "City",
            "country": "USA",
            "length_meters": 3000,
            "corner_count": 10,
            "start_finish_lat": 40.0,
            "start_finish_lon": -80.0,
            "corners": {
                "T1": {"lat": 40.1, "lon": -80.1}
            }
        }

        track = Track.from_dict(data)
        assert track.id == "test"
        assert track.name == "Test Track"
        assert "T1" in track.corners
        assert track.corners["T1"].lat == 40.1


class TestTrackDatabase:
    """Tests for TrackDatabase class"""

    def test_database_loads_default(self):
        """Test database loads tracks from default location"""
        from src.config.tracks import TrackDatabase

        db = TrackDatabase()
        assert len(db.tracks) > 0

    def test_database_has_road_america(self):
        """Test database includes Road America"""
        from src.config.tracks import TrackDatabase

        db = TrackDatabase()
        ra = db.get("road-america")

        assert ra is not None
        assert ra.name == "Road America"
        assert ra.length_meters == 4048

    def test_get_by_id(self):
        """Test getting track by ID"""
        from src.config.tracks import TrackDatabase

        db = TrackDatabase()
        track = db.get("road-america")
        assert track is not None
        assert track.id == "road-america"

    def test_get_by_name(self):
        """Test getting track by name"""
        from src.config.tracks import TrackDatabase

        db = TrackDatabase()
        track = db.get_by_name("Road America")
        assert track is not None
        assert track.id == "road-america"

    def test_get_by_name_case_insensitive(self):
        """Test name lookup is case-insensitive"""
        from src.config.tracks import TrackDatabase

        db = TrackDatabase()
        track = db.get_by_name("road america")
        assert track is not None
        assert track.id == "road-america"

    def test_list_tracks(self):
        """Test listing all tracks"""
        from src.config.tracks import TrackDatabase

        db = TrackDatabase()
        tracks = db.list_tracks()
        assert len(tracks) >= 1
        assert any(t.id == "road-america" for t in tracks)

    def test_get_nonexistent_track(self):
        """Test getting a track that doesn't exist"""
        from src.config.tracks import TrackDatabase

        db = TrackDatabase()
        track = db.get("nonexistent")
        assert track is None

    def test_load_from_json(self):
        """Test loading tracks from JSON file"""
        from src.config.tracks import TrackDatabase

        # Use the actual tracks.json file
        json_path = Path(__file__).parent.parent / "data" / "tracks.json"
        if json_path.exists():
            db = TrackDatabase(str(json_path))
            assert len(db.tracks) >= 1

    def test_save_to_json(self):
        """Test saving tracks to JSON file"""
        from src.config.tracks import TrackDatabase, Track

        db = TrackDatabase()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            db.save_to_json(f.name)

            # Reload and verify
            with open(f.name, 'r') as rf:
                data = json.load(rf)
                assert "tracks" in data
                assert len(data["tracks"]) > 0


class TestTrackDetection:
    """Tests for auto-detecting track from GPS data"""

    def test_detect_road_america(self):
        """Test detecting Road America from GPS data"""
        from src.config.tracks import TrackDatabase

        db = TrackDatabase()

        # Generate fake GPS data within Road America bounds
        lat_data = np.linspace(43.792, 43.804, 100)
        lon_data = np.linspace(-87.990, -88.000, 100)

        track = db.detect_track(lat_data, lon_data)
        assert track is not None
        assert track.id == "road-america"

    def test_detect_watkins_glen(self):
        """Test detecting Watkins Glen from GPS data"""
        from src.config.tracks import TrackDatabase

        db = TrackDatabase()

        # Generate fake GPS data within Watkins Glen bounds
        lat_data = np.linspace(42.328, 42.340, 100)
        lon_data = np.linspace(-76.930, -76.915, 100)

        track = db.detect_track(lat_data, lon_data)
        assert track is not None
        assert track.id == "watkins-glen"

    def test_detect_unknown_location(self):
        """Test that unknown location returns None"""
        from src.config.tracks import TrackDatabase

        db = TrackDatabase()

        # GPS data from somewhere not in database
        lat_data = np.linspace(51.0, 51.01, 100)  # UK somewhere
        lon_data = np.linspace(-0.5, -0.51, 100)

        track = db.detect_track(lat_data, lon_data)
        assert track is None

    def test_detect_by_start_finish_proximity(self):
        """Test detection works near start/finish line"""
        from src.config.tracks import TrackDatabase

        db = TrackDatabase()

        # Data that passes near Road America start/finish
        ra = db.get("road-america")
        sf_lat, sf_lon = ra.start_finish_gps

        lat_data = np.array([sf_lat, sf_lat + 0.001, sf_lat + 0.002])
        lon_data = np.array([sf_lon, sf_lon + 0.001, sf_lon + 0.002])

        track = db.detect_track(lat_data, lon_data)
        assert track is not None
        assert track.id == "road-america"


class TestConvenienceFunctions:
    """Tests for module-level convenience functions"""

    def test_get_track_database(self):
        """Test get_track_database returns singleton"""
        from src.config.tracks import get_track_database

        db1 = get_track_database()
        db2 = get_track_database()
        assert db1 is db2

    def test_get_track(self):
        """Test get_track convenience function"""
        from src.config.tracks import get_track

        track = get_track("road-america")
        assert track is not None
        assert track.name == "Road America"

    def test_get_track_by_name(self):
        """Test get_track_by_name convenience function"""
        from src.config.tracks import get_track_by_name

        track = get_track_by_name("Road America")
        assert track is not None
        assert track.id == "road-america"

    def test_detect_track(self):
        """Test detect_track convenience function"""
        from src.config.tracks import detect_track

        lat_data = np.linspace(43.792, 43.804, 100)
        lon_data = np.linspace(-87.990, -88.000, 100)

        track = detect_track(lat_data, lon_data)
        assert track is not None
        assert track.id == "road-america"

    def test_get_default_track_config(self):
        """Test backward-compatible track config function"""
        from src.config.tracks import get_default_track_config

        config = get_default_track_config()
        assert "name" in config
        assert config["name"] == "Road America"
        assert "start_finish_gps" in config
        assert "turn_coordinates" in config


class TestTracksJsonFile:
    """Tests for the tracks.json data file"""

    def test_json_file_exists(self):
        """Test that tracks.json exists"""
        json_path = Path(__file__).parent.parent / "data" / "tracks.json"
        assert json_path.exists()

    def test_json_file_valid(self):
        """Test that tracks.json is valid JSON"""
        json_path = Path(__file__).parent.parent / "data" / "tracks.json"
        with open(json_path) as f:
            data = json.load(f)
        assert "tracks" in data

    def test_json_has_multiple_tracks(self):
        """Test that tracks.json has multiple tracks"""
        json_path = Path(__file__).parent.parent / "data" / "tracks.json"
        with open(json_path) as f:
            data = json.load(f)
        assert len(data["tracks"]) >= 3

    def test_all_tracks_have_required_fields(self):
        """Test that all tracks have required fields"""
        json_path = Path(__file__).parent.parent / "data" / "tracks.json"
        with open(json_path) as f:
            data = json.load(f)

        required_fields = [
            "id", "name", "length_meters", "corner_count",
            "start_finish_lat", "start_finish_lon"
        ]

        for track in data["tracks"]:
            for field in required_fields:
                assert field in track, f"Track {track.get('id', 'unknown')} missing {field}"

    def test_all_tracks_have_gps_bounds(self):
        """Test that all tracks have GPS bounds for detection"""
        json_path = Path(__file__).parent.parent / "data" / "tracks.json"
        with open(json_path) as f:
            data = json.load(f)

        for track in data["tracks"]:
            assert "gps_bounds" in track, f"Track {track['id']} missing gps_bounds"
            bounds = track["gps_bounds"]
            assert "min_lat" in bounds
            assert "max_lat" in bounds
            assert "min_lon" in bounds
            assert "max_lon" in bounds
