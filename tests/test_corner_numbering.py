"""
Tests for Phase 7: Corner Numbering and Persistence
"""

import pytest
import numpy as np
import tempfile
import json
import os
from pathlib import Path

from src.features.corner_detection import (
    CornerZone,
    calculate_gps_distance,
    load_track_corners,
    save_track_corners,
    match_corner_to_definition,
    number_corners_by_track_position,
    apply_corner_definitions,
    rename_corner
)


class TestGPSDistance:
    """Tests for GPS distance calculation"""

    def test_same_point_zero_distance(self):
        """Test that distance from a point to itself is zero"""
        lat, lon = 43.797, -87.989
        distance = calculate_gps_distance(lat, lon, lat, lon)
        assert distance < 0.01  # Should be very close to 0

    def test_known_distance(self):
        """Test with known GPS coordinates"""
        # Two points approximately 100 meters apart
        lat1, lon1 = 43.797, -87.989
        lat2, lon2 = 43.798, -87.989  # Roughly 111m north

        distance = calculate_gps_distance(lat1, lon1, lat2, lon2)
        # Should be close to 111 meters (1 degree latitude ≈ 111km)
        assert 100 < distance < 120

    def test_east_west_distance(self):
        """Test east-west distance"""
        lat, lon1, lon2 = 43.797, -87.989, -87.988

        distance = calculate_gps_distance(lat, lon1, lat, lon2)
        # Should be roughly 1 degree lon * 111km * cos(lat)
        assert distance > 0

    def test_symmetric(self):
        """Test that distance is symmetric"""
        lat1, lon1 = 43.797, -87.989
        lat2, lon2 = 43.798, -87.990

        dist1 = calculate_gps_distance(lat1, lon1, lat2, lon2)
        dist2 = calculate_gps_distance(lat2, lon2, lat1, lon1)

        assert abs(dist1 - dist2) < 0.01


class TestLoadSaveTrackCorners:
    """Tests for loading and saving track corner definitions"""

    @pytest.fixture
    def temp_corners_file(self):
        """Create a temporary corners file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            yield f.name
        # Cleanup
        if os.path.exists(f.name):
            os.unlink(f.name)

    @pytest.fixture
    def sample_corners(self):
        """Create sample corner zones"""
        return [
            CornerZone(
                name="T1",
                alias="Big Bend",
                apex_lat=43.792,
                apex_lon=-87.990,
                entry_lat=43.791,
                entry_lon=-87.989,
                exit_lat=43.793,
                exit_lon=-87.991,
                corner_type="normal",
                direction="left"
            ),
            CornerZone(
                name="T2",
                apex_lat=43.795,
                apex_lon=-87.992,
                entry_lat=43.794,
                entry_lon=-87.991,
                exit_lat=43.796,
                exit_lon=-87.993,
                corner_type="sweeper",
                direction="right"
            )
        ]

    def test_save_new_track_corners(self, temp_corners_file, sample_corners):
        """Test saving corners for a new track"""
        save_track_corners("Test Track", sample_corners, temp_corners_file)

        # Verify file was created
        assert os.path.exists(temp_corners_file)

        # Verify content
        with open(temp_corners_file, 'r') as f:
            data = json.load(f)

        assert "test_track" in data
        assert len(data["test_track"]) == 2
        assert data["test_track"][0]["name"] == "T1"
        assert data["test_track"][0]["alias"] == "Big Bend"

    def test_load_existing_track_corners(self, temp_corners_file, sample_corners):
        """Test loading corners for an existing track"""
        # Save first
        save_track_corners("Test Track", sample_corners, temp_corners_file)

        # Load
        corners = load_track_corners("Test Track", temp_corners_file)

        assert corners is not None
        assert len(corners) == 2
        assert corners[0]["name"] == "T1"
        assert corners[0]["alias"] == "Big Bend"

    def test_load_nonexistent_track(self, temp_corners_file):
        """Test loading corners for a track that doesn't exist"""
        corners = load_track_corners("Nonexistent Track", temp_corners_file)
        assert corners is None

    def test_load_from_nonexistent_file(self):
        """Test loading from a file that doesn't exist"""
        corners = load_track_corners("Any Track", "nonexistent_file.json")
        assert corners is None

    def test_save_updates_existing_track(self, temp_corners_file, sample_corners):
        """Test that saving updates an existing track's corners"""
        # Save initial
        save_track_corners("Test Track", sample_corners, temp_corners_file)

        # Update corners
        updated_corners = [sample_corners[0]]  # Only one corner now
        updated_corners[0].name = "T1_Updated"

        save_track_corners("Test Track", updated_corners, temp_corners_file)

        # Load and verify
        corners = load_track_corners("Test Track", temp_corners_file)
        assert len(corners) == 1
        assert corners[0]["name"] == "T1_Updated"

    def test_save_multiple_tracks(self, temp_corners_file, sample_corners):
        """Test saving corners for multiple tracks"""
        save_track_corners("Track A", sample_corners[:1], temp_corners_file)
        save_track_corners("Track B", sample_corners[1:], temp_corners_file)

        # Load both
        corners_a = load_track_corners("Track A", temp_corners_file)
        corners_b = load_track_corners("Track B", temp_corners_file)

        assert len(corners_a) == 1
        assert len(corners_b) == 1
        assert corners_a[0]["name"] == "T1"
        assert corners_b[0]["name"] == "T2"

    def test_track_name_normalization(self, temp_corners_file, sample_corners):
        """Test that track names are normalized (lowercase, underscores)"""
        save_track_corners("Test Track Name", sample_corners, temp_corners_file)

        # Should be able to load with different casing/spacing
        corners = load_track_corners("test track name", temp_corners_file)
        assert corners is not None


class TestMatchCornerToDefinition:
    """Tests for matching detected corners to stored definitions"""

    @pytest.fixture
    def sample_definitions(self):
        """Create sample corner definitions"""
        return [
            {
                "name": "T1",
                "alias": "Big Bend",
                "apex_lat": 43.792,
                "apex_lon": -87.990,
                "entry_lat": 43.791,
                "entry_lon": -87.989,
                "exit_lat": 43.793,
                "exit_lon": -87.991,
                "corner_type": "normal",
                "direction": "left"
            },
            {
                "name": "T2",
                "apex_lat": 43.795,
                "apex_lon": -87.992,
                "entry_lat": 43.794,
                "entry_lon": -87.991,
                "exit_lat": 43.796,
                "exit_lon": -87.993,
                "corner_type": "sweeper",
                "direction": "right"
            }
        ]

    def test_exact_match(self, sample_definitions):
        """Test matching with exact GPS coordinates"""
        corner = CornerZone(
            name="Unknown",
            apex_lat=43.792,
            apex_lon=-87.990
        )

        match = match_corner_to_definition(corner, sample_definitions)

        assert match is not None
        assert match["name"] == "T1"

    def test_close_match(self, sample_definitions):
        """Test matching with GPS coordinates within threshold"""
        # 20 meters away from T1
        corner = CornerZone(
            name="Unknown",
            apex_lat=43.7921,  # Slightly offset
            apex_lon=-87.9901
        )

        match = match_corner_to_definition(corner, sample_definitions, proximity_threshold=50.0)

        assert match is not None
        assert match["name"] == "T1"

    def test_no_match_too_far(self, sample_definitions):
        """Test that corners too far away don't match"""
        # 500 meters away
        corner = CornerZone(
            name="Unknown",
            apex_lat=43.800,
            apex_lon=-87.995
        )

        match = match_corner_to_definition(corner, sample_definitions, proximity_threshold=50.0)

        assert match is None

    def test_matches_closest(self, sample_definitions):
        """Test that the closest corner is matched"""
        # Closer to T1 than T2
        corner = CornerZone(
            name="Unknown",
            apex_lat=43.7925,
            apex_lon=-87.9905
        )

        match = match_corner_to_definition(corner, sample_definitions, proximity_threshold=100.0)

        assert match is not None
        assert match["name"] == "T1"

    def test_respects_threshold(self, sample_definitions):
        """Test that proximity threshold is respected"""
        corner = CornerZone(
            name="Unknown",
            apex_lat=43.7921,
            apex_lon=-87.9901
        )

        # Should match with large threshold
        match = match_corner_to_definition(corner, sample_definitions, proximity_threshold=100.0)
        assert match is not None

        # Should not match with very small threshold
        match = match_corner_to_definition(corner, sample_definitions, proximity_threshold=5.0)
        assert match is None


class TestNumberCornersByTrackPosition:
    """Tests for numbering corners by track position"""

    def test_numbers_corners_sequentially(self):
        """Test that corners are numbered T1, T2, T3..."""
        corners = [
            CornerZone(name="Unknown1", entry_lat=43.791, entry_lon=-87.989),
            CornerZone(name="Unknown2", entry_lat=43.792, entry_lon=-87.990),
            CornerZone(name="Unknown3", entry_lat=43.793, entry_lon=-87.991)
        ]

        numbered = number_corners_by_track_position(corners)

        assert numbered[0].name == "T1"
        assert numbered[1].name == "T2"
        assert numbered[2].name == "T3"

    def test_reorders_from_start_finish(self):
        """Test that corners are reordered starting from start/finish line"""
        corners = [
            CornerZone(name="A", entry_lat=43.793, entry_lon=-87.991),
            CornerZone(name="B", entry_lat=43.791, entry_lon=-87.989),  # Closest to start
            CornerZone(name="C", entry_lat=43.792, entry_lon=-87.990)
        ]

        # Start/finish near corner B
        numbered = number_corners_by_track_position(
            corners,
            start_lat=43.791,
            start_lon=-87.989
        )

        # Corner B should be T1 since it's closest to start
        assert numbered[0].entry_lat == 43.791
        assert numbered[0].name == "T1"

    def test_handles_empty_list(self):
        """Test with empty corner list"""
        corners = []
        numbered = number_corners_by_track_position(corners)
        assert numbered == []

    def test_preserves_other_fields(self):
        """Test that numbering preserves other corner fields"""
        corners = [
            CornerZone(
                name="Unknown",
                alias="Big Bend",
                apex_lat=43.792,
                apex_lon=-87.990,
                corner_type="hairpin",
                direction="left"
            )
        ]

        numbered = number_corners_by_track_position(corners)

        assert numbered[0].name == "T1"
        assert numbered[0].alias == "Big Bend"
        assert numbered[0].corner_type == "hairpin"
        assert numbered[0].direction == "left"


class TestApplyCornerDefinitions:
    """Tests for applying corner definitions (main Phase 7 API)"""

    @pytest.fixture
    def temp_corners_file(self):
        """Create a temporary corners file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            yield f.name
        if os.path.exists(f.name):
            os.unlink(f.name)

    def test_new_track_creates_numbered_corners(self, temp_corners_file):
        """Test that new tracks get corners numbered T1, T2, T3..."""
        corners = [
            CornerZone(name="Unknown", apex_lat=43.792, apex_lon=-87.990),
            CornerZone(name="Unknown", apex_lat=43.795, apex_lon=-87.992)
        ]

        result = apply_corner_definitions(
            corners,
            "New Track",
            temp_corners_file,
            auto_save_new=True
        )

        assert result[0].name == "T1"
        assert result[1].name == "T2"

        # Verify saved
        loaded = load_track_corners("New Track", temp_corners_file)
        assert loaded is not None
        assert len(loaded) == 2

    def test_existing_track_applies_names(self, temp_corners_file):
        """Test that existing track definitions are applied"""
        # Save some corner definitions
        saved_corners = [
            CornerZone(
                name="T1",
                alias="Carousel",
                apex_lat=43.792,
                apex_lon=-87.990
            )
        ]
        save_track_corners("Known Track", saved_corners, temp_corners_file)

        # Detect corners (same location)
        detected = [
            CornerZone(name="Unknown", apex_lat=43.7920, apex_lon=-87.9900)
        ]

        result = apply_corner_definitions(
            detected,
            "Known Track",
            temp_corners_file
        )

        assert result[0].name == "T1"
        assert result[0].alias == "Carousel"

    def test_new_corners_added_to_existing_track(self, temp_corners_file):
        """Test that new corners are added to an existing track"""
        # Save T1
        saved_corners = [
            CornerZone(name="T1", apex_lat=43.792, apex_lon=-87.990)
        ]
        save_track_corners("Track", saved_corners, temp_corners_file)

        # Detect T1 + a new corner
        detected = [
            CornerZone(name="Unknown", apex_lat=43.7920, apex_lon=-87.9900),
            CornerZone(name="Unknown", apex_lat=43.800, apex_lon=-88.000)  # New corner
        ]

        result = apply_corner_definitions(
            detected,
            "Track",
            temp_corners_file,
            auto_save_new=True
        )

        assert len(result) == 2
        assert result[0].name == "T1"  # Matched
        assert result[1].name == "T2"  # New corner

    def test_auto_save_disabled(self, temp_corners_file):
        """Test that auto_save_new=False prevents saving"""
        corners = [
            CornerZone(name="Unknown", apex_lat=43.792, apex_lon=-87.990)
        ]

        apply_corner_definitions(
            corners,
            "New Track",
            temp_corners_file,
            auto_save_new=False
        )

        # Should not be saved
        loaded = load_track_corners("New Track", temp_corners_file)
        assert loaded is None

    def test_proximity_threshold_respected(self, temp_corners_file):
        """Test that proximity threshold is used for matching"""
        # Save a corner
        saved_corners = [
            CornerZone(name="T1", apex_lat=43.792, apex_lon=-87.990)
        ]
        save_track_corners("Track", saved_corners, temp_corners_file)

        # Detect a corner 100m away
        detected_far = [
            CornerZone(name="Unknown", apex_lat=43.793, apex_lon=-87.990)
        ]

        # Should match with large threshold
        result = apply_corner_definitions(
            detected_far.copy(),
            "Track",
            temp_corners_file,
            proximity_threshold=200.0,
            auto_save_new=False  # Don't save so we can test again
        )
        assert result[0].name == "T1"

        # Should not match with small threshold (test on fresh corner object)
        detected_far2 = [
            CornerZone(name="Unknown", apex_lat=43.793, apex_lon=-87.990)
        ]
        result = apply_corner_definitions(
            detected_far2,
            "Track",
            temp_corners_file,
            proximity_threshold=10.0,
            auto_save_new=False  # Don't save unmatched corners
        )
        assert result[0].name == "T2"  # Gets numbered as new corner (T1 already exists)


class TestRenameCorner:
    """Tests for manual corner renaming"""

    @pytest.fixture
    def temp_corners_file(self):
        """Create a temporary corners file with some corners"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {
                "test_track": [
                    {
                        "name": "T1",
                        "alias": None,
                        "apex_lat": 43.792,
                        "apex_lon": -87.990,
                        "entry_lat": 43.791,
                        "entry_lon": -87.989,
                        "exit_lat": 43.793,
                        "exit_lon": -87.991,
                        "corner_type": "normal",
                        "direction": "left"
                    },
                    {
                        "name": "T2",
                        "alias": "Big Bend",
                        "apex_lat": 43.795,
                        "apex_lon": -87.992,
                        "entry_lat": 43.794,
                        "entry_lon": -87.991,
                        "exit_lat": 43.796,
                        "exit_lon": -87.993,
                        "corner_type": "sweeper",
                        "direction": "right"
                    }
                ]
            }
            json.dump(data, f)
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_rename_corner_name(self, temp_corners_file):
        """Test renaming a corner's name"""
        success = rename_corner("Test Track", "T1", new_name="Turn1", corners_file=temp_corners_file)

        assert success is True

        # Verify change
        corners = load_track_corners("Test Track", temp_corners_file)
        assert corners[0]["name"] == "Turn1"

    def test_set_corner_alias(self, temp_corners_file):
        """Test setting a corner's alias"""
        success = rename_corner("Test Track", "T1", new_alias="Carousel", corners_file=temp_corners_file)

        assert success is True

        # Verify change
        corners = load_track_corners("Test Track", temp_corners_file)
        assert corners[0]["alias"] == "Carousel"

    def test_update_both_name_and_alias(self, temp_corners_file):
        """Test updating both name and alias"""
        success = rename_corner(
            "Test Track",
            "T2",
            new_name="Turn2",
            new_alias="The Kink",
            corners_file=temp_corners_file
        )

        assert success is True

        # Verify changes
        corners = load_track_corners("Test Track", temp_corners_file)
        corner_t2 = [c for c in corners if c["name"] == "Turn2"][0]
        assert corner_t2["alias"] == "The Kink"

    def test_rename_nonexistent_corner(self, temp_corners_file):
        """Test renaming a corner that doesn't exist"""
        success = rename_corner("Test Track", "T99", new_name="Doesn't Exist", corners_file=temp_corners_file)

        assert success is False

    def test_rename_on_nonexistent_track(self, temp_corners_file):
        """Test renaming on a track that doesn't exist"""
        success = rename_corner("Fake Track", "T1", new_name="New", corners_file=temp_corners_file)

        assert success is False

    def test_rename_with_nonexistent_file(self):
        """Test renaming when file doesn't exist"""
        success = rename_corner("Any Track", "T1", new_name="New", corners_file="nonexistent.json")

        assert success is False

    def test_preserves_other_corners(self, temp_corners_file):
        """Test that renaming one corner doesn't affect others"""
        rename_corner("Test Track", "T1", new_name="Turn1", corners_file=temp_corners_file)

        corners = load_track_corners("Test Track", temp_corners_file)
        assert len(corners) == 2
        assert corners[1]["name"] == "T2"  # T2 unchanged
        assert corners[1]["alias"] == "Big Bend"


class TestIntegrationScenarios:
    """Integration tests for complete Phase 7 workflows"""

    @pytest.fixture
    def temp_corners_file(self):
        """Create a temporary corners file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            yield f.name
        if os.path.exists(f.name):
            os.unlink(f.name)

    def test_first_session_workflow(self, temp_corners_file):
        """Test workflow for first session at a new track"""
        # Detect corners (no prior knowledge)
        detected = [
            CornerZone(name="Unknown", apex_lat=43.792, apex_lon=-87.990),
            CornerZone(name="Unknown", apex_lat=43.795, apex_lon=-87.992),
            CornerZone(name="Unknown", apex_lat=43.798, apex_lon=-87.995)
        ]

        # Apply definitions (should auto-number and save)
        result = apply_corner_definitions(
            detected,
            "New Track",
            temp_corners_file,
            auto_save_new=True
        )

        # Should be numbered
        assert result[0].name == "T1"
        assert result[1].name == "T2"
        assert result[2].name == "T3"

        # Should be saved
        loaded = load_track_corners("New Track", temp_corners_file)
        assert loaded is not None
        assert len(loaded) == 3

    def test_subsequent_session_workflow(self, temp_corners_file):
        """Test workflow for subsequent sessions at a known track"""
        # First session: save corners
        first_session_corners = [
            CornerZone(name="T1", alias="Big Bend", apex_lat=43.792, apex_lon=-87.990),
            CornerZone(name="T2", apex_lat=43.795, apex_lon=-87.992)
        ]
        save_track_corners("Known Track", first_session_corners, temp_corners_file)

        # Second session: detect corners (slightly different GPS due to line variation)
        detected = [
            CornerZone(name="Unknown", apex_lat=43.7921, apex_lon=-87.9901),  # T1
            CornerZone(name="Unknown", apex_lat=43.7951, apex_lon=-87.9921)   # T2
        ]

        # Apply definitions (should match by GPS proximity)
        result = apply_corner_definitions(
            detected,
            "Known Track",
            temp_corners_file
        )

        # Should match to stored definitions
        assert result[0].name == "T1"
        assert result[0].alias == "Big Bend"
        assert result[1].name == "T2"

    def test_manual_naming_workflow(self, temp_corners_file):
        """Test workflow for manually naming corners"""
        # Detect and auto-number
        detected = [
            CornerZone(name="Unknown", apex_lat=43.792, apex_lon=-87.990)
        ]
        apply_corner_definitions(detected, "Track", temp_corners_file, auto_save_new=True)

        # Manually rename
        rename_corner("Track", "T1", new_alias="Carousel", corners_file=temp_corners_file)

        # Next session should use the alias
        detected2 = [
            CornerZone(name="Unknown", apex_lat=43.7920, apex_lon=-87.9900)
        ]
        result = apply_corner_definitions(detected2, "Track", temp_corners_file)

        assert result[0].alias == "Carousel"
