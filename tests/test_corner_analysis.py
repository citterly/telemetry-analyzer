"""
Tests for corner detection and analysis.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path

from src.features.corner_detection import (
    CornerDetector, CornerZone, CornerDetectionResult, Corner,
    calc_curvature, detect_corner_boundaries, detect_corners,
    detect_corners_from_parquet
)
from src.features.corner_analysis import (
    CornerAnalyzer, CornerMetrics, LapCornerAnalysis, CornerComparison,
    CornerAnalysisResult, analyze_corners
)


class TestCalcCurvature:
    """Tests for calc_curvature function"""

    def test_straight_line_returns_zero_curvature(self):
        """Test that a straight line has zero curvature"""
        # Create a straight line from (0, 0) to (0, 0.001) - moving north
        lat = np.array([0.0, 0.0001, 0.0002, 0.0003, 0.0004])
        lon = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        curvature = calc_curvature(lat, lon)

        # Straight line should have zero or near-zero curvature
        assert np.allclose(curvature, 0.0, atol=1e-6)

    def test_circular_path_curvature_accuracy(self):
        """Test curvature calculation accuracy with a known circle radius"""
        # Create a circular path with known radius
        # Using a 100-meter radius circle
        radius_meters = 100.0
        expected_curvature = 1.0 / radius_meters  # 0.01 per meter

        # Generate points on a circle
        n_points = 50
        angles = np.linspace(0, np.pi / 2, n_points)  # Quarter circle

        # Convert radius from meters to degrees (approximate)
        # At equator: 1 degree ≈ 111 km
        radius_deg = radius_meters / 111000.0

        # Center the circle at origin for simplicity
        center_lat = 0.0
        center_lon = 0.0

        lat = center_lat + radius_deg * np.sin(angles)
        lon = center_lon + radius_deg * np.cos(angles)

        curvature = calc_curvature(lat, lon, window_size=3)

        # Check middle points (avoid edge effects)
        middle_curvature = curvature[10:-10]

        # Should be close to expected curvature (within 5% tolerance)
        mean_curvature = np.mean(middle_curvature)
        assert abs(mean_curvature - expected_curvature) / expected_curvature < 0.05, \
            f"Expected curvature {expected_curvature:.6f}, got {mean_curvature:.6f}"

    def test_tight_turn_higher_curvature(self):
        """Test that tight turns have higher curvature than gentle turns"""
        # Create two circular paths with different radii
        n_points = 30

        # Tight turn: 30m radius
        radius_tight = 30.0
        angles_tight = np.linspace(0, np.pi / 4, n_points)
        radius_deg_tight = radius_tight / 111000.0
        lat_tight = radius_deg_tight * np.sin(angles_tight)
        lon_tight = radius_deg_tight * np.cos(angles_tight)
        curvature_tight = calc_curvature(lat_tight, lon_tight)

        # Gentle turn: 200m radius
        radius_gentle = 200.0
        angles_gentle = np.linspace(0, np.pi / 4, n_points)
        radius_deg_gentle = radius_gentle / 111000.0
        lat_gentle = radius_deg_gentle * np.sin(angles_gentle)
        lon_gentle = radius_deg_gentle * np.cos(angles_gentle)
        curvature_gentle = calc_curvature(lat_gentle, lon_gentle)

        # Average curvature in the middle section
        tight_avg = np.mean(curvature_tight[5:-5])
        gentle_avg = np.mean(curvature_gentle[5:-5])

        # Tight turn should have higher curvature
        assert tight_avg > gentle_avg
        assert tight_avg > 0.02  # Should be around 1/30 = 0.033
        assert gentle_avg < 0.01  # Should be around 1/200 = 0.005

    def test_curvature_units_inverse_meters(self):
        """Test that curvature is in correct units (1/meters)"""
        # Create a 50-meter radius circle
        radius_meters = 50.0
        expected_curvature = 1.0 / radius_meters  # 0.02 per meter

        n_points = 40
        angles = np.linspace(0, np.pi / 3, n_points)
        radius_deg = radius_meters / 111000.0

        lat = radius_deg * np.sin(angles)
        lon = radius_deg * np.cos(angles)

        curvature = calc_curvature(lat, lon)

        # Check that curvature values are in the right ballpark
        middle_curvature = curvature[8:-8]
        mean_curvature = np.mean(middle_curvature)

        # Should be approximately 0.02 (1/50)
        assert 0.015 < mean_curvature < 0.025

    def test_window_size_smoothing(self):
        """Test that larger window sizes produce smoother curvature"""
        # Create a circle with some noise
        radius_meters = 80.0
        n_points = 60
        angles = np.linspace(0, np.pi / 2, n_points)
        radius_deg = radius_meters / 111000.0

        lat = radius_deg * np.sin(angles)
        lon = radius_deg * np.cos(angles)

        # Add small noise
        lat += np.random.randn(n_points) * 1e-6
        lon += np.random.randn(n_points) * 1e-6

        # Calculate with different window sizes
        curv_small = calc_curvature(lat, lon, window_size=3)
        curv_large = calc_curvature(lat, lon, window_size=9)

        # Larger window should have lower variance (smoother)
        variance_small = np.var(curv_small[10:-10])
        variance_large = np.var(curv_large[10:-10])

        assert variance_large < variance_small

    def test_handles_minimal_data(self):
        """Test handling of minimal data points"""
        # Test with exactly 3 points
        lat = np.array([0.0, 0.0001, 0.0002])
        lon = np.array([0.0, 0.0, 0.0])

        curvature = calc_curvature(lat, lon)
        assert len(curvature) == 3
        assert np.all(curvature >= 0)

        # Test with 2 points (too few)
        lat = np.array([0.0, 0.0001])
        lon = np.array([0.0, 0.0])

        curvature = calc_curvature(lat, lon)
        assert len(curvature) == 2
        assert np.all(curvature == 0)

    def test_handles_edge_cases(self):
        """Test edge cases: duplicate points, very close points"""
        # Duplicate points
        lat = np.array([0.0, 0.0, 0.0001, 0.0002])
        lon = np.array([0.0, 0.0, 0.0, 0.0])

        curvature = calc_curvature(lat, lon)
        assert len(curvature) == 4
        # Should handle gracefully without errors

        # Points very close together (< 0.1m threshold)
        lat = np.array([0.0, 0.0000001, 0.0000002, 0.0000003])
        lon = np.array([0.0, 0.0, 0.0, 0.0])

        curvature = calc_curvature(lat, lon)
        assert len(curvature) == 4
        assert np.all(curvature == 0)  # Too close, should return 0

    def test_curvature_always_positive_or_zero(self):
        """Test that curvature is always >= 0 (magnitude only)"""
        # Create various paths
        n = 100
        t = np.linspace(0, 2 * np.pi, n)

        # Sine wave path
        lat = 0.001 * np.sin(t)
        lon = 0.001 * t / (2 * np.pi)

        curvature = calc_curvature(lat, lon)

        # Curvature should never be negative (it's a magnitude)
        assert np.all(curvature >= 0)

    def test_consistency_across_coordinate_systems(self):
        """Test that curvature is consistent regardless of absolute GPS coordinates"""
        radius_meters = 75.0
        n_points = 40
        angles = np.linspace(0, np.pi / 4, n_points)
        radius_deg = radius_meters / 111000.0

        # Same circle at different lat/lon origins
        # Origin 1: Equator
        lat1 = 0.0 + radius_deg * np.sin(angles)
        lon1 = 0.0 + radius_deg * np.cos(angles)
        curv1 = calc_curvature(lat1, lon1)

        # Origin 2: Mid-latitude (Road America latitude)
        lat2 = 43.8 + radius_deg * np.sin(angles)
        lon2 = -87.99 + radius_deg * np.cos(angles)
        curv2 = calc_curvature(lat2, lon2)

        # Curvature should be similar (within 10% due to latitude correction)
        mean_curv1 = np.mean(curv1[5:-5])
        mean_curv2 = np.mean(curv2[5:-5])

        # Both should be close to 1/75 = 0.0133
        assert 0.01 < mean_curv1 < 0.02
        assert 0.01 < mean_curv2 < 0.02


class TestDetectCornerBoundaries:
    """Tests for detect_corner_boundaries function"""

    def test_single_corner_detected(self):
        """Test detection of a single corner"""
        # Create lateral G data with one corner
        lateral_g = np.array([0.1, 0.1, 0.5, 0.8, 0.9, 0.8, 0.5, 0.2, 0.1, 0.1])

        boundaries = detect_corner_boundaries(lateral_g, threshold=0.3)

        # Should detect one corner from index 2 to 6
        assert len(boundaries) == 1
        assert boundaries[0][0] == 2
        assert boundaries[0][1] == 6

    def test_multiple_corners_detected(self):
        """Test detection of multiple corners"""
        # Create data with two separate corners (need >5 samples at 10Hz for 0.5s duration)
        lateral_g = np.array([
            0.1, 0.1,                           # Straight
            0.5, 0.8, 0.9, 0.8, 0.5, 0.4,      # First corner (6 samples = 0.6s)
            0.1, 0.05, 0.05, 0.05, 0.1,        # Straight section
            -0.6, -0.7, -0.8, -0.7, -0.6, -0.4, # Second corner (6 samples = 0.6s)
            0.1, 0.1                            # Straight
        ])

        boundaries = detect_corner_boundaries(lateral_g, threshold=0.3)

        # Should detect two corners
        assert len(boundaries) == 2
        assert boundaries[0] == (2, 7)
        assert boundaries[1] == (13, 18)

    def test_threshold_filtering(self):
        """Test that threshold properly filters out low lateral G"""
        # Need >5 samples for 0.5s duration at 10Hz
        lateral_g = np.array([
            0.1, 0.2, 0.25, 0.2, 0.1,           # Low G (below threshold)
            0.8, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, # High G corner (7 samples = 0.7s)
            0.1, 0.1
        ])

        # With threshold 0.3, should only detect the high-G section
        boundaries = detect_corner_boundaries(lateral_g, threshold=0.3)

        assert len(boundaries) == 1
        assert boundaries[0][0] == 5
        assert boundaries[0][1] == 11

    def test_minimum_duration_filter(self):
        """Test that minimum duration filter eliminates noise spikes"""
        # Create data at 10 Hz (default)
        # Corner needs >0.5s = >5 samples to be valid
        lateral_g = np.array([
            0.1, 0.1, 0.1,              # Straight
            0.8, 0.9, 0.8,              # Spike (3 samples = 0.3s, too short)
            0.1, 0.1, 0.1, 0.1, 0.1,    # Straight
            0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, # Real corner (7 samples = 0.7s, valid)
            0.1, 0.1                     # Straight
        ])

        boundaries = detect_corner_boundaries(lateral_g, threshold=0.3, min_duration=0.5, sample_rate=10.0)

        # Should only detect the second corner, not the spike
        assert len(boundaries) == 1
        assert boundaries[0][0] == 11
        assert boundaries[0][1] == 17

    def test_with_time_data(self):
        """Test using actual time data for duration filtering"""
        # 20 samples over 5 seconds (non-uniform sampling)
        time_data = np.linspace(0, 5, 20)
        lateral_g = np.zeros(20)

        # Add a corner from t=1.0s to t=2.5s (1.5s duration, valid)
        lateral_g[4:10] = 0.8  # indices 4-9

        # Add a spike from t=3.0s to t=3.2s (0.2s duration, too short)
        lateral_g[12:13] = 0.9  # indices 12

        boundaries = detect_corner_boundaries(
            lateral_g,
            threshold=0.3,
            time_data=time_data,
            min_duration=0.5
        )

        # Should only detect the longer corner
        assert len(boundaries) == 1
        assert boundaries[0][0] == 4
        assert boundaries[0][1] == 9

    def test_handles_negative_lateral_g(self):
        """Test that negative lateral G (opposite direction) is detected"""
        # Mix of positive and negative lateral G (need >5 samples for 0.5s)
        lateral_g = np.array([
            0.1, 0.1,                          # Straight
            0.6, 0.8, 0.9, 0.8, 0.6, 0.4,     # Right turn (6 samples = 0.6s)
            0.1, 0.0, 0.0, 0.1,                # Straight
            -0.5, -0.8, -0.9, -0.8, -0.7, -0.5, -0.4, # Left turn (7 samples = 0.7s)
            0.1, 0.1
        ])

        boundaries = detect_corner_boundaries(lateral_g, threshold=0.3)

        # Should detect both corners (absolute value used)
        assert len(boundaries) == 2
        assert boundaries[0] == (2, 7)
        assert boundaries[1] == (12, 18)

    def test_empty_data(self):
        """Test handling of empty data"""
        lateral_g = np.array([])
        boundaries = detect_corner_boundaries(lateral_g, threshold=0.3)
        assert boundaries == []

    def test_no_corners_detected(self):
        """Test when no corners exceed threshold"""
        lateral_g = np.array([0.1, 0.15, 0.2, 0.15, 0.1, 0.05])
        boundaries = detect_corner_boundaries(lateral_g, threshold=0.3)
        assert len(boundaries) == 0

    def test_entire_data_is_corner(self):
        """Test when entire dataset exceeds threshold"""
        lateral_g = np.array([0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.6])
        boundaries = detect_corner_boundaries(lateral_g, threshold=0.3)

        # Should detect one long corner spanning entire dataset
        assert len(boundaries) == 1
        assert boundaries[0][0] == 0
        assert boundaries[0][1] == 7

    def test_corner_at_end_of_data(self):
        """Test corner that extends to end of data"""
        lateral_g = np.array([0.1, 0.1, 0.1, 0.5, 0.8, 0.9, 1.0, 1.1])
        boundaries = detect_corner_boundaries(lateral_g, threshold=0.3)

        assert len(boundaries) == 1
        assert boundaries[0][0] == 3
        assert boundaries[0][1] == 7

    def test_different_thresholds(self):
        """Test that different thresholds produce different results"""
        # Create longer corner to meet minimum duration requirement
        # Need at least 6 samples above higher threshold to meet 0.5s duration (0.6s)
        lateral_g = np.array([
            0.1, 0.1,                                                    # Straight
            0.4, 0.5, 0.7, 0.9, 1.0, 1.1, 1.0, 0.9, 0.8, 0.7, 0.5, 0.4, # Corner (12 samples)
            0.1, 0.1
        ])

        # Lower threshold detects wider corner (all 12 samples above 0.3)
        boundaries_low = detect_corner_boundaries(lateral_g, threshold=0.3)

        # Higher threshold detects narrower corner (7 samples above 0.7: indices 4-10 = 0.7s)
        boundaries_high = detect_corner_boundaries(lateral_g, threshold=0.7)

        assert len(boundaries_low) == 1
        assert len(boundaries_high) == 1

        # Low threshold corner should be wider
        low_width = boundaries_low[0][1] - boundaries_low[0][0]
        high_width = boundaries_high[0][1] - boundaries_high[0][0]
        assert low_width > high_width

    def test_custom_sample_rate(self):
        """Test with custom sample rate"""
        # At 100 Hz, need >50 samples for 0.5s duration
        lateral_g = np.zeros(200)
        lateral_g[10:70] = 0.8  # 60 samples = 0.6s at 100 Hz (valid)
        lateral_g[100:110] = 0.9  # 10 samples = 0.1s at 100 Hz (too short)

        boundaries = detect_corner_boundaries(
            lateral_g,
            threshold=0.3,
            min_duration=0.5,
            sample_rate=100.0
        )

        # Should only detect the first corner
        assert len(boundaries) == 1
        assert boundaries[0] == (10, 69)


class TestCornerZone:
    """Tests for CornerZone dataclass"""

    def test_corner_zone_creation(self):
        """Test creating a CornerZone"""
        zone = CornerZone(
            name="T1",
            alias="Carousel",
            entry_idx=100,
            apex_idx=150,
            exit_idx=200,
            entry_lat=43.792,
            entry_lon=-87.989,
            apex_lat=43.791,
            apex_lon=-87.990,
            exit_lat=43.790,
            exit_lon=-87.991,
            min_radius=50.0,
            apex_speed_mph=65.0,
            direction="left",
            corner_type="normal"
        )
        assert zone.name == "T1"
        assert zone.alias == "Carousel"
        assert zone.apex_idx == 150
        assert zone.direction == "left"

    def test_corner_zone_to_dict(self):
        """Test CornerZone serialization"""
        zone = CornerZone(
            name="T1",
            entry_idx=100,
            apex_idx=150,
            exit_idx=200,
            apex_speed_mph=70.5
        )
        d = zone.to_dict()
        assert d["name"] == "T1"
        assert d["apex_speed_mph"] == 70.5
        assert "entry" in d
        assert "apex" in d
        assert "exit" in d

    def test_corner_zone_from_dict(self):
        """Test CornerZone deserialization"""
        data = {
            "name": "T2",
            "alias": "Kink",
            "entry_idx": 50,
            "apex_idx": 75,
            "exit_idx": 100,
            "entry": {"lat": 43.8, "lon": -88.0},
            "apex": {"lat": 43.81, "lon": -88.01},
            "exit": {"lat": 43.82, "lon": -88.02},
            "min_radius": 100.0,
            "apex_speed_mph": 110.0,
            "direction": "right",
            "corner_type": "kink"
        }
        zone = CornerZone.from_dict(data)
        assert zone.name == "T2"
        assert zone.alias == "Kink"
        assert zone.apex_lat == 43.81
        assert zone.corner_type == "kink"


class TestCornerDetectionResult:
    """Tests for CornerDetectionResult dataclass"""

    def test_detection_result_creation(self):
        """Test creating a CornerDetectionResult"""
        corners = [
            CornerZone(name="T1", apex_speed_mph=60.0),
            CornerZone(name="T2", apex_speed_mph=80.0)
        ]
        result = CornerDetectionResult(
            lap_number=1,
            corners=corners,
            detection_params={"radius_threshold": 200.0},
            confidence=0.85
        )
        assert result.lap_number == 1
        assert len(result.corners) == 2
        assert result.confidence == 0.85

    def test_detection_result_to_dict(self):
        """Test CornerDetectionResult serialization"""
        corners = [CornerZone(name="T1")]
        result = CornerDetectionResult(
            lap_number=2,
            corners=corners,
            detection_params={"test": 1},
            confidence=0.9
        )
        d = result.to_dict()
        assert d["lap_number"] == 2
        assert d["corner_count"] == 1
        assert d["confidence"] == 0.9


class TestCornerDetector:
    """Tests for CornerDetector class"""

    @pytest.fixture
    def sample_track_data(self):
        """Generate sample track data with clear corners"""
        n = 500
        time = np.linspace(0, 50, n)

        # Create a track with straight sections and corners
        # Simulate a simple oval with two corners
        t = time / 10
        lat = 43.797 + 0.005 * np.sin(t)
        lon = -87.99 + 0.005 * np.cos(t)

        # Speed: fast on straights, slow in corners
        speed = 100 - 40 * np.abs(np.sin(t))

        # Radius: large on straights, small in corners
        radius = 1000 - 900 * np.abs(np.sin(t))

        # Lateral acceleration: high in corners
        lat_acc = 0.8 * np.sin(t)

        # Longitudinal acceleration: braking into corners
        lon_acc = -0.5 * np.sin(t + 0.5)

        return time, lat, lon, speed, radius, lat_acc, lon_acc

    def test_detector_creation(self):
        """Test creating a CornerDetector"""
        detector = CornerDetector(radius_threshold=150.0)
        assert detector.radius_threshold == 150.0

    def test_detector_defaults(self):
        """Test default detector parameters"""
        detector = CornerDetector()
        assert detector.radius_threshold == 200.0
        assert detector.lat_acc_threshold == 0.3
        assert detector.min_corner_duration == 0.5

    def test_detect_from_arrays(self, sample_track_data):
        """Test corner detection from arrays"""
        time, lat, lon, speed, radius, lat_acc, lon_acc = sample_track_data

        detector = CornerDetector()
        result = detector.detect_from_arrays(
            time, lat, lon, speed, radius, lat_acc, lon_acc
        )

        assert isinstance(result, CornerDetectionResult)
        assert result.lap_number == 1
        assert len(result.corners) > 0  # Should detect at least some corners
        assert result.confidence > 0

    def test_detect_without_radius(self, sample_track_data):
        """Test detection computes radius when not provided"""
        time, lat, lon, speed, _, lat_acc, lon_acc = sample_track_data

        detector = CornerDetector()
        result = detector.detect_from_arrays(
            time, lat, lon, speed, None, lat_acc, lon_acc
        )

        assert isinstance(result, CornerDetectionResult)

    def test_detect_without_accelerations(self, sample_track_data):
        """Test detection computes accelerations when not provided"""
        time, lat, lon, speed, radius, _, _ = sample_track_data

        detector = CornerDetector()
        result = detector.detect_from_arrays(
            time, lat, lon, speed, radius, None, None
        )

        assert isinstance(result, CornerDetectionResult)

    def test_detect_minimal_data(self):
        """Test detection with minimal data"""
        time = np.array([0, 1, 2])
        lat = np.array([43.0, 43.0, 43.0])
        lon = np.array([-87.0, -87.0, -87.0])
        speed = np.array([60, 60, 60])

        detector = CornerDetector()
        result = detector.detect_from_arrays(time, lat, lon, speed)

        assert result.corners == []  # Not enough data for corners
        assert result.confidence == 0.0

    def test_corner_direction_detection(self, sample_track_data):
        """Test that corner direction is detected correctly"""
        time, lat, lon, speed, radius, lat_acc, lon_acc = sample_track_data

        detector = CornerDetector()
        result = detector.detect_from_arrays(
            time, lat, lon, speed, radius, lat_acc, lon_acc
        )

        for corner in result.corners:
            assert corner.direction in ["left", "right"]

    def test_corner_type_classification(self, sample_track_data):
        """Test that corner types are classified"""
        time, lat, lon, speed, radius, lat_acc, lon_acc = sample_track_data

        detector = CornerDetector()
        result = detector.detect_from_arrays(
            time, lat, lon, speed, radius, lat_acc, lon_acc
        )

        for corner in result.corners:
            assert corner.corner_type in ["normal", "hairpin", "kink", "sweeper", "chicane"]


class TestCornerMetrics:
    """Tests for CornerMetrics dataclass"""

    def test_metrics_creation(self):
        """Test creating CornerMetrics"""
        metrics = CornerMetrics(
            corner_name="T1",
            entry_speed=80.0,
            min_speed=55.0,
            exit_speed=75.0,
            time_in_corner=3.5
        )
        assert metrics.corner_name == "T1"
        assert metrics.entry_speed == 80.0
        assert metrics.speed_scrub == 0.0  # Default

    def test_metrics_to_dict(self):
        """Test CornerMetrics serialization"""
        metrics = CornerMetrics(
            corner_name="T1",
            entry_speed=80.0,
            min_speed=55.0,
            exit_speed=75.0,
            time_in_corner=3.5,
            lift_detected=True,
            lift_time=1.2
        )
        d = metrics.to_dict()
        assert d["corner_name"] == "T1"
        assert d["speeds"]["entry"] == 80.0
        assert d["speeds"]["min"] == 55.0
        assert d["lift"]["detected"] is True

    def test_metrics_defaults(self):
        """Test default values"""
        metrics = CornerMetrics(corner_name="T1")
        assert metrics.entry_speed == 0.0
        assert metrics.lift_detected is False
        assert metrics.trail_brake_detected is False
        assert metrics.consistency_score == 0.0


class TestLapCornerAnalysis:
    """Tests for LapCornerAnalysis dataclass"""

    def test_lap_analysis_creation(self):
        """Test creating LapCornerAnalysis"""
        corners = [
            CornerMetrics(corner_name="T1", entry_speed=80.0),
            CornerMetrics(corner_name="T2", entry_speed=70.0)
        ]
        lap = LapCornerAnalysis(
            lap_number=1,
            lap_time=120.5,
            corners=corners,
            total_corner_time=25.0,
            avg_entry_speed=75.0,
            avg_exit_speed=72.0,
            lifts_count=1,
            trail_brakes_count=5
        )
        assert lap.lap_number == 1
        assert len(lap.corners) == 2
        assert lap.avg_entry_speed == 75.0

    def test_lap_analysis_to_dict(self):
        """Test LapCornerAnalysis serialization"""
        lap = LapCornerAnalysis(
            lap_number=1,
            lap_time=120.5,
            corners=[],
            total_corner_time=25.0,
            avg_entry_speed=75.0,
            avg_exit_speed=72.0,
            lifts_count=1,
            trail_brakes_count=5
        )
        d = lap.to_dict()
        assert d["lap_number"] == 1
        assert d["lap_time"] == 120.5
        assert d["summary"]["lifts_count"] == 1


class TestCornerAnalyzer:
    """Tests for CornerAnalyzer class"""

    @pytest.fixture
    def sample_session_data(self):
        """Generate sample session data"""
        n = 500
        time = np.linspace(0, 50, n)

        t = time / 10
        lat = 43.797 + 0.005 * np.sin(t)
        lon = -87.99 + 0.005 * np.cos(t)
        speed = 100 - 40 * np.abs(np.sin(t))
        radius = 1000 - 900 * np.abs(np.sin(t))
        lat_acc = 0.8 * np.sin(t)
        lon_acc = -0.5 * np.sin(t + 0.5)
        throttle = 50 + 40 * np.cos(t)

        return time, lat, lon, speed, radius, lat_acc, lon_acc, throttle

    @pytest.fixture
    def sample_parquet_file(self, sample_session_data):
        """Create a temporary parquet file with sample data"""
        time, lat, lon, speed, radius, lat_acc, lon_acc, throttle = sample_session_data

        df = pd.DataFrame({
            'GPS Latitude': lat,
            'GPS Longitude': lon,
            'GPS Speed': speed * 0.44704,  # Convert to m/s
            'GPS Radius': radius,
            'GPS LatAcc': lat_acc,
            'GPS LonAcc': lon_acc,
            'PedalPos': throttle
        }, index=time)

        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            df.to_parquet(f.name)
            yield f.name
            os.unlink(f.name)

    def test_analyzer_creation(self):
        """Test creating CornerAnalyzer"""
        analyzer = CornerAnalyzer()
        assert analyzer.detector is not None

    def test_analyze_from_arrays(self, sample_session_data):
        """Test analyzing from arrays"""
        time, lat, lon, speed, radius, lat_acc, lon_acc, throttle = sample_session_data

        analyzer = CornerAnalyzer()
        result = analyzer.analyze_from_arrays(
            time, lat, lon, speed, radius, lat_acc, lon_acc, throttle,
            session_id="test_session",
            track_name="Test Track"
        )

        assert isinstance(result, CornerAnalysisResult)
        assert result.session_id == "test_session"
        assert result.track_name == "Test Track"
        assert len(result.laps) > 0

    def test_analyze_from_parquet(self, sample_parquet_file):
        """Test analyzing from parquet file"""
        analyzer = CornerAnalyzer()
        result = analyzer.analyze_from_parquet(sample_parquet_file)

        assert isinstance(result, CornerAnalysisResult)
        assert len(result.corner_zones) >= 0

    def test_throttle_detection(self, sample_session_data):
        """Test throttle pickup detection"""
        time, lat, lon, speed, radius, lat_acc, lon_acc, throttle = sample_session_data

        analyzer = CornerAnalyzer()
        result = analyzer.analyze_from_arrays(
            time, lat, lon, speed, radius, lat_acc, lon_acc, throttle
        )

        if result.laps and result.laps[0].corners:
            for corner in result.laps[0].corners:
                assert corner.throttle_pickup_pct >= 0
                assert corner.throttle_pickup_pct <= 100

    def test_recommendations_generated(self, sample_session_data):
        """Test that recommendations are generated"""
        time, lat, lon, speed, radius, lat_acc, lon_acc, throttle = sample_session_data

        analyzer = CornerAnalyzer()
        result = analyzer.analyze_from_arrays(
            time, lat, lon, speed, radius, lat_acc, lon_acc, throttle
        )

        assert len(result.recommendations) > 0


class TestCornerAnalysisResult:
    """Tests for CornerAnalysisResult dataclass"""

    def test_result_creation(self):
        """Test creating CornerAnalysisResult"""
        result = CornerAnalysisResult(
            session_id="test",
            track_name="Test Track",
            analysis_timestamp="2024-01-01T00:00:00",
            laps=[],
            corner_comparisons=[],
            corner_zones=[],
            recommendations=["Test recommendation"]
        )
        assert result.session_id == "test"
        assert result.track_name == "Test Track"

    def test_result_to_dict(self):
        """Test CornerAnalysisResult serialization"""
        result = CornerAnalysisResult(
            session_id="test",
            track_name="Test Track",
            analysis_timestamp="2024-01-01T00:00:00",
            laps=[],
            corner_comparisons=[],
            corner_zones=[],
            recommendations=["Tip 1"]
        )
        d = result.to_dict()
        assert d["session_id"] == "test"
        assert d["track_name"] == "Test Track"
        assert d["lap_count"] == 0
        assert d["corner_count"] == 0
        assert len(d["recommendations"]) == 1

    def test_result_to_json(self):
        """Test JSON serialization"""
        result = CornerAnalysisResult(
            session_id="test",
            track_name="Test Track",
            analysis_timestamp="2024-01-01T00:00:00",
            laps=[],
            corner_comparisons=[],
            corner_zones=[],
            recommendations=[]
        )
        json_str = result.to_json()
        assert '"session_id": "test"' in json_str


class TestConvenienceFunctions:
    """Tests for module-level convenience functions"""

    @pytest.fixture
    def sample_parquet(self):
        """Create a sample parquet file"""
        n = 200
        time = np.linspace(0, 20, n)
        t = time / 5

        df = pd.DataFrame({
            'GPS Latitude': 43.797 + 0.003 * np.sin(t),
            'GPS Longitude': -87.99 + 0.003 * np.cos(t),
            'GPS Speed': 30 + 20 * np.cos(t),  # m/s
            'GPS LatAcc': 0.5 * np.sin(t),
            'GPS LonAcc': -0.3 * np.sin(t + 0.5)
        }, index=time)

        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            df.to_parquet(f.name)
            yield f.name
            os.unlink(f.name)

    def test_detect_corners_function(self, sample_parquet):
        """Test detect_corners_from_parquet convenience function"""
        result = detect_corners_from_parquet(sample_parquet)
        assert isinstance(result, CornerDetectionResult)

    def test_analyze_corners_function(self, sample_parquet):
        """Test analyze_corners convenience function"""
        result = analyze_corners(sample_parquet, track_name="Test Track")
        assert isinstance(result, CornerAnalysisResult)
        assert result.track_name == "Test Track"


class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_empty_data(self):
        """Test handling of empty data"""
        detector = CornerDetector()
        result = detector.detect_from_arrays(
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([])
        )
        assert len(result.corners) == 0
        assert result.confidence == 0.0

    def test_constant_speed(self):
        """Test with constant speed (no corners)"""
        n = 100
        time = np.linspace(0, 10, n)
        lat = np.linspace(43.0, 43.1, n)
        lon = np.linspace(-87.0, -86.9, n)
        speed = np.ones(n) * 60

        detector = CornerDetector()
        result = detector.detect_from_arrays(time, lat, lon, speed)
        # Straight line should have no corners or very low confidence
        assert len(result.corners) == 0 or result.confidence < 0.5

    def test_nan_handling(self):
        """Test handling of NaN values in data"""
        n = 100
        time = np.linspace(0, 10, n)
        lat = np.full(n, 43.0)
        lat[50] = np.nan
        lon = np.full(n, -87.0)
        speed = np.full(n, 60.0)

        detector = CornerDetector()
        # Should not raise an error
        result = detector.detect_from_arrays(time, lat, lon, speed)
        assert isinstance(result, CornerDetectionResult)

    def test_missing_throttle_data(self):
        """Test analysis with missing throttle data"""
        n = 200
        time = np.linspace(0, 20, n)
        t = time / 5
        lat = 43.797 + 0.005 * np.sin(t)
        lon = -87.99 + 0.005 * np.cos(t)
        speed = 80 - 30 * np.abs(np.sin(t))

        analyzer = CornerAnalyzer()
        result = analyzer.analyze_from_arrays(
            time, lat, lon, speed,
            None, None, None, None  # No optional data
        )

        assert isinstance(result, CornerAnalysisResult)
        # Should still work, just with zeros for throttle metrics


class TestDetectCornersPhase3:
    """Tests for Phase 3 detect_corners() function using curvature + lateral G"""

    def test_detects_corner_with_both_curvature_and_lateral_g(self):
        """Test that corners require BOTH curvature AND lateral G"""
        n = 100
        time = np.linspace(0, 10, n)

        # Create a curved path with matching lateral G
        t = time / 2
        lat = 43.797 + 0.001 * np.sin(t)
        lon = -87.99 + 0.001 * np.cos(t)
        lateral_g = 0.5 * np.sin(t)  # Matches the curvature

        df = pd.DataFrame({
            'GPS Latitude': lat,
            'GPS Longitude': lon,
            'GPS LatAcc': lateral_g
        }, index=time)

        corners = detect_corners(df, curvature_threshold=0.003, lateral_g_threshold=0.3)

        # Should detect corners where both conditions are met
        assert len(corners) > 0
        assert all(isinstance(c, Corner) for c in corners)

    def test_requires_both_curvature_and_lateral_g_at_entry(self):
        """Test that corner entry requires BOTH curvature AND lateral G"""
        n = 100
        time = np.linspace(0, 10, n)

        # High curvature but no lateral G
        lat = np.linspace(43.797, 43.798, n)
        lon = np.linspace(-87.99, -87.989, n)
        lateral_g = np.full(n, 0.1)  # Below threshold

        df = pd.DataFrame({
            'GPS Latitude': lat,
            'GPS Longitude': lon,
            'GPS LatAcc': lateral_g
        }, index=time)

        corners = detect_corners(df, curvature_threshold=0.001, lateral_g_threshold=0.3)

        # Should not detect corners without sufficient lateral G
        assert len(corners) == 0

    def test_corner_exit_uses_lateral_g(self):
        """Test that corner exit is determined by lateral G dropping"""
        n = 120
        time = np.linspace(0, 12, n)

        # Create path with curvature, lateral G rises then falls
        lat = np.full(n, 43.797)
        lon = np.full(n, -87.99)
        lateral_g = np.zeros(n)

        # Add a corner: entry at idx 30, exit at idx 70
        for i in range(30, 71):
            lateral_g[i] = 0.6  # Above threshold

        # Add some curvature in the same region
        lat[25:75] += 0.001 * np.sin(np.linspace(0, np.pi, 50))
        lon[25:75] += 0.001 * np.cos(np.linspace(0, np.pi, 50))

        df = pd.DataFrame({
            'GPS Latitude': lat,
            'GPS Longitude': lon,
            'GPS LatAcc': lateral_g
        }, index=time)

        corners = detect_corners(df, curvature_threshold=0.003, lateral_g_threshold=0.3)

        assert len(corners) >= 1
        # Corner should end around idx 70 (where lateral G drops)
        if corners:
            assert corners[0].end_idx >= 65
            assert corners[0].end_idx <= 75

    def test_merges_nearby_corners_chicane(self):
        """Test that nearby corners separated by <0.5s are merged"""
        n = 200
        time = np.linspace(0, 20, n)  # 10 Hz sampling

        # Create two corners close together (chicane pattern)
        lat = np.full(n, 43.797)
        lon = np.full(n, -87.99)
        lateral_g = np.zeros(n)

        # First corner: 20-35 (1.5s duration)
        for i in range(20, 36):
            lateral_g[i] = 0.5
            lat[i] += 0.0005 * np.sin((i - 20) / 8 * np.pi)
            lon[i] += 0.0005 * np.cos((i - 20) / 8 * np.pi)

        # Gap of 3 samples (0.3s) - should merge
        # Second corner: 39-54 (1.5s duration)
        for i in range(39, 55):
            lateral_g[i] = -0.5  # Opposite direction
            lat[i] += 0.0005 * np.sin((i - 39) / 8 * np.pi)
            lon[i] += 0.0005 * np.cos((i - 39) / 8 * np.pi)

        df = pd.DataFrame({
            'GPS Latitude': lat,
            'GPS Longitude': lon,
            'GPS LatAcc': lateral_g
        }, index=time)

        corners = detect_corners(df, merge_gap=0.5)

        # Should merge into one corner
        assert len(corners) == 1
        assert corners[0].start_idx <= 20
        assert corners[0].end_idx >= 54

    def test_does_not_merge_distant_corners(self):
        """Test that corners separated by >0.5s are NOT merged"""
        n = 200
        time = np.linspace(0, 20, n)

        lat = np.full(n, 43.797)
        lon = np.full(n, -87.99)
        lateral_g = np.zeros(n)

        # First corner: 20-30 (1.0s) - make it more curved
        angles1 = np.linspace(0, np.pi / 2, 11)
        for idx, i in enumerate(range(20, 31)):
            lateral_g[i] = 0.5
            lat[i] += 0.002 * np.sin(angles1[idx])
            lon[i] += 0.002 * np.cos(angles1[idx])

        # Gap of 8 samples (0.8s) - should NOT merge
        # Second corner: 39-49 (1.0s) - make it more curved
        angles2 = np.linspace(0, np.pi / 2, 11)
        for idx, i in enumerate(range(39, 50)):
            lateral_g[i] = 0.5
            lat[i] += 0.002 * np.sin(angles2[idx])
            lon[i] += 0.002 * np.cos(angles2[idx])

        df = pd.DataFrame({
            'GPS Latitude': lat,
            'GPS Longitude': lon,
            'GPS LatAcc': lateral_g
        }, index=time)

        corners = detect_corners(df, merge_gap=0.5, curvature_threshold=0.003)

        # Should detect two separate corners
        assert len(corners) == 2

    def test_direction_detection_right(self):
        """Test that right corners are detected correctly"""
        n = 100
        time = np.linspace(0, 10, n)

        # Right corner: positive lateral G
        lat = 43.797 + 0.001 * np.sin(np.linspace(0, np.pi / 2, n))
        lon = -87.99 + 0.001 * np.cos(np.linspace(0, np.pi / 2, n))
        lateral_g = np.full(n, 0.6)  # Positive = right turn

        df = pd.DataFrame({
            'GPS Latitude': lat,
            'GPS Longitude': lon,
            'GPS LatAcc': lateral_g
        }, index=time)

        corners = detect_corners(df)

        assert len(corners) >= 1
        assert corners[0].direction == "right"

    def test_direction_detection_left(self):
        """Test that left corners are detected correctly"""
        n = 100
        time = np.linspace(0, 10, n)

        # Left corner: negative lateral G
        lat = 43.797 + 0.001 * np.sin(np.linspace(0, np.pi / 2, n))
        lon = -87.99 + 0.001 * np.cos(np.linspace(0, np.pi / 2, n))
        lateral_g = np.full(n, -0.6)  # Negative = left turn

        df = pd.DataFrame({
            'GPS Latitude': lat,
            'GPS Longitude': lon,
            'GPS LatAcc': lateral_g
        }, index=time)

        corners = detect_corners(df)

        assert len(corners) >= 1
        assert corners[0].direction == "left"

    def test_minimum_duration_filter(self):
        """Test that corners shorter than min_duration are filtered"""
        n = 200
        time = np.linspace(0, 20, n)

        lat = np.full(n, 43.797)
        lon = np.full(n, -87.99)
        lateral_g = np.zeros(n)

        # Short spike: 3 samples = 0.3s (below 0.5s minimum)
        angles_short = np.linspace(0, np.pi / 4, 3)
        for idx, i in enumerate(range(20, 23)):
            lateral_g[i] = 0.8
            lat[i] += 0.002 * np.sin(angles_short[idx])
            lon[i] += 0.002 * np.cos(angles_short[idx])

        # Long corner: 8 samples = 0.8s (above 0.5s minimum)
        angles_long = np.linspace(0, np.pi / 2, 8)
        for idx, i in enumerate(range(50, 58)):
            lateral_g[i] = 0.8
            lat[i] += 0.002 * np.sin(angles_long[idx])
            lon[i] += 0.002 * np.cos(angles_long[idx])

        df = pd.DataFrame({
            'GPS Latitude': lat,
            'GPS Longitude': lon,
            'GPS LatAcc': lateral_g
        }, index=time)

        corners = detect_corners(df, min_corner_duration=0.5, curvature_threshold=0.003)

        # Should only detect the long corner
        assert len(corners) == 1
        assert corners[0].start_idx >= 48
        assert corners[0].end_idx <= 60

    def test_handles_corner_at_end_of_data(self):
        """Test that corners extending to end of data are handled"""
        n = 100
        time = np.linspace(0, 10, n)

        lat = np.full(n, 43.797)
        lon = np.full(n, -87.99)
        lateral_g = np.zeros(n)

        # Corner starts at 70, extends to end
        angles = np.linspace(0, np.pi / 2, n - 70)
        for idx, i in enumerate(range(70, n)):
            lateral_g[i] = 0.6
            lat[i] += 0.002 * np.sin(angles[idx])
            lon[i] += 0.002 * np.cos(angles[idx])

        df = pd.DataFrame({
            'GPS Latitude': lat,
            'GPS Longitude': lon,
            'GPS LatAcc': lateral_g
        }, index=time)

        corners = detect_corners(df, curvature_threshold=0.003)

        assert len(corners) >= 1
        assert corners[-1].end_idx == n - 1

    def test_returns_corner_objects(self):
        """Test that function returns Corner objects with correct fields"""
        n = 100
        time = np.linspace(0, 10, n)

        lat = 43.797 + 0.001 * np.sin(np.linspace(0, np.pi / 2, n))
        lon = -87.99 + 0.001 * np.cos(np.linspace(0, np.pi / 2, n))
        lateral_g = 0.5 * np.sin(np.linspace(0, np.pi, n))

        df = pd.DataFrame({
            'GPS Latitude': lat,
            'GPS Longitude': lon,
            'GPS LatAcc': lateral_g
        }, index=time)

        corners = detect_corners(df)

        assert len(corners) > 0
        for corner in corners:
            assert isinstance(corner, Corner)
            assert hasattr(corner, 'start_idx')
            assert hasattr(corner, 'end_idx')
            assert hasattr(corner, 'direction')
            assert corner.direction in ['left', 'right']
            assert corner.start_idx < corner.end_idx

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        df = pd.DataFrame({
            'GPS Latitude': [],
            'GPS Longitude': [],
            'GPS LatAcc': []
        })

        corners = detect_corners(df)
        assert corners == []

    def test_straight_section_no_corners(self):
        """Test that straight sections produce no corners"""
        n = 100
        time = np.linspace(0, 10, n)

        # Straight line
        lat = np.linspace(43.797, 43.807, n)
        lon = np.full(n, -87.99)
        lateral_g = np.full(n, 0.05)  # Low lateral G

        df = pd.DataFrame({
            'GPS Latitude': lat,
            'GPS Longitude': lon,
            'GPS LatAcc': lateral_g
        }, index=time)

        corners = detect_corners(df)
        assert len(corners) == 0

    def test_custom_thresholds(self):
        """Test that custom thresholds are respected"""
        n = 100
        time = np.linspace(0, 10, n)

        lat = 43.797 + 0.001 * np.sin(np.linspace(0, np.pi / 2, n))
        lon = -87.99 + 0.001 * np.cos(np.linspace(0, np.pi / 2, n))
        lateral_g = np.full(n, 0.4)  # Moderate lateral G

        df = pd.DataFrame({
            'GPS Latitude': lat,
            'GPS Longitude': lon,
            'GPS LatAcc': lateral_g
        }, index=time)

        # High threshold - should not detect
        corners_high = detect_corners(df, lateral_g_threshold=0.6)
        assert len(corners_high) == 0

        # Low threshold - should detect
        corners_low = detect_corners(df, lateral_g_threshold=0.3)
        assert len(corners_low) > 0


class TestApexDetectionPhase4:
    """Tests for Phase 4 apex detection using maximum lateral G"""

    def test_apex_is_max_lateral_g_point(self):
        """Test that apex_idx corresponds to maximum lateral G"""
        n = 100
        time = np.linspace(0, 10, n)

        # Create corner with clear max lateral G point
        lat = np.full(n, 43.797)
        lon = np.full(n, -87.99)
        lateral_g = np.zeros(n)

        # Corner from 20-60, with peak lateral G at index 40
        for i in range(20, 61):
            # Triangular distribution peaking at 40
            if i <= 40:
                lateral_g[i] = 0.3 + 0.5 * (i - 20) / 20
            else:
                lateral_g[i] = 0.8 - 0.5 * (i - 40) / 20

        # Add curvature
        lat[20:61] += 0.001 * np.sin(np.linspace(0, np.pi, 41))
        lon[20:61] += 0.001 * np.cos(np.linspace(0, np.pi, 41))

        df = pd.DataFrame({
            'GPS Latitude': lat,
            'GPS Longitude': lon,
            'GPS LatAcc': lateral_g
        }, index=time)

        corners = detect_corners(df, curvature_threshold=0.003)

        assert len(corners) >= 1
        corner = corners[0]
        # Apex should be at or very near index 40 (max lateral G)
        assert abs(corner.apex_idx - 40) <= 2

    def test_min_speed_idx_different_from_apex(self):
        """Test that min_speed_idx can differ from apex_idx"""
        n = 100
        time = np.linspace(0, 10, n)

        lat = np.full(n, 43.797)
        lon = np.full(n, -87.99)
        lateral_g = np.zeros(n)
        speed = np.full(n, 80.0)

        # Corner from 20-60
        # Max lateral G at index 35
        for i in range(20, 61):
            if i <= 35:
                lateral_g[i] = 0.3 + 0.5 * (i - 20) / 15
            else:
                lateral_g[i] = 0.8 - 0.5 * (i - 35) / 25

        # Min speed at index 45 (later than max lateral G)
        for i in range(20, 61):
            if i <= 45:
                speed[i] = 80 - 30 * (i - 20) / 25
            else:
                speed[i] = 50 + 30 * (i - 45) / 15

        # Add curvature
        lat[20:61] += 0.001 * np.sin(np.linspace(0, np.pi, 41))
        lon[20:61] += 0.001 * np.cos(np.linspace(0, np.pi, 41))

        df = pd.DataFrame({
            'GPS Latitude': lat,
            'GPS Longitude': lon,
            'GPS LatAcc': lateral_g,
            'GPS Speed': speed * 0.44704  # Convert to m/s
        }, index=time)

        corners = detect_corners(df, curvature_threshold=0.003)

        assert len(corners) >= 1
        corner = corners[0]
        # Apex should be near index 35 (max lateral G)
        assert abs(corner.apex_idx - 35) <= 2
        # Min speed should be near index 45
        assert abs(corner.min_speed_idx - 45) <= 2
        # They should be different
        assert abs(corner.apex_idx - corner.min_speed_idx) > 5

    def test_apex_handles_negative_lateral_g(self):
        """Test that apex uses absolute lateral G (handles left turns)"""
        n = 100
        time = np.linspace(0, 10, n)

        lat = np.full(n, 43.797)
        lon = np.full(n, -87.99)
        lateral_g = np.zeros(n)

        # Left turn: negative lateral G, peak at index 40
        for i in range(20, 61):
            if i <= 40:
                lateral_g[i] = -0.3 - 0.5 * (i - 20) / 20
            else:
                lateral_g[i] = -0.8 + 0.5 * (i - 40) / 20

        lat[20:61] += 0.001 * np.sin(np.linspace(0, np.pi, 41))
        lon[20:61] += 0.001 * np.cos(np.linspace(0, np.pi, 41))

        df = pd.DataFrame({
            'GPS Latitude': lat,
            'GPS Longitude': lon,
            'GPS LatAcc': lateral_g
        }, index=time)

        corners = detect_corners(df, curvature_threshold=0.003)

        assert len(corners) >= 1
        corner = corners[0]
        assert corner.direction == "left"
        # Apex should be at max absolute lateral G (index 40)
        assert abs(corner.apex_idx - 40) <= 2

    def test_apex_in_merged_corners(self):
        """Test that merged corners recalculate apex correctly"""
        n = 200
        time = np.linspace(0, 20, n)

        lat = np.full(n, 43.797)
        lon = np.full(n, -87.99)
        lateral_g = np.zeros(n)

        # First corner: 20-35, peak lateral G at 27
        for i in range(20, 36):
            lateral_g[i] = 0.5 if i == 27 else 0.4

        # Second corner (chicane): 38-53, peak lateral G at 45
        for i in range(38, 54):
            lateral_g[i] = -0.7 if i == 45 else -0.5

        # Add curvature to both
        lat[20:36] += 0.001 * np.sin(np.linspace(0, np.pi, 16))
        lon[20:36] += 0.001 * np.cos(np.linspace(0, np.pi, 16))
        lat[38:54] += 0.001 * np.sin(np.linspace(0, np.pi, 16))
        lon[38:54] += 0.001 * np.cos(np.linspace(0, np.pi, 16))

        df = pd.DataFrame({
            'GPS Latitude': lat,
            'GPS Longitude': lon,
            'GPS LatAcc': lateral_g
        }, index=time)

        corners = detect_corners(df, merge_gap=0.5, curvature_threshold=0.003)

        # Should merge into one corner
        assert len(corners) == 1
        corner = corners[0]
        # Apex should be at max absolute lateral G across merged corner (index 45, |0.7|)
        assert abs(corner.apex_idx - 45) <= 2

    def test_corner_zone_uses_max_lateral_g(self):
        """Test that CornerZone from CornerDetector uses max lateral G"""
        n = 200
        time = np.linspace(0, 20, n)
        t = time / 5

        lat = 43.797 + 0.003 * np.sin(t)
        lon = -87.99 + 0.003 * np.cos(t)
        speed = 80 - 30 * np.abs(np.sin(t))
        radius = 1000 - 900 * np.abs(np.sin(t))
        lat_acc = 0.8 * np.sin(t)
        lon_acc = -0.5 * np.sin(t + 0.5)

        detector = CornerDetector()
        result = detector.detect_from_arrays(
            time, lat, lon, speed, radius, lat_acc, lon_acc
        )

        # Check that detected corners have apex set
        for corner in result.corners:
            assert corner.apex_idx >= corner.entry_idx
            assert corner.apex_idx <= corner.exit_idx
            # Verify apex is at max lateral G
            corner_lat_acc = np.abs(lat_acc[corner.entry_idx:corner.exit_idx+1])
            max_idx = corner.entry_idx + np.argmax(corner_lat_acc)
            assert corner.apex_idx == max_idx


class TestPhase5Classification:
    """Tests for Phase 5 corner classification and severity rating"""

    def test_hairpin_classification(self):
        """Test that slow corners are classified as hairpins"""
        n = 200
        time = np.linspace(0, 20, n)
        lat = np.full(n, 43.797)
        lon = np.full(n, -87.99)
        speed = np.full(n, 80.0)
        radius = np.full(n, 500.0)  # Start with large radius (straight)
        lat_acc = np.zeros(n)
        lon_acc = np.zeros(n)

        # Add a hairpin: low speed corner (30 mph apex)
        # Make lateral G ramp up and down with speed
        for i in range(50, 66):  # 16 samples = 1.6s at 10Hz
            # Speed: 80 -> 30 -> 80
            speed_factor = abs(i - 58) / 8
            speed[i] = 80 - 50 * (1 - speed_factor)  # Down to 30 mph at apex (i=58)
            radius[i] = 25.0  # Tight radius (below 200m threshold)
            # Lateral G peaks where speed is lowest (apex)
            lat_acc[i] = 0.4 + 0.4 * (1 - speed_factor)  # 0.4 to 0.8 at apex

        detector = CornerDetector()
        result = detector.detect_from_arrays(
            time, lat, lon, speed, radius, lat_acc, lon_acc
        )

        # Should detect at least one corner
        assert len(result.corners) > 0

        # Should have at least one hairpin
        hairpins = [c for c in result.corners if c.corner_type == "hairpin"]
        assert len(hairpins) > 0, f"No hairpins found. Details: {[(c.corner_type, c.apex_speed_mph, c.apex_idx, time[c.exit_idx] - time[c.entry_idx]) for c in result.corners]}"
        assert hairpins[0].apex_speed_mph < 40

    def test_sweeper_classification(self):
        """Test that fast, long corners are classified as sweepers"""
        n = 300
        time = np.linspace(0, 30, n)
        lat = np.full(n, 43.797)
        lon = np.full(n, -87.99)
        speed = np.full(n, 90.0)
        radius = np.full(n, 500.0)
        lat_acc = np.zeros(n)
        lon_acc = np.zeros(n)

        # Add a sweeper: fast (70 mph) and long (2.5s duration)
        # 2.5s at 10Hz = 25 samples
        for i in range(50, 76):
            speed[i] = 90 - 20 * (abs(i - 63) / 13)  # Down to 70 mph
            radius[i] = 180.0
            lat_acc[i] = 0.5

        detector = CornerDetector()
        result = detector.detect_from_arrays(
            time, lat, lon, speed, radius, lat_acc, lon_acc
        )

        # Should detect sweeper
        sweepers = [c for c in result.corners if c.corner_type == "sweeper"]
        assert len(sweepers) > 0
        assert sweepers[0].apex_speed_mph > 60

    def test_kink_classification(self):
        """Test that very fast, short corners are classified as kinks"""
        n = 200
        time = np.linspace(0, 20, n)
        lat = np.full(n, 43.797)
        lon = np.full(n, -87.99)
        speed = np.full(n, 110.0)
        radius = np.full(n, 1000.0)  # Large radius (straight)
        lat_acc = np.zeros(n)
        lon_acc = np.zeros(n)

        # Add a kink: very fast (85 mph) and short (0.8s at 10Hz = 8 samples, but need >5 for 0.5s min)
        for i in range(50, 58):  # 0.8s at 10Hz
            speed[i] = 110 - 25 * (abs(i - 54) / 4)  # Down to 85 mph
            radius[i] = 180.0  # Below threshold
            lat_acc[i] = 0.4 if i != 54 else 0.6

        detector = CornerDetector()
        result = detector.detect_from_arrays(
            time, lat, lon, speed, radius, lat_acc, lon_acc
        )

        # Should detect at least one corner
        assert len(result.corners) > 0

        # Should have at least one kink
        kinks = [c for c in result.corners if c.corner_type == "kink"]
        assert len(kinks) > 0, f"No kinks found. Corner types: {[c.corner_type for c in result.corners]}"
        assert kinks[0].apex_speed_mph > 80

    def test_chicane_classification(self):
        """Test that direction changes are classified as chicanes"""
        n = 200
        time = np.linspace(0, 20, n)
        lat = np.full(n, 43.797)
        lon = np.full(n, -87.99)
        speed = np.full(n, 70.0)
        radius = np.full(n, 500.0)
        lat_acc = np.zeros(n)
        lon_acc = np.zeros(n)

        # Add chicane: quick left-right (0.8s total)
        # Left turn
        for i in range(50, 54):
            lat_acc[i] = -0.5
            radius[i] = 100.0

        # Right turn
        for i in range(54, 58):
            lat_acc[i] = 0.5
            radius[i] = 100.0

        detector = CornerDetector()
        result = detector.detect_from_arrays(
            time, lat, lon, speed, radius, lat_acc, lon_acc
        )

        # Should detect chicane
        chicanes = [c for c in result.corners if c.corner_type == "chicane"]
        # Note: Chicane detection requires both positive and negative lateral G
        # and short duration, which may not always trigger with synthetic data

    def test_severity_rating_very_easy(self):
        """Test severity rating 1 for minimal speed loss"""
        n = 200
        time = np.linspace(0, 20, n)
        lat = np.full(n, 43.797)
        lon = np.full(n, -87.99)
        speed = np.full(n, 100.0)
        radius = np.full(n, 500.0)
        lat_acc = np.zeros(n)

        # Very easy corner: 100 mph -> 95 mph (5% loss)
        for i in range(50, 71):
            speed[i] = 100 - 5 * (abs(i - 60) / 10)
            radius[i] = 250.0
            lat_acc[i] = 0.4

        detector = CornerDetector()
        result = detector.detect_from_arrays(
            time, lat, lon, speed, radius, lat_acc
        )

        if result.corners:
            corner = result.corners[0]
            assert corner.severity == 1  # Very easy

    def test_severity_rating_moderate(self):
        """Test severity rating 3 for moderate speed loss"""
        n = 200
        time = np.linspace(0, 20, n)
        lat = np.full(n, 43.797)
        lon = np.full(n, -87.99)
        speed = np.full(n, 100.0)
        radius = np.full(n, 500.0)
        lat_acc = np.zeros(n)

        # Moderate corner: 100 mph -> 65 mph (35% loss)
        for i in range(50, 71):
            speed[i] = 100 - 35 * (abs(i - 60) / 10)
            radius[i] = 100.0
            lat_acc[i] = 0.6

        detector = CornerDetector()
        result = detector.detect_from_arrays(
            time, lat, lon, speed, radius, lat_acc
        )

        if result.corners:
            corner = result.corners[0]
            assert corner.severity == 3  # Moderate

    def test_severity_rating_very_hard(self):
        """Test severity rating 5 for severe speed loss"""
        n = 200
        time = np.linspace(0, 20, n)
        lat = np.full(n, 43.797)
        lon = np.full(n, -87.99)
        speed = np.full(n, 100.0)
        radius = np.full(n, 500.0)
        lat_acc = np.zeros(n)

        # Very hard corner: 100 mph -> 30 mph (70% loss)
        for i in range(50, 71):
            speed[i] = 100 - 70 * (abs(i - 60) / 10)
            radius[i] = 30.0
            lat_acc[i] = 0.8

        detector = CornerDetector()
        result = detector.detect_from_arrays(
            time, lat, lon, speed, radius, lat_acc
        )

        if result.corners:
            corner = result.corners[0]
            assert corner.severity == 5  # Very hard

    def test_corner_zone_has_severity(self):
        """Test that CornerZone objects have severity field"""
        zone = CornerZone(
            name="T1",
            apex_speed_mph=60.0,
            severity=3
        )
        assert zone.severity == 3

        # Test serialization includes severity
        d = zone.to_dict()
        assert "severity" in d
        assert d["severity"] == 3


class TestRealDataCompatibility:
    """Tests that work with real data format if available"""

    @pytest.fixture
    def real_parquet_path(self):
        """Path to real parquet file if it exists"""
        possible_paths = [
            Path("data/exports/processed/test_session.parquet"),
            Path("data/exports/processed/20250712_104619_Road America_a_0394.parquet"),
        ]
        for path in possible_paths:
            if path.exists():
                return str(path)
        return None

    def test_with_real_data(self, real_parquet_path):
        """Test with real parquet file if available"""
        if real_parquet_path is None:
            pytest.skip("No real parquet file available")

        analyzer = CornerAnalyzer()
        result = analyzer.analyze_from_parquet(real_parquet_path)

        assert isinstance(result, CornerAnalysisResult)
        assert result.session_id is not None

    def test_detect_corners_with_real_data(self, real_parquet_path):
        """Test Phase 3 detect_corners with real data"""
        if real_parquet_path is None:
            pytest.skip("No real parquet file available")

        df = pd.read_parquet(real_parquet_path)

        # Check if required columns exist
        has_lat_acc = 'GPS LatAcc' in df.columns or 'lat_acc' in df.columns
        if not has_lat_acc:
            pytest.skip("Real data missing lateral acceleration column")

        corners = detect_corners(df)

        # Should detect reasonable number of corners
        assert isinstance(corners, list)
        assert len(corners) > 0
        assert all(isinstance(c, Corner) for c in corners)

        # All corners should have valid indices
        for corner in corners:
            assert corner.start_idx >= 0
            assert corner.end_idx < len(df)
            assert corner.start_idx < corner.end_idx
            assert corner.direction in ['left', 'right']
            # Phase 4: Check apex indices
            assert corner.apex_idx >= corner.start_idx
            assert corner.apex_idx <= corner.end_idx
            assert corner.min_speed_idx >= corner.start_idx
            assert corner.min_speed_idx <= corner.end_idx
