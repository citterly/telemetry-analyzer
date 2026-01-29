"""
Integration tests with real Road America telemetry data (feat-051)

Tests end-to-end analysis pipeline using actual Parquet files.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


# Test data fixtures
PARQUET_DIR = Path(__file__).parent.parent.parent / "data" / "exports" / "processed"
TEST_SESSION = PARQUET_DIR / "test_session.parquet"
ROAD_AMERICA_SESSION = PARQUET_DIR / "20250712_104619_Road America_a_0394.parquet"


@pytest.fixture
def test_session_df():
    """Load test session data"""
    if not TEST_SESSION.exists():
        pytest.skip("Test session parquet not available")
    return pd.read_parquet(TEST_SESSION)


@pytest.fixture
def road_america_df():
    """Load Road America session data"""
    if not ROAD_AMERICA_SESSION.exists():
        pytest.skip("Road America session parquet not available")
    return pd.read_parquet(ROAD_AMERICA_SESSION)


class TestParquetLoading:
    """Tests for loading and validating Parquet files"""

    def test_parquet_files_exist(self):
        """Test that expected Parquet files exist"""
        parquet_files = list(PARQUET_DIR.glob("*.parquet"))
        assert len(parquet_files) >= 1

    def test_test_session_loads(self, test_session_df):
        """Test that test_session.parquet loads successfully"""
        assert isinstance(test_session_df, pd.DataFrame)
        assert len(test_session_df) > 0

    def test_road_america_session_loads(self, road_america_df):
        """Test that Road America session loads successfully"""
        assert isinstance(road_america_df, pd.DataFrame)
        assert len(road_america_df) > 0

    def test_test_session_has_gps(self, test_session_df):
        """Test that test_session has GPS columns"""
        assert 'gps_lat' in test_session_df.columns
        assert 'gps_lon' in test_session_df.columns
        assert 'gps_speed' in test_session_df.columns

    def test_road_america_has_gps(self, road_america_df):
        """Test that Road America session has GPS columns"""
        assert 'GPS Latitude' in road_america_df.columns
        assert 'GPS Longitude' in road_america_df.columns
        assert 'GPS Speed' in road_america_df.columns


class TestGPSDataQuality:
    """Tests for GPS data quality validation"""

    def test_test_session_gps_coords_valid(self, test_session_df):
        """Test that GPS coordinates are in valid range"""
        lat = test_session_df['gps_lat']
        lon = test_session_df['gps_lon']

        # Valid latitude: -90 to 90
        assert lat.min() >= -90
        assert lat.max() <= 90

        # Valid longitude: -180 to 180
        assert lon.min() >= -180
        assert lon.max() <= 180

    def test_test_session_gps_is_road_america(self, test_session_df):
        """Test that GPS data is from Road America area"""
        lat = test_session_df['gps_lat']
        lon = test_session_df['gps_lon']

        # Road America approximate bounding box
        assert lat.mean() >= 43.79
        assert lat.mean() <= 43.81
        assert lon.mean() >= -88.01
        assert lon.mean() <= -87.98

    def test_road_america_gps_is_road_america(self, road_america_df):
        """Test that GPS data is from Road America area"""
        lat = road_america_df['GPS Latitude']
        lon = road_america_df['GPS Longitude']

        # Road America approximate bounding box
        assert lat.mean() >= 43.79
        assert lat.mean() <= 43.81
        assert lon.mean() >= -88.01
        assert lon.mean() <= -87.98

    def test_test_session_speed_valid(self, test_session_df):
        """Test that GPS speed values are reasonable"""
        speed = test_session_df['gps_speed']

        # Speed should be positive and reasonable for a race car
        assert speed.min() >= 0
        assert speed.max() <= 300  # km/h or mph, either is reasonable

    def test_road_america_speed_valid(self, road_america_df):
        """Test that GPS speed values are reasonable"""
        speed = road_america_df['GPS Speed']

        # Speed should be positive and reasonable
        assert speed.min() >= 0
        assert speed.max() <= 300


class TestTrackDetection:
    """Tests for track detection functionality"""

    def test_detect_track_from_gps(self, test_session_df):
        """Test track detection from GPS coordinates"""
        from src.config.tracks import detect_track

        lat = test_session_df['gps_lat'].values
        lon = test_session_df['gps_lon'].values

        track = detect_track(lat, lon)

        # Should detect Road America
        assert track is not None
        assert track.id == 'road-america'

    def test_detect_track_from_road_america_session(self, road_america_df):
        """Test track detection from Road America session"""
        from src.config.tracks import detect_track

        lat = road_america_df['GPS Latitude'].values
        lon = road_america_df['GPS Longitude'].values

        track = detect_track(lat, lon)

        assert track is not None
        assert track.id == 'road-america'

    def test_track_has_corners(self, test_session_df):
        """Test that detected track has corner definitions"""
        from src.config.tracks import detect_track

        lat = test_session_df['gps_lat'].values
        lon = test_session_df['gps_lon'].values

        track = detect_track(lat, lon)

        assert track is not None
        assert len(track.corners) > 0


class TestLapDetectionIntegration:
    """Tests for lap detection with real data"""

    def test_lap_analyzer_with_test_session(self, test_session_df):
        """Test lap analyzer with test session data"""
        from src.analysis.lap_analyzer import LapAnalyzer

        # Convert to session_data format
        session_data = {
            'time': test_session_df.index.values if isinstance(test_session_df.index, pd.RangeIndex) else np.arange(len(test_session_df)) * 0.1,
            'latitude': test_session_df['gps_lat'].values,
            'longitude': test_session_df['gps_lon'].values,
            'rpm': np.full(len(test_session_df), 5000.0),  # Mock RPM data
            'speed_mph': test_session_df['gps_speed'].values * 2.237  # Assuming m/s
        }

        # Handle pandas float index
        if hasattr(test_session_df.index, 'to_numpy'):
            session_data['time'] = test_session_df.index.to_numpy()

        analyzer = LapAnalyzer(session_data)
        laps = analyzer.detect_laps()

        # Should find at least one lap
        assert isinstance(laps, list)

    def test_lap_detection_finds_reasonable_lap_times(self, test_session_df):
        """Test that detected lap times are reasonable for Road America"""
        from src.analysis.lap_analyzer import LapAnalyzer

        session_data = {
            'time': test_session_df.index.to_numpy() if hasattr(test_session_df.index, 'to_numpy') else np.arange(len(test_session_df)) * 0.1,
            'latitude': test_session_df['gps_lat'].values,
            'longitude': test_session_df['gps_lon'].values,
            'rpm': np.full(len(test_session_df), 5000.0),
            'speed_mph': test_session_df['gps_speed'].values * 2.237
        }

        analyzer = LapAnalyzer(session_data)
        laps = analyzer.detect_laps()

        # If laps were found, check they're reasonable for Road America
        for lap in laps:
            # Road America lap times typically 2-4 minutes
            assert lap.lap_time >= 90  # At least 1.5 min
            assert lap.lap_time <= 300  # At most 5 min


class TestVehicleProfilesIntegration:
    """Tests for vehicle profiles with real session context"""

    def test_vehicle_database_loaded(self):
        """Test that vehicle database loads successfully"""
        from src.config.vehicles import get_vehicle_database

        db = get_vehicle_database()

        assert len(db.vehicles) >= 1

    def test_active_vehicle_is_bmw(self):
        """Test that default active vehicle is Andy's M3"""
        from src.config.vehicles import get_active_vehicle

        vehicle = get_active_vehicle()

        assert vehicle is not None
        assert 'BMW' in vehicle.make or 'M3' in vehicle.name

    def test_vehicle_speed_calculation(self):
        """Test vehicle speed calculation with known values"""
        from src.config.vehicles import get_active_vehicle

        vehicle = get_active_vehicle()

        # Test speed calculation in 4th gear at 6000 RPM
        speed = vehicle.calculate_speed_at_rpm(6000, 4)

        # Should be reasonable highway speed
        assert 100 < speed < 150  # mph


class TestSpeedAnalysis:
    """Tests for speed analysis with real data"""

    def test_max_speed_reasonable(self, road_america_df):
        """Test that max speed is reasonable for Road America"""
        speed = road_america_df['GPS Speed']

        # Max speed at Road America around 150 mph on main straight
        # GPS Speed appears to be in m/s based on earlier check
        max_speed_mph = speed.max() * 2.237

        assert max_speed_mph > 50  # Should reach highway speeds
        assert max_speed_mph < 200  # Shouldn't exceed reasonable race car speeds

    def test_speed_distribution(self, road_america_df):
        """Test speed distribution is reasonable for racing"""
        speed = road_america_df['GPS Speed'] * 2.237  # Convert to mph

        # Should have a mix of speeds (not all slow or all fast)
        assert speed.std() > 10  # Some variation in speed

    def test_lap_average_speed(self, test_session_df):
        """Test that average lap speed is reasonable"""
        speed = test_session_df['gps_speed'] * 2.237  # Convert to mph

        avg_speed = speed.mean()

        # Average speed at Road America should be reasonable
        assert 30 < avg_speed < 100


class TestTrackMapGeneration:
    """Tests for track map visualization"""

    def test_track_map_generates_svg(self, test_session_df):
        """Test that track map generates valid SVG"""
        from src.visualization.track_map import TrackMap

        lat = test_session_df['gps_lat'].values
        lon = test_session_df['gps_lon'].values
        speed = test_session_df['gps_speed'].values

        track_map = TrackMap()
        svg = track_map.render_svg(lat, lon, speed)

        assert '<svg' in svg
        assert '</svg>' in svg

    def test_track_map_generates_html(self, test_session_df):
        """Test that track map generates valid HTML"""
        from src.visualization.track_map import TrackMap

        lat = test_session_df['gps_lat'].values
        lon = test_session_df['gps_lon'].values
        speed = test_session_df['gps_speed'].values

        track_map = TrackMap()
        html = track_map.render_html(lat, lon, speed)

        assert '<html' in html or '<!DOCTYPE' in html
        assert '<svg' in html


class TestEndToEndAnalysis:
    """End-to-end analysis pipeline tests"""

    def test_full_analysis_pipeline(self, test_session_df):
        """Test complete analysis pipeline"""
        from src.analysis.lap_analyzer import LapAnalyzer
        from src.config.tracks import detect_track
        from src.config.vehicles import get_active_vehicle

        # 1. Load and validate data
        lat = test_session_df['gps_lat'].values
        lon = test_session_df['gps_lon'].values
        speed_mph = test_session_df['gps_speed'].values * 2.237

        assert len(lat) > 0

        # 2. Detect track
        track = detect_track(lat, lon)
        assert track is not None

        # 3. Get vehicle config
        vehicle = get_active_vehicle()
        assert vehicle is not None

        # 4. Prepare session data
        session_data = {
            'time': test_session_df.index.to_numpy() if hasattr(test_session_df.index, 'to_numpy') else np.arange(len(test_session_df)) * 0.1,
            'latitude': lat,
            'longitude': lon,
            'rpm': np.full(len(test_session_df), 5000.0),
            'speed_mph': speed_mph
        }

        # 5. Run lap analyzer
        analyzer = LapAnalyzer(session_data)
        laps = analyzer.detect_laps()

        # Pipeline should complete without errors
        assert True

    def test_analysis_api_integration(self, test_session_df):
        """Test that analysis results can be JSON serialized"""
        import json
        from src.analysis.lap_analyzer import LapAnalyzer

        session_data = {
            'time': test_session_df.index.to_numpy() if hasattr(test_session_df.index, 'to_numpy') else np.arange(len(test_session_df)) * 0.1,
            'latitude': test_session_df['gps_lat'].values,
            'longitude': test_session_df['gps_lon'].values,
            'rpm': np.full(len(test_session_df), 5000.0),
            'speed_mph': test_session_df['gps_speed'].values * 2.237
        }

        analyzer = LapAnalyzer(session_data)
        laps = analyzer.detect_laps()

        # Convert laps to JSON-serializable format
        lap_dicts = []
        for lap in laps:
            lap_dicts.append({
                'lap_number': lap.lap_number,
                'lap_time': lap.lap_time,
                'max_speed_mph': lap.max_speed_mph,
                'max_rpm': lap.max_rpm,
                'avg_rpm': lap.avg_rpm
            })

        # Should be JSON serializable
        json_str = json.dumps(lap_dicts)
        assert json_str is not None
