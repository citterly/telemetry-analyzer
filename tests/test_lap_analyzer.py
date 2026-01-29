"""
Tests for lap_analyzer module (feat-050)

Unit tests for LapAnalyzer, LapInfo, and lap detection functions.
"""

import pytest
import numpy as np
from typing import Dict


class TestLapInfo:
    """Tests for LapInfo dataclass"""

    def test_lap_info_creation(self):
        """Test creating a LapInfo instance"""
        from src.analysis.lap_analyzer import LapInfo

        lap = LapInfo(
            lap_number=1,
            start_index=0,
            end_index=1000,
            start_time=0.0,
            end_time=150.0,
            lap_time=150.0,
            max_speed_mph=120.5,
            max_rpm=7200,
            avg_rpm=5500,
            sample_count=1001
        )

        assert lap.lap_number == 1
        assert lap.start_index == 0
        assert lap.end_index == 1000
        assert lap.lap_time == 150.0
        assert lap.max_speed_mph == 120.5
        assert lap.max_rpm == 7200
        assert lap.avg_rpm == 5500
        assert lap.sample_count == 1001


class TestLapAnalyzer:
    """Tests for LapAnalyzer class"""

    @pytest.fixture
    def mock_session_data(self) -> Dict:
        """Create mock session data for testing"""
        # Simulate a session with ~3 laps around a track
        # Generate GPS data that passes near a start/finish point multiple times
        n_points = 3000
        time = np.linspace(0, 450, n_points)  # 450 seconds = 7.5 minutes

        # Simulate circular track pattern
        # Start/finish is at position (0, 0)
        t_rad = time * (2 * np.pi / 150)  # Complete circle every 150 seconds
        radius = 0.01  # ~1km radius in degrees

        # Add offset so we're not centered at (0, 0)
        center_lat = 43.797875
        center_lon = -87.989638

        latitude = center_lat + radius * np.sin(t_rad)
        longitude = center_lon + radius * np.cos(t_rad)

        # RPM and speed data
        rpm = 4000 + 2000 * np.abs(np.sin(t_rad * 5))  # Varying RPM
        speed_mph = 50 + 40 * np.abs(np.sin(t_rad * 5))  # Varying speed

        return {
            'time': time,
            'latitude': latitude,
            'longitude': longitude,
            'rpm': rpm,
            'speed_mph': speed_mph,
            'speed_ms': speed_mph / 2.237
        }

    @pytest.fixture
    def simple_session_data(self) -> Dict:
        """Create simple session data with clear lap boundaries"""
        n_points = 300
        time = np.linspace(0, 300, n_points)  # 300 seconds

        # Simple linear data
        latitude = np.full(n_points, 43.797875)
        longitude = np.full(n_points, -87.989638)
        rpm = np.full(n_points, 5000.0)
        speed_mph = np.full(n_points, 80.0)

        return {
            'time': time,
            'latitude': latitude,
            'longitude': longitude,
            'rpm': rpm,
            'speed_mph': speed_mph,
            'speed_ms': speed_mph / 2.237
        }

    def test_lap_analyzer_init(self, mock_session_data):
        """Test LapAnalyzer initialization"""
        from src.analysis.lap_analyzer import LapAnalyzer

        analyzer = LapAnalyzer(mock_session_data)

        assert analyzer.session_data is mock_session_data
        assert analyzer.laps == []
        assert analyzer.fastest_lap is None

    def test_detect_laps_returns_list(self, mock_session_data):
        """Test that detect_laps returns a list"""
        from src.analysis.lap_analyzer import LapAnalyzer

        analyzer = LapAnalyzer(mock_session_data)
        laps = analyzer.detect_laps()

        assert isinstance(laps, list)

    def test_split_by_time_fallback(self, simple_session_data):
        """Test time-based lap splitting as fallback"""
        from src.analysis.lap_analyzer import LapAnalyzer

        analyzer = LapAnalyzer(simple_session_data)
        laps = analyzer._split_by_time()

        assert len(laps) >= 1
        assert all(isinstance(lap, analyzer.laps[0].__class__ if analyzer.laps else type(laps[0])) for lap in laps)

    def test_create_lap_info_valid(self, simple_session_data):
        """Test creating lap info from valid indices"""
        from src.analysis.lap_analyzer import LapAnalyzer

        analyzer = LapAnalyzer(simple_session_data)
        lap_info = analyzer._create_lap_info(1, 0, 100)

        assert lap_info is not None
        assert lap_info.lap_number == 1
        assert lap_info.start_index == 0
        assert lap_info.end_index == 100
        assert lap_info.lap_time > 0
        assert lap_info.sample_count == 101

    def test_is_valid_lap_valid(self, simple_session_data):
        """Test lap validation with valid lap"""
        from src.analysis.lap_analyzer import LapAnalyzer, LapInfo

        analyzer = LapAnalyzer(simple_session_data)
        lap = LapInfo(
            lap_number=1,
            start_index=0,
            end_index=1000,
            start_time=0.0,
            end_time=150.0,
            lap_time=150.0,
            max_speed_mph=100,
            max_rpm=6000,
            avg_rpm=5000,
            sample_count=1001
        )

        assert analyzer._is_valid_lap(lap) is True

    def test_is_valid_lap_too_short(self, simple_session_data):
        """Test lap validation rejects lap that's too short"""
        from src.analysis.lap_analyzer import LapAnalyzer, LapInfo

        analyzer = LapAnalyzer(simple_session_data)
        lap = LapInfo(
            lap_number=1,
            start_index=0,
            end_index=100,
            start_time=0.0,
            end_time=30.0,
            lap_time=30.0,  # Too short
            max_speed_mph=100,
            max_rpm=6000,
            avg_rpm=5000,
            sample_count=101
        )

        assert analyzer._is_valid_lap(lap) is False

    def test_is_valid_lap_too_long(self, simple_session_data):
        """Test lap validation rejects lap that's too long"""
        from src.analysis.lap_analyzer import LapAnalyzer, LapInfo

        analyzer = LapAnalyzer(simple_session_data)
        lap = LapInfo(
            lap_number=1,
            start_index=0,
            end_index=1000,
            start_time=0.0,
            end_time=400.0,
            lap_time=400.0,  # Too long
            max_speed_mph=100,
            max_rpm=6000,
            avg_rpm=5000,
            sample_count=1001
        )

        assert analyzer._is_valid_lap(lap) is False

    def test_is_valid_lap_too_few_samples(self, simple_session_data):
        """Test lap validation rejects lap with too few samples"""
        from src.analysis.lap_analyzer import LapAnalyzer, LapInfo

        analyzer = LapAnalyzer(simple_session_data)
        lap = LapInfo(
            lap_number=1,
            start_index=0,
            end_index=10,
            start_time=0.0,
            end_time=150.0,
            lap_time=150.0,
            max_speed_mph=100,
            max_rpm=6000,
            avg_rpm=5000,
            sample_count=11  # Too few samples
        )

        assert analyzer._is_valid_lap(lap) is False

    def test_find_fastest_lap_empty(self, simple_session_data):
        """Test find_fastest_lap with no laps"""
        from src.analysis.lap_analyzer import LapAnalyzer

        analyzer = LapAnalyzer(simple_session_data)
        # Don't detect laps first

        result = analyzer.find_fastest_lap()
        assert result is None

    def test_find_fastest_lap_single(self, simple_session_data):
        """Test find_fastest_lap with single lap"""
        from src.analysis.lap_analyzer import LapAnalyzer, LapInfo

        analyzer = LapAnalyzer(simple_session_data)
        analyzer.laps = [
            LapInfo(1, 0, 100, 0.0, 150.0, 150.0, 100, 6000, 5000, 101)
        ]

        fastest = analyzer.find_fastest_lap()

        assert fastest is not None
        assert fastest.lap_number == 1

    def test_find_fastest_lap_multiple(self, simple_session_data):
        """Test find_fastest_lap identifies fastest among multiple"""
        from src.analysis.lap_analyzer import LapAnalyzer, LapInfo

        analyzer = LapAnalyzer(simple_session_data)
        analyzer.laps = [
            LapInfo(1, 0, 100, 0.0, 155.0, 155.0, 100, 6000, 5000, 101),
            LapInfo(2, 101, 200, 155.0, 300.0, 145.0, 100, 6000, 5000, 100),  # Fastest
            LapInfo(3, 201, 300, 300.0, 452.0, 152.0, 100, 6000, 5000, 100),
        ]

        fastest = analyzer.find_fastest_lap()

        assert fastest is not None
        assert fastest.lap_number == 2
        assert fastest.lap_time == 145.0

    def test_get_lap_data(self, simple_session_data):
        """Test extracting data for a specific lap"""
        from src.analysis.lap_analyzer import LapAnalyzer, LapInfo

        analyzer = LapAnalyzer(simple_session_data)
        lap = LapInfo(1, 10, 50, 10.0, 50.0, 40.0, 100, 6000, 5000, 41)

        lap_data = analyzer.get_lap_data(lap)

        assert 'lap_info' in lap_data
        assert 'time' in lap_data
        assert 'latitude' in lap_data
        assert 'longitude' in lap_data
        assert 'rpm' in lap_data
        assert len(lap_data['time']) == 41

    def test_get_lap_data_time_starts_at_zero(self, simple_session_data):
        """Test that lap data time is normalized to start at 0"""
        from src.analysis.lap_analyzer import LapAnalyzer, LapInfo

        analyzer = LapAnalyzer(simple_session_data)
        lap = LapInfo(1, 10, 50, 10.0, 50.0, 40.0, 100, 6000, 5000, 41)

        lap_data = analyzer.get_lap_data(lap)

        assert lap_data['time'][0] == 0.0

    def test_get_fastest_lap_data_none(self, simple_session_data):
        """Test get_fastest_lap_data returns None when no laps"""
        from src.analysis.lap_analyzer import LapAnalyzer

        analyzer = LapAnalyzer(simple_session_data)

        result = analyzer.get_fastest_lap_data()
        assert result is None

    def test_get_fastest_lap_data_valid(self, simple_session_data):
        """Test get_fastest_lap_data returns data for fastest lap"""
        from src.analysis.lap_analyzer import LapAnalyzer, LapInfo

        analyzer = LapAnalyzer(simple_session_data)
        analyzer.laps = [
            LapInfo(1, 0, 100, 0.0, 100.0, 100.0, 100, 6000, 5000, 101),
        ]

        lap_data = analyzer.get_fastest_lap_data()

        assert lap_data is not None
        assert 'lap_info' in lap_data
        assert lap_data['lap_info'].lap_number == 1

    def test_print_lap_summary_no_laps(self, simple_session_data, capsys):
        """Test print_lap_summary with no laps"""
        from src.analysis.lap_analyzer import LapAnalyzer

        analyzer = LapAnalyzer(simple_session_data)
        analyzer.print_lap_summary()

        captured = capsys.readouterr()
        assert 'No laps detected' in captured.out

    def test_print_lap_summary_with_laps(self, simple_session_data, capsys):
        """Test print_lap_summary with laps"""
        from src.analysis.lap_analyzer import LapAnalyzer, LapInfo

        analyzer = LapAnalyzer(simple_session_data)
        analyzer.laps = [
            LapInfo(1, 0, 100, 0.0, 150.0, 150.0, 100, 6000, 5000, 101),
        ]
        analyzer.fastest_lap = analyzer.laps[0]

        analyzer.print_lap_summary()

        captured = capsys.readouterr()
        assert 'Lap Analysis Summary' in captured.out
        assert 'Fastest Lap' in captured.out


class TestModuleFunctions:
    """Tests for module-level functions"""

    @pytest.fixture
    def simple_session_data(self) -> Dict:
        """Create simple session data"""
        n_points = 300
        time = np.linspace(0, 300, n_points)

        return {
            'time': time,
            'latitude': np.full(n_points, 43.797875),
            'longitude': np.full(n_points, -87.989638),
            'rpm': np.full(n_points, 5000.0),
            'speed_mph': np.full(n_points, 80.0),
            'speed_ms': np.full(n_points, 80.0 / 2.237)
        }

    def test_analyze_session_laps_returns_tuple(self, simple_session_data):
        """Test analyze_session_laps returns correct tuple"""
        from src.analysis.lap_analyzer import analyze_session_laps

        laps, fastest_data = analyze_session_laps(simple_session_data)

        assert isinstance(laps, list)
        # fastest_data can be None or Dict


class TestCrossingsDetection:
    """Tests for GPS crossing detection logic"""

    @pytest.fixture
    def analyzer_with_mock_data(self):
        """Create analyzer with controlled GPS data"""
        from src.analysis.lap_analyzer import LapAnalyzer

        n_points = 500
        time = np.linspace(0, 500, n_points)

        # Create GPS path that crosses start/finish multiple times
        start_lat, start_lon = 43.797875, -87.989638

        # Simple back-and-forth pattern
        latitude = start_lat + 0.01 * np.sin(time * 0.04)
        longitude = start_lon + 0.01 * np.cos(time * 0.04)

        session_data = {
            'time': time,
            'latitude': latitude,
            'longitude': longitude,
            'rpm': np.full(n_points, 5000.0),
            'speed_mph': np.full(n_points, 80.0),
        }

        return LapAnalyzer(session_data)

    def test_find_crossings_minimum_separation(self, analyzer_with_mock_data):
        """Test that crossings must be minimum distance apart"""
        from src.config.vehicle_config import PROCESSING_CONFIG

        distances = np.abs(np.sin(np.linspace(0, 10, 500)))
        threshold = PROCESSING_CONFIG['start_finish_threshold']
        time = np.linspace(0, 500, 500)

        crossings = analyzer_with_mock_data._find_crossings(distances, threshold, time)

        # Should have at least start point
        assert len(crossings) >= 1
        assert crossings[0] == 0
