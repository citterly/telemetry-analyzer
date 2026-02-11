"""
Tests for SessionDataLoader service (arch-002)

Tests cover: loading, column discovery, speed conversion,
missing channel errors, session_data_dict conversion.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.services.session_data_loader import SessionDataLoader, SessionChannels


@pytest.fixture
def sample_parquet(tmp_path):
    """Create a sample parquet file with typical telemetry columns."""
    n = 500
    time = np.linspace(0, 50, n)

    df = pd.DataFrame({
        "GPS Latitude": 43.0 + 0.001 * np.sin(time * 0.1),
        "GPS Longitude": -88.0 + 0.001 * np.cos(time * 0.1),
        "GPS Speed": np.random.uniform(20, 45, n),  # m/s range
        "RPM": np.random.uniform(3000, 7000, n),
        "GPS LatAcc": np.random.uniform(-1.5, 1.5, n),
        "GPS LonAcc": np.random.uniform(-1.5, 1.5, n),
        "PedalPos": np.random.uniform(0, 100, n),
    }, index=time)

    path = tmp_path / "session_test.parquet"
    df.to_parquet(path)
    return str(path)


@pytest.fixture
def mph_parquet(tmp_path):
    """Create a parquet file with speed in mph range."""
    n = 300
    time = np.linspace(0, 30, n)

    df = pd.DataFrame({
        "GPS Speed": np.random.uniform(50, 140, n),  # mph range
        "RPM": np.random.uniform(3000, 7000, n),
    }, index=time)

    path = tmp_path / "mph_session.parquet"
    df.to_parquet(path)
    return str(path)


@pytest.fixture
def minimal_parquet(tmp_path):
    """Create a parquet with only speed — no GPS or RPM."""
    n = 200
    time = np.linspace(0, 20, n)

    df = pd.DataFrame({
        "GPS Speed": np.random.uniform(20, 40, n),
    }, index=time)

    path = tmp_path / "minimal.parquet"
    df.to_parquet(path)
    return str(path)


@pytest.fixture
def empty_parquet(tmp_path):
    """Create a parquet with unrecognized columns."""
    n = 100
    time = np.linspace(0, 10, n)

    df = pd.DataFrame({
        "SomeUnknownChannel": np.zeros(n),
    }, index=time)

    path = tmp_path / "empty.parquet"
    df.to_parquet(path)
    return str(path)


class TestSessionDataLoader:
    """Tests for SessionDataLoader.load()"""

    def test_load_returns_session_channels(self, sample_parquet):
        loader = SessionDataLoader()
        channels = loader.load(sample_parquet)
        assert isinstance(channels, SessionChannels)

    def test_load_time_array(self, sample_parquet):
        loader = SessionDataLoader()
        channels = loader.load(sample_parquet)
        assert channels.time is not None
        assert len(channels.time) == 500
        assert channels.sample_count == 500

    def test_load_duration(self, sample_parquet):
        loader = SessionDataLoader()
        channels = loader.load(sample_parquet)
        assert abs(channels.duration_seconds - 50.0) < 0.2

    def test_load_session_id_from_filename(self, sample_parquet):
        loader = SessionDataLoader()
        channels = loader.load(sample_parquet)
        assert channels.session_id == "session_test"

    def test_load_source_path(self, sample_parquet):
        loader = SessionDataLoader()
        channels = loader.load(sample_parquet)
        assert channels.source_path == sample_parquet

    def test_load_discovers_latitude(self, sample_parquet):
        loader = SessionDataLoader()
        channels = loader.load(sample_parquet)
        assert channels.latitude is not None
        assert len(channels.latitude) == 500
        assert "latitude" in channels.column_map
        assert channels.column_map["latitude"] == "GPS Latitude"

    def test_load_discovers_longitude(self, sample_parquet):
        loader = SessionDataLoader()
        channels = loader.load(sample_parquet)
        assert channels.longitude is not None
        assert "longitude" in channels.column_map

    def test_load_discovers_rpm(self, sample_parquet):
        loader = SessionDataLoader()
        channels = loader.load(sample_parquet)
        assert channels.rpm is not None
        assert "rpm" in channels.column_map

    def test_load_discovers_lat_acc(self, sample_parquet):
        loader = SessionDataLoader()
        channels = loader.load(sample_parquet)
        assert channels.lat_acc is not None
        assert "lat_acc" in channels.column_map

    def test_load_discovers_lon_acc(self, sample_parquet):
        loader = SessionDataLoader()
        channels = loader.load(sample_parquet)
        assert channels.lon_acc is not None
        assert "lon_acc" in channels.column_map

    def test_load_discovers_throttle(self, sample_parquet):
        loader = SessionDataLoader()
        channels = loader.load(sample_parquet)
        assert channels.throttle is not None
        assert "throttle" in channels.column_map


class TestSpeedUnitDetection:
    """Tests for speed unit detection and conversion"""

    def test_ms_speed_detected(self, sample_parquet):
        """Speed < 100 max should be detected as m/s"""
        loader = SessionDataLoader()
        channels = loader.load(sample_parquet)
        assert channels.speed_unit_detected == "m/s"

    def test_ms_provides_both_units(self, sample_parquet):
        loader = SessionDataLoader()
        channels = loader.load(sample_parquet)
        assert channels.speed_mph is not None
        assert channels.speed_ms is not None
        # mph should be larger than m/s
        assert channels.speed_mph.max() > channels.speed_ms.max()

    def test_mph_speed_detected(self, mph_parquet):
        """Speed > 100 max should be detected as mph"""
        loader = SessionDataLoader()
        channels = loader.load(mph_parquet)
        assert channels.speed_unit_detected == "mph"

    def test_mph_provides_both_units(self, mph_parquet):
        loader = SessionDataLoader()
        channels = loader.load(mph_parquet)
        assert channels.speed_mph is not None
        assert channels.speed_ms is not None
        # m/s should be smaller than mph
        assert channels.speed_ms.max() < channels.speed_mph.max()

    def test_no_speed_returns_unknown(self, empty_parquet):
        loader = SessionDataLoader()
        channels = loader.load(empty_parquet)
        assert channels.speed_unit_detected == "unknown"
        assert channels.speed_mph is None
        assert channels.speed_ms is None


class TestHasProperties:
    """Tests for convenience properties"""

    def test_has_gps(self, sample_parquet):
        loader = SessionDataLoader()
        channels = loader.load(sample_parquet)
        assert channels.has_gps is True

    def test_has_speed(self, sample_parquet):
        loader = SessionDataLoader()
        channels = loader.load(sample_parquet)
        assert channels.has_speed is True

    def test_has_rpm(self, sample_parquet):
        loader = SessionDataLoader()
        channels = loader.load(sample_parquet)
        assert channels.has_rpm is True

    def test_no_gps(self, minimal_parquet):
        loader = SessionDataLoader()
        channels = loader.load(minimal_parquet)
        assert channels.has_gps is False

    def test_no_rpm(self, minimal_parquet):
        loader = SessionDataLoader()
        channels = loader.load(minimal_parquet)
        assert channels.has_rpm is False

    def test_available_channels(self, sample_parquet):
        loader = SessionDataLoader()
        channels = loader.load(sample_parquet)
        avail = channels.available_channels
        assert "latitude" in avail
        assert "longitude" in avail
        assert "speed" in avail
        assert "rpm" in avail


class TestLoadOrRaise:
    """Tests for SessionDataLoader.load_or_raise()"""

    def test_all_required_present(self, sample_parquet):
        loader = SessionDataLoader()
        channels = loader.load_or_raise(sample_parquet, required=["speed", "rpm"])
        assert channels.speed_mph is not None
        assert channels.rpm is not None

    def test_missing_required_raises(self, minimal_parquet):
        loader = SessionDataLoader()
        with pytest.raises(ValueError, match="rpm"):
            loader.load_or_raise(minimal_parquet, required=["speed", "rpm"])

    def test_missing_gps_raises(self, minimal_parquet):
        loader = SessionDataLoader()
        with pytest.raises(ValueError, match="latitude"):
            loader.load_or_raise(minimal_parquet, required=["latitude", "longitude"])

    def test_error_lists_all_missing(self, empty_parquet):
        loader = SessionDataLoader()
        with pytest.raises(ValueError, match="speed.*rpm|rpm.*speed"):
            loader.load_or_raise(empty_parquet, required=["speed", "rpm"])


class TestToSessionDataDict:
    """Tests for legacy session_data dict conversion"""

    def test_dict_has_required_keys(self, sample_parquet):
        loader = SessionDataLoader()
        channels = loader.load(sample_parquet)
        d = loader.to_session_data_dict(channels)
        assert "time" in d
        assert "latitude" in d
        assert "longitude" in d
        assert "rpm" in d
        assert "speed_mph" in d

    def test_dict_time_matches(self, sample_parquet):
        loader = SessionDataLoader()
        channels = loader.load(sample_parquet)
        d = loader.to_session_data_dict(channels)
        np.testing.assert_array_equal(d["time"], channels.time)

    def test_dict_missing_gps_returns_empty(self, minimal_parquet):
        loader = SessionDataLoader()
        channels = loader.load(minimal_parquet)
        d = loader.to_session_data_dict(channels)
        assert len(d["latitude"]) == 0
        assert len(d["longitude"]) == 0

    def test_dict_missing_rpm_returns_zeros(self, minimal_parquet):
        loader = SessionDataLoader()
        channels = loader.load(minimal_parquet)
        d = loader.to_session_data_dict(channels)
        assert len(d["rpm"]) == channels.sample_count
        assert np.all(d["rpm"] == 0)


class TestEdgeCases:
    """Edge case tests"""

    def test_dataframe_preserved(self, sample_parquet):
        """SessionChannels.df should be the full DataFrame"""
        loader = SessionDataLoader()
        channels = loader.load(sample_parquet)
        assert isinstance(channels.df, pd.DataFrame)
        assert len(channels.df) == 500

    def test_nonexistent_file_raises(self, tmp_path):
        loader = SessionDataLoader()
        with pytest.raises(Exception):
            loader.load(str(tmp_path / "nonexistent.parquet"))

    def test_column_map_complete(self, sample_parquet):
        """Column map should have entries for all discovered channels"""
        loader = SessionDataLoader()
        channels = loader.load(sample_parquet)
        # Should have at least: latitude, longitude, speed, rpm, lat_acc, lon_acc, throttle
        assert len(channels.column_map) >= 7
