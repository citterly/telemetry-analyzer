"""
Tests for arch-004: Tiered storage foundation.

Tests native rate tracking, anti-alias decimation, tiered export,
and SessionDataLoader tier awareness.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os
from pathlib import Path

from src.session.session_builder import (
    _build_dataframe,
    _resample_channel,
    _compute_summary_windows,
    export_session_tiered,
)


class TestNativeRateTracking:
    """Verify native rates are stored in DataFrame attrs."""

    def test_native_rates_present_in_attrs(self):
        """df.attrs['native_rates'] should be populated."""
        raw = {
            "RPM": {"time": np.linspace(0, 10, 501), "values": np.random.randn(501),
                     "unit": "rpm", "unit_source": "dll"},
            "GPS": {"time": np.linspace(0, 10, 101), "values": np.random.randn(101),
                     "unit": "deg", "unit_source": "dll"},
        }
        df = _build_dataframe(raw, base_rate_hz=10)
        assert "native_rates" in df.attrs
        assert isinstance(df.attrs["native_rates"], dict)

    def test_native_rates_values_correct(self):
        """Native rates should match input channel rates."""
        raw = {
            "fast": {"time": np.linspace(0, 10, 5001), "values": np.zeros(5001),
                      "unit": "v", "unit_source": "dll"},
            "slow": {"time": np.linspace(0, 10, 101), "values": np.zeros(101),
                      "unit": "deg", "unit_source": "dll"},
        }
        df = _build_dataframe(raw, base_rate_hz=10)
        rates = df.attrs["native_rates"]
        assert abs(rates["fast"] - 500.0) < 1.0
        assert abs(rates["slow"] - 10.0) < 0.5

    def test_base_rate_hz_still_stored(self):
        """attrs['base_rate_hz'] should still be present."""
        raw = {
            "chan": {"time": np.linspace(0, 5, 51), "values": np.zeros(51),
                     "unit": "v", "unit_source": "dll"},
        }
        df = _build_dataframe(raw, base_rate_hz=10)
        assert df.attrs["base_rate_hz"] == 10


class TestAntiAliasDecimation:
    """Verify anti-alias filter is used for high-frequency downsampling."""

    def test_near_rate_uses_interp(self):
        """Channels at or below 1.5x target rate use linear interpolation."""
        t_src = np.linspace(0, 10, 151)  # ~15 Hz
        vals = np.sin(2 * np.pi * t_src)
        t_dst = np.linspace(0, 10, 101)  # 10 Hz

        result = _resample_channel(t_src, vals, t_dst, 15.0, 10.0)
        assert len(result) == 101

    def test_high_freq_uses_decimation(self):
        """Channels well above target rate use anti-alias decimation."""
        t_src = np.linspace(0, 10, 5001)  # 500 Hz
        # Create signal with high-freq component that should be filtered
        vals = np.sin(2 * np.pi * 2 * t_src) + np.sin(2 * np.pi * 100 * t_src)
        t_dst = np.linspace(0, 10, 101)  # 10 Hz

        result = _resample_channel(t_src, vals, t_dst, 500.0, 10.0)
        assert len(result) == 101

        # High-freq component (100 Hz) should be attenuated by anti-alias filter
        # The 2 Hz component should be preserved
        # Check that the result is mostly the 2 Hz signal
        expected_2hz = np.sin(2 * np.pi * 2 * t_dst)
        correlation = np.corrcoef(result, expected_2hz)[0, 1]
        assert correlation > 0.8, f"2 Hz signal should be preserved, correlation={correlation}"

    def test_decimation_removes_aliasing(self):
        """Anti-alias filter should prevent aliasing artifacts."""
        t_src = np.linspace(0, 10, 5001)  # 500 Hz
        # Pure high-freq signal above Nyquist of 10 Hz target
        vals = np.sin(2 * np.pi * 200 * t_src)
        t_dst = np.linspace(0, 10, 101)  # 10 Hz

        result = _resample_channel(t_src, vals, t_dst, 500.0, 10.0)

        # With anti-aliasing, 200 Hz signal should be heavily attenuated
        # Without filter, it would alias to a low-freq signal with significant amplitude
        rms = np.sqrt(np.mean(result ** 2))
        assert rms < 0.3, f"200 Hz should be attenuated, RMS={rms}"

    def test_zero_rate_falls_back_to_interp(self):
        """Zero source rate should fall back to interpolation."""
        t_src = np.linspace(0, 5, 51)
        vals = np.ones(51)
        t_dst = np.linspace(0, 5, 26)

        result = _resample_channel(t_src, vals, t_dst, 0.0, 5.0)
        np.testing.assert_allclose(result, 1.0)

    def test_short_signal_falls_back(self):
        """Very short signals (< 4 samples) should fall back to interp."""
        t_src = np.array([0.0, 1.0, 2.0])
        vals = np.array([1.0, 2.0, 3.0])
        t_dst = np.linspace(0, 2, 5)

        result = _resample_channel(t_src, vals, t_dst, 500.0, 10.0)
        assert len(result) == 5


class TestSummaryWindowing:
    """Tests for _compute_summary_windows()."""

    def test_window_count(self):
        """Number of windows should be len(values) // window_size."""
        values = np.arange(1000.0)  # 1000 samples
        result = _compute_summary_windows(values, native_rate=500, summary_rate=50)
        # window_size = 500/50 = 10, so 1000/10 = 100 windows
        assert len(result["mean"]) == 100

    def test_mean_correct(self):
        """Window mean should be correct for known input."""
        # 20 samples, window size 10 → 2 windows
        values = np.array([1]*10 + [3]*10, dtype=float)
        result = _compute_summary_windows(values, native_rate=100, summary_rate=10)
        np.testing.assert_allclose(result["mean"], [1.0, 3.0])

    def test_min_max_correct(self):
        """Window min/max should capture extremes."""
        values = np.array([1, 5, 2, 8, 3, 7, 4, 6, 1, 9], dtype=float)
        result = _compute_summary_windows(values, native_rate=100, summary_rate=10)
        assert result["min"][0] == 1.0
        assert result["max"][0] == 9.0

    def test_velocity_nonzero_for_changing_signal(self):
        """Velocity (RMS of derivative) should be nonzero for changing signal."""
        t = np.linspace(0, 1, 500)
        values = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
        result = _compute_summary_windows(values, native_rate=500, summary_rate=50)
        # velocity should be positive everywhere
        assert np.all(result["velocity"] > 0)

    def test_velocity_zero_for_constant(self):
        """Velocity should be zero for constant signal."""
        values = np.ones(100)
        result = _compute_summary_windows(values, native_rate=100, summary_rate=10)
        np.testing.assert_allclose(result["velocity"], 0.0)

    def test_empty_result_for_short_signal(self):
        """Short signal that doesn't fill a window should return empty arrays."""
        values = np.array([1.0, 2.0])
        result = _compute_summary_windows(values, native_rate=500, summary_rate=50)
        # window_size = 10, only 2 samples → 0 windows
        assert len(result["mean"]) == 0

    def test_all_stats_same_length(self):
        """All statistic arrays should have the same length."""
        values = np.random.randn(500)
        result = _compute_summary_windows(values, native_rate=500, summary_rate=50)
        lengths = [len(v) for v in result.values()]
        assert len(set(lengths)) == 1  # All same


class TestTieredExport:
    """Tests for export_session_tiered()."""

    @pytest.fixture
    def raw_channels(self):
        """Create multi-rate raw channels for testing."""
        duration = 10.0
        return {
            "shock_pot_1": {
                "time": np.linspace(0, duration, 5001),  # 500 Hz
                "values": np.sin(np.linspace(0, 20*np.pi, 5001)),
                "unit": "mm",
                "unit_source": "dll",
            },
            "RPM": {
                "time": np.linspace(0, duration, 501),  # 50 Hz
                "values": 4000 + 2000 * np.sin(np.linspace(0, 5*np.pi, 501)),
                "unit": "rpm",
                "unit_source": "dll",
            },
            "GPS Latitude": {
                "time": np.linspace(0, duration, 101),  # 10 Hz
                "values": 43.797 + 0.001 * np.sin(np.linspace(0, 2*np.pi, 101)),
                "unit": "deg",
                "unit_source": "dll",
            },
        }

    def test_default_exports_merged_only(self, raw_channels, tmp_path, monkeypatch):
        """Default tiers=None should export only merged."""
        monkeypatch.setattr("src.session.session_builder.EXPORTS_PATH", tmp_path)
        (tmp_path / "processed").mkdir()

        paths = export_session_tiered(raw_channels, "test_session")
        assert "merged" in paths
        assert "summary" not in paths
        assert "raw" not in paths
        assert paths["merged"].exists()

    def test_merged_tier_filename(self, raw_channels, tmp_path, monkeypatch):
        """Merged tier should use standard naming."""
        monkeypatch.setattr("src.session.session_builder.EXPORTS_PATH", tmp_path)
        paths = export_session_tiered(raw_channels, "test_session", tiers=["merged"])
        assert paths["merged"].name == "test_session.parquet"

    def test_summary_tier_filename(self, raw_channels, tmp_path, monkeypatch):
        """Summary tier should use _summary_50hz naming."""
        monkeypatch.setattr("src.session.session_builder.EXPORTS_PATH", tmp_path)
        paths = export_session_tiered(raw_channels, "test_session", tiers=["summary"])
        assert "summary" in paths
        assert paths["summary"].name == "test_session_summary_50hz.parquet"

    def test_raw_tier_filename(self, raw_channels, tmp_path, monkeypatch):
        """Raw tier should use _raw_500hz naming."""
        monkeypatch.setattr("src.session.session_builder.EXPORTS_PATH", tmp_path)
        paths = export_session_tiered(raw_channels, "test_session", tiers=["raw"])
        assert "raw" in paths
        assert paths["raw"].name == "test_session_raw_500hz.parquet"

    def test_all_tiers(self, raw_channels, tmp_path, monkeypatch):
        """Exporting all tiers should produce 3 files."""
        monkeypatch.setattr("src.session.session_builder.EXPORTS_PATH", tmp_path)
        paths = export_session_tiered(
            raw_channels, "test_session", tiers=["raw", "summary", "merged"]
        )
        assert len(paths) == 3
        for tier in ["raw", "summary", "merged"]:
            assert tier in paths
            assert paths[tier].exists()

    def test_merged_tier_is_10hz(self, raw_channels, tmp_path, monkeypatch):
        """Merged tier should have ~10 Hz sample rate."""
        monkeypatch.setattr("src.session.session_builder.EXPORTS_PATH", tmp_path)
        paths = export_session_tiered(raw_channels, "test_session", tiers=["merged"])
        df = pd.read_parquet(paths["merged"])
        # ~10 seconds at 10 Hz = ~101 rows
        assert 90 < len(df) < 120

    def test_raw_tier_preserves_high_freq(self, raw_channels, tmp_path, monkeypatch):
        """Raw tier should have many more samples than merged."""
        monkeypatch.setattr("src.session.session_builder.EXPORTS_PATH", tmp_path)
        paths = export_session_tiered(
            raw_channels, "test_session", tiers=["raw", "merged"]
        )
        df_merged = pd.read_parquet(paths["merged"])
        df_raw = pd.read_parquet(paths["raw"])
        # Raw should have ~500 Hz = ~5001 rows vs ~101 merged
        assert len(df_raw) > len(df_merged) * 10

    def test_summary_has_windowed_columns(self, raw_channels, tmp_path, monkeypatch):
        """Summary tier should have mean/min/max/velocity columns for high-freq channels."""
        monkeypatch.setattr("src.session.session_builder.EXPORTS_PATH", tmp_path)
        paths = export_session_tiered(raw_channels, "test_session", tiers=["summary"])
        df = pd.read_parquet(paths["summary"])
        # shock_pot_1 at 500 Hz should have windowed columns
        assert "shock_pot_1_mean" in df.columns
        assert "shock_pot_1_min" in df.columns
        assert "shock_pot_1_max" in df.columns
        assert "shock_pot_1_velocity" in df.columns
        # RPM at 50 Hz should be direct (not windowed)
        assert "RPM" in df.columns

    def test_no_raw_channels_skips_raw_tier(self, tmp_path, monkeypatch):
        """If no channels qualify for raw tier, it should be skipped."""
        monkeypatch.setattr("src.session.session_builder.EXPORTS_PATH", tmp_path)
        raw = {
            "GPS": {"time": np.linspace(0, 5, 51), "values": np.zeros(51),
                     "unit": "deg", "unit_source": "dll"},
        }
        paths = export_session_tiered(raw, "test_session", tiers=["raw"])
        assert "raw" not in paths


class TestSessionDataLoaderTierAware:
    """Tests for SessionDataLoader tier routing."""

    @pytest.fixture
    def tier_files(self, tmp_path):
        """Create merged + summary + raw parquet files."""
        n = 100
        time = np.linspace(0, 10, n)
        df_merged = pd.DataFrame({"speed": time * 10}, index=time)
        df_summary = pd.DataFrame({"speed": time * 20}, index=time)
        df_raw = pd.DataFrame({"speed": time * 30}, index=time)

        merged_path = tmp_path / "session.parquet"
        summary_path = tmp_path / "session_summary_50hz.parquet"
        raw_path = tmp_path / "session_raw_500hz.parquet"

        df_merged.to_parquet(merged_path)
        df_summary.to_parquet(summary_path)
        df_raw.to_parquet(raw_path)

        return str(merged_path)

    def test_default_loads_merged(self, tier_files):
        """Default tier should load merged file."""
        from src.services.session_data_loader import SessionDataLoader
        loader = SessionDataLoader()
        channels = loader.load(tier_files, tier="merged")
        assert channels.sample_count == 100

    def test_summary_tier_loads_sidecar(self, tier_files):
        """tier='summary' should load the summary sidecar."""
        from src.services.session_data_loader import SessionDataLoader
        loader = SessionDataLoader()
        channels = loader.load(tier_files, tier="summary")
        # Summary file has speed = time * 20
        assert channels.session_id == "session_summary_50hz"

    def test_raw_tier_loads_sidecar(self, tier_files):
        """tier='raw' should load the raw sidecar."""
        from src.services.session_data_loader import SessionDataLoader
        loader = SessionDataLoader()
        channels = loader.load(tier_files, tier="raw")
        assert channels.session_id == "session_raw_500hz"

    def test_missing_sidecar_falls_back(self, tmp_path):
        """If sidecar doesn't exist, should fall back to merged."""
        from src.services.session_data_loader import SessionDataLoader
        n = 50
        time = np.linspace(0, 5, n)
        df = pd.DataFrame({"speed": np.ones(n)}, index=time)
        path = tmp_path / "session.parquet"
        df.to_parquet(path)

        loader = SessionDataLoader()
        channels = loader.load(str(path), tier="raw")
        # Should fall back to merged since raw sidecar doesn't exist
        assert channels.sample_count == 50
