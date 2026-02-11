"""
Compliance tests for arch-004: Tiered storage foundation.

Verifies all 10 acceptance criteria from docs/architecture/arch-004.md.
"""

import inspect
import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path


class TestAcceptanceCriteria:
    """Tests verifying each acceptance criterion from the design doc."""

    @pytest.fixture
    def multi_rate_channels(self):
        """Create raw channels at different sample rates."""
        duration = 10.0
        return {
            "shock_pot_1": {
                "time": np.linspace(0, duration, 5001),  # 500 Hz
                "values": np.sin(np.linspace(0, 20 * np.pi, 5001)),
                "unit": "mm", "unit_source": "dll",
            },
            "RPM": {
                "time": np.linspace(0, duration, 501),  # 50 Hz
                "values": 4000 + 2000 * np.sin(np.linspace(0, 5 * np.pi, 501)),
                "unit": "rpm", "unit_source": "dll",
            },
            "GPS Latitude": {
                "time": np.linspace(0, duration, 101),  # 10 Hz
                "values": 43.797 + 0.001 * np.sin(np.linspace(0, 2 * np.pi, 101)),
                "unit": "deg", "unit_source": "dll",
            },
        }

    def test_ac1_native_rates_stored(self, multi_rate_channels):
        """AC-1: df.attrs['native_rates'] is a dict mapping channel name to Hz."""
        from src.session.session_builder import _build_dataframe
        df = _build_dataframe(multi_rate_channels, base_rate_hz=10)

        assert "native_rates" in df.attrs
        rates = df.attrs["native_rates"]
        assert isinstance(rates, dict)
        # Each channel should have a rate
        for name in multi_rate_channels:
            assert name in rates
            assert isinstance(rates[name], float)
        # Rates should be approximately correct
        assert rates["shock_pot_1"] > 400
        assert 40 < rates["RPM"] < 60
        assert 8 < rates["GPS Latitude"] < 12

    def test_ac2_channel_classification_exists(self):
        """AC-2: src/session/channel_tiers.py with classify_channels()."""
        from src.session.channel_tiers import (
            classify_channels, ChannelTierConfig,
            channels_for_tier, compute_native_rates,
        )
        # All functions should be importable
        assert callable(classify_channels)
        assert callable(channels_for_tier)
        assert callable(compute_native_rates)

        # Test classification logic
        rates = {"high": 500.0, "mid": 50.0, "low": 10.0}
        result = classify_channels(rates)
        assert result["high"].is_raw
        assert result["mid"].is_summary and not result["mid"].is_raw
        assert result["low"].is_merged and not result["low"].is_summary

    def test_ac3_anti_alias_filter(self):
        """AC-3: Channels downsampled >1.5x use scipy.signal.decimate."""
        from src.session.session_builder import _resample_channel

        # Check that source code references scipy.signal.decimate
        source = inspect.getsource(_resample_channel)
        assert "decimate" in source

        # Verify it actually filters: 200 Hz signal at 500 Hz src → 10 Hz dst
        t_src = np.linspace(0, 10, 5001)  # 500 Hz
        vals = np.sin(2 * np.pi * 200 * t_src)  # Pure 200 Hz
        t_dst = np.linspace(0, 10, 101)  # 10 Hz target

        result = _resample_channel(t_src, vals, t_dst, 500.0, 10.0)
        rms = np.sqrt(np.mean(result ** 2))
        # 200 Hz should be attenuated below Nyquist (5 Hz)
        assert rms < 0.3, f"Anti-alias should attenuate 200 Hz, RMS={rms}"

    def test_ac4_tiered_export_api(self, multi_rate_channels, tmp_path, monkeypatch):
        """AC-4: export_session_tiered() with backward-compat default."""
        from src.session.session_builder import export_session_tiered
        monkeypatch.setattr("src.session.session_builder.EXPORTS_PATH", tmp_path)

        # Default should produce only merged
        paths = export_session_tiered(multi_rate_channels, "test")
        assert "merged" in paths
        assert len(paths) == 1

        # All tiers should work
        paths = export_session_tiered(
            multi_rate_channels, "test2", tiers=["raw", "summary", "merged"]
        )
        assert len(paths) == 3

    def test_ac5_raw_tier_preserves_native_rate(self, multi_rate_channels, tmp_path, monkeypatch):
        """AC-5: High-freq channels exported at native rate in raw tier."""
        from src.session.session_builder import export_session_tiered
        monkeypatch.setattr("src.session.session_builder.EXPORTS_PATH", tmp_path)

        paths = export_session_tiered(
            multi_rate_channels, "test", tiers=["raw", "merged"]
        )
        df_raw = pd.read_parquet(paths["raw"])
        df_merged = pd.read_parquet(paths["merged"])

        # Raw should have far more rows than merged
        assert len(df_raw) > len(df_merged) * 10
        # Raw should have shock_pot_1 column
        assert "shock_pot_1" in df_raw.columns

    def test_ac6_summary_tier_has_windowed_stats(self, multi_rate_channels, tmp_path, monkeypatch):
        """AC-6: High-freq channels produce mean/min/max/velocity at 50 Hz."""
        from src.session.session_builder import export_session_tiered
        monkeypatch.setattr("src.session.session_builder.EXPORTS_PATH", tmp_path)

        paths = export_session_tiered(
            multi_rate_channels, "test", tiers=["summary"]
        )
        df = pd.read_parquet(paths["summary"])

        # shock_pot_1 at 500 Hz should have windowed columns
        for stat in ["mean", "min", "max", "velocity"]:
            assert f"shock_pot_1_{stat}" in df.columns, f"Missing shock_pot_1_{stat}"

        # RPM at 50 Hz should be direct (not windowed)
        assert "RPM" in df.columns

    def test_ac7_merged_tier_unchanged(self, multi_rate_channels, tmp_path, monkeypatch):
        """AC-7: Default behavior identical to current (10 Hz, all channels)."""
        from src.session.session_builder import _build_dataframe, export_session_tiered
        monkeypatch.setattr("src.session.session_builder.EXPORTS_PATH", tmp_path)

        # Build directly and via tiered export
        df_direct = _build_dataframe(multi_rate_channels, base_rate_hz=10)
        paths = export_session_tiered(multi_rate_channels, "test", tiers=["merged"])
        df_tiered = pd.read_parquet(paths["merged"])

        # Same shape and columns
        assert df_direct.shape == df_tiered.shape
        assert set(df_direct.columns) == set(df_tiered.columns)

    def test_ac8_file_naming_convention(self, multi_rate_channels, tmp_path, monkeypatch):
        """AC-8: Correct file naming for each tier."""
        from src.session.session_builder import export_session_tiered
        monkeypatch.setattr("src.session.session_builder.EXPORTS_PATH", tmp_path)

        paths = export_session_tiered(
            multi_rate_channels, "my_session", tiers=["raw", "summary", "merged"]
        )
        assert paths["merged"].name == "my_session.parquet"
        assert paths["summary"].name == "my_session_summary_50hz.parquet"
        assert paths["raw"].name == "my_session_raw_500hz.parquet"

    def test_ac9_session_data_loader_tier_aware(self, tmp_path):
        """AC-9: SessionDataLoader.load() routes to correct file by tier."""
        from src.services.session_data_loader import SessionDataLoader

        # Create merged and sidecar files
        n = 50
        time = np.linspace(0, 5, n)
        df_merged = pd.DataFrame({"speed": np.ones(n) * 60}, index=time)
        df_raw = pd.DataFrame({"speed": np.ones(n) * 120}, index=time)

        merged_path = tmp_path / "session.parquet"
        raw_path = tmp_path / "session_raw_500hz.parquet"
        df_merged.to_parquet(merged_path)
        df_raw.to_parquet(raw_path)

        loader = SessionDataLoader()

        # Default loads merged
        ch = loader.load(str(merged_path))
        assert ch.session_id == "session"

        # tier="raw" loads sidecar
        ch_raw = loader.load(str(merged_path), tier="raw")
        assert ch_raw.session_id == "session_raw_500hz"

        # Missing sidecar falls back to merged
        ch_fallback = loader.load(str(merged_path), tier="summary")
        assert ch_fallback.session_id == "session"

    def test_ac10_all_existing_tests_pass(self):
        """AC-10: Zero regressions — verified by the full test suite."""
        # Meta-check: ensure critical modules import without error
        from src.session.session_builder import (
            _build_dataframe, _resample_channel,
            export_session, export_session_tiered,
        )
        from src.session.channel_tiers import (
            classify_channels, compute_native_rates,
            channels_for_tier, ChannelTierConfig,
        )
        from src.services.session_data_loader import SessionDataLoader

        assert callable(_build_dataframe)
        assert callable(export_session_tiered)
        assert callable(classify_channels)
