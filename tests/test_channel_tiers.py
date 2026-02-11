"""
Tests for src/session/channel_tiers.py — channel tier classification.
"""

import numpy as np
import pytest

from src.session.channel_tiers import (
    ChannelTierConfig,
    classify_channels,
    channels_for_tier,
    compute_native_rates,
    RAW_THRESHOLD_HZ,
    SUMMARY_THRESHOLD_HZ,
)


class TestChannelTierConfig:
    """Tests for the ChannelTierConfig dataclass."""

    def test_raw_channel(self):
        config = ChannelTierConfig(name="shock_pot_1", native_rate_hz=500.0,
                                   tiers=["raw", "summary", "merged"])
        assert config.is_raw
        assert config.is_summary
        assert config.is_merged

    def test_summary_channel(self):
        config = ChannelTierConfig(name="RPM", native_rate_hz=50.0,
                                   tiers=["summary", "merged"])
        assert not config.is_raw
        assert config.is_summary
        assert config.is_merged

    def test_merged_only_channel(self):
        config = ChannelTierConfig(name="GPS Latitude", native_rate_hz=10.0,
                                   tiers=["merged"])
        assert not config.is_raw
        assert not config.is_summary
        assert config.is_merged


class TestClassifyChannels:
    """Tests for classify_channels()."""

    def test_high_freq_classified_as_raw(self):
        rates = {"shock_pot_1": 500.0, "shock_pot_2": 1000.0}
        result = classify_channels(rates)
        assert result["shock_pot_1"].is_raw
        assert result["shock_pot_2"].is_raw
        assert "raw" in result["shock_pot_1"].tiers

    def test_medium_freq_classified_as_summary(self):
        rates = {"RPM": 50.0, "CAN_Temp": 25.0}
        result = classify_channels(rates)
        for name in rates:
            assert result[name].is_summary
            assert not result[name].is_raw

    def test_low_freq_classified_as_merged_only(self):
        rates = {"GPS Latitude": 10.0, "GPS Longitude": 10.0}
        result = classify_channels(rates)
        for name in rates:
            assert result[name].is_merged
            assert not result[name].is_summary
            assert not result[name].is_raw

    def test_threshold_boundaries(self):
        rates = {
            "at_raw": RAW_THRESHOLD_HZ,       # 200 Hz → raw
            "below_raw": RAW_THRESHOLD_HZ - 1,  # 199 Hz → summary
            "at_summary": SUMMARY_THRESHOLD_HZ,  # 20 Hz → summary
            "below_summary": SUMMARY_THRESHOLD_HZ - 1,  # 19 Hz → merged
        }
        result = classify_channels(rates)
        assert result["at_raw"].is_raw
        assert not result["below_raw"].is_raw
        assert result["below_raw"].is_summary
        assert result["at_summary"].is_summary
        assert not result["below_summary"].is_summary
        assert result["below_summary"].is_merged

    def test_mixed_channels(self):
        rates = {
            "shock_pot_1": 500.0,
            "RPM": 50.0,
            "GPS Latitude": 10.0,
        }
        result = classify_channels(rates)
        assert len(result) == 3
        assert result["shock_pot_1"].is_raw
        assert result["RPM"].is_summary
        assert not result["RPM"].is_raw
        assert result["GPS Latitude"].is_merged
        assert not result["GPS Latitude"].is_summary

    def test_zero_rate(self):
        rates = {"dead_channel": 0.0}
        result = classify_channels(rates)
        assert result["dead_channel"].is_merged
        assert not result["dead_channel"].is_summary


class TestChannelsForTier:
    """Tests for channels_for_tier()."""

    def test_raw_tier_returns_high_freq_only(self):
        rates = {"shock": 500.0, "RPM": 50.0, "GPS": 10.0}
        classifications = classify_channels(rates)
        raw = channels_for_tier(classifications, "raw")
        assert raw == ["shock"]

    def test_summary_tier_includes_medium_and_high(self):
        rates = {"shock": 500.0, "RPM": 50.0, "GPS": 10.0}
        classifications = classify_channels(rates)
        summary = channels_for_tier(classifications, "summary")
        assert set(summary) == {"shock", "RPM"}

    def test_merged_tier_includes_all(self):
        rates = {"shock": 500.0, "RPM": 50.0, "GPS": 10.0}
        classifications = classify_channels(rates)
        merged = channels_for_tier(classifications, "merged")
        assert set(merged) == {"shock", "RPM", "GPS"}


class TestComputeNativeRates:
    """Tests for compute_native_rates()."""

    def test_basic_rate_computation(self):
        raw = {
            "chan_10hz": {"time": np.linspace(0, 10, 101), "values": np.zeros(101)},
            "chan_50hz": {"time": np.linspace(0, 10, 501), "values": np.zeros(501)},
        }
        rates = compute_native_rates(raw)
        assert abs(rates["chan_10hz"] - 10.0) < 0.5
        assert abs(rates["chan_50hz"] - 50.0) < 0.5

    def test_high_freq_rate(self):
        raw = {
            "shock": {"time": np.linspace(0, 1, 501), "values": np.zeros(501)},
        }
        rates = compute_native_rates(raw)
        assert abs(rates["shock"] - 500.0) < 2.0

    def test_single_sample_returns_zero(self):
        raw = {"single": {"time": np.array([0.0]), "values": np.array([1.0])}}
        rates = compute_native_rates(raw)
        assert rates["single"] == 0.0

    def test_empty_channel(self):
        raw = {"empty": {"time": np.array([]), "values": np.array([])}}
        rates = compute_native_rates(raw)
        assert rates["empty"] == 0.0

    def test_zero_duration(self):
        raw = {"static": {"time": np.array([5.0, 5.0]), "values": np.array([1.0, 2.0])}}
        rates = compute_native_rates(raw)
        assert rates["static"] == 0.0
