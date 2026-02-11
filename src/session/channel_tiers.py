"""
Channel tier classification for tiered storage architecture.

Classifies channels into tiers based on their native sample rate:
  - raw (>= 200 Hz): High-frequency data (shock pots, analog sensors)
  - summary (>= 20 Hz): Medium-frequency data (CAN bus channels)
  - merged (< 20 Hz): Low-frequency data (GPS channels)

Each channel belongs to its own tier AND all lower tiers.
"""

from dataclasses import dataclass, field
from typing import Dict, List

# Rate thresholds for tier classification
RAW_THRESHOLD_HZ = 200
SUMMARY_THRESHOLD_HZ = 20


@dataclass
class ChannelTierConfig:
    """Which tier(s) a channel belongs to."""
    name: str
    native_rate_hz: float
    tiers: List[str] = field(default_factory=list)

    @property
    def is_raw(self) -> bool:
        return "raw" in self.tiers

    @property
    def is_summary(self) -> bool:
        return "summary" in self.tiers

    @property
    def is_merged(self) -> bool:
        return "merged" in self.tiers


def classify_channels(
    native_rates: Dict[str, float],
) -> Dict[str, ChannelTierConfig]:
    """
    Auto-classify channels by native sample rate.

    Args:
        native_rates: Dict mapping channel name to native rate in Hz.

    Returns:
        Dict mapping channel name to ChannelTierConfig.
    """
    result = {}
    for name, rate in native_rates.items():
        if rate >= RAW_THRESHOLD_HZ:
            tiers = ["raw", "summary", "merged"]
        elif rate >= SUMMARY_THRESHOLD_HZ:
            tiers = ["summary", "merged"]
        else:
            tiers = ["merged"]
        result[name] = ChannelTierConfig(name=name, native_rate_hz=rate, tiers=tiers)
    return result


def channels_for_tier(
    classifications: Dict[str, ChannelTierConfig],
    tier: str,
) -> List[str]:
    """
    Get channel names that belong to a specific tier.

    Args:
        classifications: Output from classify_channels().
        tier: One of "raw", "summary", "merged".

    Returns:
        List of channel names belonging to that tier.
    """
    return [name for name, config in classifications.items() if tier in config.tiers]


def compute_native_rates(
    raw_channels: Dict[str, dict],
) -> Dict[str, float]:
    """
    Compute native sample rate for each channel from its time array.

    Args:
        raw_channels: Dict of {name: {"time": np.array, "values": np.array, ...}}

    Returns:
        Dict mapping channel name to rate in Hz.
    """
    rates = {}
    for name, chan in raw_channels.items():
        t = chan.get("time")
        if t is not None and len(t) > 1:
            duration = float(t[-1] - t[0])
            if duration > 0:
                rates[name] = round(len(t) / duration, 1)
            else:
                rates[name] = 0.0
        else:
            rates[name] = 0.0
    return rates
