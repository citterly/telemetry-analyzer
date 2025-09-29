"""
Session Builder
---------------
Transforms raw XRK extracts into canonical session datasets:
- Iterates all channels
- Normalizes time base
- Attaches units
- Exports Parquet
"""

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from src.extract.data_loader import XRKDataLoader
from src.config.config import SAMPLE_FILES_PATH
from src.config.vehicle_config import DEFAULT_SESSION


# -------------------------------------------------------------------
# Core API
# -------------------------------------------------------------------

def extract_full_session(filename: str = DEFAULT_SESSION,
                         resample_hz: int = 10) -> pd.DataFrame:
    """
    Load an XRK file and return a fully structured DataFrame.

    Columns:
        - All channels (rpm, gps_lat, gps_lon, etc.)
        - Units attached via DataFrame.attrs['units']
        - Uniform time base (resampled to resample_hz)
        - Timestamp index (absolute or session-relative)

    Args:
        filename: XRK file to process (relative or absolute path).
        resample_hz: Target frequency for resampling.

    Returns:
        pd.DataFrame: Canonical session dataset.
    """
    # 1. Extract raw data (channel dicts) from DLL
    loader = XRKDataLoader()
    if not loader.open_file(filename):
        raise FileNotFoundError(f"Could not open {filename}")

    raw_channels = _extract_all_channels(loader)
    loader.close_file()

    # 2. Normalize to a DataFrame
    df = _build_dataframe(raw_channels, resample_hz=resample_hz)

    # 3. Attach units from units.xml
    df.attrs["units"] = _load_units_map(raw_channels)

    return df


def export_session(df: pd.DataFrame, out_path: Path) -> Path:
    """
    Save canonical session DataFrame to Parquet.

    Args:
        df: Session DataFrame from extract_full_session.
        out_path: Where to save (should be under data/exports/processed/).

    Returns:
        Path to the saved Parquet file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", index=True)
    return out_path


# -------------------------------------------------------------------
# Internal Helpers
# -------------------------------------------------------------------


def _extract_all_channels(loader: XRKDataLoader) -> Dict[str, Dict[str, Any]]:
    """
    Enumerate and extract ALL channels (regular + GPS) from the open XRK file.

    Returns:
        Dict[channel_name, channel_data]
        channel_data = {"time": np.array, "values": np.array, "sample_count": int}
    """
    channels: Dict[str, Dict[str, Any]] = {}

    # ---- Regular channels ----
    try:
        chan_count = loader.dll.get_channels_count(loader.file_index)
        for i in range(chan_count):
            name_ptr = loader.dll.get_channel_name(loader.file_index, i)
            if not name_ptr:
                continue
            name = name_ptr.decode("utf-8")
            chan = loader._extract_channel_data(i, is_gps=False)
            if chan:
                channels[name] = chan
    except Exception as e:
        print(f"⚠️ Error enumerating regular channels: {e}")

    # ---- GPS channels ----
    try:
        gps_count = loader.dll.get_GPS_channels_count(loader.file_index)
        for i in range(gps_count):
            name_ptr = loader.dll.get_GPS_channel_name(loader.file_index, i)
            if not name_ptr:
                continue
            name = name_ptr.decode("utf-8")
            chan = loader._extract_channel_data(i, is_gps=True)
            if chan:
                channels[name] = chan
    except Exception as e:
        print(f"⚠️ Error enumerating GPS channels: {e}")

    return channels


def _build_dataframe(raw_channels: Dict[str, Dict[str, Any]],
                     resample_hz: int) -> pd.DataFrame:
    """
    Build a DataFrame from raw channel data.
    Currently supports RPM + GPS channels as a proof of concept.
    """

    # Pick key channels (extend later)
    key_channels = ["RPM dup 3", "GPS Speed", "GPS Latitude", "GPS Longitude"]
    data = {}
    min_time, max_time = None, None

    # Collect available channels
    for name in key_channels:
        chan = raw_channels.get(name)
        if chan:
            data[name] = pd.Series(chan["values"], index=chan["time"])
            if min_time is None or chan["time"][0] < min_time:
                min_time = chan["time"][0]
            if max_time is None or chan["time"][-1] > max_time:
                max_time = chan["time"][-1]

    if not data:
        return pd.DataFrame()

    # Build common time index
    dt = 1.0 / resample_hz
    time_index = np.arange(min_time, max_time, dt)

    # Reindex and interpolate
    df = pd.DataFrame(index=time_index)
    for name, series in data.items():
        df[name] = series.reindex(time_index, method=None).interpolate()

    # Store units (placeholder — expand with units.xml later)
    df.attrs["units"] = {
        "RPM dup 3": "rpm",
        "GPS Speed": "m/s",
        "GPS Latitude": "deg",
        "GPS Longitude": "deg"
    }

    # Rename columns to cleaner names
    df = df.rename(columns={
        "RPM dup 3": "rpm",
        "GPS Speed": "gps_speed",
        "GPS Latitude": "gps_lat",
        "GPS Longitude": "gps_lon"
    })

    return df



def _load_units_map(raw_channels: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """
    Load units for each channel from units.xml.

    Args:
        raw_channels: Dict of extracted channels.

    Returns:
        Dict[channel_name, unit_str]
    """
    units = {}
    # TODO: parse units.xml, match to channel names
    return units


# -------------------------------------------------------------------
# Smoke Test
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("🔍 Smoke test: Session Builder")
    df = extract_full_session()
    print("✅ Extracted DataFrame:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Units:", df.attrs.get("units", {}))
    out_file = export_session(df, Path("data/exports/processed/test_session.parquet"))
    print(f"📁 Exported to {out_file}")
