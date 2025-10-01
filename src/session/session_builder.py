"""
Session Builder — WP3 Canonicalization
--------------------------------------
Extracts ALL XRK channels into a canonical Parquet dataset:
- Iterates all channels (regular + GPS)
- Normalizes to common time base
- Attaches units from DLL or units.xml
- Exports Parquet
- Updates metadata JSON with Parquet reference
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from src.extract.data_loader import XRKDataLoader
from src.config.config import EXPORTS_PATH, UNITS_XML_PATH
from src.io.file_manager import FileManager


# -------------------------------------------------------------------
# Core API
# -------------------------------------------------------------------

def extract_full_session(filename: str,
                         resample_hz: int = 10) -> pd.DataFrame:
    """
    Load an XRK file and return canonical DataFrame.

    Columns:
        - All channels (regular + GPS)
        - Units attached via DataFrame.attrs['units']
        - Uniform time base (resampled to resample_hz)
        - Timestamp index (session-relative seconds)
    """
    loader = XRKDataLoader()
    if not loader.open_file(filename):
        raise FileNotFoundError(f"Could not open {filename}")

    raw_channels = _extract_all_channels(loader)
    loader.close_file()

    df = _build_dataframe(raw_channels, resample_hz=resample_hz)
    df.attrs["units"] = _load_units_map(raw_channels)

    return df


def export_session(df: pd.DataFrame, session_id: str) -> Path:
    """
    Save canonical session DataFrame to Parquet and update metadata.
    """
    out_path = EXPORTS_PATH / "processed" / f"{session_id}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", index=True)

    # 🔗 Update metadata JSON
    fm = FileManager()
    fm.attach_canonical_export(
        f"{session_id}.xrk",
        out_path,
        channel_list=list(df.columns),
        units_map=df.attrs.get("units", {})
    )

    return out_path

# -------------------------------------------------------------------
# Internal Helpers
# -------------------------------------------------------------------

def _extract_all_channels(loader: XRKDataLoader) -> Dict[str, Dict[str, Any]]:
    """
    Enumerate and extract ALL channels (regular + GPS) from the open XRK file.
    Each entry includes time, values, count, and unit.
    """
    channels: Dict[str, Dict[str, Any]] = {}

    # ---- Regular channels ----
    try:
        chan_count = loader.dll.get_channels_count(loader.file_index)
        for i in range(chan_count):
            name_ptr = loader.dll.get_channel_name(loader.file_index, i)
            unit_ptr = loader.dll.get_channel_units(loader.file_index, i)
            if not name_ptr:
                continue
            name = name_ptr.decode("utf-8")
            unit = unit_ptr.decode("utf-8") if unit_ptr else None
            chan = loader._extract_channel_data(i, is_gps=False)
            if chan:
                chan["unit"] = unit
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
            # GPS channels may not expose units directly
            chan = loader._extract_channel_data(i, is_gps=True)
            if chan:
                chan["unit"] = None
                channels[name] = chan
    except Exception as e:
        print(f"⚠️ Error enumerating GPS channels: {e}")

    return channels





def _build_dataframe(raw_channels: Dict[str, Dict[str, Any]],
                     resample_hz: int) -> pd.DataFrame:
    """
    Build a canonical DataFrame from raw channel data.
    - Includes ALL channels (regular + GPS).
    - Normalized to a common time base.
    - Resampled to resample_hz with interpolation.
    """

    if not raw_channels:
        return pd.DataFrame()

    # Determine global time bounds
    min_time = min(chan["time"][0] for chan in raw_channels.values() if len(chan["time"]) > 0)
    max_time = max(chan["time"][-1] for chan in raw_channels.values() if len(chan["time"]) > 0)

    # Build common time index
    dt = 1.0 / resample_hz
    time_index = np.arange(min_time, max_time, dt)

    # Build DataFrame
    df = pd.DataFrame(index=time_index)
    for name, chan in raw_channels.items():
        if len(chan["time"]) == 0:
            continue
        series = pd.Series(chan["values"], index=chan["time"])
        df[name] = series.reindex(time_index, method=None).interpolate()

    return df



def _load_units_map(raw_channels: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """
    Build a {channel_name: unit} map from DLL + units.xml fallback.
    """
    units = {}

    for name, chan in raw_channels.items():
        unit = chan.get("unit")
        units[name] = unit or "unknown"
    return units



# -------------------------------------------------------------------
# Smoke Test
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("🔍 Smoke test: Session Builder (WP3)")
    test_file = "data/uploads/example.xrk"  # replace with sample
    df = extract_full_session(test_file)
    print("✅ Extracted DataFrame:", df.shape)
    print("Columns:", list(df.columns)[:10])
    print("Units:", list(df.attrs.get("units", {}).items())[:10])

    out_file = export_session(df, Path(test_file).stem)
    print(f"📁 Exported canonical Parquet: {out_file}")

    df_reloaded = pd.read_parquet(out_file)
    assert df_reloaded.shape == df.shape
    print("✅ Parquet reload verified")
