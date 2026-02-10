"""
Session Builder — WP4 Canonical Metadata Extension (MVP)
--------------------------------------------------------
Extracts ALL XRK channels into a canonical Parquet dataset:
- Iterates all channels (regular + GPS)
- Normalizes to common time base
- Resolves units via DLL or units_helper
- Exports Parquet
- Updates metadata JSON with Parquet reference
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import unicodedata

from ctypes import c_char_p, cast

from src.extract.data_loader import XRKDataLoader
from src.config.config import EXPORTS_PATH
from src.io.file_manager import FileManager
from src.utils import units_helper

# -------------------------------------------------------------------
# Core API
# -------------------------------------------------------------------

def extract_full_session(filename: str,
                         resample_hz: int = 10) -> pd.DataFrame:
    """
    Load an XRK file and return canonical DataFrame.
    """
    loader = XRKDataLoader()
    if not loader.open_file(filename):
        raise FileNotFoundError(f"Could not open {filename}")

    raw_channels = _extract_all_channels(loader)
    loader.close_file()

    # Build canonical DataFrame at uniform rate
    df = _build_dataframe(raw_channels, base_rate_hz=resample_hz)

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
        units_map={k: _normalize_unit_text(v) for k, v in df.attrs.get("units", {}).items()}
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
        print(f"🔍 Regular channels reported: {chan_count}")

        for i in range(chan_count):
            name_ptr = loader.dll.get_channel_name(loader.file_index, i)
            unit_ptr = loader.dll.get_channel_units(loader.file_index, i)

            if not name_ptr:
                print(f"⚠️ Channel {i} has no name pointer")
                continue

            # --- Decode channel name ---
            try:
                if isinstance(name_ptr, (bytes, bytearray)):
                    name = name_ptr.decode("utf-8", errors="replace")
                elif isinstance(name_ptr, str):
                    name = name_ptr
                else:
                    raise TypeError(f"Unexpected name_ptr type: {type(name_ptr)}")
            except Exception as e:
                print(f"⚠️ Failed to decode channel {i} name: {e}")
                name = f"chan_{i}"

            # --- Decode channel unit ---
            unit = None
            source = "dll"
            try:
                if unit_ptr:
                    if isinstance(unit_ptr, (bytes, bytearray)):
                        unit = unit_ptr.decode("utf-8", errors="replace")
                    elif isinstance(unit_ptr, str):
                        unit = unit_ptr
                    else:
                        # DLL sometimes returns bogus ints → ignore
                        unit = None
            except Exception:
                unit = None


            # --- Normalize or fallback ---
            if unit:
                unit = _normalize_unit_text(unit)
            if not unit:
                unit, source = units_helper.guess_unit(name)

            # --- Extract data ---
            chan = loader._extract_channel_data(i, is_gps=False)
            if chan:
                chan["unit"] = unit
                chan["unit_source"] = source
                channels[name] = chan

    except Exception as e:
        print(f"❌ Error enumerating regular channels: {e}")

    # ---- GPS channels ----
    try:
        gps_count = loader.dll.get_GPS_channels_count(loader.file_index)
        print(f"🔍 GPS channels reported: {gps_count}")

        for i in range(gps_count):
            name_ptr = loader.dll.get_GPS_channel_name(loader.file_index, i)
            if not name_ptr:
                continue

            try:
                if isinstance(name_ptr, (bytes, bytearray)):
                    name = name_ptr.decode("utf-8", errors="replace")
                elif isinstance(name_ptr, str):
                    name = name_ptr
                else:
                    raise TypeError(f"Unexpected GPS name_ptr type: {type(name_ptr)}")
            except Exception as e:
                print(f"⚠️ Failed to decode GPS channel {i} name: {e}")
                name = f"gps_chan_{i}"

            chan = loader._extract_channel_data(i, is_gps=True)
            if chan:
                # GPS often lacks explicit units → heuristic
                unit, source = units_helper.guess_unit(name)
                unit = _normalize_unit_text(unit)
                chan["unit"] = unit
                chan["unit_source"] = source
                channels[name] = chan

    except Exception as e:
        print(f"⚠️ Error enumerating GPS channels: {e}")

    return channels


def _build_dataframe(raw_channels: Dict[str, Dict[str, Any]],
                     base_rate_hz: int) -> pd.DataFrame:
    """
    Build a canonical DataFrame from raw channel data.

    MVP Implementation (WP4):
    - One canonical index at base_rate_hz.
    - Linear interpolation for upsampling.
    - Naive reindex + fill for downsampling (TODO: anti-alias + decimate).
    - Units resolved via DLL or units_helper.
    - Future: add native_rate_hz tracking + hi-res sidecars.

    Returns:
        pd.DataFrame with:
          - Index: session-relative seconds (uniform)
          - Columns: all channels normalized to base_rate_hz
          - attrs["units"]: {channel: unit}
          - attrs["unit_sources"]: {channel: {unit, source}}
    """
    if not raw_channels:
        return pd.DataFrame()

    # Global time bounds
    t_min = min(c["time"][0] for c in raw_channels.values() if len(c["time"]) > 0)
    t_max = max(c["time"][-1] for c in raw_channels.values() if len(c["time"]) > 0)

    # Compute expected number of samples
    duration = t_max - t_min
    n_rows = int(round(duration * base_rate_hz)) + 1

    # Perfectly uniform index with linspace (avoids float drift from arange)
    time_index = np.linspace(t_min, t_max, n_rows)

    df = pd.DataFrame(index=time_index)
    units_map = {}

    for name, chan in raw_channels.items():
        # Use numpy interp for direct interpolation onto canonical time index
        # This avoids exact-match issues with pandas reindex
        df[name] = np.interp(time_index, chan["time"], chan["values"])

        # Unit resolution
        if chan.get("unit"):
            unit = _normalize_unit_text(chan["unit"])
            source = chan.get("unit_source", "dll")
        else:
            unit, source = units_helper.guess_unit(name)
            unit = _normalize_unit_text(unit)

        units_map[name] = {"unit": unit, "source": source}

    df.attrs["units"] = {n: u["unit"] for n, u in units_map.items()}
    df.attrs["unit_sources"] = units_map
    df.attrs["base_rate_hz"] = base_rate_hz

    return df

def _safe_decode(ptr) -> str:
    """
    Safely decode DLL-returned pointer values.
    Handles bytes, c_char_p, int, or None without segfaulting.
    """
    if not ptr:
        return ""
    try:
        if isinstance(ptr, (bytes, bytearray)):
            return ptr.decode("utf-8")
        if isinstance(ptr, str):
            return ptr  # already decoded
        if isinstance(ptr, int):
            # DLL returned an integer (likely because restype not set)
            return f"chan_{ptr}"
        # Try generic cast/decode if possible
        return str(ptr)
    except Exception as e:
        raise ValueError(f"Cannot decode DLL pointer {type(ptr)!r}: {e}")

def _normalize_unit_text(text: str) -> str:
    """
    Normalize unit strings to clean UTF-8 for storage.
    Fixes Windows-1252/UTF-8 artifacts (e.g., 'Â°F' → '°F', 'm/sÂ2' → 'm/s²').
    """
    if not text:
        return ""
    cleaned = unicodedata.normalize("NFKC", str(text))

    # Common Win1252/UTF-8 glitches
    replacements = {
        "Â°": "°",    # degrees
        "Â²": "²",    # squared (common mis-decoding)
        "Â2": "²",    # squared (alt mis-decoding)
        "Â³": "³",    # cubed
        "m/sÂ2": "m/s²",  # full sequence safety catch
        "m/sÂ²": "m/s²",  # just in case
    }
    for bad, good in replacements.items():
        cleaned = cleaned.replace(bad, good)

    return cleaned.strip()

# -------------------------------------------------------------------
# Smoke Test (manual)
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("🔍 Smoke test: Session Builder (WP4 MVP)")
    test_file = "data/uploads/20250712_104619_Road America_a_0394.xrk"  # example session
    df = extract_full_session(test_file)
    print("✅ Extracted DataFrame:", df.shape)
    print("Columns:", list(df.columns)[:10])
    print("Units:", list(df.attrs.get("units", {}).items())[:10])

    out_file = export_session(df, Path(test_file).stem)
    print(f"📁 Exported canonical Parquet: {out_file}")

    df_reloaded = pd.read_parquet(out_file)
    assert df_reloaded.shape == df.shape
    print("✅ Parquet reload verified")
