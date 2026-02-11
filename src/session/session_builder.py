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
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import unicodedata

from ctypes import c_char_p, cast

from src.extraction.data_loader import XRKDataLoader
from src.config.config import EXPORTS_PATH
from src.io.file_manager import FileManager
from src.utils import units_helper
from src.session.channel_tiers import compute_native_rates, classify_channels, channels_for_tier

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

    This exports the merged tier (default 10 Hz).
    For tiered export, use export_session_tiered().
    """
    out_path = EXPORTS_PATH / "processed" / f"{session_id}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", index=True)

    # Update metadata JSON
    fm = FileManager()
    fm.attach_canonical_export(
        f"{session_id}.xrk",
        out_path,
        channel_list=list(df.columns),
        units_map={k: _normalize_unit_text(v) for k, v in df.attrs.get("units", {}).items()}
    )

    return out_path


def export_session_tiered(
    raw_channels: Dict[str, Dict[str, Any]],
    session_id: str,
    tiers: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """
    Export session data at specified tiers.

    Args:
        raw_channels: Raw channel data with native timestamps
            {name: {"time": np.array, "values": np.array, "unit": str, ...}}
        session_id: Session identifier
        tiers: List of tiers to export. Default: ["merged"].
            Options: "raw", "summary", "merged"

    Returns:
        Dict mapping tier name to output path.
    """
    if tiers is None:
        tiers = ["merged"]

    native_rates = compute_native_rates(raw_channels)
    classifications = classify_channels(native_rates)
    output_paths = {}

    out_dir = EXPORTS_PATH / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    if "merged" in tiers:
        df_merged = _build_dataframe(raw_channels, base_rate_hz=10)
        path = out_dir / f"{session_id}.parquet"
        df_merged.to_parquet(path, engine="pyarrow", index=True)
        output_paths["merged"] = path

    if "summary" in tiers:
        df_summary = _build_summary_tier(raw_channels, native_rates, classifications)
        if df_summary is not None and len(df_summary) > 0:
            path = out_dir / f"{session_id}_summary_50hz.parquet"
            df_summary.to_parquet(path, engine="pyarrow", index=True)
            output_paths["summary"] = path

    if "raw" in tiers:
        raw_tier_channels = channels_for_tier(classifications, "raw")
        if raw_tier_channels:
            df_raw = _build_raw_tier(raw_channels, raw_tier_channels)
            if df_raw is not None and len(df_raw) > 0:
                path = out_dir / f"{session_id}_raw_500hz.parquet"
                df_raw.to_parquet(path, engine="pyarrow", index=True)
                output_paths["raw"] = path

    return output_paths


def _build_raw_tier(
    raw_channels: Dict[str, Dict[str, Any]],
    channel_names: List[str],
) -> Optional[pd.DataFrame]:
    """Build a raw-tier DataFrame preserving native sample rate."""
    if not channel_names:
        return None

    # Find max native rate among selected channels
    max_rate = 0.0
    for name in channel_names:
        chan = raw_channels.get(name)
        if chan and len(chan["time"]) > 1:
            duration = chan["time"][-1] - chan["time"][0]
            if duration > 0:
                rate = len(chan["time"]) / duration
                max_rate = max(max_rate, rate)

    if max_rate == 0:
        return None

    # Build uniform index at the max native rate
    all_times = [raw_channels[n]["time"] for n in channel_names if n in raw_channels]
    t_min = min(t[0] for t in all_times if len(t) > 0)
    t_max = max(t[-1] for t in all_times if len(t) > 0)
    n_rows = int(round((t_max - t_min) * max_rate)) + 1
    time_index = np.linspace(t_min, t_max, n_rows)

    df = pd.DataFrame(index=time_index)
    for name in channel_names:
        chan = raw_channels.get(name)
        if chan:
            df[name] = np.interp(time_index, chan["time"], chan["values"])

    df.attrs["base_rate_hz"] = round(max_rate)
    df.attrs["tier"] = "raw"
    return df


def _build_summary_tier(
    raw_channels: Dict[str, Dict[str, Any]],
    native_rates: Dict[str, float],
    classifications: Dict[str, Any],
    summary_rate: int = 50,
) -> Optional[pd.DataFrame]:
    """
    Build a summary-tier DataFrame at summary_rate Hz.

    High-frequency channels (>= 200 Hz) get windowed statistics:
    mean, min, max, and RMS velocity over each window.
    Lower-frequency channels are resampled normally.
    """
    summary_channels = channels_for_tier(classifications, "summary")
    if not summary_channels:
        return None

    # Time bounds
    valid = [raw_channels[n] for n in summary_channels
             if n in raw_channels and len(raw_channels[n]["time"]) > 0]
    if not valid:
        return None

    t_min = min(c["time"][0] for c in valid)
    t_max = max(c["time"][-1] for c in valid)
    duration = t_max - t_min
    n_rows = int(round(duration * summary_rate)) + 1
    time_index = np.linspace(t_min, t_max, n_rows)

    df = pd.DataFrame(index=time_index)

    for name in summary_channels:
        chan = raw_channels.get(name)
        if not chan:
            continue
        rate = native_rates.get(name, 0.0)

        if rate >= 200 and len(chan["values"]) > 0:
            # High-freq: compute windowed statistics
            windows = _compute_summary_windows(
                chan["values"], rate, summary_rate
            )
            n_win = len(windows["mean"])
            t_win = np.linspace(t_min, t_max, n_win) if n_win > 0 else np.array([])

            for stat_name, stat_values in windows.items():
                col_name = f"{name}_{stat_name}"
                if len(t_win) > 0 and len(stat_values) > 0:
                    df[col_name] = np.interp(time_index, t_win, stat_values)
                else:
                    df[col_name] = 0.0
        else:
            # Medium-freq: resample directly
            df[name] = _resample_channel(
                chan["time"], chan["values"], time_index, rate, summary_rate
            )

    df.attrs["base_rate_hz"] = summary_rate
    df.attrs["tier"] = "summary"
    return df


def _compute_summary_windows(values, native_rate, summary_rate=50):
    """
    Compute windowed statistics for a high-frequency channel.

    Returns dict with mean, min, max, velocity arrays.
    """
    window_size = max(1, int(native_rate / summary_rate))
    n_windows = len(values) // window_size

    if n_windows == 0:
        return {"mean": np.array([]), "min": np.array([]),
                "max": np.array([]), "velocity": np.array([])}

    # Reshape into windows for vectorized computation
    trimmed = values[:n_windows * window_size]
    windows = trimmed.reshape(n_windows, window_size)

    result = {
        "mean": np.mean(windows, axis=1),
        "min": np.min(windows, axis=1),
        "max": np.max(windows, axis=1),
    }

    # Velocity: derivative at native rate, then RMS over each window
    dt = 1.0 / native_rate
    if window_size > 1:
        diffs = np.diff(windows, axis=1) / dt
        result["velocity"] = np.sqrt(np.mean(diffs ** 2, axis=1))
    else:
        result["velocity"] = np.zeros(n_windows)

    return result

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

    Resamples all channels to base_rate_hz using anti-alias decimation
    for channels significantly above the target rate, and linear
    interpolation for channels at or below the target rate.

    Returns:
        pd.DataFrame with:
          - Index: session-relative seconds (uniform)
          - Columns: all channels normalized to base_rate_hz
          - attrs["units"]: {channel: unit}
          - attrs["unit_sources"]: {channel: {unit, source}}
          - attrs["base_rate_hz"]: the target rate
          - attrs["native_rates"]: {channel: native Hz}
    """
    if not raw_channels:
        return pd.DataFrame()

    # Compute native rates before resampling
    native_rates = compute_native_rates(raw_channels)

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
        src_rate = native_rates.get(name, 0.0)
        df[name] = _resample_channel(
            chan["time"], chan["values"], time_index, src_rate, base_rate_hz
        )

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
    df.attrs["native_rates"] = native_rates

    return df


def _resample_channel(time_src, values_src, time_dst, src_rate, dst_rate):
    """
    Resample a channel from src_rate to dst_rate.

    For upsampling or near-rate channels (src <= 1.5x dst): linear interpolation.
    For downsampling (src > 1.5x dst): anti-alias filter + decimate via scipy.
    """
    if src_rate <= dst_rate * 1.5 or src_rate == 0 or len(values_src) < 4:
        # Linear interpolation — fine for upsampling or near-rate
        return np.interp(time_dst, time_src, values_src)

    # Downsampling: apply anti-alias filter before decimation
    try:
        from scipy.signal import decimate
        factor = max(2, int(round(src_rate / dst_rate)))
        # decimate applies Chebyshev Type I low-pass before downsampling
        # zero_phase=True prevents phase distortion
        decimated = decimate(values_src, factor, ftype='iir', zero_phase=True)
        # Interpolate decimated signal onto exact target timestamps
        t_dec = np.linspace(time_src[0], time_src[-1], len(decimated))
        return np.interp(time_dst, t_dec, decimated)
    except Exception:
        # Fallback to linear interpolation if scipy fails
        return np.interp(time_dst, time_src, values_src)

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
