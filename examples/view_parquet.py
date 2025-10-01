"""
Parquet Viewer Utility
----------------------
Quickly inspect canonical session exports.

Usage:
    python examples/view_parquet.py <session_id>

Example:
    python examples/view_parquet.py 20250712_104619_Road America_a_0394
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from src.config.config import EXPORTS_PATH
from src.io.file_manager import FileManager
import re

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    if len(sys.argv) < 2:
        print("❌ Usage: python examples/view_parquet.py <session_id>")
        sys.exit(1)

    session_id = sys.argv[1].replace(".xrk", "").replace(".parquet", "")

    # Use FileManager metadata first
    fm = FileManager()
    meta = fm.get_file_metadata(f"{session_id}.xrk")

    if meta and meta.parquet_path:
        parquet_path = Path(meta.parquet_path)
    else:
        # fallback: look for any matching parquet in processed dir
        candidates = list((EXPORTS_PATH / "processed").glob(f"*{session_id}*.parquet"))
        if not candidates:
            print(f"❌ Parquet file not found for session ID: {session_id}")
            sys.exit(1)
        parquet_path = candidates[0]

    # Double-check path really exists
    if not parquet_path.exists():
        print(f"❌ Parquet file not found: {parquet_path}")
        sys.exit(1)

    # Load DataFrame
    df = pd.read_parquet(parquet_path)

    # Check timebase normalization
    diffs = df.index.to_series().diff().dropna()
    unique_steps = diffs.unique()

    # Timebase verification (data-based, no forcing)
    if "base_rate_hz" in df.attrs:
        base_rate = df.attrs["base_rate_hz"]
        expected_step = 1.0 / base_rate
        steps = np.diff(df.index.values)

        # Compute deviation stats
        deviations = np.abs(steps - expected_step)
        max_dev = deviations.max() if len(deviations) > 0 else 0.0
        mean_dev = deviations.mean() if len(deviations) > 0 else 0.0

        print("\n▶ Timebase Verification:")
        print(f"  Base rate (from attrs): {base_rate} Hz")
        print(f"  Expected step: {expected_step:.6f} s")
        print(f"  Total steps: {len(steps)}")
        print(f"  Mean deviation: {mean_dev:.3e} s")
        print(f"  Max deviation:  {max_dev:.3e} s")

        if max_dev < 1e-6:
            print("  ✅ Timebase is uniform within floating-point tolerance")
        else:
            print("  ⚠️ Timebase shows irregularities (check resampling)")


    print("="*80)
    print(f"📂 Session: {session_id}")
    print(f"📁 File: {parquet_path}")
    print(f"🔢 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    if meta:
        print(f"📝 Track: {meta.track_name}")
        print(f"📅 Date: {meta.session_date}")
    print("="*80)

    # Show first few rows
    print("\n▶ First 10 rows:")
    print(df.head(10))

    # Show channel units
    units = df.attrs.get("units", {})
    print("\n▶ Channel Units:")
    for col in list(df.columns)[:20]:  # just show first 20 columns
        print(f"  {col:25} → {units.get(col, 'unknown')}")

if __name__ == "__main__":
    main()
