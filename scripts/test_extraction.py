"""
Test extraction script
Run on Windows to verify the extraction fix

Usage:
    python scripts/test_extraction.py [xrk_file]

This will:
1. Extract all channels from the XRK file
2. Save to a new Parquet file
3. Verify data quality
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.session.session_builder import extract_full_session, export_session
from src.config.config import SAMPLE_FILES_PATH
import pandas as pd


def test_extraction(xrk_path: str):
    """Test full extraction pipeline"""

    print(f"\n{'='*60}")
    print(f"TESTING EXTRACTION: {Path(xrk_path).name}")
    print(f"{'='*60}\n")

    # Extract
    print("Step 1: Extracting session...")
    try:
        df = extract_full_session(xrk_path)
        print(f"   Extracted DataFrame: {df.shape[0]} rows x {df.shape[1]} columns")
    except Exception as e:
        print(f"   FAILED: {e}")
        return False

    # Analyze data quality
    print("\nStep 2: Analyzing data quality...")
    print("-" * 40)

    good_channels = []
    empty_channels = []

    for col in df.columns:
        valid = df[col].notna().sum()
        pct = 100 * valid / len(df)

        if valid > 0:
            good_channels.append(col)
            print(f"   {col:30} {valid:6} rows ({pct:5.1f}%)  "
                  f"range: [{df[col].min():.2f} → {df[col].max():.2f}]")
        else:
            empty_channels.append(col)

    print(f"\n   Channels with data: {len(good_channels)}")
    print(f"   Empty channels: {len(empty_channels)}")

    if empty_channels:
        print(f"   Empty: {empty_channels}")

    # Export
    print("\nStep 3: Exporting to Parquet...")
    session_id = Path(xrk_path).stem + "_test"
    try:
        out_path = export_session(df, session_id)
        print(f"   Saved to: {out_path}")
    except Exception as e:
        print(f"   Export failed: {e}")
        # Try manual save
        out_path = Path("data/exports") / f"{session_id}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)
        print(f"   Manual save to: {out_path}")

    # Verify reload
    print("\nStep 4: Verifying Parquet reload...")
    df_reload = pd.read_parquet(out_path)
    reload_good = sum(1 for col in df_reload.columns if df_reload[col].notna().sum() > 0)
    print(f"   Reloaded: {df_reload.shape[0]} rows x {df_reload.shape[1]} columns")
    print(f"   Channels with data after reload: {reload_good}")

    # Summary
    print(f"\n{'='*60}")
    print("RESULT:", "PASS" if len(good_channels) > len(empty_channels) else "FAIL")
    print(f"{'='*60}")

    success = len(good_channels) > 5  # Should have more than just GPS
    return success


def main():
    if len(sys.argv) > 1:
        xrk_file = sys.argv[1]
    else:
        # Default to the 0037 test file
        test_file = SAMPLE_FILES_PATH / "Andy McDermid_24_AS_Road America_Race_a_0037.xrk"
        if test_file.exists():
            xrk_file = str(test_file)
        else:
            xrk_files = list(SAMPLE_FILES_PATH.glob("*.xrk"))
            if not xrk_files:
                print(f"No XRK files found in {SAMPLE_FILES_PATH}")
                return
            xrk_file = str(xrk_files[0])

    print(f"Using: {xrk_file}")
    success = test_extraction(xrk_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
