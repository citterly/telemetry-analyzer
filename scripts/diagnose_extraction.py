"""
Diagnostic script for XRK extraction
Run on Windows to verify channel extraction and time bases

Usage:
    python scripts/diagnose_extraction.py [xrk_file]

Outputs:
    - Console: Summary of all channels with time ranges
    - data/diagnostics/extraction_report.json: Full diagnostic data
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extract.data_loader import XRKDataLoader
from src.config.config import SAMPLE_FILES_PATH


def diagnose_file(xrk_path: str):
    """Extract and diagnose all channels from an XRK file"""

    loader = XRKDataLoader()

    if not loader.open_file(xrk_path):
        print(f"Failed to open: {xrk_path}")
        return None

    report = {
        "file": str(xrk_path),
        "timestamp": datetime.now().isoformat(),
        "regular_channels": [],
        "gps_channels": [],
        "summary": {}
    }

    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC REPORT: {Path(xrk_path).name}")
    print(f"{'='*60}\n")

    # Regular channels
    print("REGULAR CHANNELS:")
    print("-" * 40)

    try:
        chan_count = loader.dll.get_channels_count(loader.file_index)
        print(f"Total regular channels: {chan_count}\n")

        for i in range(chan_count):
            name_ptr = loader.dll.get_channel_name(loader.file_index, i)
            if not name_ptr:
                continue

            name = name_ptr.decode("utf-8") if isinstance(name_ptr, bytes) else str(name_ptr)
            sample_count = loader.dll.get_channel_samples_count(loader.file_index, i)

            chan_info = {
                "index": i,
                "name": name,
                "sample_count": sample_count,
                "time_min": None,
                "time_max": None,
                "time_unit_guess": None,
                "value_min": None,
                "value_max": None
            }

            if sample_count > 0:
                # Extract raw data (before any conversion)
                from ctypes import c_double
                times_array = (c_double * sample_count)()
                values_array = (c_double * sample_count)()
                result = loader.dll.get_channel_samples(
                    loader.file_index, i, times_array, values_array, sample_count
                )

                if result > 0:
                    times = [times_array[j] for j in range(min(result, 10))]  # First 10
                    time_min = times_array[0]
                    time_max = times_array[result - 1]
                    value_min = min(values_array[j] for j in range(result))
                    value_max = max(values_array[j] for j in range(result))

                    chan_info["time_min"] = time_min
                    chan_info["time_max"] = time_max
                    chan_info["time_unit_guess"] = "ms" if time_max > 1000 else "s"
                    chan_info["value_min"] = value_min
                    chan_info["value_max"] = value_max
                    chan_info["first_times"] = times

                    unit_guess = chan_info["time_unit_guess"]
                    print(f"  {name:30} samples={sample_count:6}  "
                          f"time=[{time_min:10.1f} → {time_max:10.1f}] ({unit_guess})  "
                          f"values=[{value_min:.2f} → {value_max:.2f}]")
                else:
                    print(f"  {name:30} samples={sample_count:6}  ** extraction failed **")
            else:
                print(f"  {name:30} samples=0  ** empty **")

            report["regular_channels"].append(chan_info)

    except Exception as e:
        print(f"Error reading regular channels: {e}")

    # GPS channels
    print("\n\nGPS CHANNELS:")
    print("-" * 40)

    try:
        gps_count = loader.dll.get_GPS_channels_count(loader.file_index)
        print(f"Total GPS channels: {gps_count}\n")

        for i in range(gps_count):
            name_ptr = loader.dll.get_GPS_channel_name(loader.file_index, i)
            if not name_ptr:
                continue

            name = name_ptr.decode("utf-8") if isinstance(name_ptr, bytes) else str(name_ptr)
            sample_count = loader.dll.get_GPS_channel_samples_count(loader.file_index, i)

            chan_info = {
                "index": i,
                "name": name,
                "sample_count": sample_count,
                "time_min": None,
                "time_max": None,
                "time_unit_guess": None,
                "value_min": None,
                "value_max": None
            }

            if sample_count > 0:
                from ctypes import c_double
                times_array = (c_double * sample_count)()
                values_array = (c_double * sample_count)()
                result = loader.dll.get_GPS_channel_samples(
                    loader.file_index, i, times_array, values_array, sample_count
                )

                if result > 0:
                    times = [times_array[j] for j in range(min(result, 10))]
                    time_min = times_array[0]
                    time_max = times_array[result - 1]
                    value_min = min(values_array[j] for j in range(result))
                    value_max = max(values_array[j] for j in range(result))

                    chan_info["time_min"] = time_min
                    chan_info["time_max"] = time_max
                    chan_info["time_unit_guess"] = "ms" if time_max > 1000 else "s"
                    chan_info["value_min"] = value_min
                    chan_info["value_max"] = value_max
                    chan_info["first_times"] = times

                    unit_guess = chan_info["time_unit_guess"]
                    print(f"  {name:30} samples={sample_count:6}  "
                          f"time=[{time_min:10.1f} → {time_max:10.1f}] ({unit_guess})  "
                          f"values=[{value_min:.2f} → {value_max:.2f}]")
                else:
                    print(f"  {name:30} samples={sample_count:6}  ** extraction failed **")
            else:
                print(f"  {name:30} samples=0  ** empty **")

            report["gps_channels"].append(chan_info)

    except Exception as e:
        print(f"Error reading GPS channels: {e}")

    loader.close_file()

    # Summary
    reg_with_data = sum(1 for c in report["regular_channels"] if c["sample_count"] > 0)
    gps_with_data = sum(1 for c in report["gps_channels"] if c["sample_count"] > 0)

    report["summary"] = {
        "regular_total": len(report["regular_channels"]),
        "regular_with_data": reg_with_data,
        "gps_total": len(report["gps_channels"]),
        "gps_with_data": gps_with_data
    }

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Regular channels: {reg_with_data}/{len(report['regular_channels'])} have data")
    print(f"GPS channels: {gps_with_data}/{len(report['gps_channels'])} have data")

    return report


def main():
    # Default to the test file, or use command line arg
    if len(sys.argv) > 1:
        xrk_file = sys.argv[1]
    else:
        # Try to find any XRK file
        xrk_files = list(SAMPLE_FILES_PATH.glob("*.xrk"))
        if not xrk_files:
            print(f"No XRK files found in {SAMPLE_FILES_PATH}")
            print("Usage: python scripts/diagnose_extraction.py <path_to_xrk>")
            return
        xrk_file = str(xrk_files[0])
        print(f"Using: {xrk_file}")

    report = diagnose_file(xrk_file)

    if report:
        # Save report
        out_dir = Path(__file__).parent.parent / "data" / "diagnostics"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "extraction_report.json"

        with open(out_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\nFull report saved to: {out_file}")


if __name__ == "__main__":
    main()
