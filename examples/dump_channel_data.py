# examples/dump_channel_data.py
"""
Example: dump samples from XRK channels into tabular form.
Uses the AIM DLL — requires units.xml to be set up.
"""

import csv
from pathlib import Path
from src.io.dll_interface import AIMDLL
from src.utils.units_helper import ensure_units_file
from src.config.config import SAMPLE_FILES_PATH


def main():
    print("📂 XRK Channel Data Dumper")

    ensure_units_file()
    dll = AIMDLL()
    if not dll.setup():
        print("❌ DLL setup failed, cannot continue.")
        return

    # Find a file
    xrk_file = next(SAMPLE_FILES_PATH.glob("*.xrk"), None)
    if not xrk_file:
        print(f"❌ No XRK files found in {SAMPLE_FILES_PATH}")
        return

    idx = dll.open(str(xrk_file))
    print(f"✅ Opened: {xrk_file}")

    # Get channels
    channels = dll.get_channels(idx)
    print(f"📊 Found {len(channels)} channels")

    # --- NOTE ---
    # This part assumes the DLL has functions to fetch sample count and sample data.
    # If they’re named differently, we’ll need to adjust signatures in dll_interface.py.
    # ----------------

    # Get channel samples (pseudo-code, adjust once we explore DLL)
    if hasattr(dll.dll, "get_channel_sample_count") and hasattr(dll.dll, "get_channel_value"):
        dll.dll.get_channel_sample_count.argtypes = [ctypes.c_int, ctypes.c_int]
        dll.dll.get_channel_sample_count.restype = ctypes.c_int

        dll.dll.get_channel_value.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        dll.dll.get_channel_value.restype = ctypes.c_double

        output_file = SAMPLE_FILES_PATH / "channel_dump.csv"
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["Index"] + [name for name, _ in channels]
            writer.writerow(header)

            sample_counts = [
                dll.dll.get_channel_sample_count(idx, i) for i in range(len(channels))
            ]
            max_samples = max(sample_counts)

            for s in range(max_samples):
                row = [s]
                for c in range(len(channels)):
                    if s < sample_counts[c]:
                        val = dll.dll.get_channel_value(idx, c, s)
                    else:
                        val = ""
                    row.append(val)
                writer.writerow(row)

        print(f"✅ Wrote sample data to {output_file}")
    else:
        print("⚠️ DLL does not expose sample-reading functions (need to confirm names).")

    dll.close(idx)
    print("✅ Done.")


if __name__ == "__main__":
    main()
