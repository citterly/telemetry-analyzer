import sys
import argparse
import re
import numpy as np
from pathlib import Path
from src.extract.data_loader import XRKDataLoader
from src.utils.units_helper import guess_unit

def format_samples(times, values, limit=5):
    pairs = [f"({t:.2f}, {v:.2f})" for t, v in zip(times[:limit], values[:limit])]
    return " ".join(pairs)

def summarize_channel(name, chan):
    times, values = chan["time"], chan["values"]
    if len(times) == 0:
        return f"{name:30} → 0 samples"
    duration = times[-1] - times[0]
    return (f"{name:30} → {len(times)} samples, "
            f"duration {duration:.1f}s, "
            f"min {np.min(values):.2f}, max {np.max(values):.2f}")

def safe_decode(ptr) -> str:
    """Safely decode channel names from DLL pointers."""
    if not ptr:
        return ""
    try:
        return ptr.decode("utf-8") if isinstance(ptr, (bytes, bytearray)) else str(ptr)
    except Exception:
        return str(ptr)


def debug_units_xml(xml_path: Path):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for pq in root.findall("PhysicalQuantity"):
        print("PQ:", pq.get("Text"))
        for child in pq:
            print("  ", child.tag, child.attrib)

def main():
    parser = argparse.ArgumentParser(description="XRK Explorer Utility")

    # parser.add_argument("xrk_file", help="Path to XRK file")
    parser.add_argument("xrk_file", nargs="?", help="Path to XRK file (not required with --debug-xml)")

    parser.add_argument("--dump-samples", type=int, default=0,
                        help="Dump first N samples per channel (default=0)")
    parser.add_argument("--only", type=str, default=None,
                        help="Regex to filter channel names")
    parser.add_argument("--summary", action="store_true",
                        help="Show summary only (counts, sample stats)")
    parser.add_argument("--debug-xml", action="store_true",
                    help="Dump the structure of units.xml for troubleshooting")

    args = parser.parse_args()

    if not args.xrk_file:
        parser.error("xrk_file is required unless --debug-xml is used")

    xrk_file = Path(args.xrk_file)
    loader = XRKDataLoader()
    if not loader.open_file(str(xrk_file)):
        print(f"❌ Could not open XRK file: {xrk_file}")
        sys.exit(1)

    regex = re.compile(args.only, re.IGNORECASE) if args.only else None

    # ---- Regular channels ----
    reg_count = loader.dll.get_channels_count(loader.file_index)
    print(f"📊 Found {reg_count} regular channels")
    for i in range(reg_count):
        name_ptr = loader.dll.get_channel_name(loader.file_index, i)

        name = safe_decode(name_ptr).strip()
        if not name:
            if args.debug:
                print(f"⚠️ Skipped channel {i}: no name")
            continue
        if regex and not regex.search(name):
            if args.debug:
                print(f"⚠️ Skipped channel {i}: '{name}' did not match regex {args.only}")
            continue

        chan = loader._extract_channel_data(i, is_gps=False)

        # unit comes from the new override/heuristic system
        unit, source = guess_unit(name)

        if args.summary:
            if chan:
                print(summarize_channel(name, chan), f"[{unit} | {source}]")
        else:
            line = f"{i:3d}: {name:30} [{unit} | {source}]"
            if args.dump_samples > 0 and chan:
                line += " " + format_samples(chan["time"], chan["values"], args.dump_samples)
            print(line)

    # ---- GPS channels ----
    gps_count = loader.dll.get_GPS_channels_count(loader.file_index)
    print(f"\n📡 Found {gps_count} GPS channels")
    for i in range(gps_count):
        name_ptr = loader.dll.get_GPS_channel_name(loader.file_index, i)

        name = safe_decode(name_ptr).strip()
        if not name:
            if args.debug:
                print(f"⚠️ Skipped channel {i}: no name")
            continue
        if regex and not regex.search(name):
            if args.debug:
                print(f"⚠️ Skipped channel {i}: '{name}' did not match regex {args.only}")
            continue

        chan = loader._extract_channel_data(i, is_gps=False)

        # unit comes from the new override/heuristic system
        unit, source = guess_unit(name)

        if args.summary:
            if chan:
                print(summarize_channel(name, chan), f"[{unit} | {source}]")
        else:
            line = f"{i:3d}: {name:30} [{unit} | {source}]"
            if args.dump_samples > 0 and chan:
                line += " " + format_samples(chan["time"], chan["values"], args.dump_samples)
            print(line)


    loader.close_file()
    print("✅ File closed")

if __name__ == "__main__":
    main()
