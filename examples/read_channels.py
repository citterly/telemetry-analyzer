# examples/read_channels.py
"""
Example: open an XRK file and print its channels using the AIM DLL.
This is not a test — just a utility for exploring your data.
"""

from pathlib import Path
from src.io.dll_interface import AIMDLL
from src.utils.units_helper import ensure_units_file
from src.config.config import SAMPLE_FILES_PATH


def main():
    print("📂 XRK File Reader Example")

    # Make sure units.xml is available
    ensure_units_file()

    # Setup DLL
    dll = AIMDLL()
    if not dll.setup():
        print("❌ DLL setup failed, cannot continue.")
        return

    # Grab first XRK file in uploads
    xrk_file = next(SAMPLE_FILES_PATH.glob("*.xrk"), None)
    if not xrk_file:
        print(f"❌ No XRK files found in {SAMPLE_FILES_PATH}")
        return

    # Open file
    idx = dll.open(str(xrk_file))
    print(f"✅ Opened: {xrk_file}")

    # List channels
    channels = dll.get_channels(idx)
    print(f"📊 Found {len(channels)} channels")
    for i, (name, units) in enumerate(channels, 1):
        print(f"{i:3d}. {name} [{units}]")

    # Close
    dll.close(idx)
    print("✅ Done.")


if __name__ == "__main__":
    main()
