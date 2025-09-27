# tests/test_integration_dll.py
"""
Integration test for AIM DLL + units.xml setup.
Ensures the DLL loads, units.xml is found, and XRK files can be opened.
"""

from pathlib import Path
from src.io.dll_interface import AIMDLL
from src.utils.units_helper import ensure_units_file
from src.config.config import SAMPLE_FILES_PATH


def test_integration():
    print("🔧 Running integration test for DLL + units.xml")

    # Step 1: Ensure units.xml is available
    ensure_units_file()

    # Step 2: Setup DLL
    dll = AIMDLL()
    assert dll.setup(), "❌ DLL setup failed."

    # Step 3: Open an XRK file
    sample_file = next(SAMPLE_FILES_PATH.glob("*.xrk"), None)
    assert sample_file, f"❌ No XRK file found in {SAMPLE_FILES_PATH}"

    idx = dll.open(str(sample_file))
    print(f"✅ Opened XRK file: {sample_file}")

    # Step 4: Get channel list
    channels = dll.get_channels(idx)
    print(f"📊 Total channels: {len(channels)}")

    # Show first 10 channels
    for name, units in channels[:10]:
        print(f"  {name} [{units}]")

    # Step 5: Close file
    dll.close(idx)
    print("✅ Integration test complete.")


if __name__ == "__main__":
    test_integration()
