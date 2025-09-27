# src/io/dll_interface.py
import os
import ctypes
import shutil
from pathlib import Path   # ✅ add this
from src.config.config import DLL_PATH, DEPENDENCY_PATH
from src.utils.units_helper import ensure_units_file   # ✅ use helper

from src.config.config import DLL_PATH, DEPENDENCY_PATH, UNITS_XML_PATH, EXPORTS_PATH

def ensure_units_redirect():
    """
    Ensure the AIM DLL finds units.xml.
    Priority:
    1. Project-local AIM cache (preferred)
    2. Fallback copy into SDK hard-coded path
    """
    # always ensure base units.xml exists in user profile
    ensure_units_file()

    # project-local copy
    project_cache_dir = Path(EXPORTS_PATH) / "aim_cache"
    project_cache_dir.mkdir(parents=True, exist_ok=True)
    project_units = project_cache_dir / "units.xml"
    shutil.copy2(UNITS_XML_PATH, project_units)
    print(f"✅ Ensured project units.xml at {project_units}")

    # SDK hard-coded path
    sdk_profile_dir = Path("C:/Python313/user/profiles")
    sdk_profile_dir.mkdir(parents=True, exist_ok=True)
    sdk_units = sdk_profile_dir / "units.xml"

    try:
        if not sdk_units.exists():
            shutil.copy2(project_units, sdk_units)
            print(f"📄 Copied units.xml into fallback SDK path: {sdk_units}")
        else:
            print(f"✅ units.xml already present at {sdk_units}")
    except Exception as e:
        print(f"⚠️ Could not copy units.xml to SDK path: {e}")

    return project_units


class AIMDLL:
    def __init__(self):
        self.dll = None

    def setup(self):
        try:
            # Ensure units.xml is in place
            ensure_units_redirect()

            # Add DLL search path
            if os.name == "nt":
                os.add_dll_directory(str(DEPENDENCY_PATH))
            else:
                os.environ["PATH"] = str(DEPENDENCY_PATH) + os.pathsep + os.environ.get("PATH", "")

            # Load DLL
            self.dll = ctypes.WinDLL(str(DLL_PATH))
            self._configure_functions()

            return True
        except Exception as e:
            print(f"⚠️ DLL not available ({e}), falling back to binary mode")
            return False

    def _configure_functions(self):
        """Define ctypes prototypes for key DLL functions."""
        self.dll.open_file.argtypes = [ctypes.c_char_p]
        self.dll.open_file.restype = ctypes.c_int

        self.dll.close_file_i.argtypes = [ctypes.c_int]
        self.dll.close_file_i.restype = ctypes.c_int

        self.dll.get_channels_count.argtypes = [ctypes.c_int]
        self.dll.get_channels_count.restype = ctypes.c_int

        self.dll.get_channel_name.argtypes = [ctypes.c_int, ctypes.c_int]
        self.dll.get_channel_name.restype = ctypes.c_char_p

        self.dll.get_channel_units.argtypes = [ctypes.c_int, ctypes.c_int]
        self.dll.get_channel_units.restype = ctypes.c_char_p

        try:
            self.dll.AiMLib_SetUnitsFile.argtypes = [ctypes.c_char_p]
            self.dll.AiMLib_SetUnitsFile.restype = ctypes.c_int
        except AttributeError:
            print("⚠️ Warning: DLL does not export AiMLib_SetUnitsFile")

    def open(self, filepath: str) -> int:
        idx = self.dll.open_file(filepath.encode("utf-8"))
        if idx < 0:
            raise RuntimeError("Failed to open file")
        return idx

    def close(self, idx: int):
        self.dll.close_file_i(idx)

    def get_channels(self, idx: int):
        count = self.dll.get_channels_count(idx)
        channels = []
        for i in range(count):
            name = self.dll.get_channel_name(idx, i).decode()
            units = self.dll.get_channel_units(idx, i).decode()
            channels.append((name, units))
        return channels
