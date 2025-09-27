from pathlib import Path
import shutil
from src.config.config import UNITS_XML_PATH, EXPORTS_PATH

def copy_units_to_sdk_path():
    """
    Fallback: copy units.xml into the hard-coded SDK profile path
    used by the AIM DLL.
    """
    sdk_profile_dir = Path("C:/Python313/user/profiles")
    sdk_profile_dir.mkdir(parents=True, exist_ok=True)
    target = sdk_profile_dir / "units.xml"

    shutil.copy2(UNITS_XML_PATH, target)
    print(f"📄 Copied units.xml to {target}")

    return target
