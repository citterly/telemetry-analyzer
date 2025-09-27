from pathlib import Path
import shutil
from src.config.config import UNITS_XML_PATH

def ensure_units_file():
    """
    Ensure units.xml exists in the AIM cache dir.
    Returns the target path.
    """
    target_dir = Path.home() / "AppData" / "Local" / "AIM"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "units.xml"

    if not target_file.exists():
        print(f"📄 Copying units.xml → {target_file}")
        shutil.copy2(UNITS_XML_PATH, target_file)
    else:
        print(f"✅ units.xml already present at {target_file}")

    return target_file
