from pathlib import Path

def cleanup_units_redirect():
    """
    Remove old units.xml redirection files (project-local + SDK path).
    Safe to run before re-initializing AIM DLL.
    """
    removed = []
    project_cache = Path(__file__).resolve().parents[2] / "data" / "exports" / "aim_cache" / "units.xml"
    sdk_profile = Path("C:/Python313/user/profiles/units.xml")

    for target in [project_cache, sdk_profile]:
        if target.exists():
            try:
                target.unlink()
                removed.append(str(target))
            except Exception as e:
                print(f"⚠️ Could not remove {target}: {e}")
    return removed
