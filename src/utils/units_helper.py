from pathlib import Path
import json

OVERRIDE_FILE = Path(__file__).resolve().parent.parent / "config" / "units_override.json"

# Load overrides once
if OVERRIDE_FILE.exists():
    with open(OVERRIDE_FILE, "r") as f:
        overrides = {k.lower(): v for k, v in json.load(f).items()}
else:
    overrides = {}

def load_overrides():
    """Load units_override.json if available."""
    global overrides
    if not overrides and UNITS_OVERRIDE_PATH.exists():
        try:
            with open(UNITS_OVERRIDE_PATH, "r") as f:
                overrides = json.load(f)
            print(f"✅ Loaded {len(overrides)} overrides from {UNITS_OVERRIDE_PATH}")
        except Exception as e:
            print(f"⚠️ Failed to load overrides: {e}")
    return overrides


def guess_unit(channel_name: str) -> tuple[str, str]:
    """
    Return (unit, source) for a channel.
    Source is 'override', 'heuristic', or 'unknown'.
    """
    name = channel_name.lower()

    # 1. Check overrides
    load_overrides()
    if name in overrides:
        return overrides[name], "override"

    # 2. Heuristics
    if "temp" in name: return "°F", "heuristic"
    if "press" in name: return "psi", "heuristic"
    if "volt" in name: return "V", "heuristic"
    if "acc" in name: return "m/s²", "heuristic"
    if "rate" in name: return "deg/s", "heuristic"
    if "rpm" in name: return "rpm", "heuristic"
    if "speed" in name: return "mph", "heuristic"
    if "gyro" in name: return "deg/s", "heuristic"
    if "heading" in name: return "deg", "heuristic"
    if "altitude" in name: return "m", "heuristic"
    if "latitude" in name or "longitude" in name: return "deg", "heuristic"

    # 3. Fallback
    return "unknown", "unknown"
