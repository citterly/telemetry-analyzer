# arch-001: Vehicle Config Unification

**Status**: Plan
**Date**: 2026-02-10
**Scope**: Migrate from dual config system to single unified vehicle config

---

## Problem Statement

The project has two competing vehicle configuration systems:

1. **`src/config/vehicle_config.py`** — Module-level dicts (`CURRENT_SETUP`, `ENGINE_SPECS`, `TRACK_CONFIG`, `PROCESSING_CONFIG`, `TRANSMISSION_SCENARIOS`, etc.). Imported by 10+ source files and 3+ test files. Hardcoded to one vehicle (Andy's BMW M3). No switching capability.

2. **`src/config/vehicles.py`** — OOP system (`Vehicle`, `EngineSpec`, `TransmissionSetup`, `VehicleDatabase`). Loaded from `data/vehicles.json`. Supports 5 vehicles with runtime switching. Has backward-compat shims (`get_current_setup()`, `get_engine_specs()`).

When the user switches the active vehicle via the UI, only code using `vehicles.py` sees the change. Code importing `vehicle_config.py` dicts continues using hardcoded BMW values.

---

## Current State: All Consumers

### Source files importing vehicle_config.py

| File | Imports | What it uses |
|------|---------|-------------|
| `src/analysis/gear_calculator.py:10` | `CURRENT_SETUP, TIRE_CIRCUMFERENCE_METERS, theoretical_speed_at_rpm, theoretical_rpm_at_speed` | Gear ratio matching, speed/RPM conversion |
| `src/analysis/lap_analyzer.py:10` | `PROCESSING_CONFIG, TRACK_CONFIG` | Lap time bounds, start/finish GPS, threshold |
| `src/analysis/lap_analyzer.py:58` | `TRACK_CONFIG` (lazy import inside method) | Fallback track config |
| `src/analysis/__init__.py:3` | `PROCESSING_CONFIG, TRACK_CONFIG` | Re-export for package users |
| `src/features/shift_analysis.py:17` | `CURRENT_SETUP, TRANSMISSION_SCENARIOS` | Gear ratios for shift analysis |
| `src/features/shift_analysis.py:288,345` | `ENGINE_SPECS` (lazy imports) | RPM limits for shift quality |
| `src/features/lap_analysis.py:17` | `TRACK_CONFIG, PROCESSING_CONFIG` | Track name, lap time bounds |
| `src/features/gear_analysis.py:17` | `CURRENT_SETUP, ENGINE_SPECS, TIRE_CIRCUMFERENCE_METERS, TRANSMISSION_SCENARIOS, TRACK_CONFIG, theoretical_rpm_at_speed` | Gear detection, RPM analysis, track sections |
| `src/features/gear_analysis.py:713` | `theoretical_rpm_at_speed` (lazy import) | Speed/RPM conversion |
| `src/features/power_analysis.py:17` | `ENGINE_SPECS` | RPM limits for power band |
| `src/features/session_report.py:20` | `TRACK_CONFIG, CURRENT_SETUP` | Defaults for track name, vehicle setup name |
| `src/features/transmission_comparison.py:15` | `CURRENT_SETUP, TIRE_CIRCUMFERENCE_METERS, theoretical_speed_at_rpm, ENGINE_SPECS, TRANSMISSION_SCENARIOS` | Full transmission comparison |
| `src/extraction/data_loader.py:14` | `DEFAULT_SESSION` | Default XRK filename |

### Test files importing vehicle_config.py

| File | Imports |
|------|---------|
| `tests/test_session_builder.py:8` | `DEFAULT_SESSION` |
| `tests/test_transmission_comparison.py:20` | `TRANSMISSION_SCENARIOS` |
| `tests/test_lap_analyzer.py:399` | `PROCESSING_CONFIG` |
| `init.sh:58` | `TRANSMISSION_SCENARIOS` (smoke test) |

### What vehicle_config.py exports (that are actually used)

| Export | Consumers | vehicles.py equivalent |
|--------|-----------|----------------------|
| `CURRENT_SETUP` (dict) | gear_calculator, shift_analysis, gear_analysis, session_report, transmission_comparison | `get_active_vehicle().current_setup.to_dict()` |
| `ENGINE_SPECS` (dict) | shift_analysis, gear_analysis, power_analysis, transmission_comparison | `get_active_vehicle().engine.to_dict()` |
| `TRACK_CONFIG` (dict) | lap_analyzer, lap_analysis, gear_analysis, session_report, analysis/__init__ | **No equivalent** — tracks.py has `TrackDatabase` but not wired in |
| `PROCESSING_CONFIG` (dict) | lap_analyzer, lap_analysis, analysis/__init__ | **No equivalent** — needs to be added |
| `TRANSMISSION_SCENARIOS` (list) | shift_analysis, transmission_comparison, init.sh, test | `get_active_vehicle().all_setups` (returns `TransmissionSetup` objects) |
| `TIRE_CIRCUMFERENCE_METERS` (float) | gear_calculator, gear_analysis, transmission_comparison | `get_active_vehicle().tire_circumference_meters` |
| `DEFAULT_SESSION` (str) | data_loader, test_session_builder | Not vehicle-specific — move to config.py or constants |
| `theoretical_speed_at_rpm()` | gear_calculator, transmission_comparison | `Vehicle.calculate_speed_at_rpm()` (mph not m/s) |
| `theoretical_rpm_at_speed()` | gear_calculator, gear_analysis | `Vehicle.calculate_rpm_at_speed()` (mph not m/s) |
| `calculate_tire_circumference()` | Only used in `__main__` block | Keep as utility or move to Vehicle class |
| `get_scenario_by_name()` | Not used externally | `Vehicle.get_setup_by_name()` |
| `PLOT_CONFIG` | Not used by any source file | Delete |
| `RPM_ZONES` | Not used by any source file | Delete |

---

## Proposed Changes

### Strategy: Bridge then migrate

Rather than rewriting all consumers at once, add a bridge layer in `vehicle_config.py` that delegates to `vehicles.py`. Then migrate consumers one by one. Finally remove the bridge.

### Step 1: Add missing equivalents to vehicles.py / tracks.py

**`src/config/vehicles.py`** — Add:
- `get_processing_config() -> dict` — Returns processing config from active vehicle or defaults
- `get_track_config() -> dict` — Delegates to tracks.py for active track

**`src/config/tracks.py`** — Add:
- `get_active_track_config() -> dict` — Returns TRACK_CONFIG-compatible dict from active track
- Wire `get_active_track()` to return the track detected from recent session or default

**`Vehicle` class** — Add:
- `theoretical_speed_at_rpm(rpm, gear_ratio, final_drive) -> float` (m/s version for backward compat)
- `theoretical_rpm_at_speed(speed_ms, gear_ratio, final_drive) -> float` (m/s version)

Or better: add module-level functions to vehicles.py that match the old signatures.

### Step 2: Convert vehicle_config.py to bridge module

Replace all hardcoded dicts with property-style delegation:

```python
# vehicle_config.py — BRIDGE (to be removed after migration)

from .vehicles import get_active_vehicle, get_vehicle_database
from .tracks import get_active_track_config

def _get_vehicle():
    """Get active vehicle with fallback to defaults."""
    v = get_active_vehicle()
    if v is None:
        # Fallback for tests/bootstrap
        return _DEFAULTS
    return v

# These become dynamic properties
@property-like pattern using module __getattr__:

def __getattr__(name):
    if name == 'CURRENT_SETUP':
        return _get_current_setup_dict()
    if name == 'ENGINE_SPECS':
        return _get_engine_specs_dict()
    ...
```

Actually, Python supports module-level `__getattr__` (PEP 562). This lets us make the module-level "constants" dynamic without changing any consumer import statements.

### Step 3: Migrate consumers to import from vehicles.py directly

For each consumer file:
1. Replace `from ..config.vehicle_config import X` with `from ..config.vehicles import ...`
2. Update code to use Vehicle objects instead of dicts where beneficial
3. Keep dict access patterns where the change would be too invasive

Priority order (least risk first):
1. `power_analysis.py` — only uses ENGINE_SPECS
2. `session_report.py` — only uses TRACK_CONFIG, CURRENT_SETUP for defaults
3. `lap_analysis.py` — uses TRACK_CONFIG, PROCESSING_CONFIG
4. `shift_analysis.py` — uses CURRENT_SETUP, TRANSMISSION_SCENARIOS, ENGINE_SPECS
5. `gear_analysis.py` — heaviest user, uses 6 exports
6. `gear_calculator.py` — uses speed/RPM functions
7. `transmission_comparison.py` — full comparison logic
8. `data_loader.py` — just DEFAULT_SESSION
9. `lap_analyzer.py` — uses PROCESSING_CONFIG, TRACK_CONFIG

### Step 4: Move non-vehicle config to appropriate homes

| Item | Current location | New location |
|------|-----------------|-------------|
| `DEFAULT_SESSION` | vehicle_config.py | `src/config/config.py` or `src/config/constants.py` |
| `PROCESSING_CONFIG` | vehicle_config.py | `src/config/tracks.py` (per-track processing params) or `src/config/constants.py` |
| `PLOT_CONFIG` | vehicle_config.py | Delete (unused) |
| `RPM_ZONES` | vehicle_config.py | Delete (unused) |
| `TRACK_CONFIG` | vehicle_config.py | `src/config/tracks.py` (already has TrackDatabase) |

### Step 5: Delete vehicle_config.py

Once all consumers are migrated and tests pass, remove the file.

---

## Acceptance Criteria

1. **No source file imports from vehicle_config.py** — `grep -r "vehicle_config" src/` returns zero hits (excluding comments/docstrings)
2. **vehicle_config.py is deleted** or contains only a deprecation warning + re-exports
3. **Vehicle switching works end-to-end** — changing active vehicle via API changes the config used by all analyzers
4. **All existing tests pass** — zero regressions
5. **TRACK_CONFIG comes from tracks.py** — not hardcoded to Road America
6. **PROCESSING_CONFIG has a proper home** — not floating in a vehicle config file
7. **DEFAULT_SESSION has a proper home** — not in vehicle config
8. **Speed/RPM utility functions accessible** — `theoretical_speed_at_rpm()` and `theoretical_rpm_at_speed()` still available (can be on Vehicle class or as module functions in vehicles.py)
9. **init.sh smoke test still works** — update the import path
10. **Test files updated** — no test imports vehicle_config.py

---

## Migration Path

The bridge approach (Step 2) means we can migrate one file at a time. If a migration breaks something, we can revert that single file. The bridge keeps everything working during the transition.

Order of operations:
1. Add missing equivalents (Step 1) — new code only, no breakage risk
2. Install bridge (Step 2) — vehicle_config.py becomes thin wrapper, all behavior preserved
3. Migrate consumers (Step 3) — one file per commit, tests after each
4. Move non-vehicle config (Step 4) — small, focused moves
5. Delete bridge (Step 5) — final cleanup

---

## Risks

1. **Singleton initialization order** — VehicleDatabase singleton loads from JSON on first access. If accessed before data/vehicles.json exists (e.g., during test setup), it falls back to built-in defaults. This is already handled but needs testing.

2. **Test isolation** — Tests that modify vehicle config could affect other tests. The singleton pattern means state leaks between tests. Mitigation: tests should use fixtures that reset the singleton.

3. **tracks.py integration** — TRACK_CONFIG is currently hardcoded to Road America. Moving to tracks.py means we need to decide: does the "active track" come from session data (auto-detect) or manual selection? For now, keep Road America as default and detect from data when available.

4. **Speed unit mismatch** — vehicle_config.py functions use m/s, Vehicle class methods use mph. Need to ensure all consumers are updated to match.

---

## Files Modified (estimated)

| File | Change type |
|------|------------|
| `src/config/vehicles.py` | Add bridge functions, speed/RPM utilities |
| `src/config/tracks.py` | Add get_active_track_config() |
| `src/config/vehicle_config.py` | Convert to bridge, then delete |
| `src/analysis/gear_calculator.py` | Update imports |
| `src/analysis/lap_analyzer.py` | Update imports |
| `src/analysis/__init__.py` | Update imports |
| `src/features/shift_analysis.py` | Update imports |
| `src/features/lap_analysis.py` | Update imports |
| `src/features/gear_analysis.py` | Update imports |
| `src/features/power_analysis.py` | Update imports |
| `src/features/session_report.py` | Update imports |
| `src/features/transmission_comparison.py` | Update imports |
| `src/extraction/data_loader.py` | Update imports |
| `tests/test_session_builder.py` | Update imports |
| `tests/test_transmission_comparison.py` | Update imports |
| `tests/test_lap_analyzer.py` | Update imports |
| `init.sh` | Update smoke test import |
