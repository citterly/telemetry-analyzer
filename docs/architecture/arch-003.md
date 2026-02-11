# arch-003: Analyzer Registry and Plugin Pattern

**Status**: Plan
**Date**: 2026-02-10
**Scope**: Replace hardcoded analyzer instantiation with a registry pattern

---

## Problem Statement

`SessionReportGenerator` hardcodes 4 sub-analyzers in its `__init__`:

```python
self.lap_analyzer = LapAnalysis(track_name=self.track_name)
self.shift_analyzer = ShiftAnalyzer()
self.gear_analyzer = GearAnalysis(track_name=self.track_name)
self.power_analyzer = PowerAnalysis(vehicle_mass_kg=vehicle_mass_kg)
```

And has 4 separate `_run_*_analysis()` methods. Adding a new analyzer (suspension, brake temperature, tire grip) requires:

1. Editing `session_report.py` imports
2. Adding a new `__init__` line
3. Adding a new `_run_*_analysis()` method
4. Adding the result to the `SessionReport` dataclass
5. Updating `to_dict()` serialization
6. Updating trace cross-validation

This violates the Open/Closed Principle. A registry pattern would allow new analyzers to register themselves and be automatically included.

---

## Current State

### Analyzers that inherit BaseAnalyzer

| Analyzer | Key | Required channels | Config params |
|----------|-----|-------------------|---------------|
| `ShiftAnalyzer` | shifts | rpm, speed | — |
| `LapAnalysis` | laps | lat, lon, speed, rpm | track_name |
| `GearAnalysis` | gears | rpm, speed, lat, lon | track_name |
| `PowerAnalysis` | power | speed, rpm (optional) | vehicle_mass_kg |
| `GGAnalyzer` | gg | lat_acc, lon_acc, speed | — |
| `CornerAnalyzer` | corners | lat, lon, speed, throttle | — |
| `SessionReportGenerator` | report | (orchestrator) | track_name, vehicle_mass_kg |

### How session_report.py uses analyzers

1. `generate_from_arrays()` receives pre-extracted numpy arrays
2. Calls `_run_lap_analysis()` → `self.lap_analyzer.analyze_from_arrays(...)`
3. Calls `_run_shift_analysis()` → `self.shift_analyzer.analyze_from_arrays(...)`
4. Calls `_run_gear_analysis()` → `self.gear_analyzer.analyze_from_arrays(...)`
5. Calls `_run_power_analysis()` → `self.power_analyzer.analyze_from_arrays(...)`
6. Each wrapped in try/except, failures recorded as warnings
7. Results assembled into `SessionReport` dataclass

### Interface variations

Each analyzer's `analyze_from_arrays()` takes different parameters:
- `ShiftAnalyzer(rpm, speed, time, session_id)`
- `LapAnalysis(time, lat, lon, rpm, speed, session_id)`
- `GearAnalysis(time, rpm, speed, lat, lon, session_id)`
- `PowerAnalysis(time, speed, rpm, session_id)`

This inconsistency makes a uniform loop difficult — each needs a custom mapping.

---

## Proposed Changes

### Analyzer metadata via class attributes

Each analyzer declares its requirements as class-level metadata:

```python
class ShiftAnalyzer(BaseAnalyzer):
    registry_key = "shifts"
    required_channels = ["rpm", "speed"]
    optional_channels = []
    config_params = []  # No special config needed
```

```python
class LapAnalysis(BaseAnalyzer):
    registry_key = "laps"
    required_channels = ["latitude", "longitude", "speed"]
    optional_channels = ["rpm"]
    config_params = ["track_name"]
```

### AnalyzerRegistry class

Location: `src/features/registry.py`

```python
@dataclass
class AnalyzerRegistration:
    key: str                       # "shifts", "laps", etc.
    analyzer_class: type           # ShiftAnalyzer, LapAnalysis, etc.
    required_channels: List[str]   # Channels needed
    optional_channels: List[str]   # Channels used if available
    config_params: List[str]       # Constructor kwargs from session config

class AnalyzerRegistry:
    _analyzers: Dict[str, AnalyzerRegistration]

    def register(self, analyzer_class: type) -> None:
        """Register an analyzer class. Reads metadata from class attrs."""

    def get(self, key: str) -> AnalyzerRegistration:
        """Get registration by key."""

    def list_registered(self) -> List[str]:
        """List all registered analyzer keys."""

    def create_instance(self, key: str, **config) -> BaseAnalyzer:
        """Create an analyzer with config params."""

    def run_all(
        self,
        channels: SessionChannels,
        session_id: str,
        config: Dict,
        include_trace: bool = False,
    ) -> Dict[str, BaseAnalysisReport]:
        """Run all registered analyzers, returning {key: report}."""
```

### Auto-registration

Analyzers register themselves at import time:

```python
# In shift_analysis.py:
from .registry import analyzer_registry

class ShiftAnalyzer(BaseAnalyzer):
    registry_key = "shifts"
    required_channels = ["rpm", "speed"]
    ...

analyzer_registry.register(ShiftAnalyzer)
```

### Channel mapping for analyze_from_arrays

Each analyzer needs a `_map_channels()` classmethod that extracts its needed arrays from `SessionChannels`:

```python
class ShiftAnalyzer(BaseAnalyzer):
    @classmethod
    def _map_channels(cls, channels: SessionChannels) -> dict:
        return {
            "rpm_data": channels.rpm,
            "speed_data": channels.speed_mph,
            "time_data": channels.time,
        }
```

Or simpler: add `analyze_from_channels(channels: SessionChannels)` method to each analyzer that wraps `analyze_from_arrays()` with the correct argument mapping.

### Updated SessionReportGenerator

```python
class SessionReportGenerator(BaseAnalyzer):
    def generate_from_arrays(self, time, lat, lon, rpm, speed, session_id):
        results = {}
        for key, reg in analyzer_registry.items():
            if not self._has_required_channels(reg, ...):
                continue
            try:
                instance = reg.create_instance(**self.config)
                report = instance.analyze_from_arrays(...)  # via channel mapping
                results[key] = report
            except Exception as e:
                self.warnings.append(f"{key}: {e}")

        return SessionReport(
            session_id=session_id,
            sub_reports=results,
            ...
        )
```

---

## Migration Strategy

### Phase 1: Create registry infrastructure (no consumer changes)

1. Create `src/features/registry.py` with `AnalyzerRegistry`, `AnalyzerRegistration`
2. Define metadata attributes on `BaseAnalyzer`
3. Create singleton `analyzer_registry`
4. Write tests

### Phase 2: Add metadata to existing analyzers

For each of the 6 analyzers:
1. Add `registry_key`, `required_channels`, `optional_channels`, `config_params`
2. Add `analyze_from_channels(channels: SessionChannels)` method
3. Register with `analyzer_registry`
4. Write tests for channel mapping

### Phase 3: Update SessionReportGenerator

1. Replace hardcoded analyzer instantiation with registry iteration
2. Replace `_run_*_analysis()` methods with a generic `_run_analyzer(key, channels)`
3. Update SessionReport dataclass to use dynamic `sub_reports` dict
4. Update trace cross-validation to iterate over dynamic results
5. Maintain backward compatibility for `report.lap_analysis` etc. via properties

### Phase 4: Update router auto-discovery (optional)

This is a stretch goal:
1. Routers query registry for available analyzers
2. Auto-generate endpoints for each registered analyzer
3. Reduces router boilerplate further

---

## Acceptance Criteria

1. **AnalyzerRegistry exists** at `src/features/registry.py`
2. **All 6 analyzers registered** — registry.list_registered() returns 6 keys
3. **Metadata on each analyzer** — `registry_key`, `required_channels`, `config_params`
4. **`analyze_from_channels()` method** on each analyzer accepts SessionChannels
5. **SessionReportGenerator uses registry** — no hardcoded analyzer imports in generate method
6. **Dynamic report assembly** — `SessionReport.sub_reports` dict keyed by registry_key
7. **Backward compatible** — `report.lap_analysis`, `report.shift_analysis` still work (via properties or __getattr__)
8. **New analyzer registration is trivial** — adding a new analyzer requires only: create class + register
9. **All existing tests pass** — zero regressions
10. **Trace cross-validation works dynamically** — iterates over registered results

---

## Risks

1. **Parameter mapping complexity** — Each analyzer's `analyze_from_arrays()` takes different arguments. The channel mapping layer adds indirection. Mitigation: `analyze_from_channels()` is a simple 5-line method on each class.

2. **Import order** — Auto-registration at import time means the registry module must be imported before analyzers. Circular imports are a risk. Mitigation: registry module has no dependencies on analyzers; analyzers import from registry.

3. **SessionReport backward compatibility** — Existing code accesses `report.lap_analysis`, `report.shift_analysis` etc. Changing to `report.sub_reports["laps"]` would break consumers. Mitigation: add `@property` accessors or `__getattr__` fallback.

4. **Test isolation** — Registry is a singleton; tests may interfere. Mitigation: provide `registry.reset()` for tests, or use a fixture that clears registrations.

---

## Files Modified (estimated)

| File | Change type |
|------|------------|
| `src/features/registry.py` | **New** — AnalyzerRegistry, AnalyzerRegistration |
| `src/features/base_analyzer.py` | Add metadata class attributes |
| `src/features/shift_analysis.py` | Add metadata + analyze_from_channels + register |
| `src/features/lap_analysis.py` | Add metadata + analyze_from_channels + register |
| `src/features/gear_analysis.py` | Add metadata + analyze_from_channels + register |
| `src/features/power_analysis.py` | Add metadata + analyze_from_channels + register |
| `src/features/gg_analysis.py` | Add metadata + analyze_from_channels + register |
| `src/features/corner_analysis.py` | Add metadata + analyze_from_channels + register |
| `src/features/session_report.py` | Replace hardcoded analyzers with registry loop |
| `src/features/__init__.py` | Import registry, trigger auto-registration |
| `tests/test_analyzer_registry.py` | **New** — registry tests |
