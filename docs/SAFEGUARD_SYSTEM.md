# Safeguard System: Calculation Traceability & Sanity Checks

**Date**: 2026-02-10
**Scope**: Add opt-in traceability and automatic sanity checks to all 7 analyzers
**Principle**: Every number shown to the user should be traceable back to its inputs, and every calculation should check itself against physical reality.

---

## Motivation

The telemetry analyzer produces numbers that drive real-world decisions: shift points, braking zones, power estimates, corner speeds. A wrong number (e.g., 800 HP from a Miata, 3.0g lateral from street tires) could lead a driver to trust analysis that doesn't match their car.

Current problems:
1. **No traceability** — When power shows 450 HP, there's no way to see that it used mass=1565 kg, detected speed in m/s, and applied `P = m × a × v`
2. **No self-validation** — If vehicle config says max_lateral_g=1.2 but analysis finds 2.1g, nothing flags it
3. **Silent misconfiguration** — Wrong vehicle selected, wrong units assumed, wrong gear ratios — analysis completes successfully with garbage results

The safeguard system solves this with two mechanisms:
- **CalculationTrace**: Records what inputs, constants, and intermediate values produced each result
- **SanityCheck**: Validates results against physical constraints and vehicle configuration

Both are **opt-in** via `include_trace=True` — zero overhead in normal operation.

---

## Architecture

### Core Classes (`src/utils/calculation_trace.py` — NEW)

```python
@dataclass
class SanityCheck:
    """A single validation check on a calculation result."""
    name: str                    # e.g. "power_plausibility"
    status: str                  # "pass", "warn", "fail"
    message: str                 # Human-readable explanation
    expected: Optional[Any]      # What we expected (from config, physics)
    actual: Optional[Any]        # What we got
    severity: str = "warning"    # "info", "warning", "error"

    def to_dict(self) -> dict: ...


@dataclass
class CalculationTrace:
    """Records the full trace of a calculation for debugging."""
    analyzer_name: str           # e.g. "PowerAnalysis"
    timestamp: str               # ISO 8601 when analysis ran
    inputs: Dict[str, Any]       # Column names used, sample counts, units detected
    config: Dict[str, Any]       # Vehicle mass, gear ratios, thresholds applied
    intermediates: Dict[str, Any]  # Key intermediate values (not exhaustive)
    sanity_checks: List[SanityCheck]
    warnings: List[str]          # Non-check warnings (e.g. "RPM column missing, skipped")

    def to_dict(self) -> dict: ...

    @property
    def has_failures(self) -> bool:
        return any(c.status == "fail" for c in self.sanity_checks)

    @property
    def has_warnings(self) -> bool:
        return any(c.status == "warn" for c in self.sanity_checks)

    def add_check(self, name, status, message, expected=None, actual=None, severity="warning"):
        """Convenience method to append a SanityCheck."""
        self.sanity_checks.append(SanityCheck(
            name=name, status=status, message=message,
            expected=expected, actual=actual, severity=severity
        ))

    def record_input(self, key, value):
        """Record an input used in the calculation."""
        self.inputs[key] = value

    def record_config(self, key, value):
        """Record a config value applied."""
        self.config[key] = value

    def record_intermediate(self, key, value):
        """Record a key intermediate value."""
        self.intermediates[key] = value
```

### BaseAnalyzer Updates (`src/features/base_analyzer.py`)

```python
class BaseAnalysisReport:
    """Updated to support optional trace field."""

    trace: Optional[CalculationTrace] = None  # Added field

    def to_dict(self) -> dict:
        raise NotImplementedError

    def _trace_dict(self) -> dict:
        """Helper for subclasses: returns trace dict if present, empty dict if not."""
        if self.trace is not None:
            return {"_trace": self.trace.to_dict()}
        return {}


class BaseAnalyzer(ABC):
    @abstractmethod
    def analyze_from_parquet(
        self,
        parquet_path: str,
        session_id: Optional[str] = None,
        include_trace: bool = False,     # NEW parameter
        **kwargs,
    ) -> BaseAnalysisReport: ...

    def _create_trace(self, analyzer_name: str) -> CalculationTrace:
        """Create a new trace object. Called by subclasses when include_trace=True."""
        return CalculationTrace(
            analyzer_name=analyzer_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            inputs={},
            config={},
            intermediates={},
            sanity_checks=[],
            warnings=[],
        )
```

### Integration Pattern

Each analyzer follows this pattern:

```python
class SomeAnalyzer(BaseAnalyzer):
    def analyze_from_parquet(self, parquet_path, session_id=None, include_trace=False, **kwargs):
        trace = self._create_trace("SomeAnalyzer") if include_trace else None

        # 1. Load data — record inputs
        df = pd.read_parquet(parquet_path)
        speed_data = find_column(df, ['GPS Speed', 'speed'])
        if trace:
            trace.record_input("speed_column", "GPS Speed")
            trace.record_input("speed_samples", len(speed_data))

        # 2. Apply config — record what was used
        if trace:
            trace.record_config("vehicle_mass_kg", self.vehicle_mass_kg)

        # 3. Calculate — record key intermediates
        power = mass * accel * velocity
        if trace:
            trace.record_intermediate("max_raw_power_watts", float(np.max(power)))

        # 4. Run sanity checks
        if trace:
            self._run_sanity_checks(trace, result)

        # 5. Attach trace to report
        report = SomeReport(...)
        report.trace = trace
        return report
```

### API Integration

Endpoints accept `?trace=true` query parameter:

```python
@router.get("/api/analyze/power/{filename}")
async def analyze_power(filename: str, trace: bool = False):
    analyzer = PowerAnalysis()
    report = analyzer.analyze_from_parquet(path, include_trace=trace)
    result = report.to_dict()  # trace included automatically if present
    return JSONResponse(result)
```

When `trace=false` (default): response is identical to current behavior.
When `trace=true`: response includes `_trace` key with full trace object.

---

## Phase Breakdown

### Phase 1: Trace Infrastructure (safeguard-001)

**Files modified:**
- `src/utils/calculation_trace.py` — **NEW**: `CalculationTrace`, `SanityCheck` dataclasses
- `src/features/base_analyzer.py` — Add `include_trace` parameter, `_create_trace()` helper, trace field on `BaseAnalysisReport`

**Files created:**
- `tests/test_calculation_trace.py` — Unit tests for trace infrastructure

**Deliverables:**
1. `SanityCheck` dataclass with `to_dict()` serialization
2. `CalculationTrace` dataclass with convenience methods (`add_check`, `record_input`, `record_config`, `record_intermediate`)
3. `has_failures` and `has_warnings` properties
4. `BaseAnalysisReport.trace` optional field with `_trace_dict()` helper
5. `BaseAnalyzer.analyze_from_parquet()` gains `include_trace=False` parameter
6. `BaseAnalyzer._create_trace()` factory method

**Tests:**
- SanityCheck creation and serialization
- CalculationTrace creation, recording, serialization
- has_failures / has_warnings properties
- BaseAnalysisReport._trace_dict() with and without trace
- Backward compatibility: all existing tests pass unchanged

**Constraint:** The `trace` field on `BaseAnalysisReport` must not break `@dataclass` subclasses. Implementation: add as a plain attribute (not dataclass field) via `__init_subclass__` or set in `__post_init__`, OR document that subclasses add `trace: Optional[CalculationTrace] = field(default=None, repr=False)` to their own dataclass fields.

---

### Phase 2: PowerAnalysis Trace + Checks (safeguard-002)

**Reference implementation** — establishes the pattern all other analyzers follow.

**File modified:** `src/features/power_analysis.py`

**Trace inputs recorded:**
| Key | Value | Why |
|-----|-------|-----|
| `speed_column` | Column name found | Verify correct channel used |
| `speed_unit_detected` | `"m/s"` or `"mph"` | Speed unit heuristic is fragile |
| `speed_max_raw` | Max before conversion | Diagnose unit detection |
| `rpm_column` | Column name found (or `null`) | RPM is optional here |
| `sample_count` | Number of data points | Verify sufficient data |
| `dt_mean` | Mean time step | Verify sample rate |

**Config recorded:**
| Key | Value |
|-----|-------|
| `vehicle_mass_kg` | Mass used in `P = mav` |
| `power_band_min_rpm` | From ENGINE_SPECS |
| `power_band_max_rpm` | From ENGINE_SPECS |
| `safe_rpm_limit` | From ENGINE_SPECS |
| `smoothing_window` | Savitzky-Golay window size (11) |
| `smoothing_polyorder` | Savitzky-Golay poly order (3) |

**Key intermediates recorded:**
| Key | Value |
|-----|-------|
| `max_raw_accel_g` | Before event filtering |
| `max_raw_power_hp` | Before event filtering |
| `accel_event_count` | Number of acceleration events detected |
| `braking_event_count` | Number of braking events detected |
| `pct_in_power_band` | % of RPM samples in power band |

**Sanity checks (5):**

| # | Name | Logic | Status |
|---|------|-------|--------|
| 2.1 | `mass_matches_config` | `vehicle_mass_kg` within 10% of active vehicle's weight_lbs/2.205 | pass/warn |
| 2.2 | `max_power_plausible` | max_power_hp < 800 (no street car exceeds this) | pass/fail |
| 2.3 | `power_weight_ratio` | max_hp / mass_kg < 0.5 HP/kg (= ~370 HP/ton, supercar territory) | pass/warn |
| 2.4 | `speed_unit_confidence` | If max raw speed is 45-110 (ambiguous zone), warn | pass/warn |
| 2.5 | `sufficient_data` | sample_count > 100 and duration > 10s | pass/fail |

**Tests:**
- Trace recorded when `include_trace=True`
- Trace NOT recorded when `include_trace=False` (default)
- All 5 sanity checks trigger correctly (provide edge-case test data)
- `to_dict()` includes `_trace` when present, omits when absent
- Existing PowerAnalysis tests still pass unchanged

---

### Phase 3: ShiftAnalyzer Trace + Checks (safeguard-003)

**File modified:** `src/features/shift_analysis.py`

**Trace inputs recorded:**
| Key | Value |
|-----|-------|
| `rpm_column` | Column name found |
| `speed_column` | Column name found |
| `speed_unit_detected` | `"m/s"` or `"mph"` |
| `sample_count` | Data points |
| `shift_count` | Number of shifts detected |

**Config recorded:**
| Key | Value |
|-----|-------|
| `transmission_ratios` | Gear ratios used |
| `final_drive` | Final drive ratio |
| `optimal_shift_rpm_min` | 6000 |
| `optimal_shift_rpm_max` | 6800 |
| `early_shift_rpm` | 5500 |
| `over_rev_rpm` | 7200 |

**Key intermediates recorded:**
| Key | Value |
|-----|-------|
| `gears_detected` | Unique gear numbers found |
| `shifts_per_gear` | Count of shifts per gear pair |
| `pct_optimal` | % of shifts in optimal range |
| `pct_early` | % early shifts |
| `pct_over_rev` | % over-rev shifts |

**Sanity checks (4):**

| # | Name | Logic | Status |
|---|------|-------|--------|
| 3.1 | `gear_count_matches_config` | Number of gears detected ≤ number of ratios in config | pass/warn |
| 3.2 | `shift_rpm_below_redline` | No shift RPM > safe_rpm_limit × 1.05 (allow 5% sensor noise) | pass/fail |
| 3.3 | `shift_confidence` | Mean gear confidence > 0.5 (from GearCalculator) | pass/warn |
| 3.4 | `sufficient_shifts` | At least 3 shifts detected for meaningful analysis | pass/warn |

---

### Phase 4: GGAnalyzer Trace + Checks (safeguard-004)

**File modified:** `src/features/gg_analysis.py`

**Trace inputs recorded:**
| Key | Value |
|-----|-------|
| `lat_acc_column` | Column name |
| `lon_acc_column` | Column name |
| `speed_column` | Column name |
| `throttle_column` | Column name (or `null`) |
| `sample_count` | After NaN filtering |
| `nan_pct` | % of samples dropped as NaN |
| `lap_filter` | Which lap was filtered (if any) |

**Config recorded:**
| Key | Value |
|-----|-------|
| `max_g_reference` | Lateral G reference |
| `max_braking_g` | Braking G reference |
| `power_limited_accel_g` | Power-limited threshold |
| `vehicle_name` | Active vehicle name |

**Key intermediates recorded:**
| Key | Value |
|-----|-------|
| `max_combined_g` | Peak combined G-force |
| `p95_combined_g` | 95th percentile |
| `utilization_pct` | Overall grip utilization |
| `corner_utilization_pct` | In-corner grip utilization |
| `braking_to_lateral_ratio` | Brake/lateral balance |

**Sanity checks (4):**

| # | Name | Logic | Status |
|---|------|-------|--------|
| 4.1 | `config_matches_vehicle` | max_g_reference matches active vehicle's max_lateral_g | pass/warn |
| 4.2 | `data_quality` | nan_pct < 20% (too many NaN = unreliable accelerometer) | pass/warn/fail |
| 4.3 | `g_force_plausible` | max_combined_g < 3.0 (no street tire exceeds this) | pass/fail |
| 4.4 | `utilization_plausible` | utilization_pct between 10% and 100% | pass/warn |

---

### Phase 5: LapAnalysis Trace + Checks (safeguard-005)

**File modified:** `src/features/lap_analysis.py`

**Trace inputs recorded:**
| Key | Value |
|-----|-------|
| `latitude_column` | Column name |
| `longitude_column` | Column name |
| `speed_column` | Column name |
| `speed_unit_detected` | `"m/s"` or `"mph"` |
| `sample_count` | Data points |
| `track_detected` | Track name from auto-detection |

**Config recorded:**
| Key | Value |
|-----|-------|
| `track_name` | TRACK_CONFIG name |
| `start_finish_gps` | GPS coords used for lap detection |
| `classification_thresholds` | HOT_LAP gap, WARM_UP criteria, etc. |

**Key intermediates recorded:**
| Key | Value |
|-----|-------|
| `laps_detected` | Number of laps |
| `fastest_lap_time` | In seconds |
| `slowest_lap_time` | In seconds |
| `avg_lap_time` | In seconds |
| `estimated_lap_distance_miles` | From speed integration |

**Sanity checks (4):**

| # | Name | Logic | Status |
|---|------|-------|--------|
| 5.1 | `speed_unit_consistent` | If speed was converted, verify max speed after conversion is 30-200 mph | pass/warn |
| 5.2 | `lap_distance_plausible` | Estimated lap distance within 20% of known track length (if track detected) | pass/warn |
| 5.3 | `lap_time_plausible` | Fastest lap > 30s (no road course has sub-30s laps) and slowest < 600s | pass/warn |
| 5.4 | `gps_coordinates_valid` | Lat in [-90, 90], Lon in [-180, 180], not all zeros | pass/fail |

---

### Phase 6: GearAnalysis + CornerAnalyzer Trace + Checks (safeguard-006)

Two smaller analyzers combined into one phase.

#### GearAnalysis (`src/features/gear_analysis.py`)

**Trace inputs recorded:**
| Key | Value |
|-----|-------|
| `rpm_column` | Column name |
| `speed_column` | Column name |
| `speed_unit_detected` | `"m/s"` or `"mph"` |
| `sample_count` | Data points |

**Config recorded:**
| Key | Value |
|-----|-------|
| `transmission_ratios` | Ratios used |
| `final_drive` | Final drive |
| `safe_rpm_limit` | From ENGINE_SPECS |
| `power_band_min` | From ENGINE_SPECS |
| `power_band_max` | From ENGINE_SPECS |
| `track_name` | If section analysis used |

**Key intermediates recorded:**
| Key | Value |
|-----|-------|
| `gears_detected` | Unique gears found |
| `pct_over_safe_limit` | % time over RPM limit |
| `pct_in_power_band` | % time in power band |
| `total_shifts` | Number of gear changes |

**Sanity checks (3):**

| # | Name | Logic | Status |
|---|------|-------|--------|
| 6.1 | `gear_count_matches_config` | Detected gears ≤ transmission ratio count | pass/warn |
| 6.2 | `rpm_data_sufficient` | > 50% of samples have RPM > 1000 (engine running) | pass/warn |
| 6.3 | `gear_usage_balanced` | No single gear > 80% usage (suggests stuck detection or highway session) | pass/warn |

#### CornerAnalyzer (`src/features/corner_analysis.py`)

**Trace inputs recorded:**
| Key | Value |
|-----|-------|
| `speed_column` | Column name |
| `lat_column` | Column name |
| `lon_column` | Column name |
| `throttle_column` | Column name (or `null`) |
| `sample_count` | Data points |
| `corners_detected` | From CornerDetector |

**Config recorded:**
| Key | Value |
|-----|-------|
| `throttle_pickup_threshold` | 10% |
| `lift_threshold` | 5% |
| `trail_brake_threshold` | -0.15g |
| `track_name` | If provided |

**Key intermediates recorded:**
| Key | Value |
|-----|-------|
| `corner_count` | Number of corners analyzed |
| `avg_entry_speed` | Mean entry speed |
| `avg_min_speed` | Mean apex speed |
| `lift_count` | Corners with detected lift |
| `trail_brake_count` | Corners with trail braking |

**Sanity checks (3):**

| # | Name | Logic | Status |
|---|------|-------|--------|
| 6.4 | `gps_quality` | GPS lat/lon not all zeros, reasonable variance (std > 0.0001 degrees) | pass/fail |
| 6.5 | `corner_count_plausible` | 3-50 corners per lap (no road course has fewer/more) | pass/warn |
| 6.6 | `corner_speeds_plausible` | All apex speeds > 5 mph and < 180 mph | pass/warn |

---

### Phase 7: SessionReport Aggregation + API (safeguard-007)

**Files modified:**
- `src/features/session_report.py` — Aggregate traces from sub-analyzers
- `src/main/routers/analysis.py` — Add `trace: bool = False` parameter to all endpoints
- `src/main/routers/visualization.py` — Add `trace: bool = False` parameter to GG and corner endpoints

#### SessionReport Aggregation

`SessionReportGenerator` already calls PowerAnalysis, LapAnalysis, ShiftAnalyzer, and GearAnalysis. When `include_trace=True`, it:

1. Passes `include_trace=True` to each sub-analyzer
2. Collects all traces into `report.traces: List[CalculationTrace]`
3. Runs cross-validation checks between sub-analyzers

**Cross-validation sanity checks (3):**

| # | Name | Logic | Status |
|---|------|-------|--------|
| 7.1 | `speed_unit_consensus` | All sub-analyzers detected the same speed unit | pass/fail |
| 7.2 | `sample_count_consistent` | All sub-analyzers processed similar sample counts (within 5%) | pass/warn |
| 7.3 | `config_consistent` | Vehicle config values identical across all sub-analyzers | pass/fail |

#### API Endpoint Updates

All analysis endpoints gain `trace: bool = False` query parameter:

| Endpoint | Router file |
|----------|-------------|
| `GET /api/analyze/shifts/{filename}` | `routers/analysis.py` |
| `GET /api/analyze/laps/{filename}` | `routers/analysis.py` |
| `GET /api/analyze/gears/{filename}` | `routers/analysis.py` |
| `GET /api/analyze/power/{filename}` | `routers/analysis.py` |
| `GET /api/analyze/report/{filename}` | `routers/analysis.py` |
| `GET /api/gg-diagram/{filename}` | `routers/visualization.py` |
| `GET /api/corner-analysis/{filename}` | `routers/visualization.py` |

**Tests:**
- `?trace=true` returns `_trace` in response JSON
- `?trace=false` (or omitted) returns identical response to current behavior
- Cross-validation checks fire correctly
- Trace serialization round-trips through JSON

---

### Phase 8: Memory Update (safeguard-008)

Save the safeguard principle to auto-memory as a permanent project premise.

**File modified:** `~/.claude/projects/-home-chris-projects-telemetry-analyzer/memory/MEMORY.md`

**Content to save:**
- Safeguard system principle: all calculations must be traceable
- Pattern: `include_trace=True` opt-in parameter
- Rule: new analyzers must implement trace recording and sanity checks
- Reference: `docs/SAFEGUARD_SYSTEM.md` for full spec

---

## Test Strategy

Each phase has its own test file or test class. Tests verify:

1. **Backward compatibility**: All existing tests pass without modification
2. **Opt-in behavior**: `include_trace=False` produces identical output to pre-safeguard code
3. **Trace completeness**: When enabled, trace contains all documented inputs/config/intermediates
4. **Check correctness**: Each sanity check triggers on known-bad data
5. **Serialization**: `to_dict()` produces valid JSON with trace included

### Test Data Patterns

For sanity check testing, use synthetic data that triggers specific conditions:
- `mass_matches_config`: Pass mass=5000 kg with BMW config (1565 kg)
- `max_power_plausible`: Create speed/accel data that yields > 800 HP
- `g_force_plausible`: Create accel data with 4.0g spikes
- `lap_distance_plausible`: Create laps with avg_speed suggesting 0.5 mile laps at Road America (4.0 miles)
- `speed_unit_consistent`: Create data with max speed of 80 (ambiguous m/s vs mph)

---

## Issue Count Summary

| Phase | New Checks | Files Modified | Files Created |
|-------|-----------|----------------|---------------|
| safeguard-001 | 0 (infrastructure) | 1 | 2 |
| safeguard-002 | 5 | 1 | 0 |
| safeguard-003 | 4 | 1 | 0 |
| safeguard-004 | 4 | 1 | 0 |
| safeguard-005 | 4 | 1 | 0 |
| safeguard-006 | 6 (3+3) | 2 | 0 |
| safeguard-007 | 3 (cross-validation) | 3 | 0 |
| safeguard-008 | 0 (documentation) | 1 | 0 |
| **Total** | **26 checks** | **11 files** | **2 files** |
