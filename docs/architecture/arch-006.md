# arch-006: Audit Mode UI

## Context

The safeguard system (26 sanity checks across 7 analyzers + 3 cross-validation) currently lives behind `?trace=true` API parameters with no web UI. Users must curl the API to see whether their analysis results can be trusted. This feature adds a visual audit layer to all three analysis pages.

## What We're Building

1. **Per-page audit toggle** — A switch on each analysis page that enables trace fetching. State persists in localStorage with a global default.
2. **Traffic light indicators** — Green/yellow/red dots next to analysis section headers showing check status at a glance.
3. **Expandable audit panel** — Full trace details (inputs, config, intermediates, sanity checks with impact descriptions) shown regardless of pass/warn/fail status.
4. **Impact descriptions** — New `impact` field on each of the 26 sanity checks explaining what downstream calculations are affected.

## Key Finding: Full Report Sub-Traces

The full report currently does NOT propagate `include_trace` to sub-analyzers — `_run_registry_analyzers()` calls `analyze_from_channels()` which ignores the trace parameter. This means:
- Individual analysis pages (shifts, laps, power, etc.) show full per-analyzer traces
- Full report page shows only the 3 cross-validation checks

**Decision for v1:** Fix this by passing `include_trace` through `_run_registry_analyzers()` so the full report includes all sub-analyzer traces. This is a small change (2 lines in session_report.py + updating each `analyze_from_channels()` to forward the param).

---

## Implementation Steps

### Step 1: Backend — `impact` field on SanityCheck

**File: `src/utils/calculation_trace.py`**
- Add `impact: str = ""` field to `SanityCheck` dataclass
- Add `"impact"` to `to_dict()` output
- Update `CalculationTrace.add_check()` to accept `impact=` kwarg

### Step 2: Impact descriptions on all 26 checks

**Files: 7 analyzer modules**

Add `impact=` to every `trace.add_check()` call. The 26 descriptions:

| Analyzer | Check | Impact |
|----------|-------|--------|
| Power | `mass_matches_config` | "Vehicle mass is used in the P=mav power equation. A mass error produces a proportional error in all power and acceleration values." |
| Power | `max_power_plausible` | "Maximum power drives the power curve and acceleration events. An implausible value means speed or mass inputs are likely wrong, making all power metrics unreliable." |
| Power | `power_weight_ratio` | "Power-to-weight ratio is used for performance benchmarking. An unusually high ratio suggests mass is too low or power is overestimated." |
| Power | `speed_unit_confidence` | "Speed unit detection affects every downstream calculation: power, acceleration, braking, and corner speeds. A wrong unit makes all speed-derived values off by ~2.2x." |
| Power | `sufficient_data` | "With too few samples or too short a session, statistical measures have high variance and smoothing filters may produce artifacts." |
| Shift | `gear_count_matches_config` | "Gear detection matches RPM/speed ratios to configured transmission ratios. Extra gears mean some data is assigned to nonexistent gears, corrupting shift RPM statistics." |
| Shift | `shift_rpm_below_redline` | "A shift above redline indicates sensor noise, wrong gear assignment, or engine over-rev. Shift quality ratings and optimal shift point recommendations become unreliable." |
| Shift | `shift_confidence` | "Non-adjacent shifts suggest unreliable gear detection. Shift timing, quality ratings, and per-gear RPM statistics may be based on incorrect gear assignments." |
| Shift | `sufficient_shifts` | "Fewer than 3 shifts means insufficient data for meaningful statistical analysis. Shift RPM averages and quality distributions will have high uncertainty." |
| GG | `config_matches_vehicle` | "The G-force reference value determines the friction circle boundary. A wrong reference makes utilization percentages and quadrant breakdowns use the wrong baseline." |
| GG | `data_quality` | "High NaN percentage means many data points were dropped. Remaining points may not represent the full range of driving, biasing utilization and peak G calculations." |
| GG | `g_force_plausible` | "G-forces above 3.0g are physically impossible on street tires and indicate sensor error. All friction circle analysis and utilization percentages become meaningless." |
| GG | `utilization_plausible` | "Utilization outside 10-100% suggests misconfigured reference G, data errors, or non-track driving. Quadrant analysis and improvement recommendations may be misleading." |
| Lap | `speed_unit_consistent` | "Lap analysis uses speed for time validation and distance estimation. A wrong speed unit makes lap distances off by ~2.2x and may cause incorrect lap classifications." |
| Lap | `lap_distance_plausible` | "Estimated lap distance is checked against known track lengths. An implausible distance means GPS is unreliable or start/finish detection is wrong, affecting all per-lap statistics." |
| Lap | `lap_time_plausible` | "Lap times outside 30-600s indicate GPS detection issues. This affects fastest lap identification, lap classification, and all per-lap metrics." |
| Lap | `gps_coordinates_valid` | "All lap detection depends on GPS for start/finish line crossing. Invalid GPS means no laps can be detected and all per-lap metrics are unavailable." |
| Gear | `gear_count_matches_config` | "Gear usage analysis allocates time and RPM to each gear. Extra detected gears mean some data goes to nonexistent gears, corrupting usage percentages and time-in-gear statistics." |
| Gear | `rpm_data_sufficient` | "RPM below 1000 means engine off or idling. If most samples are idle, gear usage statistics are unrepresentative of actual driving behavior." |
| Gear | `gear_usage_balanced` | "One gear dominating over 80% suggests highway driving or stuck gear detection. Per-gear RPM analysis lacks statistical significance for under-represented gears." |
| Corner | `gps_quality` | "Corner detection relies on GPS to calculate path radius and identify turns. Without GPS movement, no corners can be detected and all per-corner metrics are unavailable." |
| Corner | `corner_count_plausible` | "Most road courses have 8-20 corners per lap. A count outside 3-50 suggests detection parameters don't match this track, making corner comparisons unreliable." |
| Corner | `corner_speeds_plausible` | "Corner speeds below 5 or above 180 mph are unlikely for production race cars. Implausible values indicate a speed unit error or incorrect corner detection." |
| Report | `speed_unit_consensus` | "All sub-analyzers receive the same speed data. If the speed unit is wrong, every sub-analysis has proportionally incorrect speed-derived results." |
| Report | `sample_count_consistent` | "Sub-analyzers should process the same dataset. A large difference means some analyzers got differently filtered data, making cross-analysis comparisons unreliable." |
| Report | `config_consistent` | "Track and vehicle config must be identical across sub-analyzers. A mismatch means some results use wrong reference values, producing internally inconsistent reports." |

### Step 3: Propagate trace through full report

**File: `src/features/session_report.py`**
- `generate_from_parquet()`: pass `include_trace` to `_run_registry_analyzers()`
- `_run_registry_analyzers()`: accept `include_trace` param, pass to `analyze_from_channels()`

**Files: 6 analyzer `analyze_from_channels()` methods**
- Each currently ignores `include_trace` — update to create trace, record inputs/config, run sanity checks, and attach to report when enabled. Reuses existing `_run_sanity_checks()` method.

### Step 4: Shared UI module

**New file: `static/js/audit.js`**
- `AuditManager` class:
  - `enabled` getter/setter backed by `localStorage('telemetry_audit_enabled')`
  - `traceParam(prefix)` → `'&trace=true'` or `''`
  - `renderToggleButton()` → Bootstrap switch HTML
  - `getStatus(checks)` → `'green'`/`'yellow'`/`'red'`
  - `renderDot(status)` → colored dot span
  - `renderSectionIndicator(trace)` → dot + summary count
  - `renderPanel(trace)` → full expandable panel HTML with inputs/config/intermediates/checks tables
- Global instance: `window._auditManager = new AuditManager()`
- Page callback pattern: pages define `window.onAuditToggle(enabled)` to re-fetch

**New file: `static/css/audit.css`**
- `.audit-dot` — 10px colored circle
- `.audit-toggle` — switch styling
- `.audit-panel` — expandable card with colored left border (green/yellow/red)
- `.audit-panel-header` — clickable header with expand arrow
- `.audit-panel-body` — hidden by default, shown when `.expanded`
- `.audit-section-title` — uppercase labels for Inputs/Config/Intermediates/Checks
- `.audit-table` — key/value table styling
- `.audit-check-row` — check row with dot, name, message, impact

Uses existing theme colors: `#00b894` (green), `#fdcb6e` (yellow), `#e17055` (red), `#2d2d2d` (card bg), `#404040` (borders).

### Step 5: Base template integration

**File: `templates/base.html`**
- Add `<link rel="stylesheet" href="/static/css/audit.css">` in `<head>`
- Add `<script src="/static/js/audit.js"></script>` before the custom JS block

### Step 6: Analysis page integration

**File: `templates/analysis.html`**
- Add audit toggle button in the "Select Session" card header area
- Modify `runAnalysis(type)`:
  - Append `window._auditManager.traceParam('?')` to fetch URL
- Modify `displayResults(type, data)`:
  - After rendering analysis, if audit enabled and `data._trace` exists, render audit panel
  - For full report: render cross-validation panel + per-sub-analyzer panels from `data.shift_analysis._trace`, `data.lap_analysis._trace`, etc.
  - For individual analyses: render single audit panel
- Add traffic light dots next to card headers in each `display*()` function
- Define `window.onAuditToggle = (enabled) => { runAnalysis(currentType); }` to re-fetch

### Step 7: GG diagram page integration

**File: `templates/gg_diagram.html`**
- Add audit toggle in control panel area
- Modify `loadGGDiagram()`: append `&trace=true` when enabled
- After loading, render single audit panel below stats cards

### Step 8: Corner analysis page integration

**File: `templates/corner_analysis.html`**
- Add audit toggle in control panel area
- Modify fetch: append `?trace=true` when enabled
- After loading, render single audit panel below stats cards

---

## Files Modified/Created

| File | Action |
|------|--------|
| `src/utils/calculation_trace.py` | MODIFY — add `impact` field |
| `src/features/power_analysis.py` | MODIFY — add impact strings to 5 checks |
| `src/features/shift_analysis.py` | MODIFY — add impact strings to 4 checks |
| `src/features/gg_analysis.py` | MODIFY — add impact strings to 4 checks |
| `src/features/lap_analysis.py` | MODIFY — add impact strings to 4 checks |
| `src/features/gear_analysis.py` | MODIFY — add impact strings to 3 checks |
| `src/features/corner_analysis.py` | MODIFY — add impact strings to 3 checks |
| `src/features/session_report.py` | MODIFY — add impact strings to 3 checks + propagate trace |
| `static/css/audit.css` | CREATE — audit UI styles |
| `static/js/audit.js` | CREATE — AuditManager class |
| `templates/base.html` | MODIFY — include audit.css and audit.js |
| `templates/analysis.html` | MODIFY — toggle, trace fetch, panel rendering |
| `templates/gg_diagram.html` | MODIFY — toggle, trace fetch, panel rendering |
| `templates/corner_analysis.html` | MODIFY — toggle, trace fetch, panel rendering |

## Acceptance Criteria

1. `SanityCheck` has `impact` field; `to_dict()` includes it
2. All 26 checks have non-empty impact descriptions
3. Full report `?trace=true` includes sub-analyzer traces (not just cross-validation)
4. All 3 analysis pages have an audit toggle switch
5. Toggle state persists in localStorage across page loads
6. When audit OFF, no `?trace=true` in fetch requests
7. When audit ON, traffic light dots appear next to section headers
8. When audit ON, expandable audit panels show inputs, config, intermediates, and checks
9. Every check row shows impact description regardless of pass/warn/fail status
10. All existing 1011+ tests pass unchanged

## Verification

1. `pytest tests/ -q` — all tests pass
2. Manual: load analysis page → toggle audit ON → run Full Report → verify cross-validation panel + sub-analyzer panels
3. Manual: run individual Power analysis with audit ON → verify 5 checks with impacts visible
4. Manual: toggle audit OFF → re-run → verify no audit UI, no trace param in network tab
5. Manual: refresh page → verify toggle state preserved
6. Manual: GG diagram + corner analysis pages same pattern
