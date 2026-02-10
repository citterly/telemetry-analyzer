# Telemetry Analyzer - Codebase Cleanup Audit

**Date**: 2025-02-10
**Scope**: Full codebase review - silent failures, duplication, architecture, API quality
**Files**: 44 Python source files, ~8,480 lines in features/analysis layer

---

## Executive Summary

The codebase has grown organically through AI-assisted development and works end-to-end (28/28 features passing, 674 tests passing), but carries significant technical debt. The core problems fall into four categories:

1. **Fail-open error handling** - Missing data silently becomes zeros, producing plausible-looking but wrong analysis
2. **Massive duplication** - The same helpers (`_find_column`, `_safe_float`, speed conversion) are copy-pasted across 8-9 files
3. **No shared abstractions** - Each analyzer has its own interface, its own parquet loading, its own column discovery
4. **God object** - `app.py` is 1,776 lines with 49 endpoints, all concerns mixed together

---

## Part 1: Silent Failures & Bad Error Handling

### CRITICAL: Zero Defaults for Missing GPS Data

**Files**: `src/main/app.py` lines 706-714

```python
if lat_data is None:
    lat_data = np.zeros(len(time_data))  # Session located at 0,0 (Gulf of Guinea)
if lon_data is None:
    lon_data = np.zeros(len(time_data))
```

Missing GPS silently becomes (0, 0). Track identification returns nothing, lap detection uses default config, laps are split incorrectly. The API returns 200 with garbage. **This is the most dangerous pattern in the codebase** - corrupted analysis looks valid.

Same pattern for RPM and speed: engine appears "off" for entire session, all gear calculations break.

### CRITICAL: Bare `except:` Catches KeyboardInterrupt

**File**: `src/io/file_manager.py` line 365

```python
try:
    date_str = filename_parts[0]
    metadata.session_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
except:
    pass
```

Catches `SystemExit`, `KeyboardInterrupt`, `GeneratorExit`. User can't Ctrl+C out of the program during this code path.

### CRITICAL: NaN Array Length Desync

**File**: `src/features/gg_analysis.py` lines 245-250

```python
valid_mask = ~(np.isnan(lat_acc_data) | np.isnan(lon_acc_data))
lat_acc = lat_acc_data[valid_mask]
lon_acc = lon_acc_data[valid_mask]
time = time_data[valid_mask]  # Now different length from speed_data, throttle_data, etc.
```

After filtering NaN, `time`, `lat_acc`, `lon_acc` have fewer elements than `speed_data` or `throttle_data`. Later indexing desynchronizes arrays.

### HIGH: Speed Unit Heuristic at Fragile Boundary

**File**: `src/session/validator.py` lines 217-229

```python
if max_speed < 100:
    return "m/s"  # 99 mph session misclassified as m/s
```

A 99 mph session gets multiplied by 2.237 (becomes 221 mph). A 50 m/s session (112 mph) gets treated as mph. The threshold is arbitrary and doesn't account for edge cases.

### HIGH: Broad Exception Swallowing

**Files**: Multiple locations

| File | Line | Pattern | Impact |
|------|------|---------|--------|
| `extraction/windows_service.py` | 72-82 | `except Exception: return False` | DLL setup fails silently |
| `extraction/windows_service.py` | 249 | `except Exception: pass` | Temp file cleanup fails silently |
| `io/file_manager.py` | 321 | `except Exception: print(...)` | Metadata loading fails, file unindexed |
| `session/importer.py` | 173 | `except ValueError: pass` | Invalid enum silently becomes None |
| `main/app.py` | 1020 | `except Exception: pass` | Lap filtering fails, wrong laps shown |

### HIGH: DLL Pointer Garbage Fallback

**File**: `src/session/session_builder.py` line 244

```python
except Exception:
    return str(ptr)  # Returns "0x7fff1234" as channel name
```

Garbled DLL pointers become memory addresses as channel names. Calibration channels become unidentifiable.

### MEDIUM: Job Status Race Condition

**File**: `src/extraction/worker.py` lines 140-172

If `mark_completed()` fails after extraction succeeds, the job stays in PROCESSING forever. No error handling on the error handler.

### MEDIUM: Interpolation Without Range Check

**File**: `src/extract/data_loader.py` lines 278-280

`np.interp` silently extrapolates beyond input range, creating phantom RPM values where no data existed.

### MEDIUM: Auto-Created Metadata with Dummy Values

**File**: `src/io/file_manager.py` lines 139-164

Auto-creating FileMetadata entries with `file_size_bytes=0` and `file_hash=""` breaks deduplication (empty hash always matches).

---

## Part 2: Code Duplication

### CRITICAL: `_find_column()` Duplicated in 9 Files

The exact same column-finding logic exists independently in:

1. `src/main/app.py`
2. `src/features/lap_analysis.py`
3. `src/features/gear_analysis.py`
4. `src/features/gg_analysis.py`
5. `src/features/power_analysis.py`
6. `src/features/corner_analysis.py`
7. `src/features/corner_detection.py`
8. `src/features/session_report.py`
9. `src/session/validator.py`

**Fix**: Extract to `src/utils/dataframe_helpers.py`, import everywhere.

### HIGH: Speed Conversion Magic Number `2.237` in 9+ Places

The m/s-to-mph conversion factor appears as a bare literal:

- `app.py` lines 540, 599, 639, 674, 716, 786, 880, 984, 1146
- `features/lap_analysis.py` line 190
- Various feature modules

**Fix**: Define `SPEED_MS_TO_MPH = 2.237` constant and `ensure_speed_mph()` helper.

### HIGH: `analyze_from_parquet` Boilerplate in 6 Feature Classes

Every analyzer repeats:
```python
def analyze_from_parquet(self, path):
    df = pd.read_parquet(path)
    # find columns with slightly different names
    # convert speed if needed
    # call analyze_from_arrays(...)
```

With slightly different column name lists each time.

### MEDIUM: NaN Handling Duplicated

- `_safe_float()` in `lap_analysis.py` (individual floats)
- `_sanitize_for_json()` in `app.py` (recursive, handles numpy types)

Two different approaches to the same problem. Some endpoints use the sanitizer, others don't.

### MEDIUM: Duplicate Imports in `data_loader.py`

**File**: `src/extract/data_loader.py` lines 6-24

The exact same import block appears twice (copy-paste artifact):
```python
import ctypes
from ctypes import c_char_p, c_int, c_double, POINTER
import numpy as np
# ... repeated at lines 16-24
```

### LOW: Redundant Corner Detection Wrappers

`detect_corners_simple()` just calls `detect_corners_from_parquet()` which just calls `CornerDetector().detect_from_parquet()`. Three layers for one operation.

### LOW: Dead Code

- `src/analysis/acceleration_analyzer.py` (192 lines) - not imported by anything
- `src/analysis/xrk_metadata_analyzer.py` (1,300+ lines) - standalone utility, not integrated

---

## Part 3: Architecture & Consistency

### CRITICAL: Duplicate LapClassification Enum with Conflicting Values

**File 1**: `src/session/models.py`
```python
class LapClassification(str, Enum):
    OUT_LAP = "out_lap"
    IN_LAP = "in_lap"
    WARM_UP = "warm_up"
    COOL_DOWN = "cool_down"
    HOT_LAP = "hot_lap"
    NORMAL = "normal"
```

**File 2**: `src/features/lap_analysis.py`
```python
class LapClassification(str, Enum):
    HOT_LAP = "hot_lap"
    RACE_PACE = "race_pace"
    WARM_UP = "warm_up"
    COOL_DOWN = "cool_down"
    OUT_LAP = "out_lap"
    INCOMPLETE = "incomplete"
```

Different enum members (`RACE_PACE` vs `NORMAL`, `IN_LAP` vs `INCOMPLETE`). Code importing from one location will break when receiving values from the other.

### HIGH: God Object `app.py` (1,776 lines, 49 endpoints)

Mixed concerns:
- File management (upload, process, list, delete)
- Parquet viewing
- 6 different analysis types
- Visualization (SVG rendering)
- Queue management
- Session database v2 API
- Vehicle configuration
- Health check

Helper functions defined at the END (line 1304+) but used throughout.

### HIGH: No Common Analyzer Interface

| Class | Module | Method | Input |
|-------|--------|--------|-------|
| `ShiftAnalyzer` | shift_analysis.py | `analyze_session()` | Arrays |
| `GGAnalyzer` | gg_analysis.py | `analyze_from_parquet()` | File path |
| `CornerAnalyzer` | corner_analysis.py | `analyze_from_parquet()` | File path |
| `GearAnalysis` | gear_analysis.py | `analyze_from_arrays()` | Arrays |
| `LapAnalysis` | lap_analysis.py | `analyze_from_arrays()` | Arrays |
| `PowerAnalysis` | power_analysis.py | `analyze_from_arrays()` | Arrays |

No common base class. Inconsistent naming (`GearAnalysis` vs `ShiftAnalyzer`). Each has different method signatures.

### HIGH: Competing Vehicle Configuration Systems

**File 1**: `src/config/vehicle_config.py` - Module-level dicts (`CURRENT_SETUP`, `TRANSMISSION_SCENARIOS`)
**File 2**: `src/config/vehicles.py` - OOP approach (`Vehicle`, `VehicleDatabase`, dataclasses)

Both are actively imported by different parts of the codebase. They define overlapping data in incompatible formats.

### MEDIUM: Config Scattered Across 4 Files

| File | Purpose |
|------|---------|
| `src/config/config.py` | Flask/paths/upload limits |
| `src/config/vehicle_config.py` | Analysis parameters (legacy dicts) |
| `src/config/vehicles.py` | Vehicle specs (OOP) |
| `src/config/tracks.py` | Track definitions |

No unified configuration layer.

### MEDIUM: Inconsistent Data Flow

Some endpoints read parquet directly, others go through `SessionImporter`, others through `FileManager`. Three different entry points to the same data.

### MEDIUM: `extract/` vs `extraction/` Directories

Two separate directories with confusing names:
- `src/extract/` - Core data loading (XRKDataLoader)
- `src/extraction/` - Job queue and worker management

### LOW: Inconsistent Naming Throughout

| Concept | Names Used |
|---------|-----------|
| A loaded file | `session_data`, `file`, `parquet`, `xrk_filename`, `parquet_path` |
| Analysis result | `report`, `result`, `analysis_result`, `data` |
| Lap type | `session_type`, `classification`, `import_status` |

---

## Part 4: app.py Endpoint Quality

### HIGH: Parquet Loading Boilerplate Repeated 6-9 Times

Every analysis endpoint repeats:
```python
file_path = _find_parquet_file(filename)
if not file_path:
    raise HTTPException(...)
df = pd.read_parquet(file_path)
time_data = df.index.values
rpm_data = _find_column(df, ['RPM', 'rpm'])
speed_data = _find_column(df, ['GPS Speed', 'speed', 'Speed'])
if speed_data.max() < 100:
    speed_data = speed_data * 2.237
```

Should be a FastAPI `Depends()` or shared loader.

### HIGH: `dtype` String Comparison Bugs

**Lines 417-418, 469, 477**:
```python
if df[col].dtype in ['float64', 'int64']:  # WRONG - dtype is numpy type, not string
```

Should be `df[col].dtype.kind in 'fi'` or compare against `np.float64`.

### HIGH: Potential Null Dereference on Speed Data

**Lines 539, 598, 638, 673, 715, 785, 879, 983, 1145**:
```python
if speed_data.max() < 100:  # AttributeError if speed_data is None
```

Some paths check for None first, others don't.

### MEDIUM: Inconsistent Response Formats

- Some endpoints use `_sanitize_for_json()`, others don't
- Track map returns HTMLResponse, JSONResponse, or raw SVG depending on parameters
- No standard envelope format

### MEDIUM: Missing Bounds Validation

- `segments: int = 10` - no check for 0, negative, or absurdly large values
- `limit: int = 100` - no upper bound
- No validation that `low_threshold < high_threshold`

### LOW: Inline Imports

Pandas and numpy are imported inside almost every endpoint function (acceptable lazy-loading pattern), but `Path` and `compare_laps_detailed` are imported twice (line 12 + 334, line 511 + 754).

---

## Prioritized Cleanup Roadmap

### Phase 1: Safety-Critical Fixes (1-2 sessions)

These are bugs that produce wrong results or crash:

| # | Fix | Files | Effort |
|---|-----|-------|--------|
| 1.1 | Replace zero-defaults with proper error responses when GPS/RPM/speed missing | app.py | Low |
| 1.2 | Fix bare `except:` to `except Exception:` | file_manager.py | Trivial |
| 1.3 | Fix NaN array length desync in GG analysis | gg_analysis.py | Low |
| 1.4 | Fix `dtype` string comparison bugs | app.py lines 417, 469, 477 | Trivial |
| 1.5 | Add null check before `speed_data.max()` in all 9 locations | app.py | Low |
| 1.6 | Fix DLL pointer fallback to raise instead of returning garbage | session_builder.py | Low |

### Phase 2: Extract Shared Utilities (1-2 sessions)

Consolidate the most-duplicated code:

| # | Fix | Impact |
|---|-----|--------|
| 2.1 | Create `src/utils/dataframe_helpers.py` with `find_column()` | Removes duplication from 9 files |
| 2.2 | Add `SPEED_MS_TO_MPH` constant and `ensure_speed_mph()` helper | Removes 9 magic numbers |
| 2.3 | Create shared `sanitize_for_json()` utility | Unifies NaN handling |
| 2.4 | Create column name templates dict (`KNOWN_COLUMNS`) | Centralizes column discovery |
| 2.5 | Remove duplicate imports in `data_loader.py` | Trivial cleanup |
| 2.6 | Delete dead code (`acceleration_analyzer.py`) | Remove 192 lines |

### Phase 3: Consolidate Enums & Config (1 session)

| # | Fix | Impact |
|---|-----|--------|
| 3.1 | Single `LapClassification` source of truth (in `session/models.py`) | Prevents data corruption |
| 3.2 | Deprecate `vehicle_config.py`, route everything through `vehicles.py` | Single config system |
| 3.3 | Merge `extract/` and `extraction/` directories | Clear structure |

### Phase 4: Break Up app.py (2-3 sessions)

| # | Fix | Impact |
|---|-----|--------|
| 4.1 | Create `SessionDataLoader` class (parquet + column finding + speed conversion) | Eliminates boilerplate from all endpoints |
| 4.2 | Create FastAPI `Depends()` for file resolution | Removes 11 identical if-checks |
| 4.3 | Split into routers: `api/files.py`, `api/analysis.py`, `api/visualization.py`, `api/sessions.py`, `api/vehicles.py` | Manageable file sizes |
| 4.4 | Move helper functions to top of file (interim) or into utility modules | Better readability |

### Phase 5: Common Analyzer Interface (1-2 sessions)

| # | Fix | Impact |
|---|-----|--------|
| 5.1 | Create `BaseAnalyzer` ABC with standard `analyze()` method | Consistent interface |
| 5.2 | Rename `GearAnalysis` -> `GearAnalyzer`, `LapAnalysis` -> `LapAnalyzer` (feature version) | Consistent naming |
| 5.3 | Standardize report `to_dict()` with common base | Consistent serialization |
| 5.4 | Create `SessionService` facade for DB access | Decouple API from DB schema |

---

## Quick Wins (Can Do Right Now)

These are trivial fixes that improve safety immediately:

1. `except:` -> `except Exception:` in `file_manager.py:365`
2. Delete `src/analysis/acceleration_analyzer.py` (dead code)
3. Remove duplicate imports in `src/extract/data_loader.py:16-24`
4. Fix dtype comparison: `df[col].dtype in ['float64', 'int64']` -> `df[col].dtype.kind in 'fi'`
5. Add `if speed_data is not None:` guard before `.max()` calls

---

## Issue Count Summary

| Category | Critical | High | Medium | Low |
|----------|----------|------|--------|-----|
| Silent Failures | 3 | 4 | 3 | 0 |
| Code Duplication | 1 | 2 | 2 | 3 |
| Architecture | 2 | 3 | 3 | 1 |
| API Quality | 0 | 3 | 2 | 1 |
| **Total** | **6** | **12** | **10** | **5** |
