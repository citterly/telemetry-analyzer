# arch-002: SessionDataLoader Service Layer

**Status**: Plan
**Date**: 2026-02-10
**Scope**: Centralize parquet loading, column discovery, and speed unit conversion into a reusable service

---

## Problem Statement

Every router endpoint and analyzer's `analyze_from_parquet()` repeats the same boilerplate:

1. Load parquet file via `pd.read_parquet(path)`
2. Extract time index from `df.index.values`
3. Find columns by fuzzy name matching (`find_column(df, ['GPS Speed', 'speed', ...])`)
4. Detect speed units and convert m/s → mph
5. Validate required columns exist (raise HTTP 400/422 if missing)

This pattern appears **30+ times** across:
- 5 router endpoints in `analysis.py`
- 6 endpoints in `visualization.py`
- 5 endpoints in `parquet.py`
- 8 analyzer `analyze_from_parquet()` methods

Each implementation is slightly different: different column candidate lists, inconsistent error handling, duplicated speed heuristics. Adding a new data source (CSV, live CAN, different logger) would require changing all 30+ locations.

---

## Current State

### What's already centralized

| Component | Location | What it does |
|-----------|----------|-------------|
| `find_column()` | `utils/dataframe_helpers.py` | Fuzzy column name matching |
| `find_column_name()` | `utils/dataframe_helpers.py` | Returns column name (not values) |
| `KNOWN_COLUMNS` | `utils/dataframe_helpers.py` | Standard candidate lists for 11 channels |
| `ensure_speed_mph()` | `utils/dataframe_helpers.py` | Speed unit detection + conversion |
| `SPEED_MS_TO_MPH` | `utils/dataframe_helpers.py` | Conversion constant |
| `find_parquet_file()` | `main/deps.py` | File path resolution |
| `ParquetValidator` | `session/validator.py` | Comprehensive validation + unit detection |

### What's duplicated

| Pattern | Occurrences | Where |
|---------|------------|-------|
| `pd.read_parquet(path)` | 30+ | All analyzers + all routers |
| Column discovery loops | 20+ | Each endpoint finds its own columns |
| `speed_data.max() < 100` heuristic | 15+ | Inline in routers and analyzers |
| Speed unit conversion | 15+ | Some use `ensure_speed_mph()`, most inline |
| Missing column → HTTP error | 10+ | Repeated per endpoint |
| Session data dict construction | 5+ | `{'time': ..., 'latitude': ..., ...}` |

### Consumers of the new service

| Consumer | What it needs | Currently does |
|----------|--------------|----------------|
| `routers/analysis.py` (5 endpoints) | time, speed, rpm, lat, lon arrays | Inline load + find_column |
| `routers/visualization.py` (6 endpoints) | Same + session_data dict for LapAnalyzer | Inline load + dict construction |
| `ShiftAnalyzer.analyze_from_parquet()` | time, rpm, speed | Inline pd.read_parquet |
| `GearAnalysis.analyze_from_parquet()` | time, rpm, speed, lat, lon | Inline pd.read_parquet |
| `LapAnalysis.analyze_from_parquet()` | time, lat, lon, rpm, speed | Inline pd.read_parquet |
| `PowerAnalysis.analyze_from_parquet()` | time, speed, rpm | Inline pd.read_parquet |
| `GGAnalyzer.analyze_from_parquet()` | lat_acc, lon_acc, speed, lat, lon | Inline pd.read_parquet |
| `CornerAnalyzer.analyze_from_parquet()` | lat, lon, speed, throttle | Inline pd.read_parquet |
| `SessionReportGenerator.generate_from_parquet()` | time, lat, lon, rpm, speed | Inline pd.read_parquet |

---

## Proposed Changes

### New class: `SessionDataLoader`

Location: `src/services/session_data_loader.py`

```python
@dataclass
class SessionChannels:
    """Resolved channel data from a parquet file."""
    time: np.ndarray
    df: pd.DataFrame                    # Full dataframe (for analyzers that need it)
    source_path: str                    # Original file path
    session_id: str                     # Derived from filename
    sample_count: int
    duration_seconds: float
    speed_unit_detected: str            # "mph", "m/s", or "unknown"

    # Core channels (None if not found)
    latitude: Optional[np.ndarray] = None
    longitude: Optional[np.ndarray] = None
    speed_mph: Optional[np.ndarray] = None
    speed_ms: Optional[np.ndarray] = None
    rpm: Optional[np.ndarray] = None
    lat_acc: Optional[np.ndarray] = None
    lon_acc: Optional[np.ndarray] = None
    throttle: Optional[np.ndarray] = None

    # Column name mapping (logical → actual column name found)
    column_map: Dict[str, str] = field(default_factory=dict)


class SessionDataLoader:
    """Unified parquet loading with column discovery and unit conversion."""

    def load(self, parquet_path: str) -> SessionChannels:
        """
        Load a parquet file and resolve all known channels.

        - Reads the parquet file
        - Discovers columns using KNOWN_COLUMNS candidates
        - Detects speed units and provides both mph and m/s
        - Builds column_map for traceability

        Returns:
            SessionChannels with all discovered data
        """

    def load_or_raise(
        self,
        parquet_path: str,
        required: List[str],
    ) -> SessionChannels:
        """
        Load and validate required channels exist.

        Args:
            parquet_path: Path to parquet file
            required: List of logical channel names that must be present
                      e.g., ["speed", "rpm", "latitude", "longitude"]

        Raises:
            ValueError: If any required channel is missing

        Returns:
            SessionChannels with validated data
        """

    def to_session_data_dict(self, channels: SessionChannels) -> Dict:
        """
        Build the legacy session_data dict used by LapAnalyzer.

        Returns dict with keys: time, latitude, longitude, rpm,
        speed_mph, speed_ms (matching existing LapAnalyzer contract).
        """
```

### Integration with BaseAnalyzer

Update `BaseAnalyzer.analyze_from_parquet()` to optionally accept a `SessionDataLoader`:

```python
class BaseAnalyzer(ABC):
    def analyze_from_parquet(
        self,
        parquet_path: str,
        session_id: Optional[str] = None,
        include_trace: bool = False,
        loader: Optional[SessionDataLoader] = None,   # NEW
        channels: Optional[SessionChannels] = None,    # NEW
        **kwargs,
    ) -> BaseAnalysisReport:
        ...
```

When `channels` is provided, skip the internal `pd.read_parquet()` call and use the pre-loaded data. This eliminates redundant file reads when `SessionReportGenerator` runs multiple sub-analyzers on the same file.

### Integration with routers

Replace the inline boilerplate in each router endpoint:

```python
# BEFORE (repeated per endpoint):
df = pd.read_parquet(file_path)
time_data = df.index.values
rpm_data = find_column(df, ['RPM', 'rpm'])
speed_data = find_column(df, ['GPS Speed', 'speed', 'Speed'])
if speed_data is None:
    raise HTTPException(status_code=400, detail="Speed data not found")
if speed_data.max() < 100:
    speed_data = speed_data * SPEED_MS_TO_MPH

# AFTER:
loader = SessionDataLoader()
channels = loader.load_or_raise(str(file_path), required=["speed", "rpm"])
```

### Router helper in deps.py

Add a helper that combines file lookup + loading + HTTP error translation:

```python
def load_session(
    filename: str,
    required: List[str] = None,
) -> SessionChannels:
    """
    Find parquet file, load session, validate required channels.
    Raises HTTPException on failure.
    """
    file_path = find_parquet_file(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")

    loader = SessionDataLoader()
    try:
        if required:
            return loader.load_or_raise(str(file_path), required=required)
        return loader.load(str(file_path))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
```

---

## Migration Strategy

### Phase 1: Create SessionDataLoader (no changes to consumers)

1. Create `src/services/__init__.py`
2. Create `src/services/session_data_loader.py` with `SessionChannels` and `SessionDataLoader`
3. Write comprehensive tests

### Phase 2: Migrate router endpoints

For each router endpoint:
1. Replace inline loading with `load_session()` from deps.py
2. Use `channels.speed_mph`, `channels.rpm`, etc. instead of manual find_column calls
3. Remove inline speed conversion
4. Keep `trace=True` path using `analyze_from_parquet()`

Order (least complex first):
1. `analysis.py` — straightforward channel usage
2. `visualization.py` — needs `to_session_data_dict()` for LapAnalyzer paths
3. `parquet.py` — mostly reads raw df, minimal channel extraction

### Phase 3: Migrate analyzer `analyze_from_parquet()` methods

For each analyzer's `analyze_from_parquet()`:
1. Accept optional `channels: SessionChannels` parameter
2. If channels provided, use them; otherwise load internally (backward compat)
3. This allows `SessionReportGenerator` to load once and pass to sub-analyzers

Order:
1. `PowerAnalysis` (simplest: just speed + rpm)
2. `ShiftAnalyzer` (speed + rpm)
3. `LapAnalysis` (lat + lon + speed + rpm)
4. `GearAnalysis` (rpm + speed + lat + lon)
5. `GGAnalyzer` (lat_acc + lon_acc + speed)
6. `CornerAnalyzer` (lat + lon + speed + throttle)
7. `SessionReportGenerator` (orchestrator — loads once, passes channels)

### Phase 4: Optimize SessionReportGenerator

Currently `SessionReportGenerator.generate_from_parquet()` reads the parquet file once, then each sub-analyzer's `analyze_from_parquet()` reads it again. After Phase 3, the generator can:
1. Load once via `SessionDataLoader`
2. Pass `channels` to each sub-analyzer
3. Eliminate 4+ redundant file reads per report generation

---

## Acceptance Criteria

1. **SessionDataLoader exists** at `src/services/session_data_loader.py`
2. **SessionChannels dataclass** has all core channels (time, lat, lon, speed_mph, speed_ms, rpm, lat_acc, lon_acc, throttle)
3. **Column discovery uses KNOWN_COLUMNS** — single source of truth for channel name candidates
4. **Speed unit detection is centralized** — `speed_unit_detected` field in SessionChannels
5. **Both mph and m/s available** — `speed_mph` and `speed_ms` always populated when speed found
6. **Router endpoints simplified** — `analysis.py` uses `load_session()` instead of inline loading
7. **Backward compatible** — analyzers still work with direct `parquet_path` argument
8. **SessionReportGenerator loads once** — uses single SessionDataLoader call for all sub-analyzers
9. **All existing tests pass** — zero regressions
10. **New tests cover SessionDataLoader** — load, column discovery, speed conversion, missing channel errors, session_data_dict conversion

---

## Risks

1. **Speed unit edge cases** — The `max < 100` heuristic can fail for slow sessions (e.g., pit lane only) where mph values are legitimately below 100. The validator's `df.attrs["units"]` approach is more reliable but only available for data processed by our pipeline. Mitigation: try attrs first, fall back to heuristic.

2. **Memory for large files** — Loading the full DataFrame into `SessionChannels` means it stays in memory. For the current 10 Hz data at ~3000 samples/session this is negligible. For future 500 Hz shock data, may need lazy loading. Mitigation: defer to arch-004 (tiered storage).

3. **Breaking analyzer signatures** — Adding `channels` parameter to `analyze_from_parquet()` changes the abstract method signature. All concrete implementations must accept `**kwargs` already (they do), but explicit parameter is cleaner. Mitigation: make it optional with default None.

4. **Test isolation** — Tests that mock `pd.read_parquet` will need updating when the load path changes. Mitigation: mock at the `SessionDataLoader.load()` level instead.

---

## Files Modified (estimated)

| File | Change type |
|------|------------|
| `src/services/__init__.py` | **New** — package init |
| `src/services/session_data_loader.py` | **New** — SessionDataLoader + SessionChannels |
| `src/main/deps.py` | Add `load_session()` helper |
| `src/main/routers/analysis.py` | Replace inline loading with `load_session()` |
| `src/main/routers/visualization.py` | Replace inline loading with `load_session()` |
| `src/features/base_analyzer.py` | Add `channels` parameter to `analyze_from_parquet()` |
| `src/features/power_analysis.py` | Accept channels in `analyze_from_parquet()` |
| `src/features/shift_analysis.py` | Accept channels in `analyze_from_parquet()` |
| `src/features/lap_analysis.py` | Accept channels in `analyze_from_parquet()` |
| `src/features/gear_analysis.py` | Accept channels in `analyze_from_parquet()` |
| `src/features/gg_analysis.py` | Accept channels in `analyze_from_parquet()` |
| `src/features/corner_analysis.py` | Accept channels in `analyze_from_parquet()` |
| `src/features/session_report.py` | Load once + pass channels to sub-analyzers |
| `tests/test_session_data_loader.py` | **New** — comprehensive loader tests |
