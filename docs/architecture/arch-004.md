# arch-004: Tiered Storage Foundation

**Status**: Plan
**Date**: 2026-02-10
**Scope**: Preserve native channel rates and provide tiered export (raw/summary/merged)

---

## Problem Statement

`session_builder.py` extracts per-channel native timestamps from the DLL but immediately discards them during `_build_dataframe()`, resampling everything to 10 Hz via naive `np.interp()`. This has three consequences:

1. **Lost high-frequency data** — Shock pots at 500-1000 Hz are downsampled to 10 Hz. Damper velocity histograms (0-2 in/sec range) require native-rate calculation; downsampling first aliases motion into noise.

2. **No anti-aliasing** — `np.interp()` is linear interpolation with no low-pass filter. Channels above Nyquist (5 Hz for 10 Hz output) fold back as aliasing artifacts.

3. **Single-tier export** — Only one parquet file at one rate. The custom suspension system produces 500 Hz data that needs its own analysis path before merging with the 10 Hz AiM data.

The target architecture (from CLAUDE.md) is three tiers:
```
Raw (500-1000 Hz) → Summary (50 Hz) → Merged (10 Hz)
```

---

## Current State

### Data flow in session_builder.py

```
XRK file
  → XRKDataLoader.open_file()
  → _extract_all_channels(loader)       # Returns {name: {time, values, unit, sample_count}}
  → _build_dataframe(raw_channels, 10)   # Resamples ALL channels to 10 Hz
  → export_session(df, session_id)       # Writes single parquet
```

### What native rate information exists

`_extract_all_channels()` returns per-channel data with native timestamps:
```python
{"RPM": {"time": np.array([0.0, 0.02, 0.04, ...]),    # 50 Hz native
          "values": np.array([4000, 4050, ...]),
          "unit": "rpm",
          "sample_count": 1650},
 "GPS Latitude": {"time": np.array([0.0, 0.1, 0.2, ...]),  # 10 Hz native
                   "values": np.array([43.797, ...]),
                   "unit": "deg",
                   "sample_count": 330}}
```

Native rate can be derived: `sample_count / (time[-1] - time[0])`.

### What _build_dataframe discards

- All per-channel timestamps (replaced by uniform `np.linspace`)
- Native sample counts
- Relationship between channel groups (CAN bus vs GPS vs analog)
- High-frequency content above 5 Hz (10 Hz output, Nyquist = 5 Hz)

### TODO markers in codebase

- Line 180: `"Future: add native_rate_hz tracking + hi-res sidecars."`

---

## Proposed Changes

### Phase 1: Track native rates (no export changes)

Add `native_rates` to DataFrame attrs before interpolation in `_build_dataframe()`:

```python
native_rates = {}
for name, chan in raw_channels.items():
    t = chan["time"]
    if len(t) > 1:
        native_rates[name] = round(len(t) / (t[-1] - t[0]), 1)
    else:
        native_rates[name] = 0.0

df.attrs["native_rates"] = native_rates
```

This preserves the information for downstream consumers and costs nothing.

### Phase 2: Channel classification

Classify channels by their tier membership based on native rate:

```python
# In a new module: src/session/channel_tiers.py

@dataclass
class ChannelTierConfig:
    """Which tier(s) a channel belongs to."""
    name: str
    native_rate_hz: float
    tiers: List[str]  # ["raw", "summary", "merged"] etc.

def classify_channels(native_rates: Dict[str, float]) -> Dict[str, ChannelTierConfig]:
    """Auto-classify channels by rate."""
    result = {}
    for name, rate in native_rates.items():
        if rate >= 200:
            tiers = ["raw", "summary", "merged"]
        elif rate >= 20:
            tiers = ["summary", "merged"]
        else:
            tiers = ["merged"]
        result[name] = ChannelTierConfig(name=name, native_rate_hz=rate, tiers=tiers)
    return result
```

Rate thresholds:
- **>= 200 Hz**: Raw tier (shock pots, high-freq analog) → all tiers
- **>= 20 Hz**: Summary tier (CAN channels at 50-100 Hz) → summary + merged
- **< 20 Hz**: Merged tier only (GPS at 10 Hz) → merged only

### Phase 3: Anti-alias decimation filter

Replace naive `np.interp()` downsampling with proper decimation for channels above the target rate:

```python
from scipy.signal import decimate

def _resample_channel(time_src, values_src, time_dst, src_rate, dst_rate):
    """Resample a channel from src_rate to dst_rate with anti-aliasing."""
    if src_rate <= dst_rate * 1.5:
        # Close enough — linear interpolation is fine
        return np.interp(time_dst, time_src, values_src)
    else:
        # Downsample: anti-alias filter then decimate
        factor = int(round(src_rate / dst_rate))
        # scipy.signal.decimate applies Chebyshev Type I low-pass before downsampling
        decimated = decimate(values_src, factor, ftype='iir', zero_phase=True)
        # Interpolate decimated signal onto exact target timestamps
        t_dec = np.linspace(time_src[0], time_src[-1], len(decimated))
        return np.interp(time_dst, t_dec, decimated)
```

Key design choices:
- Use `scipy.signal.decimate` with IIR filter (Chebyshev Type I, order 8)
- `zero_phase=True` prevents phase distortion
- Cutoff at `dst_rate / (2 * factor)` happens automatically
- Only apply when `src_rate > 1.5 * dst_rate` (avoid unnecessary filtering for near-rate channels)

### Phase 4: Tiered export API

Extend `export_session()` to support tier selection:

```python
def export_session(
    raw_channels: Dict[str, Dict],
    session_id: str,
    tiers: List[str] = None,
) -> Dict[str, Path]:
    """
    Export session data at specified tiers.

    Args:
        raw_channels: Raw channel data with native timestamps
        session_id: Session identifier
        tiers: List of tiers to export. Default: ["merged"] for backward compat.
               Options: "raw", "summary", "merged"

    Returns:
        Dict mapping tier name to output path
    """
```

File naming convention:
```
data/exports/processed/
  {session_id}.parquet                    # merged tier (10 Hz) — backward compat
  {session_id}_summary_50hz.parquet       # summary tier
  {session_id}_raw_500hz.parquet          # raw tier (only high-freq channels)
```

### Phase 5: Summary tier windowing

For the 50 Hz summary tier, high-frequency channels get windowed statistics over 20-sample windows at their native rate:

```python
def _compute_summary_windows(values, native_rate, summary_rate=50):
    """Compute min/max/mean/velocity over windows."""
    window_size = int(native_rate / summary_rate)
    n_windows = len(values) // window_size

    result = {
        "mean": np.zeros(n_windows),
        "min": np.zeros(n_windows),
        "max": np.zeros(n_windows),
        "velocity": np.zeros(n_windows),  # For shock pots
    }

    for i in range(n_windows):
        chunk = values[i * window_size : (i + 1) * window_size]
        result["mean"][i] = np.mean(chunk)
        result["min"][i] = np.min(chunk)
        result["max"][i] = np.max(chunk)
        # Velocity = derivative at native rate, then RMS over window
        if len(chunk) > 1:
            dt = 1.0 / native_rate
            vel = np.diff(chunk) / dt
            result["velocity"][i] = np.sqrt(np.mean(vel ** 2))

    return result
```

For a shock pot at 500 Hz → 50 Hz summary:
- Window = 10 samples (500/50)
- Output columns: `shock_pot_1_mean`, `shock_pot_1_min`, `shock_pot_1_max`, `shock_pot_1_velocity`

---

## Migration Strategy

### Step 1: Native rate tracking (non-breaking)

- Modify `_build_dataframe()` to compute and store `df.attrs["native_rates"]`
- Create `src/session/channel_tiers.py` with `classify_channels()`
- No changes to export format — existing parquet files unchanged
- **Tests**: verify native_rates present in attrs, verify classification logic

### Step 2: Anti-alias filter (improves existing path)

- Add `scipy` dependency (already in venv for other uses)
- Replace `np.interp()` calls in `_build_dataframe()` with `_resample_channel()`
- Existing 10 Hz export now uses proper decimation for high-freq channels
- **Tests**: verify no aliasing in downsampled signals, verify filter cutoff

### Step 3: Tiered export (additive)

- Add `tiers` parameter to `export_session()`
- Default `tiers=["merged"]` preserves backward compatibility
- Add `extract_full_session()` variant that returns raw channels + metadata
- **Tests**: verify file naming, verify tier content, verify merged tier unchanged

### Step 4: Summary windowing (additive)

- Add `_compute_summary_windows()` for high-freq channel aggregation
- Summary tier includes window stats (mean/min/max/velocity)
- **Tests**: verify window sizes, verify velocity computation

### Step 5: SessionDataLoader tier awareness (integration)

- Extend `SessionDataLoader.load()` with `tier` parameter
- Default `tier="merged"` preserves all existing behavior
- `tier="raw"` loads high-frequency sidecar if available
- Update `deps.py` `load_session()` to accept tier
- **Tests**: verify tier routing, verify fallback to merged

---

## Acceptance Criteria

1. **Native rates stored** — `df.attrs["native_rates"]` is a dict mapping channel name to Hz for every exported parquet
2. **Channel classification exists** — `src/session/channel_tiers.py` with `classify_channels()` auto-categorizing by rate threshold
3. **Anti-alias filter** — Channels downsampled by more than 1.5x use `scipy.signal.decimate` (not naive `np.interp`)
4. **Tiered export API** — `export_session(raw_channels, session_id, tiers=["merged"])` with backward-compat default
5. **Raw tier preserves native rate** — High-freq channels exported at their native rate, not resampled
6. **Summary tier has windowed stats** — Each high-freq channel produces mean/min/max/velocity columns at 50 Hz
7. **Merged tier unchanged** — Default behavior identical to current (10 Hz, all channels interpolated)
8. **File naming convention** — `{session_id}.parquet` (merged), `{session_id}_summary_50hz.parquet`, `{session_id}_raw_500hz.parquet`
9. **SessionDataLoader tier-aware** — `load(path, tier="merged")` routes to correct file
10. **All existing tests pass** — Zero regressions in current 894 tests

---

## Risks

1. **No high-frequency data yet** — The custom suspension system isn't built. All current AiM channels are 10-50 Hz. The raw tier has no data to exercise until hardware arrives. Mitigation: test with synthetic high-freq data; the infrastructure is ready when hardware is.

2. **scipy dependency** — `scipy.signal.decimate` adds a dependency. Mitigation: scipy is already available in the venv and is a standard scientific Python package.

3. **File size explosion** — Raw tier at 500 Hz for 4 shock channels over a 30-minute session ≈ 3.6M samples × 4 = 14.4M rows. At ~50 bytes/row in parquet with compression, this is ~50-100 MB per session. Mitigation: raw tier is opt-in, not default; parquet columnar compression handles sparse data well.

4. **Windows DLL availability** — `_extract_all_channels()` requires the XRK DLL (Windows only). Native rate tracking happens in `_build_dataframe()` which receives already-extracted data, so it works cross-platform. Mitigation: native_rates computed from time array length, not DLL metadata.

5. **Merge/sync complexity** — GPS Speed cross-correlation for time alignment between AiM and suspension data is complex signal processing. Mitigation: defer sync to separate feature. This arch-004 only builds the storage tiers; arch-006+ would handle cross-device synchronization.

---

## Files Modified (estimated)

| File | Change type |
|------|------------|
| `src/session/channel_tiers.py` | **New** — classify_channels(), ChannelTierConfig |
| `src/session/session_builder.py` | Modify — native_rates tracking, anti-alias decimation, tiered export |
| `src/services/session_data_loader.py` | Modify — tier parameter in load() |
| `src/main/deps.py` | Modify — tier parameter in load_session() |
| `tests/test_channel_tiers.py` | **New** — tier classification tests |
| `tests/test_session_builder.py` | Modify — native_rates, decimation, tier export tests |
| `tests/test_session_data_loader.py` | Modify — tier routing tests |

---

## Dependencies

- **arch-002** (SessionDataLoader): Complete. Tier awareness extends the existing service.
- **arch-003** (Analyzer registry): Complete. Analyzers can declare which tier they need.
- No external hardware dependencies for implementation — synthetic data suffices for testing.
