# Result: Windows Test - Extraction Fix

**Tested:** 2026-01-28
**From:** Windows test machine
**Status:** SUCCESS

## Summary

The original fix (time base conversion in `data_loader.py`) was correct but insufficient. The **real bug** was in `session_builder.py`'s `_build_dataframe()` function.

## Root Cause

The interpolation logic used `pandas.reindex(method=None)` which requires **exact** timestamp matches. Regular channels sampled at 20Hz (times: 0.05, 0.10, 0.15...) don't exactly match the 10Hz canonical index (0.0, 0.1, 0.2...) due to floating-point precision, so all values became NaN.

## Fix Applied

Changed `session_builder.py` line 206-214 from:
```python
series = pd.Series(chan["values"], index=chan["time"])
df[name] = (
    series.reindex(time_index, method=None)
          .interpolate(method="linear")
          .bfill()
          .ffill()
)
```

To:
```python
# Use numpy interp for direct interpolation onto canonical time index
df[name] = np.interp(time_index, chan["time"], chan["values"])
```

## Test Results

**File:** `20250712_104619_Road America_a_0394.xrk`

### Before Fix
- GPS channels: 13/13 have data (100%)
- Regular channels: 0/13 have data (0%) - ALL EMPTY

### After Fix
- GPS channels: 13/13 have data (100%)
- Regular channels: 13/13 have data (100%)

### Channel Data Verified
| Channel | Rows | Range |
|---------|------|-------|
| OIL PRESSURE | 9780 (100%) | 28.90 - 85.71 |
| WATER TEMP | 9780 (100%) | 145.28 - 169.03 |
| OIL TEMP | 9780 (100%) | 96.24 - 173.52 |
| RPM dup 3 | 9780 (100%) | 997.84 - 7650.52 |
| FUEL PRESSURE | 9780 (100%) | 51.11 - 58.96 |
| PedalPos | 9780 (100%) | 0.00 - 100.00 |
| GPS Speed | 9780 (100%) | 0.04 - 67.46 |
| GPS Latitude | 9780 (100%) | 43.79 - 43.81 |
| ... | ... | ... |

**Total:** 26 channels with 100% data, 0 empty

## Parquet Export

Successfully exported to: `data/exports/processed/20250712_104619_Road America_a_0394_test.parquet`

Verified reload: 9780 rows x 26 columns, all data intact.

## Files Changed

- `src/session/session_builder.py` - Fixed interpolation method (this session)
- `src/extract/data_loader.py` - Time base fix (previous session, still valid)
