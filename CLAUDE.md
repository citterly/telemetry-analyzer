
## Autonomous Harness

This project uses the autonomous agent harness pattern.

### Boot-up Ritual
When starting work or told to "boot up":
1. Read `features.json` to see project status
2. Read `claude-progress.txt` for recent history
3. Run `bash init.sh` to start environment
4. Pick highest-priority failing feature
5. Implement it, run tests, update artifacts, commit
6. Continue to next feature automatically

### Commands
- "boot up" or "run boot-up ritual" → execute the ritual above
- "keep going" → continue working through features autonomously
- "status" → show features.json summary

### Rules
- Work on ONE feature at a time
- Only stop if: all features pass, or hit unresolvable blocker
- Always update features.json and claude-progress.txt before committing

---

## Data Architecture Spec (Long-Term Target)

### AiM Logger Baseline
- 26 channels at 10 Hz (GPS-limited)
- Captures: GPS dynamics (LatAcc, LonAcc, Speed, Gyro), engine vitals (RPM, temps, pressures), throttle (PedalPos), environment
- Missing: suspension, wheel speeds, brake pressure, steering angle
- These gaps are filled by the custom high-frequency suspension system

### Tiered Storage Architecture
```
Suspension Analysis Pipeline:
  shock_raw_500hz.parquet    →  Velocity histograms, FFT, damper correlation
         ↓ (downsample)
  shock_summary_50hz.parquet →  Real-time CAN telemetry, AiM merge candidate
         ↓ (merge on GPS Speed)
  session_combined_10hz.parquet → Lap comparison, g-g diagrams, driver coaching
```

- **500-1000 Hz**: Raw suspension pot data - required for velocity histograms and damper analysis
- **50 Hz**: CAN summary tier (min/max/velocity over 20-sample windows) - natural checkpoint from WP-007
- **10 Hz**: AiM-compatible merged tier for lap analysis

### Critical Rule: Calculate at Native Rate
Velocity histograms need full 500-1000 Hz to capture 0-2 in/sec damper velocity range. Downsampling before velocity calculation aliases damper motion into noise. Always: **calculate velocity at native rate, then bin into histograms**.

### Sync Strategy
- Primary: GPS Speed cross-correlation (once GPS available)
- Fallback: RPM correlation (if tapping Ford CAN)
- Manual: Lap marker alignment

### Current Parser Limitation (session_builder.py)
- `_extract_all_channels()` retrieves per-channel native timestamps from DLL
- `_build_dataframe()` discards them during `np.interp` resampling to 10 Hz
- Native rates lost: CAN channels (50-100 Hz) downsampled without anti-aliasing
- TODO at line 180: "Future: add native_rate_hz tracking + hi-res sidecars"

### Required Enhancements
1. **Store native rates**: Add `df.attrs["native_rates"]` before interpolation
2. **Hi-res sidecar export**: Parallel path for channels that shouldn't be resampled
3. **Anti-alias filter**: Proper decimation for downsampled channels (not naive interp)
4. **Tiered export API**: `export_session(df, tier="raw"|"summary"|"merged")`
