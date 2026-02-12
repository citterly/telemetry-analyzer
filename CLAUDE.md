
## Autonomous Harness

This project uses the autonomous agent harness pattern.
See `docs/PROCESS.md` for the full development process specification.

### Boot-up Ritual
When starting work or told to "boot up":
1. Read `features.json` to see project status
2. Read `claude-progress.txt` for recent history
3. Run `bash init.sh` to start environment
4. Pick highest-priority **unblocked** feature (see priority rules below)
5. Assess model requirement (see model selection below)
6. Execute it according to its phase type (see phase rules below)
7. Update features.json and claude-progress.txt, commit
8. Continue to next feature automatically

### Model Selection
After picking the next feature, recommend the appropriate model before starting work.
Print the recommendation so the user can switch with `/model` if needed.

| Situation | Model | Rationale |
|---|---|---|
| Failing features (bug fixes) | **Opus** | Multi-file debugging needs deep reasoning |
| Plan phase (`-plan`) | **Opus** | Architectural decisions drive everything downstream |
| Exec phase (`-exec`) | **Sonnet** | Following an approved spec; escalate to Opus if stuck |
| Test phase (`-test`) | **Sonnet** | Writing tests from acceptance criteria is well-scoped |

**Escalation rule**: If Sonnet hits a wall during exec (e.g., subtle cross-cutting bug,
complex multi-file coordination, or repeated failed attempts), print:
`⚠ MODEL ESCALATION: This task needs Opus — switch with /model`

### Priority Rules
Pick work in this order:
1. **Failing features** — fix regressions first
2. **Planned features, unblocked, highest priority** — the main work queue
3. **All passing** — stop and report status

A feature is "unblocked" when all features it depends on are "passing".
Dependencies follow the three-phase pipeline:
- `-plan` features have no blockers (always eligible)
- `-exec` features are blocked until their `-plan` is passing
- `-test` features are blocked until their `-exec` is passing

### Phase Rules
Each feature follows one of three phase types:

**Plan phase** (`-plan`):
- Read all affected files, explore existing patterns
- Write a design doc in `docs/architecture/{feature-id}.md`
- NO code changes — only the design doc
- Commit the doc, mark feature passing

**Execute phase** (`-exec`):
- Re-read the design doc from the plan phase
- Implement changes following the spec
- Run full test suite to verify no regressions
- Commit code changes, mark feature passing

**Test phase** (`-test`):
- Re-read acceptance criteria from the design doc
- Write tests verifying each criterion
- Run tests, fix any failures
- Commit tests, mark feature passing

### Commands
- "boot up" or "run boot-up ritual" → execute the ritual above
- "keep going" → continue working through features autonomously
- "status" → show features.json summary

### Rules
- Work on ONE feature at a time
- Only stop if: all features pass, or hit unresolvable blocker
- Always update features.json and claude-progress.txt before committing
- Plan phases produce design docs, NOT code
- Exec phases follow design docs, NOT ad-hoc decisions
- Test phases verify acceptance criteria from design docs

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
