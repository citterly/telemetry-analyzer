### 📄 `docs/phases.md`
# Telemetry Analyzer — Development Phases

## Phase 1: Foundation ✅
- Project restructure with `src/`, `tests/`, `examples/`.
- Config system with portable paths.
- DLL interface wrapper (AIMDLL).
- Units.xml containment.
- Example scripts: read channels, dump metadata.

---

## Phase 2: Data Access (in progress)
- Implement sample-reading functions:
  - Get sample count per channel
  - Get sample values + timestamps
- Export full tabular datasets to CSV/JSON
- Validate against Race Studio outputs

---

## Phase 3: Analysis Layer
- Lap/sector detection via GPS or time markers
- Derived channels:
  - Acceleration/braking flags
  - RPM zones
  - Gear estimation
- Consistency checks & data smoothing

---

## Phase 4: Visualization
- Plotting (Matplotlib/Plotly)
- Comparison of laps, sessions, drivers
- Generate HTML/PDF reports

---

## Phase 5: Extended Features
- Web frontend for interactive analysis
- Integration with ML models (e.g., predictive lap times)
- Cloud/offline sync for sharing data

---

## Phase 6: Stretch Goals
- Real-time telemetry streaming
- Track map reconstruction
- Custom dashboards for in-car display
```

