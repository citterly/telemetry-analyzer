# AIM Telemetry Analyzer - Feature Roadmap

**Last Updated:** 2025-02-01  
**Project Goal:** Data-driven race car optimization delivering 2+ seconds per lap through systematic instrumentation and shock correlation

---

## Feature Status Overview

| ID | Feature | Priority | Status | Complexity |
|----|---------|----------|--------|------------|
| feat-061b | G-G Diagram Enhanced | 19 | ✅ DONE | Medium |
| feat-062 | Corner Detection & Analysis | 20 | 📋 Planned | High |
| feat-070 | Cross-Session Corner Comparison | 21 | 📋 Planned | Medium |
| feat-071 | Optimal Execution Prediction | 22 | 📋 Planned | High |
| feat-072 | Similar Corner Matching | 23 | 📋 Planned | High |
| feat-063 | Sector Splits | 24 | 📋 Planned | Low |
| feat-060 | Delta Track Map | 25 | 📋 Planned | Medium |
| feat-064 | Speed vs Distance | 26 | 📋 Planned | Low |
| feat-065 | Lap Consistency | 27 | 📋 Planned | Medium |
| feat-066 | Throttle Smoothness | 28 | 📋 Planned | Medium |
| feat-067 | Reliability Monitoring | 29 | 📋 Planned | Medium |
| feat-068 | Line Variation | 30 | 📋 Planned | Medium |
| feat-069 | Elevation Profile | 31 | 📋 Planned | Low |

---

## Dependency Graph

```
feat-062 (Corner Detection)
    │
    ├──► feat-070 (Cross-Session Comparison)
    │        │
    │        └──► feat-071 (Optimal Prediction)
    │                  │
    │                  └──► feat-072 (Similar Corner Matching)
    │
    └──► feat-063 (Sector Splits) [can use corner boundaries]

feat-061b (G-G Diagram) ✅
    │
    └──► feat-066 (Throttle Smoothness) [uses G-G quadrant concepts]

feat-064 (Speed vs Distance)
    │
    └──► feat-060 (Delta Track Map) [uses distance-based alignment]

feat-068 (Line Variation)
    │
    └──► feat-072 (Similar Corner Matching) [GPS line similarity]
```

---

## Completed Features

### feat-061b: G-G Diagram Enhanced ✅

**Status:** DONE (2025-02-01)

**Delivered:**
- Session Overview with 4 color-coded quadrant cards
- Biggest Opportunities ranked by potential time gain
- Per-quadrant reference values from vehicle config
- Track map with GPS trace and zone highlighting
- Quadrant Deep Dive view with scatter plots
- Zone Detail view with specific corner analysis
- Compare Laps / Compare Sessions mode
- G-G scatter with color options (speed/throttle/gear)

**Files Changed:** gg_analysis.py, app.py, gg_diagram.html

---

## Planned Features

### feat-062: Corner Detection & Analysis

**Priority:** 20 (Next up)  
**Complexity:** High  
**Dependencies:** None  
**Estimated Phases:** 8-10

**Vision:**
Automatically detect corners from GPS data using curvature analysis, then extract comprehensive per-corner metrics. This is the foundation for all advanced analysis - comparing corners across sessions, predicting optimal execution, and finding similar corners on different tracks.

**Key Deliverables:**
- Corner detection algorithm using GPS curvature + lateral G thresholds
- Corner classification (hairpin, sweeper, chicane, kink)
- Per-corner metrics extraction:
  - Entry speed, apex speed, exit speed
  - Min/max lateral G
  - Braking point (distance from corner)
  - Throttle application point
  - Time in corner
  - Line deviation from geometric apex
- Corner numbering and naming (persisted per track)
- Visual corner overlay on track map
- Corner comparison table within session

**Technical Approach:**
- Sliding window curvature calculation from GPS lat/lon
- Lateral G confirmation (>0.3g threshold for corner entry)
- Corner boundaries: where lateral G crosses threshold
- Apex detection: max lateral G point
- Store corner definitions in track database for reuse

**UI Concept:**
```
┌─────────────────────────────────────────────────────────┐
│ Corner Analysis - Gingerman Raceway                     │
├─────────────────────────────────────────────────────────┤
│ ┌─────────────────────┐  ┌────────────────────────────┐ │
│ │   [Track Map]       │  │ Corner Details: Turn 3     │ │
│ │                     │  │ Type: Sweeper (Right)      │ │
│ │   T1 ● ───── T2 ●   │  │ ─────────────────────────  │ │
│ │        \           │  │ Entry Speed:  82 mph       │ │
│ │         T3 ●       │  │ Apex Speed:   71 mph       │ │
│ │          \         │  │ Exit Speed:   85 mph       │ │
│ │           T4 ●     │  │ Max Lat G:    1.21         │ │
│ │                     │  │ Brake Point:  145 ft      │ │
│ │   Click corner      │  │ Throttle On:  apex +12 ft │ │
│ │   to select         │  │ Time:         2.34s       │ │
│ └─────────────────────┘  └────────────────────────────┘ │
│                                                         │
│ Corner Comparison (Lap 5 vs Best)                       │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Corner │ Entry Δ │ Apex Δ │ Exit Δ │ Time Δ │ Notes │ │
│ │ T1     │ +2 mph  │ -1 mph │ +3 mph │ -0.12s │       │ │
│ │ T3     │ -4 mph  │ -3 mph │ -2 mph │ +0.31s │ ⚠️    │ │
│ │ T7     │ +1 mph  │ +2 mph │ +4 mph │ -0.18s │ ✓     │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

### feat-070: Cross-Session Corner Comparison

**Priority:** 21  
**Complexity:** Medium  
**Dependencies:** feat-062 (Corner Detection)  
**Estimated Phases:** 5-6

**Vision:**
Compare the same corner across different sessions to track improvement over time and understand how setup changes affect corner performance. Essential for shock correlation work - see how damper changes affect Turn 3 entry stability.

**Key Deliverables:**
- Session selector (multi-select for comparison)
- Corner-locked comparison view
- Metrics trend over sessions (chart)
- Setup notes correlation (if logged)
- Statistical analysis (mean, std dev, best)
- Export comparison data

**UI Concept:**
```
┌─────────────────────────────────────────────────────────┐
│ Cross-Session: Turn 3 @ Gingerman                       │
├─────────────────────────────────────────────────────────┤
│ Sessions: [✓] Mar 15 [✓] Mar 22 [✓] Apr 5 [ ] Apr 12   │
│                                                         │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Entry Speed Trend                    Best: 84 mph   │ │
│ │ 85┤                              ●                  │ │
│ │ 82┤         ●          ●                            │ │
│ │ 79┤    ●                                            │ │
│ │   └────┴──────────┴──────────┴──────────           │ │
│ │      Mar 15    Mar 22     Apr 5                     │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ Session Notes:                                          │
│ • Mar 22: +2 clicks rear rebound → better exit speed   │
│ • Apr 5: Softer front spring → earlier throttle        │
└─────────────────────────────────────────────────────────┘
```

---

### feat-071: Optimal Execution Prediction

**Priority:** 22  
**Complexity:** High  
**Dependencies:** feat-070 (Cross-Session Comparison)  
**Estimated Phases:** 6-8

**Vision:**
Predict the theoretically best possible lap time by combining the best execution of each corner from all recorded laps. Shows the gap between current performance and potential, and identifies which corners have the most improvement opportunity.

**Key Deliverables:**
- Best-of-best corner aggregation
- Theoretical optimal lap time calculation
- Per-corner gap analysis (you vs optimal)
- Priority ranking for practice focus
- Validation against physics (can't exceed tire grip)
- Visual "ghost line" comparison

**Technical Approach:**
- For each corner, find lap with best time-through-corner
- Sum optimal corner times + straight times
- Validate: combined lateral/longitudinal G must be achievable
- Handle corner-to-corner transitions (exit speed affects next entry)

**Key Output:**
```
Current Best Lap:    1:42.3
Theoretical Optimal: 1:40.1
Gap:                 2.2 seconds

Biggest Opportunities:
1. Turn 5: -0.45s (brake 15ft later, you have grip)
2. Turn 9: -0.38s (carry 3mph more through apex)
3. Turn 2: -0.31s (earlier throttle, better exit)
```

---

### feat-072: Similar Corner Matching

**Priority:** 23  
**Complexity:** High  
**Dependencies:** feat-071, feat-068 (Line Variation)  
**Estimated Phases:** 6-8

**Vision:**
Find corners on different tracks that have similar characteristics - radius, speed, camber direction. Transfer learning from one track to another: "Turn 3 at Gingerman is similar to Turn 7 at Mid-Ohio, use similar setup."

**Key Deliverables:**
- Corner signature extraction (radius, length, direction, typical speed)
- Similarity scoring algorithm
- Cross-track corner matching
- Setup recommendation transfer
- Visual comparison overlay

**Technical Approach:**
- Normalize corner profiles to unit scale
- Compare: entry speed ratio, apex speed ratio, radius, arc length
- Weight by similarity confidence
- Account for surface grip differences between tracks

---

### feat-063: Sector Splits

**Priority:** 24  
**Complexity:** Low  
**Dependencies:** None (but benefits from feat-062)  
**Estimated Phases:** 4-5

**Vision:**
Traditional sector timing analysis - divide track into sectors, show split times, identify where time is gained/lost. Simpler than full corner analysis but immediately useful.

**Key Deliverables:**
- Sector boundary definition (manual or auto from corners)
- Per-sector time display
- Delta to best sector (green/red)
- Cumulative delta through lap
- Sector time history chart
- Theoretical best from best sectors

**UI Concept:**
```
Lap 7 Sector Times (vs Best)
┌────────┬─────────┬─────────┬─────────┐
│ Sector │ Time    │ Best    │ Delta   │
├────────┼─────────┼─────────┼─────────┤
│ S1     │ 28.42   │ 28.15   │ +0.27 🔴│
│ S2     │ 34.18   │ 34.31   │ -0.13 🟢│
│ S3     │ 39.71   │ 39.55   │ +0.16 🔴│
├────────┼─────────┼─────────┼─────────┤
│ Total  │ 1:42.31 │ 1:42.01 │ +0.30   │
└────────┴─────────┴─────────┴─────────┘
```

---

### feat-060: Delta Track Map

**Priority:** 25  
**Complexity:** Medium  
**Dependencies:** feat-064 (Speed vs Distance) helps  
**Estimated Phases:** 5-6

**Vision:**
Show time delta on the track map itself - color the racing line green where you're gaining time vs reference lap, red where losing. Instant visual of where to focus.

**Key Deliverables:**
- Distance-based lap alignment
- Running delta calculation
- Color gradient track line (green/red)
- Delta magnitude shown by color intensity
- Hover for exact delta value
- Reference lap selector

**UI Concept:**
```
Track map with racing line colored:
- Bright green: gaining 0.1s+
- Light green: gaining 0-0.1s  
- Gray: neutral
- Light red: losing 0-0.1s
- Bright red: losing 0.1s+

Hover over any point: "Delta: -0.23s (braking zone T3)"
```

---

### feat-064: Speed vs Distance

**Priority:** 26  
**Complexity:** Low  
**Dependencies:** None  
**Estimated Phases:** 3-4

**Vision:**
Classic telemetry view - speed trace plotted against track distance, with multiple laps overlaid. Foundation for understanding where speed is gained/lost.

**Key Deliverables:**
- Distance calculation from GPS
- Speed vs distance plot
- Multi-lap overlay with lap selector
- Zoom to specific track section
- Channel selector (add throttle, brake, gear)
- Min/max/delta annotations

**UI Concept:**
```
Speed vs Distance - Gingerman
┌─────────────────────────────────────────────────────────┐
│ 120┤                          ████                      │
│    │    ████                 █    █        ████         │
│ 100┤   █    █               █      █      █    █        │
│    │  █      █             █        █    █      █       │
│  80┤ █        █           █          █  █        █      │
│    │█          █         █            ██          █     │
│  60┤            █       █                          █    │
│    │             █     █                            █   │
│  40┤              █████                              █  │
│   └┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴─  │
│    0     500    1000   1500   2000   2500   3000   3500 │
│                     Distance (ft)                       │
│                                                         │
│ ── Lap 5 (best)  ── Lap 7  ── Lap 12                   │
└─────────────────────────────────────────────────────────┘
```

---

### feat-065: Lap Consistency

**Priority:** 27  
**Complexity:** Medium  
**Dependencies:** None  
**Estimated Phases:** 4-5

**Vision:**
Measure driver consistency lap-to-lap. Consistent drivers are fast drivers. Show variance in lap times, sector times, and specific metrics (braking points, turn-in points).

**Key Deliverables:**
- Lap time standard deviation
- Consistency score (0-100)
- Per-sector consistency breakdown
- Outlier detection (offs, mistakes)
- Trend over session (getting more consistent?)
- Comparison to reference driver if available

**Key Metrics:**
- Lap time σ (target: <0.5s for club racer)
- Braking point σ per corner (target: <10ft)
- Apex speed σ per corner (target: <2mph)
- Throttle application point σ (target: <15ft)

---

### feat-066: Throttle Smoothness

**Priority:** 28  
**Complexity:** Medium  
**Dependencies:** feat-061b (G-G concepts)  
**Estimated Phases:** 4-5

**Vision:**
Analyze driver input quality. Smooth inputs = faster, easier on tires, more consistent. Detect stabbing throttle, abrupt steering, harsh braking.

**Key Deliverables:**
- Throttle application rate analysis
- Steering smoothness score
- Brake modulation quality
- Input aggression index
- Comparison to smooth reference
- Per-corner input quality breakdown

**Technical Approach:**
- Calculate derivative of throttle/brake/steering
- High d/dt = abrupt input
- Compare to achievable smooth profile
- Weight by corner criticality

---

### feat-067: Reliability Monitoring

**Priority:** 29  
**Complexity:** Medium  
**Dependencies:** None  
**Estimated Phases:** 5-6

**Vision:**
Track mechanical health through telemetry trends. Catch problems before failures. Essential for endurance and avoiding DNFs.

**Key Deliverables:**
- Oil temp vs ambient trending
- Water temp vs lap intensity correlation
- Oil pressure vs RPM curve (compare to baseline)
- Voltage monitoring
- Anomaly detection alerts
- Session-over-session comparison

**Alert Examples:**
- "Oil pressure 5psi lower than baseline at 6000 RPM"
- "Water temp climbing 2°F/lap above normal"
- "Voltage dropping under load - check alternator"

---

### feat-068: Line Variation

**Priority:** 30  
**Complexity:** Medium  
**Dependencies:** None  
**Estimated Phases:** 4-5

**Vision:**
Compare racing lines lap-to-lap using GPS traces. Are you hitting the same marks? Where does your line vary most?

**Key Deliverables:**
- GPS trace overlay (multiple laps)
- Line deviation heatmap
- Per-corner line consistency
- Ideal line extraction (from best lap)
- Deviation from ideal visualization

---

### feat-069: Elevation Profile

**Priority:** 31  
**Complexity:** Low  
**Dependencies:** None  
**Estimated Phases:** 3-4

**Vision:**
Extract and display track elevation from GPS altitude data. Understanding elevation helps with brake bias, suspension setup, and strategy.

**Key Deliverables:**
- Elevation vs distance plot
- Grade percentage calculation
- Uphill/downhill corner tagging
- Elevation change per lap summary
- Smoothed profile (GPS altitude is noisy)

---

## Implementation Notes

### Autonomous Harness Usage

Features are implemented using the autonomous CC harness:
- Full specs in `feat-XXX-spec.json` (phases, acceptance criteria)
- Vision docs in `feat-XXX-spec.md` (mockups, workflows)
- Run: `python3 run.py auto ~/projects/racing/aim-telemetry feat-XXX --model sonnet`

### Spec Development Process

1. Copy feature description from this roadmap
2. Create detailed mockups (ASCII or Figma)
3. Define 6-10 phases with acceptance criteria
4. Create JSON spec file
5. Create markdown spec file
6. Run harness

### Testing Strategy

- Each feature gets integration tests
- Manual testing against real Gingerman data
- Performance validation (must handle 50+ lap sessions)

---

## Revision History

| Date | Changes |
|------|---------|
| 2025-02-01 | Initial roadmap created. feat-061b completed. |
