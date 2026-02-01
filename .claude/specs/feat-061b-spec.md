# G-G Diagram Feature Specification
**Feature ID:** feat-061b  
**Feature Name:** G-G diagram bug fixes and enhancements  
**Status:** planned  
**Priority:** 19

---

## Purpose

The G-G diagram answers one question: **Where am I leaving grip on the table?**

From that answer, driver and engineer build an action plan before the next session. Time between sessions is 20-45 minutes. Analysis must be complete in under 5 minutes.

---

## User Workflow

```
Session ends
    ↓
Open G-G Diagram (2 clicks max from data download)
    ↓
See overall picture (instant - no waiting)
    "I'm at 61% grip utilization, weak in braking"
    ↓
Drill into weak area (1 click)
    "Braking is weak in T5, T12, T14"
    ↓
See exactly where on track (visual, not numbers)
    "T5 braking zone - I'm at 0.8g, could be 1.2g"
    ↓
Compare to reference (best lap, teammate, previous session)
    "On my fast lap I braked at 1.1g here"
    ↓
Action plan
    "T5, T12, T14 - brake later and harder. Focus on T5 first, longest zone."
    ↓
Back on track
```

**Total time: 3-5 minutes**

---

## Phases

### Phase 1: Fix lap filter
**Acceptance Criteria:**
- [ ] Lap dropdown populates with individual lap numbers
- [ ] Selecting a lap re-fetches data from API with lap filter
- [ ] All metrics update to reflect selected lap only
- [ ] Scatter plot shows only selected lap's data points

---

### Phase 2: Track map renders GPS trace
**Acceptance Criteria:**
- [ ] Track Position box shows GPS trace of session
- [ ] Trace is recognizable as the track layout
- [ ] Trace renders in < 1 second

---

### Phase 3: Click zone highlights on map
**Acceptance Criteria:**
- [ ] Clicking utilization zone in list highlights corresponding area on track map
- [ ] Highlighted zone visually distinct (color, pulse, or border)
- [ ] Clicking different zone moves highlight
- [ ] Clicking same zone toggles highlight off

---

### Phase 4: Quadrant reference values from vehicle config
**Acceptance Criteria:**
- [ ] vehicles.json has fields: max_lateral_g, max_braking_g, power_limited_accel_g
- [ ] Lat Left/Right quadrants use max_lateral_g as reference
- [ ] Braking quadrant uses max_braking_g as reference
- [ ] Acceleration quadrant uses power_limited_accel_g as reference
- [ ] Utilization percentages recalculate with correct per-quadrant references
- [ ] Warning displayed if max lateral from data exceeds vehicle config

---

### Phase 5: Session Overview layout per spec
**Acceptance Criteria:**
- [ ] Landing page shows 4 quadrant cards with status color (red < 70%, yellow 70-85%, green > 85%)
- [ ] "Biggest Opportunities" section shows top 3 ranked by time gain
- [ ] Time opportunity shown in seconds, not just percentage
- [ ] No scrolling required to see summary on standard laptop screen
- [ ] "Grip Utilization excluding power-limited zones" shown as primary metric
- [ ] "Corner-only utilization" metric displayed
- [ ] Braking vs Lateral ratio shown (flag if braking << lateral)

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│  SESSION SUMMARY - [Track] - [Date]                        │
│                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ BRAKING  │ │ LEFT     │ │ RIGHT    │ │ ACCEL    │       │
│  │  78%     │ │  61%     │ │  57%     │ │ POWER    │       │
│  │ ▲ weak   │ │ ○ ok     │ │ ○ ok     │ │ LIMITED  │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│                                                             │
│  BIGGEST OPPORTUNITIES:          TIME AVAILABLE:           │
│  1. Braking - 0.4g headroom      ~0.8 sec                  │
│  2. Left corners - 0.2g          ~0.3 sec                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### Phase 6: Quadrant Deep Dive view
**Acceptance Criteria:**
- [ ] Clicking quadrant card opens deep dive view
- [ ] Track map shows all zones of that type, color-coded by utilization
- [ ] Worst zone highlighted/pulsing to draw attention
- [ ] Ranked list shows zones with time opportunity in seconds
- [ ] Back button returns to overview
- [ ] One click to reach this view from landing

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│  BRAKING ANALYSIS                          [← Back]        │
│                                                             │
│  ┌─────────────────────┐  ┌─────────────────────────────┐  │
│  │                     │  │  BRAKING ZONES RANKED       │  │
│  │   TRACK MAP         │  │                             │  │
│  │   (braking zones    │  │  T5  ████████░░ 62%  0.4s  │  │
│  │    highlighted)     │  │  T14 ███████░░░ 58%  0.3s  │  │
│  │                     │  │  T12 ██████████ 81%  0.1s  │  │
│  │   [T5 pulsing]      │  │  T1  ██████████ 89%  --    │  │
│  │                     │  │                             │  │
│  └─────────────────────┘  └─────────────────────────────┘  │
│                                                             │
│  YOUR BRAKING: Avg 0.78g | Max 1.13g | Possible 1.25g      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### Phase 7: Zone Detail view
**Acceptance Criteria:**
- [ ] Clicking zone from list or map opens detail view
- [ ] Zoomed map shows just this corner/zone approach
- [ ] Side-by-side comparison: current lap vs reference (best lap)
- [ ] Specific numbers displayed: entry speed (mph), brake point (ft), peak g, duration (sec)
- [ ] Plain English action statement generated (e.g., "Brake 23 ft later at 1.0g+")
- [ ] Mini G-G scatter shows just this zone's data overlaid with reference
- [ ] Two clicks to reach this view from landing page

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│  T5 BRAKING ZONE - "Canada Corner"         [← Back]        │
│                                                             │
│  ┌─────────────────────┐  ┌─────────────────────────────┐  │
│  │                     │  │  THIS SESSION (Lap 3)       │  │
│  │  ZOOMED MAP         │  │  Entry: 142 mph             │  │
│  │  showing T5         │  │  Brake point: 485 ft        │  │
│  │  approach           │  │  Peak braking: 0.91g        │  │
│  │                     │  │  Brake duration: 2.1s       │  │
│  │  [your line shown]  │  │                             │  │
│  │                     │  │  vs YOUR BEST (Lap 6)       │  │
│  └─────────────────────┘  │  Entry: 144 mph             │  │
│                           │  Brake point: 462 ft ←LATER │  │
│  ┌─────────────────────┐  │  Peak braking: 1.08g ←HARDER│  │
│  │  G-G SCATTER        │  │                             │  │
│  │  (just this zone)   │  │  DELTA: -0.3 sec           │  │
│  └─────────────────────┘  └─────────────────────────────┘  │
│                                                             │
│  ACTION: "Brake 23 ft later at 1.0g+ to match your best"   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### Phase 8: Comparison mode
**Acceptance Criteria:**
- [ ] Compare button accessible from any view
- [ ] Can select: this lap vs another lap (same session)
- [ ] Can select: this session vs previous session (same track)
- [ ] G-G overlay shows two datasets in different colors
- [ ] Summary table shows which quadrant improved/degraded
- [ ] Delta per corner displayed

---

### Phase 9: Performance validation
**Acceptance Criteria:**
- [ ] Session overview loads in < 2 seconds
- [ ] Quadrant deep dive loads in < 1 second
- [ ] Zone detail loads in < 1 second
- [ ] Comparison overlay loads in < 2 seconds
- [ ] Lap switching completes in < 0.5 seconds

---

## Data Requirements

**API must provide:**

```json
{
  "session_summary": {
    "quadrants": [
      {
        "name": "braking",
        "utilization_pct": 78,
        "avg_g": 0.78,
        "max_g": 1.13,
        "reference_g": 1.25,
        "time_opportunity_sec": 0.8,
        "status": "weak"
      }
    ],
    "grip_utilization_excluding_power_limited": 62,
    "corner_only_utilization": 68,
    "braking_vs_lateral_ratio": 0.85,
    "power_limited_pct": 19
  },
  "zones": [
    {
      "id": "T5_brake",
      "name": "T5 - Canada Corner",
      "type": "braking",
      "utilization_pct": 62,
      "time_opportunity_sec": 0.4,
      "gps_polygon": [[lat, lon], ...]
    }
  ],
  "zone_detail": {
    "zone_id": "T5_brake",
    "laps": [
      {
        "lap": 3,
        "entry_speed_mph": 142,
        "brake_point_ft": 485,
        "peak_g": 0.91,
        "duration_sec": 2.1,
        "gps_trace": [[lat, lon], ...],
        "gg_points": [[lat_g, lon_g], ...]
      }
    ],
    "reference": {
      "source": "Lap 6 (best)",
      "entry_speed_mph": 144,
      "brake_point_ft": 462,
      "peak_g": 1.08,
      "duration_sec": 1.9
    },
    "action": "Brake 23 ft later at 1.0g+ to match your best lap"
  },
  "lap_numbers": [1, 2, 3, 4, 5, 6],
  "gps_bounds": {
    "min_lat": 43.79,
    "max_lat": 43.81,
    "min_lon": -87.99,
    "max_lon": -87.97
  }
}
```

---

## Vehicle Config Additions

Add to `data/vehicles.json` for each vehicle:

```json
{
  "max_lateral_g": 1.30,
  "max_braking_g": 1.40,
  "power_limited_accel_g": 0.45
}
```

---

## UI Principles

1. **No scrolling for primary insights** - summary fits on one screen
2. **Color means something** - Red/Yellow/Green = action needed/ok/good
3. **Biggest opportunity highlighted** - don't make user hunt
4. **Plain English actions** - "Brake 23ft later" not "Increase longitudinal deceleration"
5. **Track map always visible** - racers think spatially
6. **One click deeper, one click back** - no getting lost
7. **Compare always available** - A/B is how racers think

---

## Performance Requirements

| Action | Max Time |
|--------|----------|
| Load session overview | 2 sec |
| Click to quadrant detail | 1 sec |
| Click to zone detail | 1 sec |
| Load comparison overlay | 2 sec |
| Switch between laps | 0.5 sec |

---

## Questions This Must Answer

**Driver (in 3 minutes):**
- Am I using all available grip? → Overview
- Where am I weakest? → Overview quadrants
- Which corners specifically? → Deep dive
- What should I do differently? → Zone detail action

**Engineer (in 5 minutes):**
- Is car balanced left/right? → Compare quadrants
- Did setup change help? → Comparison mode
- Which corners need setup help vs driver help? → Zone detail + comparison

---

## Output After Using Tool

User should have:
1. Top 3 opportunities ranked by time gain
2. Specific corners where each opportunity exists
3. Specific action for each
4. Reference point to compare against

Written on pit board:
```
T5:  Brake 20ft later, hit 1.0g
T14: Brake 15ft later
T8:  Carry 3mph more entry
```

This is racing.
