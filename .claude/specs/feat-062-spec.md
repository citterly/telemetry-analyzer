# Feature Specification: Corner Detection & Analysis

**Feature ID:** feat-062  
**Priority:** 20  
**Status:** Ready for Implementation  
**Complexity:** High  
**Estimated Phases:** 10  
**Dependencies:** None (this is foundational)

---

## Vision

Corner Detection & Analysis automatically identifies corners from GPS telemetry data and extracts comprehensive per-corner metrics. This is the **foundation feature** for the entire advanced analysis suite - without reliable corner detection, we cannot:

- Compare the same corner across different sessions (feat-070)
- Predict optimal lap times from best corner executions (feat-071)
- Find similar corners across different tracks (feat-072)
- Generate meaningful sector splits (feat-063)

The goal is **zero manual corner definition** - drop in a session file from any track and corners are automatically detected, classified, and analyzed.

---

## User Stories

1. **As a driver**, I want to see which corners I'm losing time in so I know where to focus practice.

2. **As a driver**, I want to compare my corner entry speeds to my best lap so I can identify braking point opportunities.

3. **As a crew chief**, I want to see corner-by-corner metrics to correlate setup changes with corner performance.

4. **As a data engineer**, I want corner definitions to persist so the same corner numbers work across sessions.

---

## Technical Approach

### Corner Detection Algorithm

**Step 1: Curvature Calculation**
```python
def calc_curvature(lat, lon, window=5):
    """
    Calculate path curvature using sliding window.
    
    Curvature κ = 1/R where R is turning radius.
    High curvature = tight corner, low curvature = straight.
    
    Uses three-point circle fitting:
    - Point A: window samples back
    - Point B: current point  
    - Point C: window samples forward
    
    Returns curvature in 1/meters.
    """
```

**Step 2: Lateral G Confirmation**
```python
def detect_corner_boundaries(lateral_g, threshold=0.3):
    """
    Find corner entry/exit using lateral G threshold crossing.
    
    Corner Entry: lateral_g crosses above threshold
    Corner Exit: lateral_g crosses below threshold
    
    Filters:
    - Minimum duration 0.5s (eliminates noise)
    - Merge corners separated by <0.5s (chicane handling)
    """
```

**Step 3: Combined Detection**
```python
def detect_corners(df):
    """
    Robust corner detection using curvature AND lateral G.
    
    A corner is valid when:
    1. Curvature exceeds minimum threshold, AND
    2. Lateral G exceeds 0.3g
    
    This eliminates false positives from:
    - GPS noise on straights (fails curvature check)
    - Car rotation without turning (fails lateral G check)
    """
```

### Corner Classification

| Type | Criteria | Example |
|------|----------|---------|
| Hairpin | Apex speed < 40 mph | Gingerman T5 |
| Sweeper | Apex speed > 60 mph, duration > 2s | Road America carousel |
| Chicane | Direction change within 1s | Mid-Ohio chicane |
| Kink | Apex speed > 80 mph, duration < 1s | High-speed esses |
| Standard | Everything else | Most corners |

### Corner Metrics

| Metric | Calculation | Units |
|--------|-------------|-------|
| Entry Speed | Speed at corner start boundary | mph |
| Apex Speed | Speed at max lateral G point | mph |
| Exit Speed | Speed at corner end boundary | mph |
| Max Lateral G | Peak abs(lateral_g) in corner | g |
| Max Braking G | Peak longitudinal_g (negative) before apex | g |
| Max Accel G | Peak longitudinal_g (positive) after apex | g |
| Brake Point | Distance from brake application to apex | ft |
| Throttle Point | Distance from apex to throttle application | ft |
| Time in Corner | Duration from entry to exit | seconds |

### Corner Persistence

Corners are stored in `data/track_corners.json`:

```json
{
  "gingerman_raceway": {
    "corners": [
      {
        "id": "T1",
        "name": "Turn 1",
        "apex_lat": 42.12345,
        "apex_lon": -86.12345,
        "direction": "right",
        "type": "standard",
        "typical_apex_speed": 72
      },
      ...
    ],
    "created": "2025-02-01",
    "lap_count_used": 15
  }
}
```

On session load:
1. Extract track name from session metadata or GPS coordinates
2. If track exists in JSON, match detected corners to stored definitions by GPS proximity (<50m)
3. If no match, create new track entry from detected corners
4. Allow manual rename (updates JSON)

---

## UI Design

### Main Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Corner Analysis                                    Session: gingerman_1 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────┐  ┌──────────────────────────────┐  │
│  │                                 │  │ Corner Details: T3           │  │
│  │         [TRACK MAP]             │  │ ════════════════════════════ │  │
│  │                                 │  │                              │  │
│  │    T1●────────T2●               │  │ Type: Right Sweeper          │  │
│  │         \                       │  │ Severity: ███░░ (3/5)        │  │
│  │          \                      │  │                              │  │
│  │          T3●  ← selected        │  │ ┌──────────────────────────┐ │  │
│  │            \                    │  │ │ Entry    Apex     Exit   │ │  │
│  │             \                   │  │ │  82  ──→  71  ──→  85    │ │  │
│  │             T4●                 │  │ │  mph      mph      mph   │ │  │
│  │              |                  │  │ └──────────────────────────┘ │  │
│  │             T5●                 │  │                              │  │
│  │             /                   │  │ Max Lateral G:    1.21g     │  │
│  │           T6●                   │  │ Max Braking G:    1.35g     │  │
│  │           /                     │  │ Brake Point:      145 ft    │  │
│  │    T11●──T10●──T9●──T8●──T7●    │  │ Throttle On:      +12 ft    │  │
│  │                                 │  │ Time in Corner:   2.34s     │  │
│  │   ● Corner apex (click to       │  │                              │  │
│  │     select)                     │  │ vs Best Lap:                │  │
│  │   ━ Corner boundary             │  │   Entry: +2 mph  🟢         │  │
│  │   ━ Selected corner             │  │   Apex:  -1 mph  🔴         │  │
│  │                                 │  │   Exit:  +3 mph  🟢         │  │
│  └─────────────────────────────────┘  └──────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Corner Comparison: Lap 5 vs Best                     [Lap: ▼ 5 ]  │  │
│  ├───────┬─────────┬─────────┬─────────┬─────────┬─────────┬────────┤  │
│  │Corner │ Entry Δ │ Apex Δ  │ Exit Δ  │ Time Δ  │ Lat G Δ │ Notes  │  │
│  ├───────┼─────────┼─────────┼─────────┼─────────┼─────────┼────────┤  │
│  │ T1    │ +2 mph  │ +1 mph  │ +3 mph  │ -0.12s  │ +0.05g  │   ✓    │  │
│  │ T2    │  0 mph  │ -1 mph  │ +1 mph  │ -0.03s  │  0.00g  │        │  │
│  │ T3    │ -4 mph  │ -3 mph  │ -2 mph  │ +0.31s  │ -0.12g  │   ⚠️   │  │
│  │ T4    │ +1 mph  │ +2 mph  │ +2 mph  │ -0.08s  │ +0.03g  │   ✓    │  │
│  │ T5    │ -2 mph  │ -1 mph  │  0 mph  │ +0.15s  │ -0.05g  │   ⚠️   │  │
│  │ ...   │         │         │         │         │         │        │  │
│  ├───────┼─────────┼─────────┼─────────┼─────────┼─────────┼────────┤  │
│  │ TOTAL │         │         │         │ +0.23s  │         │        │  │
│  └───────┴─────────┴─────────┴─────────┴─────────┴─────────┴────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Track Map Interactions

**Corner Markers:**
- Numbered circles at apex positions (T1, T2, etc.)
- Size indicates severity (larger = slower corner)
- Color indicates performance vs best:
  - 🟢 Green outline: gaining time
  - 🔴 Red outline: losing time
  - ⚪ Gray outline: neutral

**Corner Boundaries:**
- Highlighted GPS trace segment during corner
- Darker shade = higher lateral G
- Selected corner has distinct color (yellow/gold)

**Click Behavior:**
- Click marker → select corner, show details panel
- Click elsewhere → deselect
- Hover marker → tooltip with corner name and type

### Corner Details Panel

```
┌──────────────────────────────────────┐
│ T3: Turn 3                    [edit] │
│ ──────────────────────────────────── │
│                                      │
│ Classification                       │
│ ┌──────────────────────────────────┐ │
│ │ Type:      Right Sweeper         │ │
│ │ Severity:  ███░░ (3/5)           │ │
│ │ Duration:  2.34s                 │ │
│ └──────────────────────────────────┘ │
│                                      │
│ Speed Profile                        │
│ ┌──────────────────────────────────┐ │
│ │     Entry      Apex       Exit   │ │
│ │                                  │ │
│ │  85 ─┐                    ┌─ 88  │ │
│ │      └──────┐    ┌───────┘      │ │
│ │             └────┘ 71           │ │
│ │                                  │ │
│ │  mph ────────────────────── mph │ │
│ └──────────────────────────────────┘ │
│                                      │
│ Key Metrics                          │
│ ┌──────────────────────────────────┐ │
│ │ Max Lateral G     1.21g          │ │
│ │ Max Braking G     1.35g          │ │
│ │ Max Accel G       0.45g          │ │
│ │ Brake Point       145 ft         │ │
│ │ Throttle On       apex +12 ft    │ │
│ └──────────────────────────────────┘ │
│                                      │
│ vs Session Best                      │
│ ┌──────────────────────────────────┐ │
│ │ Entry Speed   +2 mph      🟢     │ │
│ │ Apex Speed    -1 mph      🔴     │ │
│ │ Exit Speed    +3 mph      🟢     │ │
│ │ Time          -0.08s      🟢     │ │
│ └──────────────────────────────────┘ │
│                                      │
│ [Compare to Other Laps ▼]            │
└──────────────────────────────────────┘
```

---

## API Endpoints

### GET /api/corners/{session_id}

Returns detected corners for a session.

**Response:**
```json
{
  "session_id": "gingerman_20250315",
  "track": "gingerman_raceway",
  "corners": [
    {
      "id": "T1",
      "name": "Turn 1", 
      "type": "standard",
      "direction": "right",
      "severity": 3,
      "apex_lat": 42.12345,
      "apex_lon": -86.12345,
      "boundaries": {
        "start_distance": 1250,
        "apex_distance": 1380,
        "end_distance": 1520
      }
    },
    ...
  ],
  "detection_params": {
    "lateral_g_threshold": 0.3,
    "min_duration": 0.5,
    "curvature_threshold": 0.01
  }
}
```

### GET /api/corners/{session_id}/lap/{lap_number}

Returns per-corner metrics for a specific lap.

**Response:**
```json
{
  "lap_number": 5,
  "lap_time": 102.34,
  "corners": [
    {
      "corner_id": "T1",
      "entry_speed": 82.3,
      "apex_speed": 71.2,
      "exit_speed": 85.1,
      "max_lateral_g": 1.21,
      "max_braking_g": 1.35,
      "max_accel_g": 0.45,
      "brake_point_ft": 145,
      "throttle_on_ft": 12,
      "time_in_corner": 2.34,
      "delta_vs_best": {
        "entry_speed": 2.1,
        "apex_speed": -1.3,
        "exit_speed": 3.2,
        "time": -0.08
      }
    },
    ...
  ]
}
```

### PUT /api/corners/{session_id}/{corner_id}

Update corner name or properties.

**Request:**
```json
{
  "name": "The Kink"
}
```

---

## File Structure

```
src/features/
├── corner_analysis.py      # Core detection and metrics
│   ├── calc_curvature()
│   ├── detect_corner_boundaries()
│   ├── detect_corners()
│   ├── classify_corner()
│   ├── extract_corner_metrics()
│   ├── Corner (dataclass)
│   └── CornerMetrics (dataclass)

templates/
├── corner_analysis.html    # Main UI template

data/
├── track_corners.json      # Persisted corner definitions

src/main/
├── app.py                  # Add routes for corner endpoints
```

---

## Testing Strategy

### Unit Tests

```python
def test_curvature_calculation():
    """Known circle should return expected curvature."""
    # Generate points on circle with 100m radius
    # Curvature should be ~0.01 (1/100)
    
def test_corner_detection_straight():
    """Straight section should detect no corners."""
    
def test_corner_detection_hairpin():
    """Hairpin should be detected as single corner."""
    
def test_corner_classification():
    """Corners classified correctly by speed/duration."""
```

### Integration Tests

```python
def test_gingerman_corner_count():
    """Gingerman should have approximately 11 corners."""
    session = load_session("gingerman_test.xrk")
    corners = detect_corners(session)
    assert 10 <= len(corners) <= 12
    
def test_corner_persistence():
    """Corners should persist across page reload."""
```

### Manual Validation

1. Load Gingerman session
2. Verify corners match visual track map
3. Verify apex positions are at actual apexes
4. Verify metrics match manual spot-check
5. Verify comparison table sums correctly

---

## Performance Requirements

- Corner detection: < 500ms for 50-lap session
- Page load with map: < 3 seconds
- Corner selection response: < 100ms
- Lap comparison update: < 500ms

---

## Edge Cases

1. **Track with no corners** (oval): Should detect curved sections even if low G
2. **Track not in database**: Auto-generate corner definitions
3. **Missing lateral G channel**: Fall back to curvature-only detection
4. **Chicane handling**: Merge rapid direction changes into single "complex"
5. **Wet session**: Lower G thresholds may need adjustment
6. **GPS dropout**: Skip affected sections, don't create false corners

---

## Future Enhancements (Out of Scope)

- Manual corner boundary adjustment (drag handles)
- Corner-specific video sync
- Ideal line overlay from best lap
- Corner-specific setup recommendations
- Machine learning corner detection refinement

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2025-02-01 | Claude | Initial specification |
