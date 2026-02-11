# User Acceptance Test Script - Recent Features

**Project**: Telemetry Analyzer
**Date**: 2026-02-11
**Coverage**: arch-001 through arch-006 + feedback widget
**Estimated Time**: 30-45 minutes

---

## Prerequisites

- [ ] Server running at http://127.0.0.1:8000
- [ ] At least one parquet file in `data/exports/processed/`
- [ ] Feedback service running at http://127.0.0.1:8100 (for feat-070 test)

**Start the server**:
```bash
cd ~/projects/telemetry-analyzer
source venv/bin/activate
bash init.sh
```

---

## 1. Vehicle Switching (arch-001)

**Feature**: Runtime vehicle switching affects all analyzers

### Test 1.1: Vehicle Settings Page
- [ ] Navigate to http://127.0.0.1:8000/vehicles
- [ ] **Verify**: 5 vehicles listed (BMW M3, NC Miata, 996 GT3, S2000, Spec E30)
- [ ] **Verify**: One vehicle has green "Active" badge
- [ ] Click on "NC Miata" card
- [ ] **Verify**: Vehicle parameters load in right panel
- [ ] **Verify**: G-Force Limits section shows max_lateral_g, max_braking_g, power_limited_accel_g
- [ ] **Verify**: Engine Specs shows max_rpm, shift_rpm, power_band_min_rpm, power_band_max_rpm
- [ ] **Verify**: Transmission ratios table shows all 6 gears + final drive
- [ ] Click "Set as Active" button
- [ ] **Verify**: Success message appears
- [ ] **Verify**: NC Miata card now has green "Active" badge
- [ ] **Verify**: Other vehicles no longer active

### Test 1.2: Verify Vehicle Switch Affects Analysis
- [ ] Navigate to http://127.0.0.1:8000/analysis
- [ ] Select a session from dropdown
- [ ] Click "Full Report"
- [ ] **Verify**: "Vehicle Setup" in report shows "NC Miata Stock 6MT" (or current setup name)
- [ ] Switch back to Vehicles page (http://127.0.0.1:8000/vehicles)
- [ ] Set "BMW E46 M3" as active
- [ ] Return to Analysis page, run Full Report again
- [ ] **Verify**: "Vehicle Setup" now shows "Current Setup" or BMW-related setup

**Expected Result**: Vehicle switching works across pages, analysis uses correct vehicle config

---

## 2. Analysis Dashboard (arch-002 integration)

**Feature**: SessionDataLoader centralizes data loading with speed unit detection

### Test 2.1: Basic Analysis Flow
- [ ] Navigate to http://127.0.0.1:8000/analysis
- [ ] Open browser DevTools → Network tab
- [ ] Select a session file from dropdown
- [ ] Click "Shifts"
- [ ] **Verify**: Results load (shift table, chart, statistics)
- [ ] **Verify**: No console errors
- [ ] **Verify**: Speed values appear reasonable (30-150 mph range for track driving)
- [ ] Click "Laps"
- [ ] **Verify**: Lap times table appears
- [ ] **Verify**: Fastest lap highlighted
- [ ] **Verify**: Lap times in reasonable range (30s - 10min)
- [ ] Click "Gears"
- [ ] **Verify**: Gear usage doughnut chart displays
- [ ] **Verify**: Time in gear percentages sum to ~100%
- [ ] Click "Power"
- [ ] **Verify**: Max power/acceleration stats display
- [ ] **Verify**: Acceleration events table appears

### Test 2.2: Full Report Integration
- [ ] Click "Full Report"
- [ ] **Verify**: All 4 sub-sections load (Laps, Shifts, Gears, Power)
- [ ] **Verify**: Track name appears at top
- [ ] **Verify**: Vehicle setup name appears
- [ ] **Verify**: Recommendations section displays
- [ ] Scroll through entire report
- [ ] **Verify**: No missing data or "undefined" values
- [ ] **Verify**: Charts render correctly

**Expected Result**: All analysis types load data correctly, speed unit conversion works

---

## 3. Audit Mode UI (arch-006) - CRITICAL NEW FEATURE

**Feature**: Toggle-able audit panels show calculation traceability and sanity checks

### Test 3.1: Analysis Page Audit Mode
- [ ] Navigate to http://127.0.0.1:8000/analysis
- [ ] **Verify**: Audit toggle switch visible in "Select Session" card header
- [ ] **Verify**: Toggle is OFF by default (gray)
- [ ] Select a session file
- [ ] Click "Power" analysis type
- [ ] **Verify**: Analysis results display normally
- [ ] **Verify**: NO audit panels visible
- [ ] Click audit toggle to ON (should turn green/blue)
- [ ] **Verify**: Page reloads or re-fetches data
- [ ] **Verify**: Audit panel appears below Power analysis results
- [ ] **Verify**: Panel has colored left border (green = all pass, yellow = warnings, red = failures)
- [ ] **Verify**: Panel header shows "PowerAnalysis Calculation Trace"
- [ ] Click panel header to expand
- [ ] **Verify**: Four sections appear: Inputs, Configuration, Intermediates, Sanity Checks
- [ ] **Verify**: "Sanity Checks" section shows 5 checks with status dots (green/yellow/red)
- [ ] **Verify**: Each check has an "Impact" description in gray italic text
- [ ] Click each check to read impact
- [ ] **Verify**: Impact describes what goes wrong if check fails

### Test 3.2: Full Report Audit Mode
- [ ] Keep audit toggle ON
- [ ] Click "Full Report"
- [ ] **Verify**: Multiple audit panels appear
- [ ] **Verify**: First panel is "SessionReport Cross-Validation" (3 checks)
- [ ] **Verify**: Additional panels for each sub-analyzer (Shifts, Laps, Gears, Power)
- [ ] Expand "SessionReport Cross-Validation" panel
- [ ] **Verify**: Check #1: "speed_unit_consensus"
- [ ] **Verify**: Check #2: "sample_count_consistent"
- [ ] **Verify**: Check #3: "config_consistent"
- [ ] **Verify**: All checks show "pass" status (green dots)
- [ ] Expand "ShiftAnalyzer" panel
- [ ] **Verify**: 4 checks visible (gear_count_matches_config, shift_rpm_below_redline, shift_confidence, sufficient_shifts)
- [ ] Expand "LapAnalysis" panel
- [ ] **Verify**: 4 checks visible
- [ ] Click audit toggle to OFF
- [ ] **Verify**: All audit panels disappear immediately
- [ ] **Verify**: Only analysis results remain
- [ ] Refresh page
- [ ] **Verify**: Toggle state persists (still OFF after refresh)

### Test 3.3: G-G Diagram Audit Mode
- [ ] Navigate to http://127.0.0.1:8000/gg-diagram
- [ ] **Verify**: Audit toggle visible in control panel area
- [ ] Select a session file
- [ ] **Verify**: G-G diagram loads
- [ ] **Verify**: NO audit panel visible (toggle OFF by default)
- [ ] Turn audit toggle ON
- [ ] **Verify**: Page re-fetches data with trace
- [ ] **Verify**: Audit panel appears below scatter plot
- [ ] **Verify**: Panel header shows "GGAnalyzer Calculation Trace"
- [ ] Expand panel
- [ ] **Verify**: 4 sanity checks visible:
  - config_matches_vehicle
  - data_quality
  - g_force_plausible
  - utilization_plausible
- [ ] **Verify**: Each check has impact description
- [ ] Read "g_force_plausible" impact
- [ ] **Verify**: Impact mentions "G-forces above 3.0g are physically impossible on street tires"

### Test 3.4: Corner Analysis Audit Mode
- [ ] Navigate to http://127.0.0.1:8000/corner-analysis
- [ ] **Verify**: Audit toggle visible in control panel
- [ ] Select a session file
- [ ] **Verify**: Corner analysis loads (cards or table view)
- [ ] Turn audit toggle ON
- [ ] **Verify**: Audit panel appears below results
- [ ] Expand panel
- [ ] **Verify**: 3 sanity checks visible:
  - gps_quality
  - corner_count_plausible
  - corner_speeds_plausible
- [ ] **Verify**: Each has impact description
- [ ] Toggle OFF
- [ ] **Verify**: Panel disappears

### Test 3.5: Audit Toggle Persistence
- [ ] With audit toggle ON, navigate away to http://127.0.0.1:8000/vehicles
- [ ] Return to http://127.0.0.1:8000/analysis
- [ ] **Verify**: Audit toggle is still ON (state persisted in localStorage)
- [ ] Run any analysis
- [ ] **Verify**: Audit panel appears automatically
- [ ] Turn toggle OFF
- [ ] Open browser DevTools → Application → Local Storage → http://127.0.0.1:8000
- [ ] **Verify**: Key "telemetry_audit_enabled" exists with value "false"

### Test 3.6: Network Traffic Verification
- [ ] Open DevTools → Network tab
- [ ] Turn audit toggle OFF
- [ ] Select session, run "Power" analysis
- [ ] **Verify**: Request URL is `/api/analyze/power/{filename}` (NO trace parameter)
- [ ] Turn audit toggle ON
- [ ] Run "Power" analysis again
- [ ] **Verify**: Request URL is `/api/analyze/power/{filename}?trace=true`
- [ ] Check response in Network tab
- [ ] **Verify**: Response JSON has `_trace` key with `inputs`, `config`, `intermediates`, `sanity_checks`

**Expected Result**: Audit mode works on all 3 pages, toggle persists, trace data appears when enabled

---

## 4. G-G Diagram Enhancements (feat-061, feat-061a, feat-061b)

**Feature**: Friction circle analysis with quadrant breakdown and lap comparison

### Test 4.1: Basic G-G Diagram
- [ ] Navigate to http://127.0.0.1:8000/gg-diagram
- [ ] Select a session file
- [ ] **Verify**: Scatter plot displays lat_acc vs lon_acc
- [ ] **Verify**: Reference circle overlay (max g capability)
- [ ] **Verify**: Data points colored by speed gradient (blue to red)
- [ ] **Verify**: Four quadrant cards show:
  - Left Lateral (turning left)
  - Right Lateral (turning right)
  - Braking
  - Acceleration
- [ ] **Verify**: Each quadrant shows utilization % and max g

### Test 4.2: Lap Filter
- [ ] **Verify**: "Filter by Lap" dropdown appears above diagram
- [ ] **Verify**: Dropdown populates with lap numbers (Lap 1, Lap 2, etc.)
- [ ] Select "Lap 2"
- [ ] **Verify**: Diagram updates showing only Lap 2 data
- [ ] **Verify**: Quadrant stats update
- [ ] Select "All Laps"
- [ ] **Verify**: Full session data returns

### Test 4.3: Track Map Integration (Low Utilization Zones)
- [ ] Scroll down to "Low Utilization Zones" section
- [ ] **Verify**: Mini track map displays
- [ ] **Verify**: Track outline visible (GPS trace)
- [ ] **Verify**: Zone markers appear on map (if low utilization zones exist)
- [ ] Click a zone marker
- [ ] **Verify**: Zone highlights on map
- [ ] **Verify**: Corresponding zone in list highlights

### Test 4.4: Lap Comparison
- [ ] Scroll to "Lap Comparison" section
- [ ] Select Lap 1 in first dropdown
- [ ] Select Lap 3 in second dropdown
- [ ] Click "Compare"
- [ ] **Verify**: Overlay chart shows both laps (different colors)
- [ ] **Verify**: Comparison stats table appears
- [ ] **Verify**: Differences shown (ΔLat G, ΔLon G, ΔUtilization)

**Expected Result**: G-G diagram shows friction circle, lap filtering works, track map renders

---

## 5. Corner Analysis (feat-062)

**Feature**: Auto-detected corners with entry/apex/exit speeds and driver behavior

### Test 5.1: Corner Cards View
- [ ] Navigate to http://127.0.0.1:8000/corner-analysis
- [ ] Select a session file
- [ ] **Verify**: "View Mode" toggle at top (Cards / Table / Chart)
- [ ] Ensure "Cards" mode selected
- [ ] **Verify**: Multiple corner cards display (one per corner)
- [ ] **Verify**: Each card shows:
  - Corner number
  - Entry speed (green bar)
  - Apex speed (yellow bar)
  - Exit speed (blue bar)
  - Time in corner
  - Throttle pickup point %
- [ ] **Verify**: Cards have "Lift" or "Trail Brake" badges if applicable
- [ ] Scroll through all corners
- [ ] **Verify**: No "undefined" or "NaN" values

### Test 5.2: Comparison Table View
- [ ] Click "Table" view mode
- [ ] **Verify**: Table displays all corners in rows
- [ ] **Verify**: Columns: Corner #, Entry Speed, Apex, Exit, Time, Throttle Pickup, Notes
- [ ] **Verify**: Sortable by clicking column headers
- [ ] Sort by "Exit Speed"
- [ ] **Verify**: Table reorders (highest exit speed on top)

### Test 5.3: Speed Chart View
- [ ] Click "Chart" view mode
- [ ] **Verify**: Bar chart displays corner speeds
- [ ] **Verify**: Three bars per corner (entry, apex, exit)
- [ ] **Verify**: X-axis shows corner numbers
- [ ] **Verify**: Y-axis shows speed in mph
- [ ] **Verify**: Legend shows Entry/Apex/Exit colors

### Test 5.4: Corner Statistics
- [ ] Scroll to statistics cards at top
- [ ] **Verify**: Four cards display:
  - Total Corners
  - Average Entry Speed
  - Average Apex Speed
  - Average Exit Speed
- [ ] **Verify**: Values are reasonable
- [ ] **Verify**: Card with "Lifts Detected" shows count
- [ ] **Verify**: Card with "Trail Brakes" shows count

**Expected Result**: Corner analysis auto-detects corners, displays speeds, shows driver behavior

---

## 6. Delta Track Map (feat-060)

**Feature**: Lap comparison showing time gained/lost overlaid on track map

### Test 6.1: Access from Lap Analysis
- [ ] Navigate to http://127.0.0.1:8000/analysis
- [ ] Select session, click "Laps"
- [ ] **Verify**: Lap table displays with lap numbers and times
- [ ] Scroll to "Lap Comparison" section
- [ ] Select two laps (e.g., Lap 2 vs Lap 3)
- [ ] **Verify**: Delta track map appears
- [ ] **Verify**: Track outline colored in gradient (green to red)
- [ ] **Verify**: Green sections = reference lap faster
- [ ] **Verify**: Red sections = comparison lap faster
- [ ] **Verify**: Legend shows color scale with time delta range
- [ ] **Verify**: Lap labels show which is which (e.g., "Lap 2 vs Lap 3")

### Test 6.2: Interpretation
- [ ] Identify a green section on track
- [ ] **Verify**: Tooltip or legend indicates reference lap was faster here
- [ ] Identify a red section
- [ ] **Verify**: Comparison lap was faster here
- [ ] **Verify**: Track shape matches actual circuit (Road America, etc.)

**Expected Result**: Delta map visually shows where time was gained/lost on track

---

## 7. Feedback Widget (feat-070)

**Feature**: Embedded feedback widget on all pages

### Test 7.1: Widget Visibility
- [ ] Navigate to http://127.0.0.1:8000/analysis
- [ ] **Verify**: Floating action button (FAB) visible in bottom-right corner
- [ ] **Verify**: FAB has feedback icon or text
- [ ] Hover over FAB
- [ ] **Verify**: Tooltip or color change on hover

### Test 7.2: Widget Interaction
- [ ] Click FAB
- [ ] **Verify**: Feedback panel opens (slides in or modal appears)
- [ ] **Verify**: Panel shows "Telemetry Analyzer" as app name
- [ ] **Verify**: Screenshot preview visible (current page)
- [ ] **Verify**: Text area for feedback
- [ ] **Verify**: Option for voice recording (if Whisper available)
- [ ] **Verify**: Auto-context shows current page URL
- [ ] Type "Test feedback message"
- [ ] **Verify**: Text appears in textarea
- [ ] Click "Submit" or "Send"
- [ ] **Verify**: Success message appears OR widget closes
- [ ] **Verify**: No console errors

### Test 7.3: Widget on All Pages
- [ ] Navigate to http://127.0.0.1:8000/vehicles
- [ ] **Verify**: FAB visible
- [ ] Navigate to http://127.0.0.1:8000/gg-diagram
- [ ] **Verify**: FAB visible
- [ ] Navigate to http://127.0.0.1:8000/corner-analysis
- [ ] **Verify**: FAB visible
- [ ] Navigate to http://127.0.0.1:8000/parquet
- [ ] **Verify**: FAB visible

**Expected Result**: Feedback widget appears on all pages, can submit feedback

**Note**: If feedback service not running, widget may show connection error - this is expected.

---

## 8. Multi-Tier Storage (arch-004) - Backend Verification

**Feature**: Raw/summary/merged parquet tiers for high-frequency data

### Test 8.1: Check Parquet File Attributes
- [ ] Navigate to http://127.0.0.1:8000/parquet
- [ ] Select any processed parquet file
- [ ] Click "View Data"
- [ ] Scroll to "File Metadata" section (if available)
- [ ] **Verify**: "native_rates" attribute exists
- [ ] **Verify**: Shows Hz values for each channel
- [ ] **Alternative**: Check via Python REPL:
  ```python
  import pandas as pd
  df = pd.read_parquet("data/exports/processed/test_session.parquet")
  print(df.attrs.get("native_rates"))
  # Should print dict like: {'GPS Speed': 10.0, 'RPM': 50.0, ...}
  ```

### Test 8.2: Verify No Regression in Normal Operation
- [ ] Run any analysis (shifts, laps, power)
- [ ] **Verify**: All analyses work normally
- [ ] **Verify**: No "tier" errors in console
- [ ] **Verify**: Speed/RPM data loads correctly

**Expected Result**: Tier infrastructure present but invisible to normal operation

---

## 9. Session Report Integration (arch-003)

**Feature**: Analyzer registry automatically includes all analyzers in reports

### Test 9.1: Full Report Includes All Analyzers
- [ ] Navigate to http://127.0.0.1:8000/analysis
- [ ] Select session, click "Full Report"
- [ ] **Verify**: Report includes at minimum:
  - Lap Analysis section
  - Shift Analysis section
  - Gear Analysis section
  - Power Analysis section
- [ ] With audit mode ON
- [ ] **Verify**: Sub-analyzer traces appear for each section
- [ ] **Verify**: No "corners" or "gg" sections (they have separate pages)

### Test 9.2: Backward Compatibility
- [ ] Check Full Report JSON via API:
  ```bash
  curl http://127.0.0.1:8000/api/analyze/report/test_session.parquet | jq '.'
  ```
- [ ] **Verify**: JSON has keys: `lap_analysis`, `shift_analysis`, `gear_analysis`, `power_analysis`
- [ ] **Verify**: JSON also has `sub_reports` dict with same data

**Expected Result**: Registry pattern works, backward compatibility maintained

---

## 10. Error Handling & Edge Cases

### Test 10.1: Missing Data Scenarios
- [ ] Navigate to http://127.0.0.1:8000/analysis
- [ ] Select a session known to have no GPS data
- [ ] Click "Laps"
- [ ] **Verify**: Error message appears (not a crash)
- [ ] **Verify**: Message says "GPS data not found" or similar (HTTP 422)
- [ ] **Verify**: Page remains functional

### Test 10.2: Invalid Session File
- [ ] In URL bar, manually change filename to nonexistent file:
  ```
  http://127.0.0.1:8000/analysis?session=nonexistent.parquet
  ```
- [ ] Click any analysis type
- [ ] **Verify**: "File not found" error (HTTP 404)
- [ ] **Verify**: No server crash
- [ ] **Verify**: Can still select valid session from dropdown

### Test 10.3: Audit Mode with Failing Checks
- [ ] If you have a session with anomalies (e.g., over-rev, implausible speeds):
  - [ ] Turn audit mode ON
  - [ ] Run analysis
  - [ ] **Verify**: Sanity checks show "fail" or "warn" status (red/yellow dots)
  - [ ] **Verify**: Impact description explains what's wrong
  - [ ] **Verify**: Panel header has red or yellow left border
- [ ] If no anomalous data available, this is expected behavior (all checks pass)

**Expected Result**: Errors handled gracefully, no server crashes

---

## 11. Performance & UX

### Test 11.1: Page Load Times
- [ ] Open DevTools → Network tab, disable cache
- [ ] Navigate to http://127.0.0.1:8000/analysis
- [ ] **Verify**: Page loads in < 2 seconds
- [ ] Select session, run "Full Report"
- [ ] **Verify**: Report loads in < 5 seconds (for typical session)
- [ ] Turn audit mode ON, run Full Report again
- [ ] **Verify**: Trace overhead is acceptable (< 2x slower than without trace)

### Test 11.2: Responsiveness
- [ ] Resize browser window to narrow width (mobile simulation)
- [ ] **Verify**: Navigation collapses to hamburger menu
- [ ] **Verify**: Charts remain readable
- [ ] **Verify**: Audit panels stack vertically
- [ ] **Verify**: No horizontal scrolling required

### Test 11.3: Console Errors
- [ ] Open DevTools → Console
- [ ] Navigate through all pages: analysis, gg-diagram, corner-analysis, vehicles, parquet, queue
- [ ] Run various analyses with audit mode ON and OFF
- [ ] **Verify**: No red console errors
- [ ] **Verify**: No uncaught exceptions
- [ ] **Verify**: Warnings (if any) are minor (e.g., Chart.js deprecation notices)

**Expected Result**: Smooth performance, no console errors, responsive design

---

## 12. Cross-Browser Testing (Optional)

If time permits, repeat key tests in:
- [ ] Chrome/Chromium
- [ ] Firefox
- [ ] Safari (if on macOS)

**Focus on**:
- Audit toggle functionality
- Chart.js rendering (G-G diagram, corner analysis charts)
- localStorage persistence (audit toggle state)

---

## Summary Checklist

### Critical Paths (Must Pass):
- [ ] Vehicle switching affects analysis
- [ ] Analysis dashboard loads all 4 types (shifts, laps, gears, power)
- [ ] Audit mode toggle appears on 3 pages (analysis, gg-diagram, corner-analysis)
- [ ] Audit panels show sanity checks with impact descriptions
- [ ] Audit toggle state persists in localStorage
- [ ] G-G diagram displays friction circle
- [ ] Corner analysis auto-detects corners
- [ ] Full report includes all sub-analyzers
- [ ] No console errors on any page

### Nice-to-Have (Should Pass):
- [ ] Feedback widget appears and is clickable
- [ ] Delta track map shows lap comparison
- [ ] Lap filter in G-G diagram works
- [ ] Track map in corner analysis renders
- [ ] Quadrant breakdown in G-G diagram
- [ ] Error messages are user-friendly

---

## Bug Reporting Template

If you find issues, report using this format:

**Bug Title**: [Brief description]

**Steps to Reproduce**:
1. Navigate to [page]
2. Click [element]
3. Observe [issue]

**Expected Result**: [What should happen]

**Actual Result**: [What actually happened]

**Browser**: [Chrome/Firefox/Safari version]

**Console Errors**: [Copy any red errors from DevTools Console]

**Screenshot**: [If applicable]

---

## Notes

- This UAT script covers **user-facing features only**
- Backend features (tiered storage, analyzer registry) are verified via API/code
- Total test time: **30-45 minutes** for full walkthrough
- Can be split across multiple sessions
- Prioritize "Critical Paths" section for quick smoke test

**Questions or issues?** Check browser DevTools Console for errors, review http://127.0.0.1:8000/health for service status.
