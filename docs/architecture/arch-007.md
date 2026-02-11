# arch-007: Session-Centric Analysis UX

## Status
- **Phase**: Plan
- **Created**: 2026-02-11
- **Dependencies**: arch-001 (vehicle config), arch-002 (SessionDataLoader), arch-004 (tiered storage)

## Problem Statement

The current analysis workflow is **file-centric** when it should be **session-centric**:

### Current Pain Points
1. **Repetitive file selection**: Every analysis page (GG diagram, corner analysis, lap analysis) has its own file dropdown
2. **No analysis context**: User must re-select the same file when navigating between analysis types
3. **No cross-session analysis**: Can't easily answer "show me all Road America Turn 3 data from 2025"
4. **Disconnected import flow**: File import → session browse → analysis are separate, unlinked workflows
5. **Limited session metadata**: Import doesn't capture track, conditions, driver, setup details
6. **No comparison mode**: Can't easily baseline vs current, or multi-session comparison
7. **No shareability**: Can't bookmark or share a specific analysis state

### User's Vision
> "I don't want to pick the file I'm going to use every time. From the very beginning it's: what are you looking at? What are you trying to accomplish? It might be a specific session, multiple sessions, or a year's worth of Road America Turn 3."

The workflow should be:
1. **Define scope**: Pick session(s) or filter criteria
2. **Analyze**: All analysis tools operate on that scope
3. **Compare**: Optionally add baseline or comparison sessions
4. **Share**: URL captures the complete analysis state

---

## Current System Architecture

### What Exists (Leverage These)

**SessionDatabase** (`src/session/session_database.py`):
- SQLite backend (`data/sessions.db`)
- Tables: `sessions`, `laps`, `stints`, `setups`
- Session fields: `id, parquet_path, track_id, vehicle_id, session_date, session_type, import_status, notes, best_lap_time, total_laps`
- API: `GET /api/v2/sessions` with filtering, `POST /api/v2/sessions/import`, `GET /api/v2/sessions/{id}`

**SessionDataLoader** (`src/services/session_data_loader.py`):
- Central parquet loading abstraction
- Auto-discovers channels, resolves speed units
- Returns `SessionChannels` dataclass with `df, time, speed_mph, lat, lon, rpm, throttle, lat_acc, lon_acc, column_map`
- Tier support: `load(path, tier="merged"|"summary"|"raw")`

**VehicleDatabase** (`src/config/vehicles.py`):
- 5 vehicle profiles with engine specs, transmission setups, weight, tire size
- Active vehicle switching API
- `vehicle.current_setup` + `vehicle.alternative_setups` for configuration tracking

**TrackDatabase** (`src/config/tracks.py`):
- 6 tracks with GPS bounds, corners, lap time ranges
- Auto-detection: `detect_track(lat, lon)` via bounding box + start/finish proximity

**Analysis Framework** (`src/features/`):
- 7 analyzers inheriting `BaseAnalyzer`
- All support `analyze_from_parquet(path, include_trace=bool)`
- 26 sanity checks across all analyzers

### What's Fragmented (Fix These)

**File Selection Pattern** (repeated in 3+ pages):
```javascript
// analysis.html, gg_diagram.html, corner_analysis.html
async function loadFileList() {
  const files = await fetch('/api/parquet/list');
  // Populate dropdown with filenames
}
function runAnalysis() {
  const filename = document.getElementById('file-select').value;
  fetch(`/api/analyze/report/${filename}`);
}
```

**No Analysis Context**:
- Each page loads files independently
- No persistence when navigating analysis.html → gg_diagram.html
- No "working session" concept

**Limited Session Metadata Capture**:
- Import captures: `track_id, vehicle_id, session_date, session_type`
- **Missing**: driver, weather/conditions, setup snapshot, session notes, run number

**No Comparison Support**:
- Can't select "baseline" vs "current" sessions
- Can't overlay multiple sessions in analysis

---

## Proposed Architecture

### 1. Analysis Context System

**New Context Model**:
```python
# src/context/analysis_context.py
@dataclass
class AnalysisScope:
    """Defines what data to analyze"""
    mode: str  # "single" | "multi" | "filtered"
    session_ids: List[str]  # Primary session(s)
    baseline_session_id: Optional[str] = None  # For comparison
    filters: Optional[Dict] = None  # For cross-session queries

@dataclass
class AnalysisContext:
    """User's current analysis state"""
    scope: AnalysisScope
    active_session_id: str  # Which session is "primary"
    created_at: datetime
    last_accessed: datetime
```

**Context Storage**:
- **Backend**: Session-based storage (Flask session or Redis for multi-user)
- **Frontend**: `sessionStorage` for current tab, `localStorage` for cross-tab persistence
- **URL**: Query params for shareability: `?session=abc123&baseline=def456&view=gg`

**Context API** (`src/main/routers/context.py`):
```python
POST /api/context/set
  Body: {scope: {mode: "single", session_ids: ["abc123"]}}

GET /api/context/current
  Returns: {scope: {...}, active_session_id: "abc123"}

POST /api/context/add-comparison
  Body: {session_id: "def456", role: "baseline"}

DELETE /api/context/clear
```

---

### 2. Enhanced Session Metadata

**Extended Session Model**:
```python
# Additions to src/session/session_database.py
class Session:
    # Existing fields...
    id: str
    parquet_path: str
    track_id: Optional[str]
    vehicle_id: Optional[str]
    session_date: datetime
    session_type: str  # "practice" | "qualifying" | "race" | "test"
    import_status: str

    # NEW FIELDS
    driver_name: Optional[str]  # "Chris" | "Guest Driver"
    run_number: Optional[int]  # 1, 2, 3... for same day
    weather_conditions: Optional[str]  # "Dry, 72F, Sunny"
    track_conditions: Optional[str]  # "Green" | "Rubbered In" | "Damp"
    setup_snapshot: Optional[Dict]  # Copy of vehicle setup at time of session
    tire_pressures: Optional[Dict]  # {"FL": 32, "FR": 32, "RL": 28, "RR": 28}
    notes: Optional[str]  # Free-form user notes
    tags: Optional[List[str]]  # ["baseline", "new-dampers", "wet-setup"]
```

**Setup Snapshot Strategy**:
- On import, capture `vehicle.current_setup.to_dict()` → store in `setup_snapshot` field
- Allows post-hoc comparison even if vehicle config changes later
- Display in session detail: "Setup: 2024 Gearing (3.73 final drive, 3450 lbs)"

**Migration Plan**:
```sql
ALTER TABLE sessions ADD COLUMN driver_name TEXT;
ALTER TABLE sessions ADD COLUMN run_number INTEGER;
ALTER TABLE sessions ADD COLUMN weather_conditions TEXT;
ALTER TABLE sessions ADD COLUMN track_conditions TEXT;
ALTER TABLE sessions ADD COLUMN setup_snapshot TEXT;  -- JSON blob
ALTER TABLE sessions ADD COLUMN tire_pressures TEXT;  -- JSON blob
ALTER TABLE sessions ADD COLUMN tags TEXT;  -- JSON array
```

---

### 3. Session-Centric Import Flow

**New Import Wizard** (replace `/sessions/import`):

**Step 1: File Selection**
- Upload new XRK (queues extraction) OR select existing parquet
- Auto-detect track from GPS → pre-fill track dropdown
- Show parquet preview: duration, channels, sample count

**Step 2: Session Details** (NEW - expanded form)
```
Track: [Road America ▼]  (auto-detected)
Vehicle: [2019 Mustang GT ▼]  (from active vehicle)
Date: [2025-06-15] Time: [14:30]
Session Type: [Practice ▼] Run #: [3]

Driver: [Chris ▼]  (dropdown from recent drivers + "Add new")
Weather: [Dry, 75°F, Partly Cloudy]
Track Condition: [Rubbered In ▼]

Setup: [Current Setup (2024 Gearing) ▼]  (from vehicle.all_setups)
Tire Pressures:
  FL: [32] FR: [32]  psi
  RL: [28] RR: [28]  psi

Tags: [baseline] [new-dampers] [±]

Notes:
[First session with new Penske shocks, ran 10W front / 60W rear...]
```

**Step 3: Confirmation**
- Show summary card
- "Import & Analyze" → creates session + redirects to analysis with context set
- "Import Only" → creates session, returns to session list

**API Enhancement**:
```python
POST /api/v2/sessions/import
Body: {
  parquet_path: "...",
  track_id: "road-america",
  vehicle_id: "mustang-gt-2019",
  session_date: "2025-06-15T14:30:00",
  session_type: "practice",
  driver_name: "Chris",
  run_number: 3,
  weather_conditions: "Dry, 75F, Partly Cloudy",
  track_conditions: "Rubbered In",
  setup_snapshot: {...},  # Auto-captured from vehicle
  tire_pressures: {"FL": 32, "FR": 32, "RL": 28, "RR": 28},
  tags: ["baseline", "new-dampers"],
  notes: "First session with new Penske shocks..."
}
```

---

### 4. Session Selector Component

**New Global UI Component** (replaces all file dropdowns):

```html
<!-- templates/components/session_selector.html -->
<div id="session-selector" class="session-selector">
  <div class="selector-header">
    <h3>Analysis Scope</h3>
    <button id="change-scope-btn">Change</button>
  </div>

  <div class="current-scope">
    <!-- Single session mode -->
    <div class="session-card mini">
      <span class="track-badge">Road America</span>
      <span class="date">Jun 15, 2025 - Run #3</span>
      <span class="vehicle">2019 Mustang GT</span>
      <span class="meta">23 laps, 45:32 duration</span>
    </div>

    <!-- Comparison mode -->
    <div class="comparison-indicator">
      <span>vs</span>
      <div class="session-card mini baseline">
        <span class="track-badge">Road America</span>
        <span class="date">Jun 1, 2025 - Run #2</span>
        <span class="tag">baseline</span>
      </div>
    </div>
  </div>
</div>
```

**Scope Selection Modal**:
```html
<div id="scope-modal" class="modal">
  <h2>Select Analysis Scope</h2>

  <!-- Tab 1: Single Session -->
  <div class="tab-content" id="single-session">
    <div class="session-browser">
      <div class="filters">
        Track: [All ▼]  Vehicle: [All ▼]  Driver: [All ▼]
        Date Range: [Last 30 days ▼]  Tags: [Any]

        <input type="search" placeholder="Search sessions...">
      </div>

      <div class="session-list">
        <!-- Session cards with: track, date, vehicle, laps, best lap, tags -->
        <div class="session-card" data-session-id="abc123">
          <div class="card-header">
            <span class="track-badge">Road America</span>
            <span class="date">Jun 15, 2025 14:30</span>
          </div>
          <div class="card-body">
            <span class="vehicle">2019 Mustang GT</span>
            <span class="driver">Chris - Run #3</span>
            <span class="stats">23 laps, best 2:14.3</span>
            <div class="tags">
              <span class="tag">new-dampers</span>
            </div>
          </div>
          <div class="card-actions">
            <button class="select-primary">Select</button>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Tab 2: Comparison Mode -->
  <div class="tab-content" id="comparison-mode">
    <div class="comparison-builder">
      <div class="role-section">
        <h4>Primary Session</h4>
        <div class="selected-session">
          <!-- Same card format -->
        </div>
      </div>

      <div class="role-section">
        <h4>Baseline (optional)</h4>
        <button id="add-baseline">+ Add Baseline</button>
        <!-- Session browser filtered to same track -->
      </div>

      <div class="role-section">
        <h4>Additional Comparisons (up to 3 total)</h4>
        <button id="add-comparison">+ Add Session</button>
      </div>
    </div>
  </div>

  <!-- Tab 3: Cross-Session Query (Advanced) -->
  <div class="tab-content" id="filtered-mode">
    <div class="query-builder">
      <h4>Analyze across multiple sessions:</h4>
      Track: [Road America ▼]
      Corner: [Turn 3 ▼]  (from track.corners)
      Date Range: [2025-01-01] to [2025-12-31]
      Vehicle: [All ▼]
      Tags: [baseline] [competition] (multi-select)

      <div class="preview">
        Found 47 sessions matching criteria
        <button id="preview-sessions">Preview List</button>
      </div>
    </div>
  </div>
</div>
```

---

### 5. Analysis Pages Refactor

**Before** (analysis.html, gg_diagram.html, corner_analysis.html):
```javascript
// Each page loads files independently
async function loadFileList() {
  const files = await fetch('/api/parquet/list');
  populateDropdown(files);
}
```

**After**:
```javascript
// All pages use shared context
async function initPage() {
  const context = await fetch('/api/context/current');

  if (!context.scope) {
    // No context set - prompt to select
    showScopeSelector();
    return;
  }

  // Render scope indicator (sticky header)
  renderScopeIndicator(context);

  // Load analysis using context
  runAnalysis(context.active_session_id);
}

// Shared scope indicator component
function renderScopeIndicator(context) {
  const indicator = document.getElementById('scope-indicator');
  indicator.innerHTML = `
    <div class="scope-summary">
      Analyzing: <strong>${context.scope.session_name}</strong>
      ${context.scope.baseline ? ` vs ${context.scope.baseline_name}` : ''}
      <button onclick="changeScopeModal()">Change</button>
    </div>
  `;
}
```

**Analysis API Changes**:

**Current**:
```python
GET /api/analyze/report/{filename}
GET /api/gg-diagram/{filename}
GET /api/corner-analysis/{filename}?track_name={track}
```

**New** (backward compatible):
```python
# Session-based (preferred)
GET /api/analyze/report?session_id={id}&trace={bool}
GET /api/gg-diagram?session_id={id}&lap={n}&trace={bool}
GET /api/corner-analysis?session_id={id}&trace={bool}

# Context-based (uses active context)
GET /api/analyze/report?use_context=true&trace={bool}
GET /api/gg-diagram?use_context=true&lap={n}&trace={bool}

# Comparison mode
GET /api/analyze/report?session_id={id}&baseline_id={id2}
GET /api/gg-diagram?session_id={id}&overlay_sessions={id2,id3}

# Legacy (still works)
GET /api/analyze/report/{filename}
```

**Backend Implementation**:
```python
# src/main/routers/analysis.py
@router.get("/api/analyze/report")
async def get_report(
    session_id: Optional[str] = None,
    filename: Optional[str] = None,  # Legacy
    baseline_id: Optional[str] = None,
    use_context: bool = False,
    trace: bool = False
):
    if use_context:
        context = get_current_context()  # From session/Redis
        session_id = context.active_session_id
        baseline_id = context.scope.baseline_session_id

    if session_id:
        session = session_db.get_session(session_id)
        parquet_path = session.parquet_path
    elif filename:
        parquet_path = find_parquet_file(filename)
    else:
        raise HTTPException(400, "Must provide session_id, filename, or use_context")

    # Load session data
    channels = SessionDataLoader.load(parquet_path)

    # Run analysis
    report = SessionReportGenerator().analyze_from_arrays(
        channels.time, channels.speed_mph, ...,
        filename=session.display_name if session_id else filename
    )

    # Add session metadata to response
    result = report.to_dict()
    if session_id:
        result['_session'] = {
            'id': session.id,
            'track': session.track_name,
            'vehicle': session.vehicle_id,
            'date': session.session_date,
            'driver': session.driver_name,
            'setup': session.setup_snapshot.get('name') if session.setup_snapshot else None
        }

    # Handle baseline comparison
    if baseline_id:
        baseline_session = session_db.get_session(baseline_id)
        baseline_channels = SessionDataLoader.load(baseline_session.parquet_path)
        baseline_report = SessionReportGenerator().analyze_from_arrays(
            baseline_channels.time, baseline_channels.speed_mph, ...
        )
        result['_baseline'] = baseline_report.to_dict()
        result['_baseline']['_session'] = {...}  # Baseline session metadata

    return result
```

---

### 6. URL State & Shareability

**URL Structure**:
```
# Single session
/analysis?session=abc123&view=report&trace=true

# Comparison mode
/gg-diagram?session=abc123&baseline=def456&lap=5

# Cross-session query
/corner-analysis?track=road-america&corner=3&date_start=2025-01-01&date_end=2025-12-31

# Legacy (still works)
/analysis?file=session_001.parquet
```

**Implementation**:
```javascript
// On page load
const params = new URLSearchParams(window.location.search);
if (params.has('session')) {
  // Set context from URL
  await fetch('/api/context/set', {
    method: 'POST',
    body: JSON.stringify({
      scope: {
        mode: 'single',
        session_ids: [params.get('session')],
        baseline_session_id: params.get('baseline')
      }
    })
  });
}

// On context change
function updateURL(context) {
  const url = new URL(window.location);
  url.searchParams.set('session', context.active_session_id);
  if (context.scope.baseline_session_id) {
    url.searchParams.set('baseline', context.scope.baseline_session_id);
  }
  window.history.pushState({}, '', url);
}
```

---

### 7. Quick Access Features

**Recent Sessions**:
```python
GET /api/v2/sessions/recent?limit=10
# Returns last 10 accessed sessions
# Tracks access via `last_accessed` timestamp update on any API call
```

**Favorites/Bookmarks**:
```python
# New table: session_bookmarks
# Columns: id, session_id, user_id, label, created_at

POST /api/v2/sessions/{session_id}/bookmark
Body: {label: "Best baseline lap - pre damper change"}

GET /api/v2/sessions/bookmarks
# Returns user's bookmarked sessions
```

**Quick Filters** (frontend):
```javascript
// Dropdown in navbar
<div class="quick-access">
  Recent:
  - Road America, Jun 15 (23 laps)
  - Blackhawk Farms, Jun 8 (18 laps)

  Bookmarks:
  - ⭐ Best baseline - Road America
  - ⭐ First session with new dampers

  Quick Filters:
  - All Road America sessions (47)
  - This month's sessions (12)
  - Tagged "baseline" (8)
</div>
```

---

## Implementation Plan

### Phase 1: Foundation (arch-007-exec-phase1)
**Backend Context System**
1. Create `AnalysisContext` and `AnalysisScope` models
2. Implement context storage (Flask session + Redis optional)
3. Create Context API router: `/api/context/{set,current,add-comparison,clear}`
4. Add session ID resolution to all analysis endpoints (maintain legacy filename support)

**Database Migration**
5. Add new session metadata columns (driver, run_number, weather, setup_snapshot, etc.)
6. Create migration script for existing sessions
7. Update `SessionDatabase` ORM methods

**Duration**: 2-3 days

---

### Phase 2: Session Import Enhancement (arch-007-exec-phase2)
**Import Wizard**
1. Redesign `/sessions/import` template with 3-step wizard
2. Auto-detect track, pre-fill vehicle from active vehicle
3. Add metadata form fields (driver, weather, setup, tires, tags)
4. Capture setup snapshot from `vehicle.current_setup.to_dict()`
5. Add "Import & Analyze" flow (creates session + sets context + redirects)

**API Updates**
6. Enhance `POST /api/v2/sessions/import` to accept new metadata fields
7. Add driver autocomplete endpoint: `GET /api/v2/drivers/recent`

**Duration**: 2 days

---

### Phase 3: Session Selector Component (arch-007-exec-phase3)
**Frontend Component**
1. Create `templates/components/session_selector.html`
2. Create `static/js/session_selector.js` (modal, filters, search)
3. Create `static/css/session_selector.css`
4. Implement 3 tabs: Single Session, Comparison Mode, Filtered Query
5. Add session card rendering with track badge, date, vehicle, stats, tags

**Integration**
6. Add scope indicator to all analysis pages (sticky header)
7. **DELETE** file dropdowns from analysis.html, gg_diagram.html, corner_analysis.html
8. Wire up context API calls
9. **Run data migration script** to import existing parquet files as sessions

**Duration**: 3 days

---

### Phase 4: Analysis Pages Refactor (arch-007-exec-phase4)
**Page Updates**
1. Remove file selection dropdowns from `analysis.html`, `gg_diagram.html`, `corner_analysis.html`
2. Add `initPage()` flow: check context → prompt if missing → load analysis
3. Update all analysis JavaScript to use context API
4. Add session metadata display to analysis results

**API Enhancements**
5. Add `?use_context=true` parameter to all analysis endpoints
6. Add `?baseline_id={id}` comparison support to report/gg-diagram endpoints
7. Include `_session` metadata block in all responses

**Duration**: 2-3 days

---

### Phase 5: URL State & Quick Access (arch-007-exec-phase5)
**URL State**
1. Add URL parameter parsing on page load (session, baseline, view, trace)
2. Update URL on context changes (pushState)
3. Add share button with copy-to-clipboard for current URL

**Quick Access**
4. Add `last_accessed` timestamp tracking to session API
5. Create `GET /api/v2/sessions/recent` endpoint
6. Create session bookmarks table + API
7. Add quick access dropdown to navbar (recent + bookmarks + filters)

**Duration**: 2 days

---

### Phase 6: Cross-Session Analysis (arch-007-exec-phase6)
**Advanced Filtering**
1. Implement filtered query builder (track + corner + date range + tags)
2. Create `POST /api/analyze/cross-session` endpoint
   - Accepts filter criteria
   - Loads matching sessions
   - Aggregates data (e.g., all Turn 3 corners from 47 sessions)
   - Returns combined analysis

**Example Use Case**: "Show me corner speed progression for Road America Turn 3 across 2025"
- Filter: track=road-america, corner=3, date_start=2025-01-01, date_end=2025-12-31
- Result: Chart with 47 data points (one per session), trend line, best/worst markers

**Duration**: 3 days

---

## Testing Strategy

### Unit Tests
- `test_context.py` - AnalysisContext model, storage, retrieval
- `test_session_metadata.py` - Extended session fields, setup snapshot capture
- `test_session_selector_api.py` - Context API endpoints

### Integration Tests
- `test_session_import_flow.py` - Full wizard workflow, metadata capture
- `test_analysis_with_context.py` - All analysis endpoints with session_id + use_context
- `test_comparison_mode.py` - Baseline comparison in report/GG diagram
- `test_url_state.py` - URL parsing, context restoration

### Frontend Tests (Manual + Playwright)
- Session selector modal: filtering, search, selection
- Scope indicator: display, change button, persistence
- Analysis pages: context load, no-context prompt, comparison display
- URL sharing: copy URL, open in new tab, restore state

---

## Success Criteria

### User Experience
1. ✅ User imports a session with full metadata (driver, weather, setup, tags) in one flow
2. ✅ User selects a session once, navigates between analysis/GG/corner pages without re-selecting
3. ✅ User adds a baseline session, sees comparison data in all analysis views
4. ✅ User bookmarks a session, returns to it later from quick access menu
5. ✅ User shares an analysis URL with another user, who sees identical state
6. ✅ User queries "all Road America Turn 3 sessions from 2025", gets aggregated analysis

### Technical
1. ✅ All analysis endpoints accept `session_id` OR `filename` (backward compatible)
2. ✅ Context persists across page navigation (session storage)
3. ✅ URL state fully captures analysis scope (session, baseline, filters, view)
4. ✅ Session metadata includes: driver, run#, weather, track conditions, setup snapshot, tire pressures, tags
5. ✅ Setup snapshot captured from vehicle config at import time
6. ✅ Zero regressions in existing tests (1021 tests still passing)

---

## Migration Path

### Complete Replacement Strategy
- **No side-by-side operation** - rip out old file-centric UI completely
- File dropdowns removed in Phase 3 (not hidden with feature flags)
- API maintains `/{filename}` endpoints for external tools only (not exposed in UI)

### Data Migration
**One-time script**: Import existing parquet files as sessions
```python
# scripts/migrate_existing_sessions.py
# For each parquet in data/exports/processed/
#   - Auto-detect track from GPS
#   - Set vehicle_id from config.ACTIVE_VEHICLE
#   - Prompt for: driver, date, session_type, tags
#   - Create session entry with best-guess metadata
```

**Execution**: Run migration script before deploying Phase 3 (session selector)

---

## Open Questions

1. **Multi-user support**: Do we need per-user contexts (requires auth)?
   - **Current assumption**: Single-user desktop app, shared context OK
   - **Future**: Add `user_id` column if multi-user needed

2. **Context expiration**: How long should context persist?
   - **Current assumption**: Session storage (until browser closed) + localStorage backup
   - **Option**: Add "Pin Context" feature for long-term persistence

3. **Cross-session data volume**: Analyzing 47 sessions at once - performance?
   - **Current assumption**: Load all into memory, compute aggregates
   - **Future**: If >100 sessions, implement lazy loading or pre-computed summaries

4. **Setup changes mid-session**: What if user changes vehicle setup during a track day?
   - **Current assumption**: Setup snapshot is per-session, captures config at import time
   - **Option**: Add "stint" support (multiple setup snapshots per session)

5. **Track corner definitions**: Currently in `tracks.json`, but user might want custom corner markers?
   - **Current assumption**: Use pre-defined corners from track database
   - **Future**: Add "custom corner" overlay tool

---

## Dependencies

### Leverages (Existing)
- ✅ SessionDatabase (arch-002)
- ✅ SessionDataLoader (arch-002)
- ✅ VehicleDatabase (arch-001)
- ✅ TrackDatabase (arch-001)
- ✅ BaseAnalyzer trace support (arch-006)
- ✅ Tiered storage foundation (arch-004)

### Blocks (Future)
- `feat-071`: "Session comparison dashboard" (needs comparison mode API from this)
- `feat-072`: "Setup impact analysis" (needs setup_snapshot from this)
- `feat-073`: "Track day summary report" (needs cross-session query from this)

---

## Acceptance Criteria (for arch-007-test)

1. **Session Import Flow**
   - [ ] User can import parquet with driver, weather, setup, tire pressures, tags
   - [ ] Setup snapshot auto-captured from vehicle config
   - [ ] "Import & Analyze" button sets context and redirects

2. **Session Selector**
   - [ ] Modal has 3 tabs: Single, Comparison, Filtered
   - [ ] Session cards show track, date, vehicle, laps, tags
   - [ ] Filters work: track, vehicle, driver, date range, tags, search
   - [ ] Can select primary + baseline sessions

3. **Analysis Context**
   - [ ] Scope indicator shows current session on all analysis pages
   - [ ] "Change" button opens selector modal
   - [ ] Context persists when navigating analysis → GG → corner pages
   - [ ] URL captures session_id + baseline_id parameters

4. **Analysis API**
   - [ ] All endpoints accept `?session_id={id}` parameter
   - [ ] All endpoints accept `?use_context=true` parameter
   - [ ] Legacy `/{filename}` endpoints still work
   - [ ] Comparison endpoints return `_baseline` data block
   - [ ] All responses include `_session` metadata block

5. **Quick Access**
   - [ ] Recent sessions endpoint returns last 10 accessed
   - [ ] Bookmark API creates/lists/deletes bookmarks
   - [ ] Navbar dropdown shows recent + bookmarks

6. **URL Sharing**
   - [ ] Opening `/analysis?session=abc123` sets context and loads analysis
   - [ ] Opening `/gg-diagram?session=abc123&baseline=def456` loads comparison
   - [ ] Share button copies current URL to clipboard

7. **Test Compatibility**
   - [ ] All tests updated to use session-based API (file-based deprecated)
   - [ ] Existing tests pass with session IDs (1021+ tests)
   - [ ] No regressions in analysis output quality

---

## Estimated Effort

- **Planning**: 1 day (this doc)
- **Execution**: 12-15 days (6 phases)
- **Testing**: 3-4 days (unit + integration + manual)
- **Total**: ~3 weeks for complete session-centric UX transformation

---

## Notes

This is the largest UX refactor since the project's inception. It transforms the system from a **file viewer** into a **session analysis platform**. The three-phase pipeline ensures we:
1. Design the complete vision first (this doc)
2. Execute in manageable phases (6 implementation phases)
3. Validate against acceptance criteria (comprehensive test coverage)

The payoff: users spend less time selecting files and more time analyzing data. The workflow becomes: **Select once → Analyze anywhere → Compare easily → Share instantly**.
