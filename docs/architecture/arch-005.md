# arch-005: API Integration Test Coverage

**Status**: Plan
**Date**: 2026-02-10
**Scope**: TestClient-based endpoint tests for all 6 routers + file management APIs

---

## Problem Statement

The API surface has 33+ endpoints across 6 routers and the main app, but only 2 routers have dedicated endpoint tests:
- **analysis** (5 endpoints): 5 tests in `test_analysis_api.py`
- **sessions** (7 endpoints): 10+ tests in `test_session_api.py`

Missing coverage:
- **visualization** (5 endpoints): 0 endpoint tests
- **parquet** (3 endpoints): 0 endpoint tests
- **queue** (7 endpoints): 0 endpoint tests
- **vehicles** (4 endpoints): 0 endpoint tests
- **file management** (7 endpoints in app.py): 0 endpoint tests

Most bugs in production APIs are wiring bugs (wrong parameter name, missing import, incorrect response structure), not logic bugs. These are caught by integration tests, not unit tests.

---

## Current Test Coverage

### Tested endpoints (analysis router)
- `GET /api/analyze/shifts/{filename}` — file not found (404), missing data (422), success
- `GET /api/analyze/laps/{filename}` — file not found (404), success
- `GET /api/analyze/gears/{filename}` — file not found (404), success
- `GET /api/analyze/power/{filename}` — file not found (404), success
- `GET /api/analyze/report/{filename}` — file not found (404), success

### Tested endpoints (sessions router)
- Full CRUD: import, list, get, laps, classify, confirm, setup get/save

### Untested endpoints (28 total)
1. Analysis: lap compare, trace parameter
2. Visualization: track-map (SVG/JSON), delta track-map, gg-diagram, corner-analysis, corner-track-map
3. Parquet: list, view, summary
4. Queue: stats, jobs list, job detail, retry, retry-all, delete, clear-completed
5. Vehicles: list, get, update, set active
6. App: upload, files list, process, file details, delete, stats, health

---

## Proposed Test Structure

### File organization

```
tests/
  test_api_visualization.py     # NEW — 5 visualization endpoints
  test_api_parquet.py            # NEW — 3 parquet endpoints
  test_api_queue.py              # NEW — 7 queue endpoints
  test_api_vehicles.py           # NEW — 4 vehicle endpoints
  test_api_health.py             # NEW — health check + stats
```

### Shared fixtures

All test files use FastAPI `TestClient` with shared fixtures:

```python
@pytest.fixture
def client():
    """TestClient for the FastAPI app."""
    from src.main.app import app
    return TestClient(app)

@pytest.fixture
def sample_parquet(tmp_path):
    """Create a synthetic parquet file and make it discoverable."""
    # Includes: GPS Lat/Lon, RPM, Speed, LatAcc, LonAcc, throttle
    # Located in tmp_path, monkeypatched into find_parquet_file
```

### Testing strategy per endpoint type

**GET endpoints (read-only)**:
1. Happy path: valid params → 200 + expected response keys
2. File not found: invalid filename → 404
3. Missing data: parquet without required columns → 422 or graceful degradation
4. Optional params: verify default behavior and with each option

**POST/PUT endpoints (state-changing)**:
1. Happy path: valid body → 200/201 + expected response
2. Missing required fields → 400/422
3. Invalid values → 400/422
4. Resource not found → 404

**DELETE endpoints**:
1. Happy path: existing resource → 200
2. Not found → 404

---

## Endpoint Coverage Plan

### Visualization Router (5 endpoints)

| Test | Endpoint | Scenario |
|------|----------|----------|
| test_track_map_svg | GET /api/track-map/{f} | SVG response with GPS data |
| test_track_map_json | GET /api/track-map/{f} | format=json |
| test_track_map_not_found | GET /api/track-map/{f} | File not found → 404 |
| test_track_map_color_by | GET /api/track-map/{f} | color_by=rpm |
| test_delta_map | GET /api/track-map/delta/{f} | lap_a=1&lap_b=2 |
| test_delta_map_not_found | GET /api/track-map/delta/{f} | File not found → 404 |
| test_gg_diagram_json | GET /api/gg-diagram/{f} | format=json |
| test_gg_diagram_not_found | GET /api/gg-diagram/{f} | File not found → 404 |
| test_gg_diagram_trace | GET /api/gg-diagram/{f} | trace=true |
| test_corner_analysis | GET /api/corner-analysis/{f} | Valid parquet |
| test_corner_analysis_not_found | GET /api/corner-analysis/{f} | File not found → 404 |
| test_corner_track_map | GET /api/corner-track-map/{f} | SVG response |

### Parquet Router (3 endpoints)

| Test | Endpoint | Scenario |
|------|----------|----------|
| test_parquet_list | GET /api/parquet/list | Returns list of files |
| test_parquet_view | GET /api/parquet/view/{f} | Returns rows + columns |
| test_parquet_view_pagination | GET /api/parquet/view/{f} | limit=10&offset=5 |
| test_parquet_view_not_found | GET /api/parquet/view/{f} | File not found → 404 |
| test_parquet_summary | GET /api/parquet/summary/{f} | Returns stats |
| test_parquet_summary_not_found | GET /api/parquet/summary/{f} | File not found → 404 |

### Queue Router (7 endpoints)

| Test | Endpoint | Scenario |
|------|----------|----------|
| test_queue_stats | GET /api/queue/stats | Returns count by status |
| test_queue_jobs_list | GET /api/queue/jobs | Returns job list |
| test_queue_jobs_filter | GET /api/queue/jobs | status=pending |
| test_queue_job_detail | GET /api/queue/jobs/{id} | Returns single job |
| test_queue_job_not_found | GET /api/queue/jobs/{id} | 404 |
| test_queue_retry | POST /api/queue/jobs/{id}/retry | Retry failed job |
| test_queue_retry_all | POST /api/queue/retry-all | Retry all failed |
| test_queue_delete | DELETE /api/queue/jobs/{id} | Delete job |
| test_queue_clear | POST /api/queue/clear-completed | Clear completed |

### Vehicles Router (4 endpoints)

| Test | Endpoint | Scenario |
|------|----------|----------|
| test_vehicles_list | GET /api/vehicles | Returns 5 vehicles + active |
| test_vehicle_get | GET /api/vehicles/{id} | Returns vehicle config |
| test_vehicle_not_found | GET /api/vehicles/{id} | 404 |
| test_vehicle_update | PUT /api/vehicles/{id} | Update weight → success |
| test_vehicle_set_active | PUT /api/vehicles/active | Switch vehicle |
| test_vehicle_set_active_invalid | PUT /api/vehicles/active | Invalid ID → 404 |

### Health + Stats (app.py)

| Test | Endpoint | Scenario |
|------|----------|----------|
| test_health_check | GET /health | Returns status=healthy |
| test_stats | GET /api/stats | Returns stats dict |

---

## Acceptance Criteria

1. **Visualization tests exist** — `test_api_visualization.py` with >= 10 tests covering all 5 endpoints
2. **Parquet tests exist** — `test_api_parquet.py` with >= 5 tests covering all 3 endpoints
3. **Queue tests exist** — `test_api_queue.py` with >= 7 tests covering all 7 endpoints
4. **Vehicle tests exist** — `test_api_vehicles.py` with >= 5 tests covering all 4 endpoints
5. **Health tests exist** — `test_api_health.py` with >= 2 tests
6. **All tests use TestClient** — FastAPI TestClient, no manual HTTP calls
7. **404 scenarios covered** — Every endpoint that takes a filename/ID has a not-found test
8. **Response structure validated** — Tests check for expected keys in JSON responses
9. **No test pollution** — Tests use fixtures/mocks to avoid modifying real data
10. **All existing tests pass** — Zero regressions in current 949 tests

---

## Risks

1. **Test isolation for queue** — Queue uses SQLite singleton. Tests must use temporary DB or mock. Mitigation: monkeypatch `get_queue()` to return fresh queue with temp DB.

2. **Vehicle state mutation** — `set_active_vehicle` modifies global state. Tests must restore original active vehicle. Mitigation: save/restore active vehicle in fixture teardown.

3. **Parquet file discovery** — `find_parquet_file()` in deps.py searches real data directory. Tests must mock this. Mitigation: monkeypatch to return temp fixtures.

4. **HTML page tests** — Page routes return HTML templates. Testing structure (not rendering) is sufficient. Mitigation: just check status_code == 200 and content_type.

---

## Files Modified (estimated)

| File | Change type |
|------|------------|
| `tests/test_api_visualization.py` | **New** — visualization endpoint tests |
| `tests/test_api_parquet.py` | **New** — parquet endpoint tests |
| `tests/test_api_queue.py` | **New** — queue endpoint tests |
| `tests/test_api_vehicles.py` | **New** — vehicle endpoint tests |
| `tests/test_api_health.py` | **New** — health + stats tests |

---

## Dependencies

- No code changes required — only new test files.
- Uses existing `TestClient` pattern from `test_analysis_api.py` and `test_session_api.py`.
