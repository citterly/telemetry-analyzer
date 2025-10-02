# 🚦 Telemetry Analyzer Roadmap (Revised)

This roadmap tracks project milestones, phases, work packages, and backlog.  
Status codes: 📝 Planned | ⏳ In Progress | ✅ Done  

---

## 🎯 Milestones

- **v0.1 — Foundation 🚀**  
  DLL working, FileManager imports metadata, unit tests in place.  

- **v0.2 — Quick Reports & RPM Analysis 📊**  
  Automated session reports (laps, RPM/shift analysis, gears, track map).  

- **v0.3 — Extended Data Access 🧩**  
  Normalized schema and cross-session database.  

- **v0.4 — Advanced Analysis 🧪**  
  Power/accel analysis, lap overlays, sector times.  

- **v0.5 — Visualization & Reporting 🎨**  
  Interactive UI + polished reports.  

- **v1.0 — Stable Release 🏁**  
  End-to-end tool with UI, CI/CD, docs complete.  

---

# 📍 Project Roadmap

## Phase 1 — Foundation 🚀
Focus: establish reliable IO layer and canonical data export.

- **WP1 — DLL Setup & Smoke Test ✅**  
  Scope: Verify AIM DLL can load, units.xml is available, XRK can open, channels list accessible.  
  Status: Completed. .xrk opens, channels listed, DLL closes cleanly.  

- **WP2 — File Manager Import & Metadata ✅**  
  Scope: Import XRK files, deduplicate by hash, persist metadata, generate session summaries.  
  Status: Completed. FileManager implemented, metadata JSON persisted, duplicate detection verified.  

- **WP3 — Channel Extraction & Unit Resolution 🏗️ In Progress**  
  Scope: Extract all XRK channels with solid unit mapping (heuristics + overrides). Confirm units are attached to DataFrame.  
  Status: In Progress. Exploratory script (`explore_xrk.py`) now displays channels with unit source (`override`, `heuristic`).  

- **WP4 — Canonical Metadata Extension ⏳ In Progress**  
  Scope: Normalize all channels to a common time base, export canonical Parquet, and extend metadata JSON with pointer to Parquet file and verified units.  
  Files: `src/session/session_builder.py`, `src/io/file_manager.py`, `src/utils/units_helper.py`, `src/config/config.py`  
  Artifacts: Canonical Parquet file(s) with normalized index, updated metadata JSON (with channel list + units map + Parquet path), smoke test confirmation.  
  Acceptance:  
   - All channels exported to Parquet with common, normalized index  
   - Units included in `df.attrs["units"]`  
   - Metadata JSON updated with canonical Parquet path  
   - Smoke test reload succeeds with no data/metadata loss  
  Status: In Progress  


---

## Phase 2: Quick Reports & RPM Analysis 📊

- **WP3: Lap Detection & Fastest Lap** 📝 Planned  
  - **Scope:** Split sessions into laps, identify fastest lap(s).  
  - **Files:** `analysis/lap_analyzer.py`, `file_manager.py`  
  - **Artifacts:** Lap summary JSON, fastest lap flag.  
  - **Acceptance:** Lap count matches stopwatch/manual within ±1 lap.  
  - **Milestone:** v0.2  

- **WP4: RPM Tracking & Shift Analysis** 📝 Planned  
  - **Scope:** Track RPM traces, detect shift points, compare fastest laps vs. slower laps.  
  - **Files:** `analysis/gear_calculator.py`, `config/config_analysis.py`  
  - **Artifacts:** RPM charts, shift point summary.  
  - **Acceptance:** Detected shifts align with onboard video within ±0.5s.  
  - **Milestone:** v0.2  

- **WP5: Quick Report Generator** 📝 Planned  
  - **Scope:** Automate generation of HTML/PDF reports including laps, gears, RPM/shift analysis, and track map.  
  - **Files:** `analysis/acceleration_analyzer.py`, plotting utilities, reporting scripts  
  - **Artifacts:** One report per XRK file, exportable.  
  - **Acceptance:** Drop XRK → report produced in one command.  
  - **Milestone:** v0.2  

---

## Phase 3: Extended Data Access 🧩

- **WP6: Normalized Schema & Export** 📝 Planned  
  - **Scope:** Ingest XRK data into normalized schema (Sessions, Laps, Channels, Samples).  
  - **Files:** `analysis/data_loader.py`, DB layer (SQLite).  
  - **Artifacts:** Normalized DB, CSV/JSON exports.  
  - **Acceptance:** Exports match Race Studio reference outputs.  
  - **Milestone:** v0.3  

---

## Phase 4: Advanced Analysis 🧪

- **WP7: Power/Acceleration Analysis** 📝 Planned  
  - **Scope:** Compute acceleration/power approximations (F=ma).  
  - **Files:** `analysis/acceleration_analyzer.py`  
  - **Artifacts:** Power vs. speed plots, accel zone summaries.  
  - **Acceptance:** Outputs consistent with previous manual analysis.  
  - **Milestone:** v0.4  

- **WP8: Lap Overlays & Sector Times** 📝 Planned  
  - **Scope:** Generate lap overlays and compute sector deltas.  
  - **Files:** Analysis modules + visualization utilities.  
  - **Artifacts:** Sector report, overlay plots.  
  - **Acceptance:** Sector deltas align with manual timing.  
  - **Milestone:** v0.4  

---

## Phase 5: Visualization & Reporting 🎨

- **WP9: Interactive UI** 📝 Planned  
  - **Scope:** Add FastAPI/Blazor-based UI for file upload and dashboard views.  
  - **Files:** `app.py`, UI templates.  
  - **Artifacts:** Local web interface.  
  - **Acceptance:** User can interact via browser.  
  - **Milestone:** v0.5  

- **WP10: Cross-Session Comparison Reports** 📝 Planned  
  - **Scope:** Compare sessions (e.g., qualifying vs. race).  
  - **Files:** Normalized DB + reporting utilities.  
  - **Artifacts:** Comparison reports (HTML/PDF).  
  - **Acceptance:** Reports show consistent session deltas.  
  - **Milestone:** v0.5  

---

## 📋 Release Readiness Checklists

### v0.1 Foundation 🚀
- [ ] DLL tested with sample XRK  
- [ ] File import working  
- [ ] Metadata persisted  
- [ ] Unit tests pass  
- [ ] Inventory & roadmap updated  

### v0.2 Quick Reports & RPM Analysis 📊
- [ ] Laps split, fastest lap identified  
- [ ] RPM traces plotted, shift points detected  
- [ ] Gear usage summarized  
- [ ] Track map visualization working  
- [ ] HTML/PDF quick report generated  

### v0.3 Extended Data Access 🧩
- [ ] Schema implemented  
- [ ] Full sample export validated  
- [ ] DB populated  

### v0.4 Advanced Analysis 🧪
- [ ] Power/accel analysis validated  
- [ ] Lap overlays functional  
- [ ] Sector deltas accurate  

### v0.5 Visualization & Reporting 🎨
- [ ] Interactive UI functional  
- [ ] Cross-session comparison available  
- [ ] Reports stable  

### v1.0 Stable Release 🏁
- [ ] CI/CD pipeline green  
- [ ] Documentation complete  
- [ ] Release tagged  

---

## Backlog Tracker

| Title                               | Type      | Phase             | Milestone | Status   |
|-------------------------------------|-----------|------------------|-----------|----------|
| Add unit tests for FileManager      | Feature   | Phase 1 (Foundation) | v0.1 | 📝 Planned |
| CI/CD Setup with GitHub Actions     | Tech Debt | Phase 1 (Foundation) | v0.1 | 📝 Planned |
| Consistent Config Handling Across Modules | Refactor  | Phase 3 (Data Access) | v0.3 | 📝 Planned |
| Add Visualization Tests with Static Data | Feature  | Phase 5 (Visualization) | v0.5 | 📝 Planned |
| Developer Documentation Cleanup     | Tech Debt | Ongoing          | v1.0 | 📝 Planned |

### Status Legend
- 📝 Planned  
- ⏳ In Progress  
- ✅ Done  

---

### Backlog — Technical Debt & Fixes

- **Fix DLL Channel Name & Unit Decode**  
  - **Type:** Bug  
  - **Phase:** Phase 1 — Foundation 🚀  
  - **Files Affected:**  
    - `src/session/session_builder.py` (`_extract_all_channels`)  
    - `src/io/dll_interface.py` (binding declarations)  
  - **Scope:** Ensure channel name and unit strings returned by the AIM DLL are properly declared as `c_char_p` and decoded as `bytes → str`.  
    - Currently DLL returns `int` (memory address) instead of `bytes`.  
    - `_safe_decode()` is a workaround but not the final fix.  
  - **Acceptance Criteria:**  
    - No decode warnings in smoke tests.  
    - All channels (regular + GPS) return human-readable names and units.  
    - `_safe_decode()` no longer needed.

- **Replace Deprecated `fillna(method=...)`** ✅ **Done (WP4)**  
  - **Type:** Tech Debt  
  - **Phase:** Phase 1 — Foundation 🚀  
  - **Files Affected:** `src/session/session_builder.py` (`_build_dataframe`)  
  - **Scope:** Replace `.fillna(method="ffill")` and `.fillna(method="bfill")` with `.ffill()` and `.bfill()`.  
  - **Acceptance Criteria:**  
    - No FutureWarning during smoke test.  
    - DataFrame gaps still fill consistently in both directions.  
  - **Status:** Completed in WP4.

- **Fix DLL Restype Declarations**  
  - **Type:** Tech Debt  
  - **Phase:** Phase 1 — Foundation 🚀  
  - **Files Affected:**  
    - `src/io/dll_interface.py`  
    - `src/extract/data_loader.py`  
  - **Scope:** Correctly declare AIM DLL functions with `restype = c_char_p` for string-returning methods (`get_channel_name`, `get_channel_units`, `get_GPS_channel_name`).  
  - **Acceptance Criteria:**  
    - Channel names/units consistently returned as decoded strings.  
    - No fallback to `"chan_1234"` names.  
    - Smoke test passes cleanly without warnings.


# Backlog — Post-WP4 Polishing & Regression Coverage

## 🐞 Test Improvements
- **Mock XRK Import Test**
  - **Type:** Bug/Polish
  - **Phase:** Phase 1 — Foundation 🚀
  - **Files:** `tests/test_file_manager.py`
  - **Scope:** Ensure mock `.xrk` import is expected to fail (don’t allow false “Imported (25 bytes)” messages).
  - **Acceptance Criteria:**
    - Mock XRK import raises a controlled error.
    - Test output is clean (no misleading "Imported" message).

## 🧪 Regression Coverage
- **Metadata File Persistence**
  - **Type:** Test Coverage
  - **Phase:** Phase 1 — Foundation 🚀
  - **Files:** `tests/test_session_builder.py`, `tests/test_session_builder_smoke.py`
  - **Scope:** Add explicit assertions that `data/metadata/*.json` is created after `export_session()`.
  - **Acceptance Criteria:**
    - Metadata JSON file exists on disk after export.
    - File contents include expected keys (`parquet_path`, `channel_list`, `units_map`).

## ⚠️ DLL Warnings
- **AiMLib_SetUnitsFile Export**
  - **Type:** Tech Debt
  - **Phase:** Phase 1 — Foundation 🚀
  - **Files:** `src/io/dll_interface.py`
  - **Scope:** Investigate missing `AiMLib_SetUnitsFile` in current AIM DLL. Decide whether:
    - Safe to ignore (no functional impact), OR
    - Provide fallback handling.
  - **Acceptance Criteria:**
    - Warning silenced or properly logged.
    - No runtime issues during DLL integration tests.

---



### Backlog — Future Enhancements

- **Hi-Res Sidecar Export**  
  - **Type:** Feature  
  - **Phase:** Phase 3 — Extended Data Access 🧩  
  - **Files Affected:** `src/session/session_builder.py`, `src/io/file_manager.py`, `src/config/config.py`  
  - **Scope:** Add dual-tier export. Channels with native_rate_hz > HIRES_THRESH_HZ are downsampled for canonical Parquet (with anti-alias filter) and also written to hi-res sidecars at native rate.  
  - **Artifacts:** Per-channel hi-res Parquet files, metadata JSON updated with `native_rate_hz` and `hires_paths`.  
  - **Acceptance Criteria:**  
    - Canonical file remains consistent with current WP4 format.  
    - Hi-res files reload successfully.  
    - Metadata JSON includes correct references and per-channel rates.  
  - **Notes:** MVP in WP4 uses naive reindex/interpolate. Future work replaces downsample step with `scipy.signal.decimate` (polyphase filter) and introduces hi-res sidecar export.

- **Title:** Add unit tests for FileManager  
- **Type:** Feature  
- **Phase:** Phase 1 (Foundation)  
- **Milestone:** v0.1  
- **Files Affected:** `file_manager.py`, `tests/test_file_manager.py`  
- **Acceptance Criteria:** FileManager import, duplicate detection, metadata persistence, and processing have automated test coverage with `pytest`.  

- **Title:** CI/CD Setup with GitHub Actions  
- **Type:** Tech Debt  
- **Phase:** Phase 1 (Foundation)  
- **Milestone:** v0.1  
- **Files Affected:** `tests/`, `requirements.txt`, `.github/workflows/` (new)  
- **Acceptance Criteria:** On each push, CI runs `pytest` and basic smoke tests; build fails if tests fail.  

- **Title:** Consistent Config Handling Across Modules  
- **Type:** Refactor  
- **Phase:** Phase 3 (Data Access)  
- **Milestone:** v0.3  
- **Files Affected:** `config/config.py`, `config/config_analysis.py`, all analysis modules  
- **Acceptance Criteria:** All modules import configuration via a unified interface; duplicate config constants removed.  

- **Title:** Add Visualization Tests with Static Data  
- **Type:** Feature  
- **Phase:** Phase 5 (Visualization)  
- **Milestone:** v0.5  
- **Files Affected:** `analysis/acceleration_analyzer.py`, plotting utilities (future)  
- **Acceptance Criteria:** Given a known dataset, visualization functions generate consistent plot outputs; test compares image hash or metadata.  

- **Title:** Developer Documentation Cleanup  
- **Type:** Tech Debt  
- **Phase:** Ongoing  
- **Milestone:** v1.0  
- **Files Affected:** `docs/overview.md`, `docs/inventory.md`, `docs/rules.md`  
- **Acceptance Criteria:** Docs reflect current file structure, roadmap, and governance process; outdated references removed.  
