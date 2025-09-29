# 📑 Telemetry Analyzer — Project File Inventory
This document lists all core source code and documentation files in the project, with 2–3 sentence descriptions of their role and purpose. Irrelevant files (data, venv, build artifacts) are excluded.

## Analysis Modules

## IO Layer

- **`src/io/dll_interface.py`**  
  DLL wrapper for AIM XRK interface. Handles loading the DLL, ensuring `units.xml` is available, and defining ctypes prototypes for key functions (open, close, get channels). Forms the foundation of the `io/` layer.  

- **`src/io/fallback_units.py`**  
  Provides fallback logic to copy `units.xml` into the hard-coded SDK profile path required by the AIM DLL. Called when automatic setup fails.  

- **`src/io/file_manager.py`**  
  Manages XRK file imports, duplicate detection (via hash), metadata extraction, and metadata persistence. Provides command-line interface for import/list/process/stats. Integrates with analysis modules for lap and session data.  

---
## Session Layer

- **`src/session/session_builder.py`**  
  Extracts all XRK channel data using the DLL, normalizes to a common time base, attaches units, and exports canonical Parquet. Designed as the bridge between raw XRK files and higher-level analysis.  

- **`data/exports/processed/`**  
  Directory for canonical Parquet exports, one file per session. Each Parquet file contains all available channels, aligned timestamps, and attached units. Metadata JSON in `data/metadata/` points to these exports.  

---
## Analysis Modules (deprecated)

These modules were used during early development for exploring XRK data and quick checks.  
They are being phased out in favor of `session/` and `features/` modules but remain for reference until migration is complete.

- **`analysis/data_loader.py`**  
  Legacy loader for XRK telemetry files using the AIM DLL. Superseded by the session builder. 

- **`analysis/lap_analyzer.py`**  
  Splits sessions into laps using GPS start/finish detection or fallback heuristics. Produces LapInfo objects, tracks lap count, calculates per-lap stats (time, max speed, RPM), and identifies the fastest lap.

- **`analysis/gear_calculator.py`**  
  Derives gear usage from RPM and speed using transmission ratios, final drive, and tire circumference. Detects shifts, builds summaries, and estimates theoretical top speeds for analysis.

- **`analysis/acceleration_analyzer.py`**  
  Performs simplified acceleration and power analysis from speed/time traces. Uses basic F=ma relationships, applies smoothing, and generates visualizations of power vs. speed or track segments.

- **`analysis/xrk_metadata_analyzer.py`**  
  Lightweight inspector for XRK files. Retained only as a reference during WP2/WP3. 

- **`analysis/units_helper.py`**  
  Utility that ensures the required units.xml file exists in the AIM cache directory. Copies the file from project config if missing and returns its path.

- **`analysis/sdk_cleanup.py`**  
  Cleans up stray SDK unit redirection files. Kept for reference.

- **`analysis/__init__.py`**  
  Previously re-exported config constants. No longer needed with dedicated config modules. 

## Configuration

- **`config/config.py`**  
  Base configuration utilities for managing environment variables and file paths. Provides central constants for project directories and data storage.

- **`config/vehicle_config.py`**  
  Defines vehicle and track configuration values such as gear ratios, tire size, engine RPM zones, and track coordinates. Includes helpers for theoretical speed/RPM calculations.  

## Core Management

- **`file_manager.py`**  
  Manages XRK file import, deduplication, metadata persistence, and analysis processing. Central service that coordinates session loading, lap analysis, and stats. Provides search and file deletion.

## Application Entry Points

- **`app.py`**  
  Main FastAPI application exposing telemetry analysis features via a web interface. Implements routes for dashboard, file upload, processing, statistics, and file management. Uses Jinja2 templates for HTML output.

- **`run.py`**  
  Bootstrap script for starting the Telemetry Analyzer. Ensures Python version, sets up virtualenv, installs dependencies, validates analysis modules, and launches FastAPI app.

## Tests

- **`tests/test_file_manager.py`**  
  Unit tests for the FileManager. Validates file import, metadata extraction, duplicate detection, search, statistics, and processing pipeline using XRK files.

- **`tests/test_integration_dll.py`**  
  Integration test verifying AIM DLL setup. Ensures units.xml is available, DLL loads successfully, XRK files can be opened, channels enumerated, and file closed.

## Documentation

- **`README.md`**  
  Top-level documentation for the project, describing purpose, usage, and structure.

- **`docs/overview.md`**  
  High-level overview of the Telemetry Analyzer project, its goals, and system context.

- **`docs/phases.md`**  
  Development roadmap document. Outlines incremental phases from foundation and data access through analysis, visualization, and extended features.

