### 📄 `docs/overview.md`
# Telemetry Analyzer — System Overview

## Purpose
The **Telemetry Analyzer** provides a flexible, programmable framework for working with AIM `.xrk` telemetry files. Instead of relying on Race Studio, this system enables direct access to raw data and structured exports for:

- Vehicle performance analysis
- Custom calculations
- Integration with other tools (e.g., machine learning, simulation)

---

## Core Components
- **Config Layer (`src/config/`)**
  - Defines paths, DLLs, environment setup.
  - Stores vehicle- and track-specific parameters (gear ratios, tire size, coordinates).
- **I/O Layer (`src/io/`)**
  - Interfaces with AIM SDK DLL.
  - Handles file import, deduplication, metadata persistence.
- **Session Layer (`src/session/`)**
  - Builds canonical session exports.
  - Extracts all channels, normalizes time base, attaches units.
  - Exports Parquet for downstream analysis.
- **Analysis Layer (`src/analysis/`, deprecated)**
  - Early prototypes for lap detection, gear calculation, acceleration analysis.
  - Being phased out in favor of structured `session/` and future `features/` modules.
- **Export Layer**
  - Outputs canonical Parquet, JSON metadata, and future formats (CSV, HTML).
- **Tests (`tests/`)**
  - Ensure stability after changes.
- **Examples (`examples/`)**
  - Show how to load files, process sessions, and inspect exported data.

---

## Current Capabilities
- Organized project structure for scalability.
- DLL wrapper working:
  - Load `.xrk` files
  - List channels + units
- File Manager operational:
  - Import, deduplicate, persist metadata
  - Generate summaries and manage session inventory
- Session Builder operational:
  - Open `.xrk` files
  - Extract RPM + GPS
  - Export canonical Parquet with aligned time base
- Units.xml containment solved (DLL finds correct units).
- Example tests confirm integration.

---

## Planned Capabilities
- Expand channel extraction to include all telemetry fields.
- Attach units to exported Parquet columns from `units.xml`.
- Replace relative seconds with datetime index in exports.
- Derived analysis (RPM bands, acceleration, braking, horsepower).
- Lap/sector segmentation.
- Visualization (plots, dashboards).
- Optional web frontend.
