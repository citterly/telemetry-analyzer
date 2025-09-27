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
- **I/O Layer (`src/io/`)**
  - Interfaces with AIM SDK DLL.
  - Handles file open/close, channel enumeration, units.
- **Analysis Layer (`src/analysis/`)**
  - Higher-level processing: lap detection, sector times, derived channels.
- **Export Layer**
  - Outputs JSON, CSV, HTML for reports & external use.
- **Tests (`tests/`)**
  - Ensure stability after changes.
- **Examples (`examples/`)**
  - Show how to load files, read channels, dump data.

---

## Current Capabilities
- Organized project structure for scalability.
- DLL wrapper working:
  - Load `.xrk` files
  - List channels + units
- Units.xml containment solved (DLL finds correct units).
- Example scripts confirm integration.

---

## Planned Capabilities
- Sample data extraction (tabular values).
- Derived analysis (RPM bands, acceleration, braking).
- Lap/sector segmentation.
- Visualization (plots, dashboards).
- Optional web frontend.
