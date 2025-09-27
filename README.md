### 📄 `README.md`
# Telemetry Analyzer

Custom telemetry analysis framework for AIM `.xrk` files.

## Purpose
Race Studio is powerful but not intuitive. This project provides a **code-driven approach**:
- Import `.xrk` files through AIM’s DLL.
- Extract metadata, channels, and sample data.
- Perform custom analysis & visualization.

The system is designed to grow over time — starting with reliable data access, then expanding into higher-level analysis and visual reporting.

---

## Project Structure

telemetry-analyzer/
├── data/
│   ├── exports/          # Exported reports, DLL cache
│   │   └── aim_cache/
│   │       └── units.xml
│   ├── metadata/         # Generated metadata JSON
│   ├── raw/              # Unmodified raw files
│   └── uploads/          # Active session uploads
├── docs/
│   ├── overview.md       # System overview
│   └── phases.md         # Roadmap & phases
├── examples/             # Example scripts
│   ├── dump_channel_data.py
│   ├── read_channels.py
├── reports/              # Generated reports (json/csv/html)
├── scripts/              # Utility scripts (e.g., setup/cleanup)
├── src/
│   ├── analysis/         # Analysis logic
│   ├── config/           # Configuration
│   ├── io/               # DLL + file I/O
│   ├── main/             # App entrypoints
│   └── utils/            # Shared helpers
├── static/               # Web assets
├── templates/            # Web templates
├── tests/                # Tests (unit/integration)
├── third-party/          # AIM SDK DLLs, headers, samples
└── requirements.txt

---

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
````

2. Place `.xrk` files in `data/uploads/`.

3. Run an example:

   ```bash
   python -m examples.read_channels
   ```

---

## Current Status

* ✅ Project restructured with clean config and modules
* ✅ DLL integration confirmed (channel names + units)
* ✅ Units.xml containment handled
* ⚠️ Sample data extraction (tabular view) is next

---

## Roadmap

See [docs/phases.md](docs/phases.md).



