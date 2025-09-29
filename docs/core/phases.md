# 📑 Telemetry Analyzer — Phases (Revised)

This document defines the phased development plan for the Telemetry Analyzer project.  
It emphasizes actionable insights (e.g., RPM/shift analysis) early for on-track use, while leaving room for deeper cross-session analytics later.  

---

## Phase 1: Foundation 🚀
**Goal:** Get the system running with basic ingestion and verification.  
- DLL integration and smoke tests (open XRK, list channels).  
- Metadata extraction (session info, channel list, file stats).  
- FileManager basics: import, deduplication, metadata persistence.  
- Unit tests for ingestion and metadata handling.  
- Simple command-line reporting.  

**Deliverable:** Confidence that XRK files can be reliably opened, indexed, and stored with metadata. 
-- WP2: File Manager Import & Metadata (in progress)
+- WP2: File Manager Import & Metadata ✅ (completed)

+-- WP3: Session Canonicalization (upcoming)
+   Goal: Canonical session export (Parquet) with all channels, units, and aligned time base. 

---

## Phase 2: Quick Reports & RPM Analysis 📊
**Goal:** Provide actionable, automated reports for trackside use (Runoffs-ready).  
- Lap detection and fastest lap identification.  
- RPM tracking and shift point analysis (focus on corners where shifts matter).  
- Gear usage summaries (per lap and per session).  
- Basic track map visualization (GPS trace color-coded by speed or gear).  
- HTML/PDF “Quick Reports” generated automatically on ingestion.  

**Deliverable:** Drop XRK file → receive a session report with laps, gears, RPM/shift analysis, and track map.  

---

## Phase 3: Extended Data Access 🧩
**Goal:** Enable structured, normalized data for cross-session analysis.  
- Implement normalized schema (Sessions, Laps, Channels, Samples).  
- Export full channel samples (CSV/JSON).  
- Populate SQLite with normalized telemetry for long-term comparisons.  
- Validate exports against Race Studio.  

**Deliverable:** Ability to run queries and cross-file comparisons using normalized data.  

---

## Phase 4: Advanced Analysis 🧪
**Goal:** Add higher-level metrics and deeper insights.  
- Power/acceleration analysis (F=ma approximations).  
- Sector times and lap deltas.  
- Corner radii and braking zones.  
- Comparative lap overlays (fast vs. slow).  

**Deliverable:** Analytical reports that explain *why* laps are fast or slow, not just *which* ones.  

---

## Phase 5: Visualization & Reporting 🎨
**Goal:** Enhance the usability and polish of outputs.  
- Advanced plots (RPM traces, speed-distance charts, braking maps).  
- Interactive dashboards via FastAPI UI.  
- Cross-session comparisons in UI.  
- One-click full session report generator (HTML/PDF).  

**Deliverable:** Polished interface and reporting system that can be used trackside or offline.  

---

## Phase 6: Stretch Features 🔮
**Goal:** Explore enhancements for future usability.  
- Session replay mode (time-synced telemetry playback).  
- Config-based scenario modeling (gear ratios, tire sizes).  
- Plugin API for custom analyzers.  
- AI/ML-driven pattern detection (lap clustering, anomaly detection).  

**Deliverable:** Optional advanced tools layered onto the core analyzer.  
