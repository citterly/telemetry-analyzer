# Feedback Service — Design Overview

## Context

A reusable user-testing/feedback module that can be dropped into any web application. Captures screenshots, text notes, voice transcriptions, and auto-context to identify bugs, friction points, enhancements, and feature requests. Feeds into a prioritization pipeline.

## Architecture: Two Pieces

### 1. Standalone Feedback Service (`~/projects/feedback-service`)

Own FastAPI project, own port (8100), own git repo. Three responsibilities:

**Receives feedback** — Widget POSTs screenshot (base64/file), text notes, category, voice audio + transcript, and auto-context (URL, page title, viewport, timestamp, app name). Stored in SQLite (metadata) + filesystem (binary files: screenshots, audio).

**Proxies transcription** — Voice audio goes to the existing Whisper service's drop-in component. The feedback service doesn't build its own voice capture — it integrates the existing component from the Whisper project. Whisper URL is a config value, wired up later.

**Serves dashboard + widget** — Dashboard for reviewing all feedback across all apps. Widget JS served from the service's own static files so any app includes it via one script tag.

### 2. Per-App Integration (one line)

Any app adds to its base template:
```html
<script src="http://feedback-service:8100/static/feedback-widget.js"
        data-app="telemetry-analyzer"
        data-endpoint="http://feedback-service:8100/api/feedback">
</script>
```

No other changes to the host app.

---

## Widget UX

Floating action button, bottom-right corner. Click opens a panel:

1. **Category chips** — Bug / Enhancement / Friction / Feature — one tap
2. **Screenshot** — Auto-captured on panel open (html2canvas). Click thumbnail to annotate (draw, circle, arrow)
3. **Notes** — Text field for description
4. **Voice** — Uses Whisper service's existing drop-in voice component. Record → transcribe → transcript appears in notes (editable). Original audio attached.
5. **Submit** — POSTs everything. Brief confirmation, panel closes.

**Auto-context** (captured silently): current URL/route, page title, viewport size, browser/OS, timestamp, app name from data attribute.

---

## Feedback Service API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/feedback` | Submit new feedback (multipart: screenshot, audio, JSON metadata) |
| GET | `/api/feedback` | List feedback (query: app, category, status, page, limit) |
| GET | `/api/feedback/{id}` | Single item with full detail |
| PATCH | `/api/feedback/{id}` | Update status, priority, reviewer notes |
| DELETE | `/api/feedback/{id}` | Remove item |
| GET | `/api/feedback/stats` | Summary counts by app, category, status |
| GET | `/api/feedback/apps` | List all apps that have submitted feedback |

## Data Model

```
FeedbackItem:
  id: str (UUID)
  app_name: str
  category: str (bug/enhancement/friction/feature)
  status: str (new/reviewed/prioritized/done)
  priority: int (1-5, nullable)
  text: str (user notes + voice transcript)
  screenshot_path: str (filesystem path)
  audio_path: str (filesystem path, nullable)
  transcript: str (from Whisper, nullable)
  context_url: str (page URL when submitted)
  context_page_title: str
  context_viewport: str (e.g., "1920x1080")
  context_user_agent: str
  reviewer_notes: str (nullable)
  created_at: datetime
  updated_at: datetime
```

## Dashboard

Simple review interface served at `/dashboard`:

- **List view** — All items, newest first. Columns: app, category badge, text preview, screenshot thumbnail, status, priority, timestamp
- **Filters** — By app, category, status. Search text.
- **Detail view** — Click to expand: full screenshot (annotated), full notes, play audio, auto-context details
- **Actions** — Change status (dropdown), set priority (1-5), add reviewer notes
- **No auth** — Open access for now. Auth via future identity service.

## Voice Integration

**Key decision:** Do NOT build voice recording/transcription from scratch. The Whisper project already has a drop-in component for this. The feedback widget integrates that component.

Integration approach (to be refined after exploring the Whisper project):
- The Whisper component likely provides a record button + transcription callback
- The feedback widget embeds this component in its panel
- Transcript feeds into the notes field
- Original audio is captured and stored alongside the feedback

Whisper endpoint URL is a config value in the feedback service (placeholder until provided).

## Project Structure (`~/projects/feedback-service`)

```
feedback-service/
├── CLAUDE.md              # Dev process, boot-up ritual
├── features.json          # Feature pipeline
├── init.sh                # Bootstrap script
├── src/
│   ├── main/
│   │   ├── app.py         # FastAPI app
│   │   └── routers/
│   │       ├── feedback.py    # CRUD endpoints
│   │       └── dashboard.py   # Dashboard page routes
│   ├── models/
│   │   └── feedback.py    # SQLite models
│   └── config.py          # Config (port, DB path, Whisper URL, storage dir)
├── static/
│   ├── feedback-widget.js # The drop-in widget
│   ├── feedback-widget.css
│   └── lib/
│       └── html2canvas.min.js
├── templates/
│   └── dashboard.html     # Review dashboard
├── data/
│   ├── feedback.db        # SQLite
│   ├── screenshots/       # PNG files
│   └── audio/             # Audio recordings
└── tests/
```

## Feature Pipeline (for feedback-service project)

| ID | Name | Phase | Description |
|----|------|-------|-------------|
| fb-001-plan | Plan: Data model and API | plan | Design SQLite schema, API endpoints, storage layout |
| fb-001-exec | Execute: Data model and API | exec | Implement FastAPI service with CRUD endpoints |
| fb-001-test | Test: Data model and API | test | API tests with TestClient |
| fb-002-plan | Plan: Feedback widget | plan | Design the embeddable JS widget (capture, annotate, submit) |
| fb-002-exec | Execute: Feedback widget | exec | Build feedback-widget.js with screenshot + annotation + submission |
| fb-002-test | Test: Feedback widget | test | Widget integration tests |
| fb-003-plan | Plan: Voice integration | plan | Explore Whisper project's drop-in component, design integration |
| fb-003-exec | Execute: Voice integration | exec | Integrate Whisper component into widget |
| fb-003-test | Test: Voice integration | test | End-to-end voice capture → transcription tests |
| fb-004-plan | Plan: Dashboard | plan | Design review/prioritization UI |
| fb-004-exec | Execute: Dashboard | exec | Build dashboard page with filters, detail view, status management |
| fb-004-test | Test: Dashboard | test | Dashboard functionality tests |

## Telemetry-Analyzer Integration

One feature in telemetry-analyzer's pipeline:

| ID | Name | Description |
|----|------|-------------|
| feat-070 | Integrate feedback widget | Add feedback-widget.js script tag to templates/base.html with data-app="telemetry-analyzer". One-line change. Depends on fb-002 being complete. |
