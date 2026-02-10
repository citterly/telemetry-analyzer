"""
Session management API router (v2 API).

Session CRUD, lap classification, setup management.
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from src.session.session_database import SessionDatabase
from src.session.importer import SessionImporter
from src.session.models import (
    ImportStatus,
    LapClassification,
    SessionType,
    Setup,
)
from ..deps import config

router = APIRouter()

# Session DB singleton
_session_db = None


def get_session_db() -> SessionDatabase:
    """Get or create the session database singleton."""
    global _session_db
    if _session_db is None:
        db_path = Path(config.DATA_DIR) / "sessions.db"
        _session_db = SessionDatabase(str(db_path))
    return _session_db


@router.get("/api/v2/sessions")
async def list_sessions_api(
    track_id: Optional[str] = None,
    vehicle_id: Optional[str] = None,
    session_type: Optional[str] = None,
    import_status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    """List sessions with optional filters"""
    db = get_session_db()

    s_type = None
    if session_type:
        try:
            s_type = SessionType(session_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid session_type: {session_type}")

    i_status = None
    if import_status:
        try:
            i_status = ImportStatus(import_status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid import_status: {import_status}")

    sessions = db.list_sessions(
        track_id=track_id,
        vehicle_id=vehicle_id,
        session_type=s_type,
        import_status=i_status,
        limit=limit,
        offset=offset,
    )

    return {
        "sessions": [s.to_dict() for s in sessions],
        "total": len(sessions),
        "stats": db.get_stats(),
    }


@router.post("/api/v2/sessions/import")
async def import_session_api(request: Request):
    """Import a Parquet file as a session"""
    data = await request.json()
    parquet_path = data.get("parquet_path")

    if not parquet_path:
        raise HTTPException(status_code=400, detail="parquet_path is required")

    # Resolve relative paths against data dir
    path = Path(parquet_path)
    if not path.is_absolute():
        path = Path(config.DATA_DIR) / parquet_path

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {parquet_path}")

    db = get_session_db()
    importer = SessionImporter(db)

    try:
        result = importer.import_session(
            str(path),
            vehicle_id=data.get("vehicle_id"),
            session_type=data.get("session_type"),
            notes=data.get("notes", ""),
        )
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


@router.get("/api/v2/sessions/{session_id}")
async def get_session_api(session_id: int):
    """Get session detail"""
    db = get_session_db()
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    stints = db.get_stints(session_id)
    setups = db.get_setups(session_id)

    return {
        "session": session.to_dict(),
        "stints": [s.to_dict() for s in stints],
        "setups": [s.to_dict() for s in setups],
    }


@router.get("/api/v2/sessions/{session_id}/laps")
async def get_session_laps_api(session_id: int):
    """Get laps for a session with classifications"""
    db = get_session_db()
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    laps = db.get_laps(session_id)

    # Compute gaps from best
    best_time = session.best_lap_time
    lap_dicts = []
    for lap in laps:
        d = lap.to_dict()
        if best_time and lap.lap_time > 0:
            d["gap_to_best"] = round(lap.lap_time - best_time, 3)
        else:
            d["gap_to_best"] = None
        lap_dicts.append(d)

    return {"laps": lap_dicts, "best_lap_time": best_time}


@router.put("/api/v2/sessions/{session_id}/laps/{lap_number}/classify")
async def classify_lap_api(session_id: int, lap_number: int, request: Request):
    """Override a lap's classification"""
    db = get_session_db()
    data = await request.json()

    classification_str = data.get("classification")
    if not classification_str:
        raise HTTPException(status_code=400, detail="classification is required")

    try:
        classification = LapClassification(classification_str)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid classification: {classification_str}")

    # Find the lap by session_id and lap_number
    laps = db.get_laps(session_id)
    target_lap = None
    for lap in laps:
        if lap.lap_number == lap_number:
            target_lap = lap
            break

    if not target_lap:
        raise HTTPException(status_code=404, detail=f"Lap {lap_number} not found in session {session_id}")

    success = db.update_lap_classification(
        target_lap.id,
        classification,
        confidence=1.0,
        user_override=True,
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to update classification")

    return {"status": "ok", "lap_number": lap_number, "classification": classification_str}


@router.put("/api/v2/sessions/{session_id}/confirm")
async def confirm_session_api(session_id: int, request: Request):
    """Confirm a session (mark as confirmed, optionally update metadata)"""
    db = get_session_db()
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    data = await request.json()

    session.import_status = ImportStatus.CONFIRMED
    if "session_type" in data:
        try:
            session.session_type = SessionType(data["session_type"])
        except ValueError:
            pass
    if "notes" in data:
        session.notes = data["notes"]
    if "vehicle_id" in data:
        session.vehicle_id = data["vehicle_id"]

    db.update_session(session)
    return {"status": "ok", "session": session.to_dict()}


@router.get("/api/v2/sessions/{session_id}/setup")
async def get_session_setup_api(session_id: int, setup_point: str = "pre"):
    """Get setup data for a session"""
    db = get_session_db()
    setup = db.get_setup(session_id, setup_point)
    if not setup:
        return {"setup": None}
    return {"setup": setup.to_dict()}


@router.put("/api/v2/sessions/{session_id}/setup")
async def save_session_setup_api(session_id: int, request: Request):
    """Save setup data for a session"""
    db = get_session_db()
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    data = await request.json()

    setup = Setup(
        session_id=session_id,
        setup_point=data.get("setup_point", "pre"),
        setup_data=data.get("setup_data", {}),
        notes=data.get("notes", ""),
    )
    setup = db.save_setup(setup)
    return {"status": "ok", "setup": setup.to_dict()}
