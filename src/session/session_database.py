"""
SQLite-backed session database for telemetry metadata.

Stores sessions, laps, stints, and setup data. Parquet files remain the
source of truth for time-series data; this database holds all metadata.
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .models import (
    ImportStatus,
    Lap,
    LapClassification,
    Session,
    SessionType,
    Setup,
    Stint,
)


class SessionDatabase:
    """
    SQLite-backed session metadata store.

    Thread-safe and persistent across restarts.
    """

    def __init__(self, db_path: str = "data/sessions.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parquet_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL DEFAULT '',
                    track_id TEXT,
                    track_name TEXT,
                    track_confidence REAL DEFAULT 0.0,
                    vehicle_id TEXT,
                    session_date TEXT,
                    session_type TEXT NOT NULL DEFAULT 'unknown',
                    import_status TEXT NOT NULL DEFAULT 'pending',
                    total_laps INTEGER DEFAULT 0,
                    best_lap_time REAL,
                    total_duration REAL,
                    notes TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    driver_name TEXT,
                    run_number INTEGER,
                    weather_conditions TEXT,
                    track_conditions TEXT,
                    setup_snapshot TEXT,
                    tire_pressures TEXT,
                    tags TEXT,
                    last_accessed TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_parquet
                ON sessions(parquet_path)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_hash
                ON sessions(file_hash)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_track
                ON sessions(track_id)
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS laps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    lap_number INTEGER NOT NULL,
                    stint_number INTEGER DEFAULT 0,
                    lap_time REAL NOT NULL,
                    start_time REAL DEFAULT 0.0,
                    end_time REAL DEFAULT 0.0,
                    start_index INTEGER DEFAULT 0,
                    end_index INTEGER DEFAULT 0,
                    classification TEXT NOT NULL DEFAULT 'normal',
                    classification_confidence REAL DEFAULT 0.0,
                    user_override INTEGER DEFAULT 0,
                    max_speed_mph REAL DEFAULT 0.0,
                    max_rpm REAL DEFAULT 0.0,
                    avg_rpm REAL DEFAULT 0.0,
                    sample_count INTEGER DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_laps_session
                ON laps(session_id)
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS stints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    stint_number INTEGER NOT NULL,
                    start_lap INTEGER NOT NULL,
                    end_lap INTEGER NOT NULL,
                    lap_count INTEGER DEFAULT 0,
                    best_lap_time REAL,
                    avg_lap_time REAL,
                    start_time REAL DEFAULT 0.0,
                    end_time REAL DEFAULT 0.0,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_stints_session
                ON stints(session_id)
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_setups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    setup_point TEXT NOT NULL DEFAULT 'pre',
                    setup_data TEXT NOT NULL DEFAULT '{}',
                    notes TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_setups_session
                ON session_setups(session_id)
            """)

            conn.execute("PRAGMA foreign_keys = ON")
            conn.commit()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(self, session: Session) -> Session:
        now = datetime.now(timezone.utc).isoformat()

        # Serialize JSON fields
        setup_snapshot_json = json.dumps(session.setup_snapshot) if session.setup_snapshot else None
        tire_pressures_json = json.dumps(session.tire_pressures) if session.tire_pressures else None
        tags_json = json.dumps(session.tags) if session.tags else "[]"

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO sessions
                    (parquet_path, file_hash, track_id, track_name, track_confidence,
                     vehicle_id, session_date, session_type, import_status,
                     total_laps, best_lap_time, total_duration, notes,
                     created_at, updated_at,
                     driver_name, run_number, weather_conditions, track_conditions,
                     setup_snapshot, tire_pressures, tags, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.parquet_path,
                    session.file_hash,
                    session.track_id,
                    session.track_name,
                    session.track_confidence,
                    session.vehicle_id,
                    session.session_date,
                    session.session_type.value,
                    session.import_status.value,
                    session.total_laps,
                    session.best_lap_time,
                    session.total_duration,
                    session.notes,
                    now, now,
                    # Enhanced metadata
                    session.driver_name,
                    session.run_number,
                    session.weather_conditions,
                    session.track_conditions,
                    setup_snapshot_json,
                    tire_pressures_json,
                    tags_json,
                    session.last_accessed.isoformat() if session.last_accessed else None,
                ))
                conn.commit()
                session.id = cursor.lastrowid
        return session

    def get_session(self, session_id: int) -> Optional[Session]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row:
                return Session.from_dict(dict(row))
        return None

    def get_session_by_parquet_path(self, parquet_path: str) -> Optional[Session]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE parquet_path = ?", (parquet_path,)
            ).fetchone()
            if row:
                return Session.from_dict(dict(row))
        return None

    def get_session_by_hash(self, file_hash: str) -> Optional[Session]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE file_hash = ?", (file_hash,)
            ).fetchone()
            if row:
                return Session.from_dict(dict(row))
        return None

    def list_sessions(
        self,
        track_id: Optional[str] = None,
        vehicle_id: Optional[str] = None,
        session_type: Optional[SessionType] = None,
        import_status: Optional[ImportStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Session]:
        conditions = []
        params: list = []

        if track_id:
            conditions.append("track_id = ?")
            params.append(track_id)
        if vehicle_id:
            conditions.append("vehicle_id = ?")
            params.append(vehicle_id)
        if session_type:
            conditions.append("session_type = ?")
            params.append(session_type.value)
        if import_status:
            conditions.append("import_status = ?")
            params.append(import_status.value)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT * FROM sessions {where}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [Session.from_dict(dict(r)) for r in rows]

    def update_session(self, session: Session) -> Session:
        now = datetime.now(timezone.utc).isoformat()

        # Serialize JSON fields
        setup_snapshot_json = json.dumps(session.setup_snapshot) if session.setup_snapshot else None
        tire_pressures_json = json.dumps(session.tire_pressures) if session.tire_pressures else None
        tags_json = json.dumps(session.tags) if session.tags else "[]"

        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE sessions SET
                        parquet_path = ?, file_hash = ?,
                        track_id = ?, track_name = ?, track_confidence = ?,
                        vehicle_id = ?, session_date = ?,
                        session_type = ?, import_status = ?,
                        total_laps = ?, best_lap_time = ?, total_duration = ?,
                        notes = ?, updated_at = ?,
                        driver_name = ?, run_number = ?,
                        weather_conditions = ?, track_conditions = ?,
                        setup_snapshot = ?, tire_pressures = ?, tags = ?,
                        last_accessed = ?
                    WHERE id = ?
                """, (
                    session.parquet_path, session.file_hash,
                    session.track_id, session.track_name, session.track_confidence,
                    session.vehicle_id, session.session_date,
                    session.session_type.value, session.import_status.value,
                    session.total_laps, session.best_lap_time, session.total_duration,
                    session.notes, now,
                    # Enhanced metadata
                    session.driver_name,
                    session.run_number,
                    session.weather_conditions,
                    session.track_conditions,
                    setup_snapshot_json,
                    tire_pressures_json,
                    tags_json,
                    session.last_accessed.isoformat() if session.last_accessed else None,
                    session.id,
                ))
                conn.commit()
        session.updated_at = datetime.fromisoformat(now)
        return session

    def delete_session(self, session_id: int) -> bool:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM sessions WHERE id = ?", (session_id,)
                )
                conn.commit()
                return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Laps
    # ------------------------------------------------------------------

    def create_lap(self, lap: Lap) -> Lap:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO laps
                    (session_id, lap_number, stint_number, lap_time,
                     start_time, end_time, start_index, end_index,
                     classification, classification_confidence, user_override,
                     max_speed_mph, max_rpm, avg_rpm, sample_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    lap.session_id, lap.lap_number, lap.stint_number, lap.lap_time,
                    lap.start_time, lap.end_time, lap.start_index, lap.end_index,
                    lap.classification.value, lap.classification_confidence,
                    int(lap.user_override),
                    lap.max_speed_mph, lap.max_rpm, lap.avg_rpm, lap.sample_count,
                ))
                conn.commit()
                lap.id = cursor.lastrowid
        return lap

    def bulk_create_laps(self, laps: List[Lap]) -> List[Lap]:
        with self._lock:
            with self._get_connection() as conn:
                for lap in laps:
                    cursor = conn.execute("""
                        INSERT INTO laps
                        (session_id, lap_number, stint_number, lap_time,
                         start_time, end_time, start_index, end_index,
                         classification, classification_confidence, user_override,
                         max_speed_mph, max_rpm, avg_rpm, sample_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        lap.session_id, lap.lap_number, lap.stint_number,
                        lap.lap_time, lap.start_time, lap.end_time,
                        lap.start_index, lap.end_index,
                        lap.classification.value, lap.classification_confidence,
                        int(lap.user_override),
                        lap.max_speed_mph, lap.max_rpm, lap.avg_rpm,
                        lap.sample_count,
                    ))
                    lap.id = cursor.lastrowid
                conn.commit()
        return laps

    def get_laps(self, session_id: int) -> List[Lap]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM laps WHERE session_id = ? ORDER BY lap_number",
                (session_id,),
            ).fetchall()
            return [Lap.from_dict(dict(r)) for r in rows]

    def update_lap_classification(
        self,
        lap_id: int,
        classification: LapClassification,
        confidence: float = 1.0,
        user_override: bool = True,
    ) -> bool:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    UPDATE laps SET
                        classification = ?,
                        classification_confidence = ?,
                        user_override = ?
                    WHERE id = ?
                """, (classification.value, confidence, int(user_override), lap_id))
                conn.commit()
                return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Stints
    # ------------------------------------------------------------------

    def create_stint(self, stint: Stint) -> Stint:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO stints
                    (session_id, stint_number, start_lap, end_lap, lap_count,
                     best_lap_time, avg_lap_time, start_time, end_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    stint.session_id, stint.stint_number,
                    stint.start_lap, stint.end_lap, stint.lap_count,
                    stint.best_lap_time, stint.avg_lap_time,
                    stint.start_time, stint.end_time,
                ))
                conn.commit()
                stint.id = cursor.lastrowid
        return stint

    def bulk_create_stints(self, stints: List[Stint]) -> List[Stint]:
        with self._lock:
            with self._get_connection() as conn:
                for stint in stints:
                    cursor = conn.execute("""
                        INSERT INTO stints
                        (session_id, stint_number, start_lap, end_lap, lap_count,
                         best_lap_time, avg_lap_time, start_time, end_time)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        stint.session_id, stint.stint_number,
                        stint.start_lap, stint.end_lap, stint.lap_count,
                        stint.best_lap_time, stint.avg_lap_time,
                        stint.start_time, stint.end_time,
                    ))
                    stint.id = cursor.lastrowid
                conn.commit()
        return stints

    def get_stints(self, session_id: int) -> List[Stint]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM stints WHERE session_id = ? ORDER BY stint_number",
                (session_id,),
            ).fetchall()
            return [Stint.from_dict(dict(r)) for r in rows]

    # ------------------------------------------------------------------
    # Setups
    # ------------------------------------------------------------------

    def save_setup(self, setup: Setup) -> Setup:
        now = datetime.now(timezone.utc).isoformat()
        setup_json = json.dumps(setup.setup_data) if isinstance(setup.setup_data, dict) else setup.setup_data
        with self._lock:
            with self._get_connection() as conn:
                # Upsert: replace existing setup for same session + setup_point
                existing = conn.execute("""
                    SELECT id FROM session_setups
                    WHERE session_id = ? AND setup_point = ?
                """, (setup.session_id, setup.setup_point)).fetchone()

                if existing:
                    conn.execute("""
                        UPDATE session_setups SET
                            setup_data = ?, notes = ?, created_at = ?
                        WHERE id = ?
                    """, (setup_json, setup.notes, now, existing["id"]))
                    setup.id = existing["id"]
                else:
                    cursor = conn.execute("""
                        INSERT INTO session_setups
                        (session_id, setup_point, setup_data, notes, created_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (setup.session_id, setup.setup_point, setup_json,
                          setup.notes, now))
                    setup.id = cursor.lastrowid
                conn.commit()
        return setup

    def get_setup(self, session_id: int, setup_point: str = "pre") -> Optional[Setup]:
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM session_setups
                WHERE session_id = ? AND setup_point = ?
            """, (session_id, setup_point)).fetchone()
            if row:
                data = dict(row)
                if isinstance(data.get("setup_data"), str):
                    data["setup_data"] = json.loads(data["setup_data"])
                return Setup.from_dict(data)
        return None

    def get_setups(self, session_id: int) -> List[Setup]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM session_setups WHERE session_id = ? ORDER BY setup_point",
                (session_id,),
            ).fetchall()
            results = []
            for row in rows:
                data = dict(row)
                if isinstance(data.get("setup_data"), str):
                    data["setup_data"] = json.loads(data["setup_data"])
                results.append(Setup.from_dict(data))
            return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            confirmed = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE import_status = ?",
                (ImportStatus.CONFIRMED.value,),
            ).fetchone()[0]
            pending = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE import_status = ?",
                (ImportStatus.PENDING.value,),
            ).fetchone()[0]
            total_laps = conn.execute("SELECT COUNT(*) FROM laps").fetchone()[0]

            return {
                "total_sessions": total,
                "confirmed_sessions": confirmed,
                "pending_sessions": pending,
                "total_laps": total_laps,
            }


# Singleton instance
_session_db: Optional[SessionDatabase] = None


def get_session_database(db_path: Optional[str] = None) -> SessionDatabase:
    """Get the singleton SessionDatabase instance."""
    global _session_db
    if _session_db is None:
        if db_path is None:
            from src.config.config import PROJECT_ROOT
            db_path = str(PROJECT_ROOT / "data" / "sessions.db")
        _session_db = SessionDatabase(db_path)
    return _session_db
