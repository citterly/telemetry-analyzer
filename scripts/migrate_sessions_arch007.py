#!/usr/bin/env python3
"""
Database migration for arch-007: Session-centric analysis UX.

Adds enhanced metadata columns to sessions table:
- driver_name
- run_number
- weather_conditions
- track_conditions
- setup_snapshot (JSON)
- tire_pressures (JSON)
- tags (JSON array)
- last_accessed

Safe to run multiple times (uses IF NOT EXISTS).
"""

import sqlite3
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.config import PROJECT_ROOT as CONFIG_ROOT


def migrate_database(db_path: str):
    """Run the migration."""
    print(f"Migrating database: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check current schema
        print("\nCurrent schema:")
        cursor.execute("PRAGMA table_info(sessions)")
        columns = {row[1] for row in cursor.fetchall()}
        print(f"  Existing columns: {', '.join(sorted(columns))}")

        # Add new columns (IF NOT EXISTS prevents errors if already migrated)
        migrations = [
            ("driver_name", "ALTER TABLE sessions ADD COLUMN driver_name TEXT"),
            ("run_number", "ALTER TABLE sessions ADD COLUMN run_number INTEGER"),
            ("weather_conditions", "ALTER TABLE sessions ADD COLUMN weather_conditions TEXT"),
            ("track_conditions", "ALTER TABLE sessions ADD COLUMN track_conditions TEXT"),
            ("setup_snapshot", "ALTER TABLE sessions ADD COLUMN setup_snapshot TEXT"),  # JSON blob
            ("tire_pressures", "ALTER TABLE sessions ADD COLUMN tire_pressures TEXT"),  # JSON blob
            ("tags", "ALTER TABLE sessions ADD COLUMN tags TEXT"),  # JSON array
            ("last_accessed", "ALTER TABLE sessions ADD COLUMN last_accessed TEXT"),  # ISO datetime
        ]

        print("\nApplying migrations:")
        for col_name, sql in migrations:
            if col_name not in columns:
                print(f"  Adding column: {col_name}")
                cursor.execute(sql)
            else:
                print(f"  Skipping {col_name} (already exists)")

        conn.commit()

        # Verify new schema
        print("\nNew schema:")
        cursor.execute("PRAGMA table_info(sessions)")
        new_columns = {row[1] for row in cursor.fetchall()}
        print(f"  All columns: {', '.join(sorted(new_columns))}")

        # Count sessions
        cursor.execute("SELECT COUNT(*) FROM sessions")
        count = cursor.fetchone()[0]
        print(f"\n✅ Migration complete! {count} sessions in database.")

    except Exception as e:
        conn.rollback()
        print(f"\n❌ Migration failed: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    db_path = str(CONFIG_ROOT / "data" / "sessions.db")

    # Check if database exists
    if not Path(db_path).exists():
        print(f"Database not found at {db_path}")
        print("Creating new database with updated schema...")

    migrate_database(db_path)
