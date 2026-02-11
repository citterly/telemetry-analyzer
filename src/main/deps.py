"""
Shared dependencies for FastAPI routers.

Provides config, file_manager, templates, and helper functions
that multiple routers need access to.
"""

from pathlib import Path
from typing import List, Optional

from fastapi import HTTPException

from src.config.config import get_config
from src.io.file_manager import FileManager
from src.services.session_data_loader import SessionDataLoader, SessionChannels
from src.utils.dataframe_helpers import (
    find_column,
    sanitize_for_json,
    SPEED_MS_TO_MPH,
    ensure_speed_mph,
)

# Initialize configuration (only once)
config = get_config()

# Initialize file manager
file_manager = FileManager(config.DATA_DIR)


def find_parquet_file(filename: str) -> Optional[Path]:
    """Find a Parquet file by name in the data directories."""
    data_dir = Path(config.DATA_DIR)

    # Try direct path
    file_path = data_dir / filename
    if file_path.exists():
        return file_path

    # Try with .parquet extension
    if not filename.endswith('.parquet'):
        file_path = data_dir / f"{filename}.parquet"
        if file_path.exists():
            return file_path

    # Search recursively
    for pq in data_dir.rglob(f"*{filename}*"):
        if pq.suffix == '.parquet':
            return pq

    return None


def load_session(
    filename: str,
    required: List[str] = None,
) -> SessionChannels:
    """
    Find parquet file, load session data, validate required channels.

    Combines find_parquet_file() + SessionDataLoader for use in router endpoints.

    Args:
        filename: Parquet filename or path
        required: Logical channel names that must be present,
                  e.g. ["speed", "rpm"]. If None, no validation.

    Raises:
        HTTPException 404: If parquet file not found
        HTTPException 422: If required channels missing

    Returns:
        SessionChannels with discovered data
    """
    file_path = find_parquet_file(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")

    loader = SessionDataLoader()
    try:
        if required:
            return loader.load_or_raise(str(file_path), required=required)
        return loader.load(str(file_path))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
