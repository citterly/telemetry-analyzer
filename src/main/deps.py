"""
Shared dependencies for FastAPI routers.

Provides config, file_manager, templates, and helper functions
that multiple routers need access to.
"""

from pathlib import Path
from typing import List, Optional

from src.config.config import get_config
from src.io.file_manager import FileManager
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
