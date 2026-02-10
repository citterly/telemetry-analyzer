"""
Parquet validation layer for session import.

Validates Parquet files at the import boundary: checks required columns,
detects units, builds a channel map, and computes file hashes for dedup.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.dataframe_helpers import find_column_name

logger = logging.getLogger(__name__)

# Logical channel name -> list of candidate column names (priority order)
CHANNEL_CANDIDATES = {
    "latitude": ["GPS Latitude", "GPS_Latitude", "gps_lat", "latitude", "Latitude", "lat"],
    "longitude": ["GPS Longitude", "GPS_Longitude", "gps_lon", "longitude", "Longitude", "lon"],
    "speed": ["GPS Speed", "GPS_Speed", "gps_speed", "speed", "Speed"],
    "rpm": ["RPM", "rpm", "Engine RPM", "engine_rpm"],
    "lat_acc": ["GPS LatAcc", "GPS_LatAcc", "LatAcc", "lateral_acc", "Lateral Acc"],
    "lon_acc": ["GPS LonAcc", "GPS_LonAcc", "LonAcc", "longitudinal_acc", "Longitudinal Acc"],
    "throttle": ["PedalPos", "Throttle", "throttle", "TPS"],
    "time": ["Time", "time", "timestamp"],
}

REQUIRED_CHANNELS = ["latitude", "longitude"]


@dataclass
class ValidationResult:
    """Result of validating a Parquet file."""

    is_valid: bool = False
    channel_map: Dict[str, str] = field(default_factory=dict)
    units: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    duration_seconds: float = 0.0
    file_hash: str = ""
    detected_speed_unit: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "channel_map": self.channel_map,
            "units": self.units,
            "warnings": self.warnings,
            "errors": self.errors,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "duration_seconds": self.duration_seconds,
            "file_hash": self.file_hash,
            "detected_speed_unit": self.detected_speed_unit,
        }


class ParquetValidator:
    """
    Validates Parquet files for telemetry import.

    Checks required columns, detects units from df.attrs, builds a
    channel_map, and computes SHA256 hash for dedup detection.
    """

    def __init__(self, min_rows: int = 10):
        self.min_rows = min_rows

    def validate(self, parquet_path: str) -> ValidationResult:
        """
        Validate a Parquet file for import.

        Args:
            parquet_path: Path to the Parquet file.

        Returns:
            ValidationResult with channel map, units, warnings, and pass/fail.
        """
        result = ValidationResult()
        path = Path(parquet_path)

        # Check file exists
        if not path.exists():
            result.errors.append(f"File not found: {parquet_path}")
            return result

        if not path.suffix == ".parquet":
            result.errors.append(f"Not a Parquet file: {path.name}")
            return result

        # Compute file hash
        result.file_hash = self._compute_hash(parquet_path)

        # Load the file
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            result.errors.append(f"Failed to read Parquet: {e}")
            return result

        result.row_count = len(df)
        result.column_count = len(df.columns)

        # Check non-empty
        if result.row_count < self.min_rows:
            result.errors.append(
                f"Too few rows: {result.row_count} (minimum {self.min_rows})"
            )
            return result

        # Compute duration from index
        if len(df) > 1:
            result.duration_seconds = float(df.index[-1] - df.index[0])

        # Extract units from attrs if available
        stored_units = df.attrs.get("units", {})
        if stored_units:
            result.units = dict(stored_units)

        # Build channel map
        result.channel_map = self._build_channel_map(df)

        # Check required channels
        for required in REQUIRED_CHANNELS:
            if required not in result.channel_map:
                result.errors.append(
                    f"Required channel '{required}' not found. "
                    f"Available columns: {list(df.columns)}"
                )

        # Detect speed unit
        if "speed" in result.channel_map:
            speed_col = result.channel_map["speed"]
            result.detected_speed_unit = self._detect_speed_unit(
                df, speed_col, stored_units
            )

        # Warnings for missing optional channels
        optional_nice = {"rpm": "RPM", "throttle": "Throttle", "lat_acc": "Lateral G", "lon_acc": "Longitudinal G"}
        for ch, label in optional_nice.items():
            if ch not in result.channel_map:
                result.warnings.append(f"Optional channel missing: {label}")

        # Check for data quality
        for logical, col in result.channel_map.items():
            if logical in ("latitude", "longitude"):
                data = df[col].values
                valid = np.sum(~np.isnan(data)) if np.issubdtype(data.dtype, np.floating) else len(data)
                if valid < result.row_count * 0.5:
                    result.warnings.append(
                        f"Channel '{col}' has only {valid}/{result.row_count} valid values"
                    )

        result.is_valid = len(result.errors) == 0
        return result

    def _build_channel_map(self, df: pd.DataFrame) -> Dict[str, str]:
        """Map logical channel names to actual DataFrame column names."""
        channel_map = {}
        columns_lower = {c.lower(): c for c in df.columns}

        for logical, candidates in CHANNEL_CANDIDATES.items():
            matched = find_column_name(df, candidates)
            if matched:
                channel_map[logical] = matched

        return channel_map

    def _detect_speed_unit(
        self, df: pd.DataFrame, speed_col: str, stored_units: Dict[str, str]
    ) -> str:
        """Detect whether speed is in m/s or mph using stored units or heuristics."""
        # Prefer stored units
        if speed_col in stored_units:
            unit = stored_units[speed_col].lower()
            if "mph" in unit or "mi" in unit:
                return "mph"
            if "km" in unit:
                return "km/h"
            if "m/s" in unit or "meter" in unit:
                return "m/s"

        # Heuristic fallback based on max value
        speed_data = df[speed_col].values
        valid = speed_data[~np.isnan(speed_data)] if np.issubdtype(speed_data.dtype, np.floating) else speed_data
        if len(valid) == 0:
            return "unknown"

        max_speed = float(np.max(valid))
        if max_speed < 100:
            return "m/s"
        elif max_speed < 400:
            return "mph"
        else:
            return "km/h"

    @staticmethod
    def _compute_hash(file_path: str) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
