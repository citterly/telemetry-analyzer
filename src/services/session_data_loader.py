"""
SessionDataLoader — centralized parquet loading with column discovery and unit conversion.

Replaces 30+ inline pd.read_parquet + find_column + speed conversion patterns
scattered across router endpoints and analyzer classes.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.dataframe_helpers import (
    find_column,
    find_column_name,
    ensure_speed_mph,
    KNOWN_COLUMNS,
    SPEED_MS_TO_MPH,
)


@dataclass
class SessionChannels:
    """Resolved channel data from a parquet file."""

    time: np.ndarray
    df: pd.DataFrame
    source_path: str
    session_id: str
    sample_count: int
    duration_seconds: float
    speed_unit_detected: str  # "mph", "m/s", or "unknown"

    # Core channels (None if not found in file)
    latitude: Optional[np.ndarray] = None
    longitude: Optional[np.ndarray] = None
    speed_mph: Optional[np.ndarray] = None
    speed_ms: Optional[np.ndarray] = None
    rpm: Optional[np.ndarray] = None
    lat_acc: Optional[np.ndarray] = None
    lon_acc: Optional[np.ndarray] = None
    throttle: Optional[np.ndarray] = None

    # Logical name → actual column name found in the DataFrame
    column_map: Dict[str, str] = field(default_factory=dict)

    @property
    def has_gps(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    @property
    def has_speed(self) -> bool:
        return self.speed_mph is not None

    @property
    def has_rpm(self) -> bool:
        return self.rpm is not None

    @property
    def available_channels(self) -> List[str]:
        """List logical channel names that were found."""
        return list(self.column_map.keys())


class SessionDataLoader:
    """Unified parquet loading with column discovery and unit conversion."""

    # Channels to auto-discover from KNOWN_COLUMNS
    _CHANNEL_ATTRS = {
        "latitude": "latitude",
        "longitude": "longitude",
        "speed": "_raw_speed",  # Internal; we split into speed_mph and speed_ms
        "rpm": "rpm",
        "lat_acc": "lat_acc",
        "lon_acc": "lon_acc",
        "throttle": "throttle",
    }

    def load(self, parquet_path: str) -> SessionChannels:
        """
        Load a parquet file and resolve all known channels.

        Reads the file, discovers columns using KNOWN_COLUMNS candidates,
        detects speed units, and provides both mph and m/s.

        Args:
            parquet_path: Path to parquet file

        Returns:
            SessionChannels with all discovered data
        """
        df = pd.read_parquet(parquet_path)
        time_data = df.index.values
        session_id = Path(parquet_path).stem

        sample_count = len(time_data)
        duration = float(time_data[-1] - time_data[0]) if sample_count > 1 else 0.0

        # Discover all known channels
        column_map = {}
        discovered = {}
        for logical_name, candidates in KNOWN_COLUMNS.items():
            col_name = find_column_name(df, candidates)
            if col_name is not None:
                column_map[logical_name] = col_name
                discovered[logical_name] = find_column(df, candidates)

        # Resolve speed unit and provide both mph and m/s
        raw_speed = discovered.get("speed")
        speed_mph = None
        speed_ms = None
        speed_unit = "unknown"

        if raw_speed is not None and len(raw_speed) > 0:
            # Check stored metadata first
            units_meta = getattr(df, 'attrs', {}).get('units', {})
            speed_col = column_map.get("speed", "")

            if speed_col in units_meta:
                unit_str = units_meta[speed_col].lower()
                if "mph" in unit_str:
                    speed_unit = "mph"
                    speed_mph = raw_speed
                    speed_ms = raw_speed / SPEED_MS_TO_MPH
                elif "km" in unit_str:
                    speed_unit = "km/h"
                    speed_mph = raw_speed * 0.621371
                    speed_ms = raw_speed / 3.6
                else:
                    speed_unit = "m/s"
                    speed_ms = raw_speed
                    speed_mph = raw_speed * SPEED_MS_TO_MPH
            else:
                # Heuristic: max < 100 → m/s, else mph
                max_speed = float(np.nanmax(raw_speed)) if len(raw_speed) > 0 else 0
                if max_speed < 100:
                    speed_unit = "m/s"
                    speed_ms = raw_speed
                    speed_mph = raw_speed * SPEED_MS_TO_MPH
                else:
                    speed_unit = "mph"
                    speed_mph = raw_speed
                    speed_ms = raw_speed / SPEED_MS_TO_MPH

        return SessionChannels(
            time=time_data,
            df=df,
            source_path=parquet_path,
            session_id=session_id,
            sample_count=sample_count,
            duration_seconds=duration,
            speed_unit_detected=speed_unit,
            latitude=discovered.get("latitude"),
            longitude=discovered.get("longitude"),
            speed_mph=speed_mph,
            speed_ms=speed_ms,
            rpm=discovered.get("rpm"),
            lat_acc=discovered.get("lat_acc"),
            lon_acc=discovered.get("lon_acc"),
            throttle=discovered.get("throttle"),
            column_map=column_map,
        )

    def load_or_raise(
        self,
        parquet_path: str,
        required: List[str],
    ) -> SessionChannels:
        """
        Load and validate that required channels exist.

        Args:
            parquet_path: Path to parquet file
            required: Logical channel names that must be present,
                      e.g. ["speed", "rpm", "latitude", "longitude"]

        Raises:
            ValueError: If any required channel is missing

        Returns:
            SessionChannels with validated data
        """
        channels = self.load(parquet_path)

        missing = []
        for name in required:
            value = getattr(channels, self._resolve_attr(name), None)
            if value is None:
                missing.append(name)

        if missing:
            raise ValueError(
                f"Required channels not found in {Path(parquet_path).name}: "
                + ", ".join(missing)
            )

        return channels

    def to_session_data_dict(self, channels: SessionChannels) -> Dict:
        """
        Build the legacy session_data dict used by LapAnalyzer.

        Returns dict with keys: time, latitude, longitude, rpm,
        speed_mph, speed_ms.
        """
        result = {
            "time": channels.time,
            "latitude": channels.latitude if channels.latitude is not None else np.array([]),
            "longitude": channels.longitude if channels.longitude is not None else np.array([]),
            "rpm": channels.rpm if channels.rpm is not None else np.zeros(channels.sample_count),
        }

        if channels.speed_mph is not None:
            result["speed_mph"] = channels.speed_mph
        if channels.speed_ms is not None:
            result["speed_ms"] = channels.speed_ms

        return result

    @staticmethod
    def _resolve_attr(logical_name: str) -> str:
        """Map logical channel name to SessionChannels attribute name."""
        mapping = {
            "speed": "speed_mph",
            "latitude": "latitude",
            "longitude": "longitude",
            "rpm": "rpm",
            "lat_acc": "lat_acc",
            "lon_acc": "lon_acc",
            "throttle": "throttle",
        }
        return mapping.get(logical_name, logical_name)
