"""
G-G Diagram Analysis
Friction circle analysis showing lateral vs longitudinal acceleration.

Calculates grip utilization and identifies where grip is not fully used.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json


@dataclass
class GGPoint:
    """A single G-G data point"""
    time: float
    lat_acc: float  # Lateral acceleration (g)
    lon_acc: float  # Longitudinal acceleration (g)
    total_g: float  # Combined g-force
    speed_mph: float = 0.0
    throttle_pct: float = 0.0
    gear: int = 0
    lap_number: int = 0


@dataclass
class GGStats:
    """Statistics for G-G analysis"""
    max_lateral_g: float
    max_braking_g: float
    max_acceleration_g: float
    max_combined_g: float
    avg_utilized_g: float
    utilization_pct: float  # How much of max grip is being used on average
    data_derived_max_g: float  # 95th percentile of actual data
    points_count: int


@dataclass
class GGAnalysisResult:
    """Complete G-G analysis result"""
    session_id: str
    analysis_timestamp: str
    stats: GGStats
    points: List[GGPoint]
    reference_max_g: float  # From vehicle config or data-derived
    low_utilization_zones: List[Dict]  # Corners with grip left on table

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "session_id": self.session_id,
            "analysis_timestamp": self.analysis_timestamp,
            "stats": {
                "max_lateral_g": round(self.stats.max_lateral_g, 3),
                "max_braking_g": round(self.stats.max_braking_g, 3),
                "max_acceleration_g": round(self.stats.max_acceleration_g, 3),
                "max_combined_g": round(self.stats.max_combined_g, 3),
                "avg_utilized_g": round(self.stats.avg_utilized_g, 3),
                "utilization_pct": round(self.stats.utilization_pct, 1),
                "data_derived_max_g": round(self.stats.data_derived_max_g, 3),
                "points_count": self.stats.points_count
            },
            "reference_max_g": round(self.reference_max_g, 3),
            "low_utilization_zones": self.low_utilization_zones,
            # Include downsampled points for Chart.js (every 5th point to reduce data size)
            "points": [
                {
                    "lat_acc": round(p.lat_acc, 3),
                    "lon_acc": round(p.lon_acc, 3),
                    "total_g": round(p.total_g, 3),
                    "speed_mph": round(p.speed_mph, 1),
                    "lap_number": p.lap_number
                }
                for i, p in enumerate(self.points) if i % 5 == 0
            ]
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)


class GGAnalyzer:
    """
    Analyzes G-G (friction circle) data from telemetry.

    Calculates grip utilization and identifies areas for improvement.
    """

    def __init__(self, max_g_reference: float = 1.3):
        """
        Initialize analyzer.

        Args:
            max_g_reference: Reference max g from vehicle config (default 1.3g for R-compounds)
        """
        self.max_g_reference = max_g_reference

    def analyze_from_arrays(
        self,
        time_data: np.ndarray,
        lat_acc_data: np.ndarray,
        lon_acc_data: np.ndarray,
        speed_data: np.ndarray = None,
        throttle_data: np.ndarray = None,
        gear_data: np.ndarray = None,
        lap_data: np.ndarray = None,
        session_id: str = "unknown"
    ) -> GGAnalysisResult:
        """
        Analyze G-G data from arrays.

        Args:
            time_data: Time array (seconds)
            lat_acc_data: Lateral acceleration (g, positive = right turn)
            lon_acc_data: Longitudinal acceleration (g, positive = acceleration)
            speed_data: Speed in mph (optional, for coloring)
            throttle_data: Throttle position 0-100 (optional)
            gear_data: Gear number (optional)
            lap_data: Lap number for each point (optional)
            session_id: Session identifier

        Returns:
            GGAnalysisResult with statistics and points
        """
        # Filter out invalid/zero data points
        valid_mask = ~(np.isnan(lat_acc_data) | np.isnan(lon_acc_data))

        lat_acc = lat_acc_data[valid_mask]
        lon_acc = lon_acc_data[valid_mask]
        time = time_data[valid_mask]

        # Calculate combined g-force for each point
        total_g = np.sqrt(lat_acc**2 + lon_acc**2)

        # Calculate statistics
        max_lateral_g = float(np.max(np.abs(lat_acc)))
        max_braking_g = float(np.abs(np.min(lon_acc)))  # Braking is negative
        max_acceleration_g = float(np.max(lon_acc))
        max_combined_g = float(np.max(total_g))

        # Data-derived max g (95th percentile)
        data_derived_max_g = float(np.percentile(total_g, 95))

        # Use vehicle config max or data-derived, whichever is higher
        reference_max_g = max(self.max_g_reference, data_derived_max_g)

        # Calculate utilization (what percentage of max grip is being used)
        utilization = total_g / reference_max_g
        avg_utilized_g = float(np.mean(total_g))
        utilization_pct = float(np.mean(utilization) * 100)

        # Build points list
        points = []
        for i in range(len(time)):
            p = GGPoint(
                time=float(time[i]),
                lat_acc=float(lat_acc[i]),
                lon_acc=float(lon_acc[i]),
                total_g=float(total_g[i]),
                speed_mph=float(speed_data[valid_mask][i]) if speed_data is not None else 0.0,
                throttle_pct=float(throttle_data[valid_mask][i]) if throttle_data is not None else 0.0,
                gear=int(gear_data[valid_mask][i]) if gear_data is not None else 0,
                lap_number=int(lap_data[valid_mask][i]) if lap_data is not None else 0
            )
            points.append(p)

        # Find low utilization zones (potential grip left on table)
        low_utilization_zones = self._find_low_utilization_zones(
            time, lat_acc, lon_acc, total_g, reference_max_g
        )

        stats = GGStats(
            max_lateral_g=max_lateral_g,
            max_braking_g=max_braking_g,
            max_acceleration_g=max_acceleration_g,
            max_combined_g=max_combined_g,
            avg_utilized_g=avg_utilized_g,
            utilization_pct=utilization_pct,
            data_derived_max_g=data_derived_max_g,
            points_count=len(points)
        )

        return GGAnalysisResult(
            session_id=session_id,
            analysis_timestamp=datetime.utcnow().isoformat(),
            stats=stats,
            points=points,
            reference_max_g=reference_max_g,
            low_utilization_zones=low_utilization_zones
        )

    def analyze_from_parquet(
        self,
        parquet_path: str,
        session_id: Optional[str] = None
    ) -> GGAnalysisResult:
        """
        Analyze G-G data from a Parquet file.

        Args:
            parquet_path: Path to Parquet file
            session_id: Session identifier (defaults to filename)

        Returns:
            GGAnalysisResult with statistics and points
        """
        df = pd.read_parquet(parquet_path)

        if session_id is None:
            from pathlib import Path
            session_id = Path(parquet_path).stem

        time_data = df.index.values

        # Find required columns
        lat_acc = self._find_column(df, ['GPS LatAcc', 'LatAcc', 'lateral_acc'])
        lon_acc = self._find_column(df, ['GPS LonAcc', 'LonAcc', 'longitudinal_acc'])

        if lat_acc is None or lon_acc is None:
            raise ValueError("Parquet file missing GPS LatAcc/LonAcc columns")

        # Find optional columns
        speed_data = self._find_column(df, ['GPS Speed', 'speed', 'Speed'])
        if speed_data is not None and speed_data.max() < 100:
            speed_data = speed_data * 2.237  # Convert m/s to mph

        throttle_data = self._find_column(df, ['PedalPos', 'throttle', 'Throttle'])

        return self.analyze_from_arrays(
            time_data, lat_acc, lon_acc,
            speed_data=speed_data,
            throttle_data=throttle_data,
            session_id=session_id
        )

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[np.ndarray]:
        """Find a column by trying multiple names"""
        for col in candidates:
            if col in df.columns:
                return df[col].values
            for actual_col in df.columns:
                if actual_col.lower() == col.lower():
                    return df[actual_col].values
        return None

    def _find_low_utilization_zones(
        self,
        time: np.ndarray,
        lat_acc: np.ndarray,
        lon_acc: np.ndarray,
        total_g: np.ndarray,
        max_g: float,
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Find zones where utilization is low (grip left on table).

        Args:
            time: Time array
            lat_acc: Lateral acceleration
            lon_acc: Longitudinal acceleration
            total_g: Combined g-force
            max_g: Reference maximum g
            threshold: Utilization threshold (0.5 = 50% of max grip)

        Returns:
            List of zones with low utilization
        """
        zones = []

        # Find periods where we're in a turn (significant lateral g) but low total g
        in_turn = np.abs(lat_acc) > 0.3  # In a turn
        low_g = total_g < (max_g * threshold)
        low_util_mask = in_turn & low_g

        # Find continuous zones
        zone_start = None
        for i in range(len(low_util_mask)):
            if low_util_mask[i] and zone_start is None:
                zone_start = i
            elif not low_util_mask[i] and zone_start is not None:
                if i - zone_start >= 5:  # Minimum 0.5s at 10Hz
                    zones.append({
                        "start_time": float(time[zone_start]),
                        "end_time": float(time[i-1]),
                        "duration": float(time[i-1] - time[zone_start]),
                        "avg_lat_g": float(np.mean(np.abs(lat_acc[zone_start:i]))),
                        "avg_total_g": float(np.mean(total_g[zone_start:i])),
                        "utilization_pct": float(np.mean(total_g[zone_start:i]) / max_g * 100)
                    })
                zone_start = None

        return zones[:10]  # Return top 10 zones


def analyze_gg_diagram(parquet_path: str, max_g_reference: float = 1.3) -> GGAnalysisResult:
    """
    Convenience function to analyze G-G diagram from parquet file.

    Args:
        parquet_path: Path to parquet file
        max_g_reference: Reference max g from vehicle config

    Returns:
        GGAnalysisResult
    """
    analyzer = GGAnalyzer(max_g_reference=max_g_reference)
    return analyzer.analyze_from_parquet(parquet_path)
