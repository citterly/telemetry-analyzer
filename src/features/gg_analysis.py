"""
G-G Diagram Analysis
Friction circle analysis showing lateral vs longitudinal acceleration.

Calculates grip utilization and identifies where grip is not fully used.
Enhanced with quadrant breakdown, power-limited detection, and lap comparison.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
import json

from ..utils.dataframe_helpers import find_column, SPEED_MS_TO_MPH
from .base_analyzer import BaseAnalyzer, BaseAnalysisReport
from .registry import analyzer_registry


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
    lat: float = 0.0  # GPS latitude for track map
    lon: float = 0.0  # GPS longitude for track map


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
    # New actionable metrics
    corner_utilization_pct: float = 0.0  # Grip utilization in corners only (lat_g > 0.3)
    non_power_limited_utilization_pct: float = 0.0  # Utilization excluding power-limited zones
    braking_to_lateral_ratio: float = 0.0  # max_braking / max_lateral - flag if < 0.9


@dataclass
class QuadrantStats:
    """Statistics for a single quadrant of the G-G diagram"""
    name: str  # "lat_left", "lat_right", "braking", "acceleration"
    display_name: str
    max_g: float
    avg_g: float
    utilization_pct: float
    time_spent_pct: float
    points_count: int
    color: str  # For UI display


@dataclass
class LowUtilizationZone:
    """A zone where grip is not fully utilized"""
    start_time: float
    end_time: float
    start_idx: int
    end_idx: int
    duration: float
    avg_lat_g: float
    avg_lon_g: float
    avg_total_g: float
    utilization_pct: float
    avg_lat: float  # GPS latitude for track map
    avg_lon: float  # GPS longitude for track map
    zone_type: str  # "low_grip" or "power_limited"
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "duration": round(self.duration, 2),
            "avg_lat_g": round(self.avg_lat_g, 3),
            "avg_lon_g": round(self.avg_lon_g, 3),
            "avg_total_g": round(self.avg_total_g, 3),
            "utilization_pct": round(self.utilization_pct, 1),
            "lat": round(self.avg_lat, 6),
            "lon": round(self.avg_lon, 6),
            "zone_type": self.zone_type,
            "recommendation": self.recommendation
        }


@dataclass
class GGAnalysisResult(BaseAnalysisReport):
    """Complete G-G analysis result"""
    session_id: str
    analysis_timestamp: str
    stats: GGStats
    points: List[GGPoint]
    reference_max_g: float  # From vehicle config or data-derived (for legacy compatibility)
    low_utilization_zones: List[LowUtilizationZone]
    quadrants: List[QuadrantStats] = field(default_factory=list)
    power_limited_zones: List[LowUtilizationZone] = field(default_factory=list)
    power_limited_pct: float = 0.0
    lap_numbers: List[int] = field(default_factory=list)
    gps_bounds: Dict = field(default_factory=dict)
    # Per-quadrant reference values from vehicle config
    reference_lateral_g: float = 1.3
    reference_braking_g: float = 1.4
    reference_accel_g: float = 0.5
    # Warnings and flags
    warnings: List[str] = field(default_factory=list)
    max_g_exceeds_config: bool = False  # True if data exceeds vehicle config (config wrong or GPS noise)
    braking_opportunity: bool = False  # True if braking << lateral (could brake harder)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        result = {
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
                "points_count": self.stats.points_count,
                # New actionable metrics
                "corner_utilization_pct": round(self.stats.corner_utilization_pct, 1),
                "non_power_limited_utilization_pct": round(self.stats.non_power_limited_utilization_pct, 1),
                "braking_to_lateral_ratio": round(self.stats.braking_to_lateral_ratio, 2)
            },
            "reference_max_g": round(self.reference_max_g, 3),
            # Per-quadrant reference values
            "reference_lateral_g": round(self.reference_lateral_g, 3),
            "reference_braking_g": round(self.reference_braking_g, 3),
            "reference_accel_g": round(self.reference_accel_g, 3),
            "low_utilization_zones": [z.to_dict() for z in self.low_utilization_zones],
            "power_limited_zones": [z.to_dict() for z in self.power_limited_zones],
            "power_limited_pct": round(self.power_limited_pct, 1),
            "quadrants": [
                {
                    "name": q.name,
                    "display_name": q.display_name,
                    "max_g": round(q.max_g, 3),
                    "avg_g": round(q.avg_g, 3),
                    "utilization_pct": round(q.utilization_pct, 1),
                    "time_spent_pct": round(q.time_spent_pct, 1),
                    "points_count": q.points_count,
                    "color": q.color
                }
                for q in self.quadrants
            ],
            "lap_numbers": self.lap_numbers,
            "gps_bounds": self.gps_bounds,
            # Warnings and flags
            "warnings": self.warnings,
            "max_g_exceeds_config": self.max_g_exceeds_config,
            "braking_opportunity": self.braking_opportunity,
            # Include downsampled points for Chart.js (every 5th point to reduce data size)
            "points": [
                {
                    "lat_acc": round(p.lat_acc, 3),
                    "lon_acc": round(p.lon_acc, 3),
                    "total_g": round(p.total_g, 3),
                    "speed_mph": round(p.speed_mph, 1),
                    "lap_number": p.lap_number,
                    "lat": round(p.lat, 6) if p.lat != 0 else None,
                    "lon": round(p.lon, 6) if p.lon != 0 else None
                }
                for i, p in enumerate(self.points) if i % 5 == 0
            ]
        }
        result.update(self._trace_dict())
        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)


class GGAnalyzer(BaseAnalyzer):
    """
    Analyzes G-G (friction circle) data from telemetry.

    Calculates grip utilization and identifies areas for improvement.
    Enhanced with quadrant breakdown and power-limited detection.
    """

    # Registry metadata
    registry_key = "gg"
    required_channels = ["lat_acc", "lon_acc"]
    optional_channels = ["speed", "throttle", "latitude", "longitude"]
    config_params = []

    # Power-limited detection threshold
    POWER_LIMITED_THROTTLE = 95.0  # % throttle
    POWER_LIMITED_LAT_G = 0.2      # Max lateral g to be considered on straight

    def __init__(
        self,
        max_g_reference: float = 1.3,
        max_braking_g: float = None,
        power_limited_accel_g: float = 0.4
    ):
        """
        Initialize analyzer.

        Args:
            max_g_reference: Reference max lateral g from vehicle config (default 1.3g for R-compounds)
            max_braking_g: Reference max braking g (defaults to max_g_reference * 1.1 if not set)
            power_limited_accel_g: Expected acceleration g when power-limited (based on power/weight)
        """
        self.max_g_reference = max_g_reference
        self.max_braking_g = max_braking_g if max_braking_g is not None else max_g_reference * 1.1
        self.power_limited_accel_g = power_limited_accel_g

    def analyze_from_arrays(
        self,
        time_data: np.ndarray,
        lat_acc_data: np.ndarray,
        lon_acc_data: np.ndarray,
        speed_data: np.ndarray = None,
        throttle_data: np.ndarray = None,
        gear_data: np.ndarray = None,
        lap_data: np.ndarray = None,
        lat_gps: np.ndarray = None,
        lon_gps: np.ndarray = None,
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
            lat_gps: GPS latitude (optional, for track map)
            lon_gps: GPS longitude (optional, for track map)
            session_id: Session identifier

        Returns:
            GGAnalysisResult with statistics and points
        """
        # Filter out invalid/zero data points - apply mask to ALL arrays
        valid_mask = ~(np.isnan(lat_acc_data) | np.isnan(lon_acc_data))

        lat_acc = lat_acc_data[valid_mask]
        lon_acc = lon_acc_data[valid_mask]
        time = time_data[valid_mask]
        speed_filtered = speed_data[valid_mask] if speed_data is not None else None
        gear_filtered = gear_data[valid_mask] if gear_data is not None else None
        lap_filtered = lap_data[valid_mask] if lap_data is not None else None

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

        # Get GPS data if available
        gps_lat = lat_gps[valid_mask] if lat_gps is not None else np.zeros(len(time))
        gps_lon = lon_gps[valid_mask] if lon_gps is not None else np.zeros(len(time))

        # Get throttle data if available
        throttle = throttle_data[valid_mask] if throttle_data is not None else np.zeros(len(time))

        # Build points list
        points = []
        for i in range(len(time)):
            p = GGPoint(
                time=float(time[i]),
                lat_acc=float(lat_acc[i]),
                lon_acc=float(lon_acc[i]),
                total_g=float(total_g[i]),
                speed_mph=float(speed_filtered[i]) if speed_filtered is not None else 0.0,
                throttle_pct=float(throttle[i]),
                gear=int(gear_filtered[i]) if gear_filtered is not None else 0,
                lap_number=int(lap_filtered[i]) if lap_filtered is not None else 0,
                lat=float(gps_lat[i]),
                lon=float(gps_lon[i])
            )
            points.append(p)

        # Calculate quadrant breakdown with per-quadrant reference values
        quadrants = self._calculate_quadrants(
            lat_acc, lon_acc, total_g,
            max_lateral_g=self.max_g_reference,
            max_braking_g=self.max_braking_g,
            power_limited_accel_g=self.power_limited_accel_g
        )

        # Find power-limited zones
        power_limited_zones, power_limited_pct = self._find_power_limited_zones(
            time, lat_acc, lon_acc, throttle, gps_lat, gps_lon
        )

        # Find low utilization zones (excluding power-limited)
        low_utilization_zones = self._find_low_utilization_zones(
            time, lat_acc, lon_acc, total_g, reference_max_g,
            gps_lat, gps_lon, throttle
        )

        # Get unique lap numbers
        if lap_filtered is not None:
            lap_numbers = sorted(list(set(int(x) for x in lap_filtered if x > 0)))
        else:
            lap_numbers = []

        # Calculate GPS bounds for track map
        gps_bounds = {}
        if lat_gps is not None and lon_gps is not None:
            valid_lat = gps_lat[~np.isnan(gps_lat) & (gps_lat != 0)]
            valid_lon = gps_lon[~np.isnan(gps_lon) & (gps_lon != 0)]
            if len(valid_lat) > 0 and len(valid_lon) > 0:
                gps_bounds = {
                    "min_lat": float(np.min(valid_lat)),
                    "max_lat": float(np.max(valid_lat)),
                    "min_lon": float(np.min(valid_lon)),
                    "max_lon": float(np.max(valid_lon))
                }

        # Calculate new actionable metrics
        # 1. Corner-only utilization (lat_g > 0.3g = in a corner)
        in_corner_mask = np.abs(lat_acc) > 0.3
        if np.sum(in_corner_mask) > 0:
            corner_total_g = total_g[in_corner_mask]
            corner_utilization_pct = float(np.mean(corner_total_g) / reference_max_g * 100)
        else:
            corner_utilization_pct = 0.0

        # 2. Non-power-limited utilization (exclude straights with full throttle)
        power_limited_mask = (
            (throttle >= self.POWER_LIMITED_THROTTLE) &
            (np.abs(lat_acc) < self.POWER_LIMITED_LAT_G)
        )
        non_power_limited_mask = ~power_limited_mask
        if np.sum(non_power_limited_mask) > 0:
            non_pl_total_g = total_g[non_power_limited_mask]
            non_power_limited_utilization_pct = float(np.mean(non_pl_total_g) / reference_max_g * 100)
        else:
            non_power_limited_utilization_pct = utilization_pct

        # 3. Braking to lateral ratio
        braking_to_lateral_ratio = max_braking_g / max_lateral_g if max_lateral_g > 0 else 0.0

        # 4. Warnings and flags
        warnings = []
        max_g_exceeds_config = False
        braking_opportunity = False

        # Check if max lateral exceeds vehicle config (possible GPS noise or wrong config)
        if max_lateral_g > self.max_g_reference * 1.1:
            max_g_exceeds_config = True
            warnings.append(
                f"Max lateral G ({max_lateral_g:.2f}g) exceeds vehicle config ({self.max_g_reference:.2f}g). "
                "Check vehicle settings or possible GPS noise."
            )

        # Check if max braking exceeds vehicle config
        if max_braking_g > self.max_braking_g * 1.1:
            warnings.append(
                f"Max braking G ({max_braking_g:.2f}g) exceeds vehicle config ({self.max_braking_g:.2f}g). "
                "Check vehicle settings or possible GPS noise."
            )

        # Check if braking is significantly lower than lateral (opportunity to brake harder)
        if braking_to_lateral_ratio < 0.85:
            braking_opportunity = True
            warnings.append(
                f"Braking G ({max_braking_g:.2f}g) is only {braking_to_lateral_ratio:.0%} of lateral G ({max_lateral_g:.2f}g). "
                "You may be able to brake harder."
            )

        stats = GGStats(
            max_lateral_g=max_lateral_g,
            max_braking_g=max_braking_g,
            max_acceleration_g=max_acceleration_g,
            max_combined_g=max_combined_g,
            avg_utilized_g=avg_utilized_g,
            utilization_pct=utilization_pct,
            data_derived_max_g=data_derived_max_g,
            points_count=len(points),
            corner_utilization_pct=corner_utilization_pct,
            non_power_limited_utilization_pct=non_power_limited_utilization_pct,
            braking_to_lateral_ratio=braking_to_lateral_ratio
        )

        return GGAnalysisResult(
            session_id=session_id,
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            stats=stats,
            points=points,
            reference_max_g=reference_max_g,
            reference_lateral_g=self.max_g_reference,
            reference_braking_g=self.max_braking_g,
            reference_accel_g=self.power_limited_accel_g,
            low_utilization_zones=low_utilization_zones,
            quadrants=quadrants,
            power_limited_zones=power_limited_zones,
            power_limited_pct=power_limited_pct,
            lap_numbers=lap_numbers,
            gps_bounds=gps_bounds,
            warnings=warnings,
            max_g_exceeds_config=max_g_exceeds_config,
            braking_opportunity=braking_opportunity
        )

    def analyze_from_parquet(
        self,
        parquet_path: str,
        session_id: Optional[str] = None,
        lap_filter: Optional[int] = None,
        include_trace: bool = False,
        **kwargs,
    ) -> GGAnalysisResult:
        """
        Analyze G-G data from a Parquet file.

        Args:
            parquet_path: Path to Parquet file
            session_id: Session identifier (defaults to filename)
            lap_filter: If specified, only analyze data from this lap number
            include_trace: If True, attach CalculationTrace to report.

        Returns:
            GGAnalysisResult with statistics and points
        """
        trace = self._create_trace("GGAnalyzer") if include_trace else None

        df = pd.read_parquet(parquet_path)

        if session_id is None:
            from pathlib import Path
            session_id = Path(parquet_path).stem

        time_data = df.index.values

        # Find required columns
        from ..utils.dataframe_helpers import find_column_name
        lat_acc_col = find_column_name(df, ['GPS LatAcc', 'LatAcc', 'lateral_acc'])
        lon_acc_col = find_column_name(df, ['GPS LonAcc', 'LonAcc', 'longitudinal_acc'])
        lat_acc = find_column(df, ['GPS LatAcc', 'LatAcc', 'lateral_acc'])
        lon_acc = find_column(df, ['GPS LonAcc', 'LonAcc', 'longitudinal_acc'])

        if lat_acc is None or lon_acc is None:
            raise ValueError("Parquet file missing GPS LatAcc/LonAcc columns")

        # Find optional columns
        speed_col = find_column_name(df, ['GPS Speed', 'speed', 'Speed'])
        speed_data = find_column(df, ['GPS Speed', 'speed', 'Speed'])
        if speed_data is not None and speed_data.max() < 100:
            speed_data = speed_data * SPEED_MS_TO_MPH  # Convert m/s to mph

        throttle_col = find_column_name(df, ['PedalPos', 'throttle', 'Throttle'])
        throttle_data = find_column(df, ['PedalPos', 'throttle', 'Throttle'])

        # GPS coordinates for track map
        lat_gps = find_column(df, ['GPS Latitude', 'latitude', 'gps_lat'])
        lon_gps = find_column(df, ['GPS Longitude', 'longitude', 'gps_lon'])

        # Count NaN before filtering
        total_samples = len(lat_acc)
        nan_count = int(np.sum(np.isnan(lat_acc) | np.isnan(lon_acc)))

        # Try to detect laps if not already in data
        lap_data = None
        all_lap_numbers = []
        if lat_gps is not None and lon_gps is not None:
            try:
                from ..analysis.lap_analyzer import LapAnalyzer
                session_data = {
                    'time': time_data,
                    'latitude': lat_gps,
                    'longitude': lon_gps,
                    'rpm': np.zeros(len(time_data)),
                    'speed_mph': speed_data if speed_data is not None else np.zeros(len(time_data)),
                    'speed_ms': (speed_data / SPEED_MS_TO_MPH) if speed_data is not None else np.zeros(len(time_data))
                }
                analyzer = LapAnalyzer(session_data)
                laps = analyzer.detect_laps()
                if laps:
                    lap_data = np.zeros(len(time_data))
                    for lap in laps:
                        lap_data[lap.start_index:lap.end_index+1] = lap.lap_number
                    all_lap_numbers = sorted(list(set(int(lap.lap_number) for lap in laps)))
            except Exception:
                pass  # Lap detection failed, continue without

        # Apply lap filter if specified - filter all arrays to only include data from selected lap
        if lap_filter is not None and lap_data is not None:
            lap_mask = lap_data == lap_filter
            if np.sum(lap_mask) > 0:
                time_data = time_data[lap_mask]
                lat_acc = lat_acc[lap_mask]
                lon_acc = lon_acc[lap_mask]
                if speed_data is not None:
                    speed_data = speed_data[lap_mask]
                if throttle_data is not None:
                    throttle_data = throttle_data[lap_mask]
                if lat_gps is not None:
                    lat_gps = lat_gps[lap_mask]
                if lon_gps is not None:
                    lon_gps = lon_gps[lap_mask]
                # Keep lap_data filtered but preserve the lap number
                lap_data = lap_data[lap_mask]

        result = self.analyze_from_arrays(
            time_data, lat_acc, lon_acc,
            speed_data=speed_data,
            throttle_data=throttle_data,
            lap_data=lap_data,
            lat_gps=lat_gps,
            lon_gps=lon_gps,
            session_id=session_id
        )

        # Always include all lap numbers so the dropdown can be populated
        # even when filtering to a specific lap
        if all_lap_numbers:
            result.lap_numbers = all_lap_numbers

        if trace:
            nan_pct = (nan_count / total_samples * 100) if total_samples > 0 else 0
            trace.record_input("lat_acc_column", lat_acc_col)
            trace.record_input("lon_acc_column", lon_acc_col)
            trace.record_input("speed_column", speed_col)
            trace.record_input("throttle_column", throttle_col)
            trace.record_input("sample_count", total_samples - nan_count)
            trace.record_input("nan_pct", round(nan_pct, 1))
            trace.record_input("lap_filter", lap_filter)

            trace.record_config("max_g_reference", self.max_g_reference)
            trace.record_config("max_braking_g", self.max_braking_g)
            trace.record_config("power_limited_accel_g", self.power_limited_accel_g)
            try:
                from ..config.vehicles import get_active_vehicle
                vehicle = get_active_vehicle()
                trace.record_config("vehicle_name", vehicle.name if vehicle else "unknown")
            except Exception:
                trace.record_config("vehicle_name", "unknown")

            trace.record_intermediate("max_combined_g", result.stats.max_combined_g)
            trace.record_intermediate("p95_combined_g", result.stats.data_derived_max_g)
            trace.record_intermediate("utilization_pct", result.stats.utilization_pct)
            trace.record_intermediate("corner_utilization_pct", result.stats.corner_utilization_pct)
            trace.record_intermediate("braking_to_lateral_ratio", result.stats.braking_to_lateral_ratio)

            self._run_sanity_checks(trace, result, nan_pct)
            result.trace = trace

        return result

    def _run_sanity_checks(self, trace, result: GGAnalysisResult, nan_pct: float) -> None:
        """Run sanity checks on G-G analysis results."""
        # Check 4.1: config_matches_vehicle
        try:
            from ..config.vehicles import get_active_vehicle
            vehicle = get_active_vehicle()
            if vehicle and hasattr(vehicle, 'max_lateral_g'):
                if abs(self.max_g_reference - vehicle.max_lateral_g) < 0.01:
                    trace.add_check(
                        "config_matches_vehicle", "pass",
                        f"Reference G ({self.max_g_reference}) matches vehicle config ({vehicle.max_lateral_g})",
                        expected=vehicle.max_lateral_g, actual=self.max_g_reference,
                        impact="The G-force reference value determines the friction circle boundary. A wrong reference makes utilization percentages and quadrant breakdowns use the wrong baseline.",
                    )
                else:
                    trace.add_check(
                        "config_matches_vehicle", "warn",
                        f"Reference G ({self.max_g_reference}) differs from vehicle config ({vehicle.max_lateral_g})",
                        expected=vehicle.max_lateral_g, actual=self.max_g_reference,
                        impact="The G-force reference value determines the friction circle boundary. A wrong reference makes utilization percentages and quadrant breakdowns use the wrong baseline.",
                    )
            else:
                trace.add_check(
                    "config_matches_vehicle", "warn",
                    "No active vehicle or max_lateral_g not set",
                    impact="The G-force reference value determines the friction circle boundary. A wrong reference makes utilization percentages and quadrant breakdowns use the wrong baseline.",
                )
        except Exception:
            trace.add_check(
                "config_matches_vehicle", "warn",
                "Could not load vehicle config for comparison",
                impact="The G-force reference value determines the friction circle boundary. A wrong reference makes utilization percentages and quadrant breakdowns use the wrong baseline.",
            )

        # Check 4.2: data_quality
        if nan_pct < 5:
            trace.add_check(
                "data_quality", "pass",
                f"Only {nan_pct:.1f}% NaN values in accelerometer data",
                expected="< 5%", actual=f"{nan_pct:.1f}%",
                impact="High NaN percentage means many data points were dropped. Remaining points may not represent the full range of driving, biasing utilization and peak G calculations.",
            )
        elif nan_pct < 20:
            trace.add_check(
                "data_quality", "warn",
                f"{nan_pct:.1f}% NaN values in accelerometer data, results may be less reliable",
                expected="< 5%", actual=f"{nan_pct:.1f}%",
                impact="High NaN percentage means many data points were dropped. Remaining points may not represent the full range of driving, biasing utilization and peak G calculations.",
            )
        else:
            trace.add_check(
                "data_quality", "fail",
                f"{nan_pct:.1f}% NaN values in accelerometer data, analysis unreliable",
                expected="< 20%", actual=f"{nan_pct:.1f}%",
                severity="error",
                impact="High NaN percentage means many data points were dropped. Remaining points may not represent the full range of driving, biasing utilization and peak G calculations.",
            )

        # Check 4.3: g_force_plausible
        max_combined = result.stats.max_combined_g
        if max_combined < 3.0:
            trace.add_check(
                "g_force_plausible", "pass",
                f"Max combined G-force {max_combined:.2f}g is plausible",
                expected="< 3.0", actual=round(max_combined, 2),
                impact="G-forces above 3.0g are physically impossible on street tires and indicate sensor error. All friction circle analysis and utilization percentages become meaningless.",
            )
        else:
            trace.add_check(
                "g_force_plausible", "fail",
                f"Max combined G-force {max_combined:.2f}g exceeds 3.0g limit for street tires",
                expected="< 3.0", actual=round(max_combined, 2),
                severity="error",
                impact="G-forces above 3.0g are physically impossible on street tires and indicate sensor error. All friction circle analysis and utilization percentages become meaningless.",
            )

        # Check 4.4: utilization_plausible
        util_pct = result.stats.utilization_pct
        if 10 <= util_pct <= 100:
            trace.add_check(
                "utilization_plausible", "pass",
                f"Utilization {util_pct:.1f}% is in expected range (10-100%)",
                expected="10-100%", actual=f"{util_pct:.1f}%",
                impact="Utilization outside 10-100% suggests misconfigured reference G, data errors, or non-track driving. Quadrant analysis and improvement recommendations may be misleading.",
            )
        else:
            trace.add_check(
                "utilization_plausible", "warn",
                f"Utilization {util_pct:.1f}% is outside expected range (10-100%)",
                expected="10-100%", actual=f"{util_pct:.1f}%",
                impact="Utilization outside 10-100% suggests misconfigured reference G, data errors, or non-track driving. Quadrant analysis and improvement recommendations may be misleading.",
            )

    def _calculate_quadrants(
        self,
        lat_acc: np.ndarray,
        lon_acc: np.ndarray,
        total_g: np.ndarray,
        max_lateral_g: float,
        max_braking_g: float,
        power_limited_accel_g: float
    ) -> List[QuadrantStats]:
        """
        Calculate statistics for each quadrant of the G-G diagram.

        Uses per-quadrant reference values for accurate utilization:
        - Lateral (left/right): uses max_lateral_g
        - Braking: uses max_braking_g (often higher than lateral due to weight transfer)
        - Acceleration: uses power_limited_accel_g (engine/traction limited)
        """
        quadrants = []

        # Each quadrant has its own reference g value
        definitions = [
            ("lat_left", "Left Corners", lat_acc < -0.1, "#e74c3c",
             np.abs(lat_acc), max_lateral_g),
            ("lat_right", "Right Corners", lat_acc > 0.1, "#3498db",
             np.abs(lat_acc), max_lateral_g),
            ("braking", "Braking", lon_acc < -0.1, "#f39c12",
             np.abs(lon_acc), max_braking_g),
            ("acceleration", "Acceleration", lon_acc > 0.1, "#2ecc71",
             lon_acc, power_limited_accel_g)
        ]

        for name, display_name, mask, color, g_values, ref_g in definitions:
            if np.sum(mask) > 0:
                q_g_values = g_values[mask]
                max_g_in_quadrant = float(np.max(q_g_values))
                avg_g_in_quadrant = float(np.mean(q_g_values))
                # Utilization is based on the appropriate reference for this quadrant
                utilization = float(avg_g_in_quadrant / ref_g * 100) if ref_g > 0 else 0.0
                # Cap at 100% for display (can exceed if driver is exceeding expected limits)
                utilization_capped = min(utilization, 100.0)

                quadrants.append(QuadrantStats(
                    name=name,
                    display_name=display_name,
                    max_g=max_g_in_quadrant,
                    avg_g=avg_g_in_quadrant,
                    utilization_pct=utilization_capped,
                    time_spent_pct=float(np.sum(mask) / len(lat_acc) * 100),
                    points_count=int(np.sum(mask)),
                    color=color
                ))
            else:
                quadrants.append(QuadrantStats(
                    name=name,
                    display_name=display_name,
                    max_g=0.0,
                    avg_g=0.0,
                    utilization_pct=0.0,
                    time_spent_pct=0.0,
                    points_count=0,
                    color=color
                ))

        return quadrants

    def _find_power_limited_zones(
        self,
        time: np.ndarray,
        lat_acc: np.ndarray,
        lon_acc: np.ndarray,
        throttle: np.ndarray,
        lat_gps: np.ndarray,
        lon_gps: np.ndarray
    ) -> Tuple[List[LowUtilizationZone], float]:
        """
        Find zones where the car is power-limited (full throttle on straight).

        Args:
            time: Time array
            lat_acc: Lateral acceleration
            lon_acc: Longitudinal acceleration
            throttle: Throttle position (0-100)
            lat_gps: GPS latitude
            lon_gps: GPS longitude

        Returns:
            Tuple of (zones list, percentage of session power-limited)
        """
        zones = []

        # Power limited: full throttle, on straight, low acceleration
        full_throttle = throttle >= self.POWER_LIMITED_THROTTLE
        on_straight = np.abs(lat_acc) < self.POWER_LIMITED_LAT_G
        low_accel = lon_acc < self.power_limited_accel_g

        power_limited_mask = full_throttle & on_straight & low_accel

        # Calculate percentage of session
        power_limited_pct = float(np.sum(power_limited_mask) / len(time) * 100) if len(time) > 0 else 0.0

        # Find continuous zones
        zone_start = None
        for i in range(len(power_limited_mask)):
            if power_limited_mask[i] and zone_start is None:
                zone_start = i
            elif not power_limited_mask[i] and zone_start is not None:
                if i - zone_start >= 10:  # Minimum 1s at 10Hz
                    avg_lat = float(np.mean(lat_gps[zone_start:i])) if lat_gps is not None else 0.0
                    avg_lon = float(np.mean(lon_gps[zone_start:i])) if lon_gps is not None else 0.0

                    zones.append(LowUtilizationZone(
                        start_time=float(time[zone_start]),
                        end_time=float(time[i-1]),
                        start_idx=zone_start,
                        end_idx=i-1,
                        duration=float(time[i-1] - time[zone_start]),
                        avg_lat_g=float(np.mean(np.abs(lat_acc[zone_start:i]))),
                        avg_lon_g=float(np.mean(lon_acc[zone_start:i])),
                        avg_total_g=float(np.mean(np.sqrt(lat_acc[zone_start:i]**2 + lon_acc[zone_start:i]**2))),
                        utilization_pct=100.0,  # Power limited = using all available
                        avg_lat=avg_lat,
                        avg_lon=avg_lon,
                        zone_type="power_limited",
                        recommendation="Power limited zone - engine at maximum output"
                    ))
                zone_start = None

        return zones[:10], power_limited_pct

    def _find_low_utilization_zones(
        self,
        time: np.ndarray,
        lat_acc: np.ndarray,
        lon_acc: np.ndarray,
        total_g: np.ndarray,
        max_g: float,
        lat_gps: np.ndarray,
        lon_gps: np.ndarray,
        throttle: np.ndarray,
        threshold: float = 0.5
    ) -> List[LowUtilizationZone]:
        """
        Find zones where utilization is low (grip left on table).
        Excludes power-limited zones.

        Args:
            time: Time array
            lat_acc: Lateral acceleration
            lon_acc: Longitudinal acceleration
            total_g: Combined g-force
            max_g: Reference maximum g
            lat_gps: GPS latitude
            lon_gps: GPS longitude
            throttle: Throttle position
            threshold: Utilization threshold (0.5 = 50% of max grip)

        Returns:
            List of zones with low utilization
        """
        zones = []

        # Find periods where we're in a turn (significant lateral g) but low total g
        in_turn = np.abs(lat_acc) > 0.3  # In a turn
        low_g = total_g < (max_g * threshold)

        # Exclude power-limited zones (full throttle on straight)
        not_power_limited = ~((throttle >= self.POWER_LIMITED_THROTTLE) &
                              (np.abs(lat_acc) < self.POWER_LIMITED_LAT_G))

        low_util_mask = in_turn & low_g & not_power_limited

        # Find continuous zones
        zone_start = None
        for i in range(len(low_util_mask)):
            if low_util_mask[i] and zone_start is None:
                zone_start = i
            elif not low_util_mask[i] and zone_start is not None:
                if i - zone_start >= 5:  # Minimum 0.5s at 10Hz
                    avg_total = float(np.mean(total_g[zone_start:i]))
                    avg_lat = float(np.mean(lat_gps[zone_start:i])) if lat_gps is not None else 0.0
                    avg_lon = float(np.mean(lon_gps[zone_start:i])) if lon_gps is not None else 0.0
                    avg_lat_g = float(np.mean(np.abs(lat_acc[zone_start:i])))

                    # Calculate how much more grip could be used
                    grip_available = max_g - avg_total

                    zones.append(LowUtilizationZone(
                        start_time=float(time[zone_start]),
                        end_time=float(time[i-1]),
                        start_idx=zone_start,
                        end_idx=i-1,
                        duration=float(time[i-1] - time[zone_start]),
                        avg_lat_g=avg_lat_g,
                        avg_lon_g=float(np.mean(lon_acc[zone_start:i])),
                        avg_total_g=avg_total,
                        utilization_pct=float(avg_total / max_g * 100),
                        avg_lat=avg_lat,
                        avg_lon=avg_lon,
                        zone_type="low_grip",
                        recommendation=f"Could carry {grip_available:.2f}g more lateral grip here"
                    ))
                zone_start = None

        return zones[:10]  # Return top 10 zones

    def analyze_from_channels(self, channels, session_id="unknown",
                              include_trace=False, **kwargs):
        """Analyze from pre-loaded SessionChannels."""
        trace = self._create_trace("GGAnalyzer") if include_trace else None

        result = self.analyze_from_arrays(
            time_data=channels.time,
            lat_acc_data=channels.lat_acc,
            lon_acc_data=channels.lon_acc,
            speed_data=channels.speed_mph,
            throttle_data=channels.throttle,
            lat_gps=channels.latitude,
            lon_gps=channels.longitude,
            session_id=session_id,
        )

        if trace:
            # Calculate NaN percentage
            nan_count = 0
            total_samples = channels.sample_count
            if channels.lat_acc is not None:
                nan_count += np.isnan(channels.lat_acc).sum()
            if channels.lon_acc is not None:
                nan_count += np.isnan(channels.lon_acc).sum()
            nan_pct = (nan_count / (total_samples * 2) * 100) if total_samples > 0 else 0

            trace.record_input("lat_acc_channel", "lat_acc")
            trace.record_input("lon_acc_channel", "lon_acc")
            trace.record_input("speed_channel", "speed_mph")
            trace.record_input("throttle_channel", "throttle")
            trace.record_input("sample_count", total_samples - nan_count)
            trace.record_input("nan_pct", round(nan_pct, 1))
            trace.record_input("lap_filter", None)

            trace.record_config("max_g_reference", self.max_g_reference)
            trace.record_config("max_braking_g", self.max_braking_g)
            trace.record_config("power_limited_accel_g", self.power_limited_accel_g)
            try:
                from ..config.vehicles import get_active_vehicle
                vehicle = get_active_vehicle()
                trace.record_config("vehicle_name", vehicle.name if vehicle else "unknown")
            except Exception:
                trace.record_config("vehicle_name", "unknown")

            trace.record_intermediate("max_combined_g", result.stats.max_combined_g)
            trace.record_intermediate("p95_combined_g", result.stats.data_derived_max_g)
            trace.record_intermediate("utilization_pct", result.stats.utilization_pct)
            trace.record_intermediate("corner_utilization_pct", result.stats.corner_utilization_pct)
            trace.record_intermediate("braking_to_lateral_ratio", result.stats.braking_to_lateral_ratio)

            self._run_sanity_checks(trace, result, nan_pct)
            result.trace = trace

        return result


analyzer_registry.register(GGAnalyzer)


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
