"""
Corner Analysis Module

Per-corner metrics calculation: entry/apex/exit speeds, time in corner,
throttle pickup point, lift detection, and lap-to-lap comparison.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path
import json

from .corner_detection import CornerDetector, CornerZone, CornerDetectionResult


@dataclass
class CornerMetrics:
    """Metrics for a single corner passage."""
    corner_name: str
    corner_alias: Optional[str] = None

    # Speeds (mph)
    entry_speed: float = 0.0      # Speed at turn-in
    min_speed: float = 0.0        # Minimum speed (apex)
    exit_speed: float = 0.0       # Speed at track-out
    speed_scrub: float = 0.0      # entry_speed - min_speed (how much speed lost)

    # Timing
    time_in_corner: float = 0.0   # Time from entry to exit (seconds)
    time_to_apex: float = 0.0     # Time from entry to apex
    time_from_apex: float = 0.0   # Time from apex to exit

    # Throttle analysis
    throttle_pickup_time: float = 0.0  # Time after entry when throttle applied
    throttle_pickup_pct: float = 0.0   # Distance % through corner when throttle applied
    throttle_at_apex: float = 0.0      # Throttle position at apex

    # Lift/hesitation detection
    lift_detected: bool = False        # Did driver lift throttle mid-corner?
    lift_time: float = 0.0             # When lift occurred (seconds after entry)
    lift_duration: float = 0.0         # How long the lift lasted

    # Trail braking
    trail_brake_detected: bool = False  # Braking while turning
    trail_brake_duration: float = 0.0   # Duration of trail braking

    # G-forces
    max_lateral_g: float = 0.0
    avg_lateral_g: float = 0.0
    max_braking_g: float = 0.0
    max_accel_g: float = 0.0        # Maximum acceleration (positive longitudinal g)

    # Distances (feet)
    brake_point_distance: float = 0.0   # Distance before apex where braking started
    throttle_on_distance: float = 0.0   # Distance after apex where throttle applied

    # Quality indicators
    consistency_score: float = 0.0  # 0-100, how consistent vs other laps
    optimal_line: bool = True       # Whether apex was hit optimally

    def to_dict(self) -> dict:
        return {
            "corner_name": self.corner_name,
            "corner_alias": self.corner_alias,
            "speeds": {
                "entry": round(self.entry_speed, 1),
                "min": round(self.min_speed, 1),
                "exit": round(self.exit_speed, 1),
                "scrub": round(self.speed_scrub, 1)
            },
            "timing": {
                "total": round(self.time_in_corner, 3),
                "to_apex": round(self.time_to_apex, 3),
                "from_apex": round(self.time_from_apex, 3)
            },
            "throttle": {
                "pickup_time": round(self.throttle_pickup_time, 3),
                "pickup_pct": round(self.throttle_pickup_pct, 1),
                "at_apex": round(self.throttle_at_apex, 1)
            },
            "lift": {
                "detected": self.lift_detected,
                "time": round(self.lift_time, 3) if self.lift_detected else None,
                "duration": round(self.lift_duration, 3) if self.lift_detected else None
            },
            "trail_brake": {
                "detected": self.trail_brake_detected,
                "duration": round(self.trail_brake_duration, 3) if self.trail_brake_detected else None
            },
            "g_forces": {
                "max_lateral": round(self.max_lateral_g, 2),
                "avg_lateral": round(self.avg_lateral_g, 2),
                "max_braking": round(self.max_braking_g, 2),
                "max_accel": round(self.max_accel_g, 2)
            },
            "distances": {
                "brake_point": round(self.brake_point_distance, 1),
                "throttle_on": round(self.throttle_on_distance, 1)
            },
            "quality": {
                "consistency_score": round(self.consistency_score, 1),
                "optimal_line": self.optimal_line
            }
        }


@dataclass
class LapCornerAnalysis:
    """Corner analysis for a single lap."""
    lap_number: int
    lap_time: float
    corners: List[CornerMetrics]
    total_corner_time: float
    avg_entry_speed: float
    avg_exit_speed: float
    lifts_count: int
    trail_brakes_count: int

    def to_dict(self) -> dict:
        return {
            "lap_number": self.lap_number,
            "lap_time": round(self.lap_time, 2),
            "corners": [c.to_dict() for c in self.corners],
            "summary": {
                "total_corner_time": round(self.total_corner_time, 2),
                "avg_entry_speed": round(self.avg_entry_speed, 1),
                "avg_exit_speed": round(self.avg_exit_speed, 1),
                "lifts_count": self.lifts_count,
                "trail_brakes_count": self.trail_brakes_count
            }
        }


@dataclass
class CornerComparison:
    """Comparison of a corner across multiple laps."""
    corner_name: str
    corner_alias: Optional[str]
    laps: List[int]

    # Stats across laps
    entry_speeds: List[float]
    min_speeds: List[float]
    exit_speeds: List[float]
    times: List[float]

    # Deltas from fastest
    fastest_lap: int
    fastest_time: float
    time_deltas: List[float]

    # Consistency
    entry_speed_std: float
    time_std: float

    def to_dict(self) -> dict:
        return {
            "corner_name": self.corner_name,
            "corner_alias": self.corner_alias,
            "laps": self.laps,
            "fastest": {
                "lap": self.fastest_lap,
                "time": round(self.fastest_time, 3)
            },
            "entry_speeds": {
                "values": [round(s, 1) for s in self.entry_speeds],
                "best": round(max(self.entry_speeds), 1),
                "std": round(self.entry_speed_std, 1)
            },
            "min_speeds": {
                "values": [round(s, 1) for s in self.min_speeds],
                "best": round(max(self.min_speeds), 1)
            },
            "exit_speeds": {
                "values": [round(s, 1) for s in self.exit_speeds],
                "best": round(max(self.exit_speeds), 1)
            },
            "times": {
                "values": [round(t, 3) for t in self.times],
                "deltas": [round(d, 3) for d in self.time_deltas],
                "std": round(self.time_std, 3)
            }
        }


@dataclass
class CornerAnalysisResult:
    """Complete corner analysis for a session."""
    session_id: str
    track_name: str
    analysis_timestamp: str
    laps: List[LapCornerAnalysis]
    corner_comparisons: List[CornerComparison]
    corner_zones: List[CornerZone]
    recommendations: List[str]

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "track_name": self.track_name,
            "analysis_timestamp": self.analysis_timestamp,
            "lap_count": len(self.laps),
            "corner_count": len(self.corner_zones),
            "laps": [lap.to_dict() for lap in self.laps],
            "corner_comparisons": [c.to_dict() for c in self.corner_comparisons],
            "corner_zones": [z.to_dict() for z in self.corner_zones],
            "recommendations": self.recommendations
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class CornerAnalyzer:
    """
    Analyzes corner performance from telemetry data.

    Provides per-corner metrics and lap-to-lap comparison.
    """

    # Thresholds for detection
    THROTTLE_PICKUP_THRESHOLD = 10.0    # % throttle to consider "picked up"
    LIFT_THRESHOLD = 5.0                # % throttle drop to detect lift
    TRAIL_BRAKE_LON_ACC_THRESHOLD = -0.15  # g - braking while cornering

    def __init__(self, detector: Optional[CornerDetector] = None):
        """
        Initialize analyzer.

        Args:
            detector: CornerDetector instance (optional, will create default)
        """
        self.detector = detector or CornerDetector()

    def analyze_from_parquet(
        self,
        parquet_path: str,
        session_id: Optional[str] = None,
        track_name: str = "Unknown Track"
    ) -> CornerAnalysisResult:
        """
        Analyze corners from a Parquet file.

        Args:
            parquet_path: Path to parquet file
            session_id: Session identifier (defaults to filename)
            track_name: Track name for the report

        Returns:
            CornerAnalysisResult
        """
        df = pd.read_parquet(parquet_path)

        if session_id is None:
            session_id = Path(parquet_path).stem

        # Extract data arrays
        time_data = df.index.values
        lat_data = self._find_column(df, ['GPS Latitude', 'gps_lat', 'latitude'])
        lon_data = self._find_column(df, ['GPS Longitude', 'gps_lon', 'longitude'])
        speed_data = self._find_column(df, ['GPS Speed', 'gps_speed', 'speed'])
        radius_data = self._find_column(df, ['GPS Radius', 'radius'])
        lat_acc_data = self._find_column(df, ['GPS LatAcc', 'lat_acc'])
        lon_acc_data = self._find_column(df, ['GPS LonAcc', 'lon_acc'])
        throttle_data = self._find_column(df, ['PedalPos', 'throttle', 'Throttle'])

        if lat_data is None or lon_data is None:
            raise ValueError("Missing GPS latitude/longitude columns")

        if speed_data is None:
            speed_data = np.zeros(len(time_data))
        elif speed_data.max() < 100:
            speed_data = speed_data * 2.237

        if throttle_data is None:
            throttle_data = np.zeros(len(time_data))

        return self.analyze_from_arrays(
            time_data, lat_data, lon_data, speed_data,
            radius_data, lat_acc_data, lon_acc_data, throttle_data,
            session_id, track_name
        )

    def analyze_from_arrays(
        self,
        time_data: np.ndarray,
        lat_data: np.ndarray,
        lon_data: np.ndarray,
        speed_data: np.ndarray,
        radius_data: Optional[np.ndarray],
        lat_acc_data: Optional[np.ndarray],
        lon_acc_data: Optional[np.ndarray],
        throttle_data: Optional[np.ndarray],
        session_id: str = "unknown",
        track_name: str = "Unknown Track"
    ) -> CornerAnalysisResult:
        """
        Analyze corners from raw data arrays.

        Returns:
            CornerAnalysisResult
        """
        # First, detect corners using the full session
        detection = self.detector.detect_from_arrays(
            time_data, lat_data, lon_data, speed_data,
            radius_data, lat_acc_data, lon_acc_data
        )

        if not detection.corners:
            return CornerAnalysisResult(
                session_id=session_id,
                track_name=track_name,
                analysis_timestamp=datetime.now(timezone.utc).isoformat(),
                laps=[],
                corner_comparisons=[],
                corner_zones=[],
                recommendations=["No corners detected in session data"]
            )

        # Ensure we have acceleration data
        if lat_acc_data is None:
            lat_acc_data = np.zeros(len(time_data))
        if lon_acc_data is None:
            lon_acc_data = np.zeros(len(time_data))
        if throttle_data is None:
            throttle_data = np.zeros(len(time_data))

        # Analyze each corner
        lap_analysis = LapCornerAnalysis(
            lap_number=1,
            lap_time=time_data[-1] - time_data[0],
            corners=[],
            total_corner_time=0.0,
            avg_entry_speed=0.0,
            avg_exit_speed=0.0,
            lifts_count=0,
            trail_brakes_count=0
        )

        for corner in detection.corners:
            metrics = self._analyze_corner(
                corner, time_data, speed_data,
                lat_acc_data, lon_acc_data, throttle_data,
                lat_data, lon_data
            )
            lap_analysis.corners.append(metrics)

            # Phase 6: Set metrics on corner zone for easy access via corner.metrics
            corner.set_metrics(
                entry_speed=metrics.entry_speed,
                apex_speed=metrics.min_speed,  # min_speed is apex speed
                exit_speed=metrics.exit_speed,
                max_lateral_g=metrics.max_lateral_g,
                max_braking_g=metrics.max_braking_g,
                max_accel_g=metrics.max_accel_g,
                brake_point_distance=metrics.brake_point_distance,
                throttle_on_distance=metrics.throttle_on_distance,
                time_in_corner=metrics.time_in_corner
            )

        # Calculate lap summary
        if lap_analysis.corners:
            lap_analysis.total_corner_time = sum(c.time_in_corner for c in lap_analysis.corners)
            lap_analysis.avg_entry_speed = np.mean([c.entry_speed for c in lap_analysis.corners])
            lap_analysis.avg_exit_speed = np.mean([c.exit_speed for c in lap_analysis.corners])
            lap_analysis.lifts_count = sum(1 for c in lap_analysis.corners if c.lift_detected)
            lap_analysis.trail_brakes_count = sum(1 for c in lap_analysis.corners if c.trail_brake_detected)

        # For single lap, corner comparison is trivial
        corner_comparisons = []
        for i, corner in enumerate(detection.corners):
            if i < len(lap_analysis.corners):
                metrics = lap_analysis.corners[i]
                comparison = CornerComparison(
                    corner_name=corner.name,
                    corner_alias=corner.alias,
                    laps=[1],
                    entry_speeds=[metrics.entry_speed],
                    min_speeds=[metrics.min_speed],
                    exit_speeds=[metrics.exit_speed],
                    times=[metrics.time_in_corner],
                    fastest_lap=1,
                    fastest_time=metrics.time_in_corner,
                    time_deltas=[0.0],
                    entry_speed_std=0.0,
                    time_std=0.0
                )
                corner_comparisons.append(comparison)

        # Generate recommendations
        recommendations = self._generate_recommendations(lap_analysis)

        return CornerAnalysisResult(
            session_id=session_id,
            track_name=track_name,
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            laps=[lap_analysis],
            corner_comparisons=corner_comparisons,
            corner_zones=detection.corners,
            recommendations=recommendations
        )

    def _analyze_corner(
        self,
        corner: CornerZone,
        time_data: np.ndarray,
        speed_data: np.ndarray,
        lat_acc_data: np.ndarray,
        lon_acc_data: np.ndarray,
        throttle_data: np.ndarray,
        lat_data: Optional[np.ndarray] = None,
        lon_data: Optional[np.ndarray] = None
    ) -> CornerMetrics:
        """Analyze a single corner passage."""
        entry_idx = corner.entry_idx
        apex_idx = corner.apex_idx
        exit_idx = corner.exit_idx

        # Validate indices
        n = len(time_data)
        entry_idx = max(0, min(entry_idx, n - 1))
        apex_idx = max(0, min(apex_idx, n - 1))
        exit_idx = max(0, min(exit_idx, n - 1))

        # Speeds
        entry_speed = float(speed_data[entry_idx])
        min_speed = float(speed_data[apex_idx])
        exit_speed = float(speed_data[exit_idx])
        speed_scrub = entry_speed - min_speed

        # Timing
        time_in_corner = float(time_data[exit_idx] - time_data[entry_idx])
        time_to_apex = float(time_data[apex_idx] - time_data[entry_idx])
        time_from_apex = float(time_data[exit_idx] - time_data[apex_idx])

        # Throttle analysis
        throttle_pickup_time, throttle_pickup_pct = self._find_throttle_pickup(
            throttle_data[entry_idx:exit_idx+1],
            time_data[entry_idx:exit_idx+1]
        )
        throttle_at_apex = float(throttle_data[apex_idx])

        # Lift detection
        lift_detected, lift_time, lift_duration = self._detect_lift(
            throttle_data[entry_idx:exit_idx+1],
            time_data[entry_idx:exit_idx+1]
        )

        # Trail braking detection
        trail_brake_detected, trail_brake_duration = self._detect_trail_braking(
            lat_acc_data[entry_idx:apex_idx+1],
            lon_acc_data[entry_idx:apex_idx+1],
            time_data[entry_idx:apex_idx+1]
        )

        # G-forces
        corner_lat_acc = lat_acc_data[entry_idx:exit_idx+1]
        corner_lon_acc = lon_acc_data[entry_idx:exit_idx+1]
        max_lateral_g = float(np.max(np.abs(corner_lat_acc))) if len(corner_lat_acc) > 0 else 0
        avg_lateral_g = float(np.mean(np.abs(corner_lat_acc))) if len(corner_lat_acc) > 0 else 0
        max_braking_g = float(np.abs(np.min(corner_lon_acc))) if len(corner_lon_acc) > 0 else 0
        max_accel_g = float(np.max(corner_lon_acc)) if len(corner_lon_acc) > 0 else 0

        # Distance calculations (in feet)
        brake_point_distance = 0.0
        throttle_on_distance = 0.0

        if lat_data is not None and lon_data is not None:
            # Calculate brake point distance (from brake start to apex)
            brake_start_idx = corner.brake_start_idx
            if brake_start_idx >= 0 and brake_start_idx < apex_idx:
                brake_point_distance = self._calculate_distance_feet(
                    lat_data, lon_data, brake_start_idx, apex_idx
                )

            # Calculate throttle pickup distance (from apex to throttle pickup point)
            # Find throttle pickup index relative to entry
            throttle_pickup_idx = entry_idx
            for i in range(entry_idx, exit_idx + 1):
                if i < len(throttle_data) and throttle_data[i] > self.THROTTLE_PICKUP_THRESHOLD:
                    throttle_pickup_idx = i
                    break

            if throttle_pickup_idx > apex_idx:
                throttle_on_distance = self._calculate_distance_feet(
                    lat_data, lon_data, apex_idx, throttle_pickup_idx
                )

        return CornerMetrics(
            corner_name=corner.name,
            corner_alias=corner.alias,
            entry_speed=entry_speed,
            min_speed=min_speed,
            exit_speed=exit_speed,
            speed_scrub=speed_scrub,
            time_in_corner=time_in_corner,
            time_to_apex=time_to_apex,
            time_from_apex=time_from_apex,
            throttle_pickup_time=throttle_pickup_time,
            throttle_pickup_pct=throttle_pickup_pct,
            throttle_at_apex=throttle_at_apex,
            lift_detected=lift_detected,
            lift_time=lift_time,
            lift_duration=lift_duration,
            trail_brake_detected=trail_brake_detected,
            trail_brake_duration=trail_brake_duration,
            max_lateral_g=max_lateral_g,
            avg_lateral_g=avg_lateral_g,
            max_braking_g=max_braking_g,
            max_accel_g=max_accel_g,
            brake_point_distance=brake_point_distance,
            throttle_on_distance=throttle_on_distance,
            consistency_score=100.0,  # Will be calculated in multi-lap comparison
            optimal_line=True
        )

    def _find_throttle_pickup(
        self,
        throttle_data: np.ndarray,
        time_data: np.ndarray
    ) -> Tuple[float, float]:
        """Find when throttle was picked up in corner."""
        if len(throttle_data) == 0:
            return 0.0, 0.0

        start_time = time_data[0]
        total_time = time_data[-1] - start_time if len(time_data) > 1 else 1.0

        # Find first point where throttle exceeds threshold
        for i, throttle in enumerate(throttle_data):
            if throttle > self.THROTTLE_PICKUP_THRESHOLD:
                pickup_time = time_data[i] - start_time
                pickup_pct = (pickup_time / total_time) * 100 if total_time > 0 else 0
                return pickup_time, pickup_pct

        return total_time, 100.0  # Throttle never picked up

    def _detect_lift(
        self,
        throttle_data: np.ndarray,
        time_data: np.ndarray
    ) -> Tuple[bool, float, float]:
        """Detect if driver lifted throttle mid-corner."""
        if len(throttle_data) < 3:
            return False, 0.0, 0.0

        start_time = time_data[0]

        # Look for throttle drops after initial application
        was_on_throttle = False
        lift_start = None

        for i in range(1, len(throttle_data)):
            if throttle_data[i] > self.THROTTLE_PICKUP_THRESHOLD:
                was_on_throttle = True
            elif was_on_throttle and throttle_data[i] < throttle_data[i-1] - self.LIFT_THRESHOLD:
                # Detected a lift
                if lift_start is None:
                    lift_start = i
            elif lift_start is not None and throttle_data[i] > throttle_data[i-1]:
                # Throttle resumed - end of lift
                lift_time = time_data[lift_start] - start_time
                lift_duration = time_data[i] - time_data[lift_start]
                return True, lift_time, lift_duration

        return False, 0.0, 0.0

    def _detect_trail_braking(
        self,
        lat_acc_data: np.ndarray,
        lon_acc_data: np.ndarray,
        time_data: np.ndarray
    ) -> Tuple[bool, float]:
        """Detect trail braking (braking while turning)."""
        if len(lat_acc_data) < 2:
            return False, 0.0

        # Trail braking: significant lateral g AND negative longitudinal g
        trail_brake_mask = (np.abs(lat_acc_data) > 0.3) & (lon_acc_data < self.TRAIL_BRAKE_LON_ACC_THRESHOLD)

        if not np.any(trail_brake_mask):
            return False, 0.0

        # Calculate duration
        indices = np.where(trail_brake_mask)[0]
        if len(indices) == 0:
            return False, 0.0

        duration = time_data[indices[-1]] - time_data[indices[0]]
        return True, float(duration)

    def _calculate_distance_feet(
        self,
        lat_data: np.ndarray,
        lon_data: np.ndarray,
        start_idx: int,
        end_idx: int
    ) -> float:
        """
        Calculate distance between two points in feet.

        Args:
            lat_data: GPS latitude array (degrees)
            lon_data: GPS longitude array (degrees)
            start_idx: Starting index
            end_idx: Ending index

        Returns:
            Distance in feet
        """
        if start_idx >= end_idx or start_idx < 0 or end_idx >= len(lat_data):
            return 0.0

        # Sum the distances between consecutive points
        total_distance_meters = 0.0
        mean_lat = np.mean(lat_data[start_idx:end_idx+1])

        for i in range(start_idx, end_idx):
            # Convert lat/lon to meters
            lat_m1 = lat_data[i] * 111000.0
            lon_m1 = lon_data[i] * 111000.0 * np.cos(np.radians(mean_lat))
            lat_m2 = lat_data[i+1] * 111000.0
            lon_m2 = lon_data[i+1] * 111000.0 * np.cos(np.radians(mean_lat))

            # Calculate distance
            distance = np.sqrt((lat_m2 - lat_m1)**2 + (lon_m2 - lon_m1)**2)
            total_distance_meters += distance

        # Convert meters to feet (1 meter = 3.28084 feet)
        return total_distance_meters * 3.28084

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[np.ndarray]:
        """Find a column by trying multiple names."""
        for col in candidates:
            if col in df.columns:
                return df[col].values
            for actual_col in df.columns:
                if actual_col.lower() == col.lower():
                    return df[actual_col].values
        return None

    def _generate_recommendations(self, lap: LapCornerAnalysis) -> List[str]:
        """Generate recommendations based on corner analysis."""
        recommendations = []

        if not lap.corners:
            return ["No corners analyzed"]

        # Check for lifts
        if lap.lifts_count > 0:
            lift_corners = [c.corner_name for c in lap.corners if c.lift_detected]
            recommendations.append(
                f"Throttle lift detected in {lap.lifts_count} corner(s): {', '.join(lift_corners)}. "
                "Work on smoother throttle application to maintain momentum."
            )

        # Check for late throttle pickup
        late_pickup_corners = [c for c in lap.corners if c.throttle_pickup_pct > 70]
        if late_pickup_corners:
            names = [c.corner_name for c in late_pickup_corners]
            recommendations.append(
                f"Late throttle pickup in: {', '.join(names)}. "
                "Consider earlier throttle application to improve exit speed."
            )

        # Check for excessive speed scrub
        high_scrub_corners = [c for c in lap.corners if c.speed_scrub > 40]
        if high_scrub_corners:
            names = [c.corner_name for c in high_scrub_corners]
            recommendations.append(
                f"High speed scrub (>40 mph) in: {', '.join(names)}. "
                "May indicate over-braking or late apex."
            )

        # Trail braking feedback
        if lap.trail_brakes_count > len(lap.corners) * 0.3:
            recommendations.append(
                f"Good trail braking technique detected in {lap.trail_brakes_count} corners."
            )
        elif lap.trail_brakes_count == 0 and len(lap.corners) > 3:
            recommendations.append(
                "No trail braking detected. Consider carrying brake deeper into corners "
                "to rotate the car and improve mid-corner speed."
            )

        if not recommendations:
            recommendations.append(
                f"Analyzed {len(lap.corners)} corners. "
                f"Average entry: {lap.avg_entry_speed:.1f} mph, "
                f"Average exit: {lap.avg_exit_speed:.1f} mph."
            )

        return recommendations


def analyze_corners(parquet_path: str, track_name: str = "Unknown Track") -> CornerAnalysisResult:
    """
    Convenience function to analyze corners from a parquet file.

    Args:
        parquet_path: Path to parquet file
        track_name: Track name for the report

    Returns:
        CornerAnalysisResult
    """
    analyzer = CornerAnalyzer()
    return analyzer.analyze_from_parquet(parquet_path, track_name=track_name)
