"""
Corner Detection Module

Auto-detects corners from telemetry data using GPS radius, speed, and acceleration patterns.
Provides corner zone definitions for analysis.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import math

from ..utils.dataframe_helpers import find_column, SPEED_MS_TO_MPH


def classify_corner(
    apex_speed_mph: float,
    entry_speed_mph: float,
    duration_seconds: float,
    min_radius: float,
    lat_acc_data: np.ndarray,
    entry_idx: int,
    exit_idx: int
) -> Tuple[str, int]:
    """
    Classify corner type and calculate severity rating.

    Args:
        apex_speed_mph: Speed at apex
        entry_speed_mph: Speed at corner entry
        duration_seconds: Time spent in corner
        min_radius: Minimum turning radius in meters
        lat_acc_data: Lateral acceleration data array
        entry_idx: Entry index in data
        exit_idx: Exit index in data

    Returns:
        Tuple of (corner_type, severity_rating)

    Corner types:
        - "hairpin": <40mph apex speed
        - "sweeper": >60mph apex, >2s duration
        - "chicane": Direction change detected (<1s per direction)
        - "kink": >80mph apex, <1s duration
        - "normal": Default classification

    Severity (1-5):
        Based on speed loss percentage:
        - 1: <10% speed loss (very easy)
        - 2: 10-25% speed loss (easy)
        - 3: 25-40% speed loss (moderate)
        - 4: 40-55% speed loss (hard)
        - 5: >55% speed loss (very hard)
    """
    # Calculate speed loss percentage
    if entry_speed_mph > 0:
        speed_loss_pct = ((entry_speed_mph - apex_speed_mph) / entry_speed_mph) * 100
    else:
        speed_loss_pct = 0.0

    # Classify corner type based on Phase 5 requirements
    corner_type = "normal"  # Default

    # Check for chicane: direction change in lateral G within corner
    if exit_idx > entry_idx + 1:
        corner_lat_acc = lat_acc_data[entry_idx:exit_idx+1]
        # Chicane detected if lateral G changes sign (direction change)
        has_positive = np.any(corner_lat_acc > 0.3)
        has_negative = np.any(corner_lat_acc < -0.3)
        if has_positive and has_negative and duration_seconds < 1.0:
            corner_type = "chicane"

    # Only classify as other types if not a chicane
    # Order matters: check most specific criteria first
    if corner_type == "normal":
        if apex_speed_mph < 40:
            # Hairpin: very slow apex regardless of duration
            corner_type = "hairpin"
        elif apex_speed_mph > 80 and duration_seconds < 1.0:
            # Kink: very fast and very short
            corner_type = "kink"
        elif apex_speed_mph > 60 and duration_seconds > 2.0:
            # Sweeper: fast and long (but not slow enough for hairpin)
            corner_type = "sweeper"

    # Calculate severity rating (1-5) based on speed loss
    if speed_loss_pct < 10:
        severity = 1
    elif speed_loss_pct < 25:
        severity = 2
    elif speed_loss_pct < 40:
        severity = 3
    elif speed_loss_pct < 55:
        severity = 4
    else:
        severity = 5

    return corner_type, severity


def detect_corner_boundaries(
    lateral_g: np.ndarray,
    threshold: float = 0.3,
    time_data: Optional[np.ndarray] = None,
    min_duration: float = 0.5,
    sample_rate: float = 10.0
) -> List[Tuple[int, int]]:
    """
    Detect corner boundaries where lateral G exceeds threshold.

    Args:
        lateral_g: Lateral acceleration data in g (positive or negative for left/right)
        threshold: Minimum absolute lateral G to be considered a corner (default 0.3g)
        time_data: Optional time array in seconds (for precise duration filtering)
        min_duration: Minimum corner duration in seconds (default 0.5s)
        sample_rate: Sample rate in Hz if time_data not provided (default 10.0 Hz)

    Returns:
        List of (start_idx, end_idx) tuples marking corner boundaries

    Algorithm:
        1. Find all points where |lateral_g| exceeds threshold
        2. Group consecutive points into corner zones
        3. Filter out corners shorter than min_duration
        4. Return list of (start_idx, end_idx) for each valid corner

    Example:
        >>> lat_g = np.array([0.1, 0.5, 0.8, 0.7, 0.4, 0.1, -0.6, -0.5, -0.2])
        >>> boundaries = detect_corner_boundaries(lat_g, threshold=0.3)
        >>> # Returns [(1, 4), (6, 7)] - two corners detected
    """
    n = len(lateral_g)

    # Validate inputs
    if n == 0:
        return []

    # Create mask of points exceeding threshold (use absolute value)
    corner_mask = np.abs(lateral_g) > threshold

    # Find corner boundaries
    boundaries = []
    in_corner = False
    start_idx = 0

    for i in range(n):
        if corner_mask[i] and not in_corner:
            # Start of corner
            in_corner = True
            start_idx = i
        elif not corner_mask[i] and in_corner:
            # End of corner
            in_corner = False
            end_idx = i - 1

            # Check minimum duration
            if time_data is not None:
                duration = time_data[end_idx] - time_data[start_idx]
            else:
                # Estimate duration from sample count and rate
                duration = (end_idx - start_idx + 1) / sample_rate

            if duration >= min_duration:
                boundaries.append((start_idx, end_idx))

    # Handle corner at end of data
    if in_corner:
        end_idx = n - 1
        if time_data is not None:
            duration = time_data[end_idx] - time_data[start_idx]
        else:
            duration = (end_idx - start_idx + 1) / sample_rate

        if duration >= min_duration:
            boundaries.append((start_idx, end_idx))

    return boundaries


def calc_curvature(lat: np.ndarray, lon: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Calculate path curvature from GPS coordinates using the three-point circle method.

    Curvature is the inverse of the turning radius (κ = 1/R), measured in 1/meters.
    Positive values indicate the path is curving, zero indicates straight sections,
    and larger values indicate tighter turns.

    Args:
        lat: GPS latitude array (degrees)
        lon: GPS longitude array (degrees)
        window_size: Number of points to use for curvature calculation (must be odd, default=3)
                    Larger windows smooth the curvature but reduce time resolution.

    Returns:
        np.ndarray: Curvature values in 1/meters (inverse turning radius)
                   Returns 0.0 for straight sections, positive values for curves

    Algorithm:
        For each point, fits a circle through the point and its neighbors within
        the window. The circumradius of the triangle formed by these points
        approximates the instantaneous turning radius. Curvature = 1/radius.

    Edge cases:
        - Points too close together (< 0.1m): returns 0.0 (invalid)
        - Nearly collinear points (straight): returns 0.0
        - Array boundaries: uses available points (may have edge effects)
    """
    n = len(lat)

    # Validate inputs
    if n < 3:
        return np.zeros(n)

    if window_size < 3:
        window_size = 3

    # Ensure window_size is odd for symmetric neighborhood
    if window_size % 2 == 0:
        window_size += 1

    half_window = window_size // 2

    # Initialize curvature array
    curvature = np.zeros(n)

    # Convert GPS coordinates to local metric coordinates (meters)
    # Approximate: 1 degree latitude ≈ 111 km
    # Longitude varies by latitude: 1 degree lon ≈ 111 km * cos(lat)
    mean_lat = np.mean(lat)
    lat_m = lat * 111000.0  # meters
    lon_m = lon * 111000.0 * np.cos(np.radians(mean_lat))  # meters, corrected for latitude

    # Calculate curvature for each point using neighbors within window
    for i in range(n):
        # Determine window bounds
        start_idx = max(0, i - half_window)
        end_idx = min(n - 1, i + half_window)

        # Need at least 3 points for curvature calculation
        if end_idx - start_idx < 2:
            curvature[i] = 0.0
            continue

        # Use center point and boundary points of window
        idx_prev = start_idx
        idx_curr = i
        idx_next = end_idx

        # Get the three points
        x1, y1 = lon_m[idx_prev], lat_m[idx_prev]
        x2, y2 = lon_m[idx_curr], lat_m[idx_curr]
        x3, y3 = lon_m[idx_next], lat_m[idx_next]

        # Calculate side lengths of triangle
        a = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)  # distance from prev to curr
        b = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)  # distance from curr to next
        c = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)  # distance from prev to next

        # Check if points are too close (invalid GPS data)
        if a < 0.1 or b < 0.1:
            curvature[i] = 0.0
            continue

        # Calculate area of triangle using cross product
        # Area = 0.5 * |cross product|
        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

        # Check if points are nearly collinear (straight line)
        # Use adaptive threshold based on triangle size to handle tight turns
        # For small triangles, use smaller area threshold
        perimeter = a + b + c
        # Threshold: area should be at least 0.1% of what an equilateral triangle
        # with the same perimeter would have
        equilateral_area = (perimeter ** 2) / (36 * np.sqrt(3)) if perimeter > 0 else 0
        area_threshold = max(0.001, equilateral_area * 0.001)  # At least 0.001 m²

        if area < area_threshold:  # Nearly collinear (straight)
            curvature[i] = 0.0
        else:
            # Circumradius of triangle: R = (a * b * c) / (4 * Area)
            # This is the radius of the circle passing through all three points
            radius = (a * b * c) / (4.0 * area)

            # Curvature = 1 / radius (in 1/meters)
            # Cap at reasonable maximum (radius >= 1m means curvature <= 1.0)
            if radius >= 1.0:
                curvature[i] = 1.0 / radius
            else:
                # Extremely tight turn or GPS error - cap curvature
                curvature[i] = 1.0

    return curvature


@dataclass
class CornerZone:
    """A detected corner zone with entry, apex, and exit points."""
    name: str
    alias: Optional[str] = None  # e.g., "Carousel", "Kink"

    # Key indices in the data
    entry_idx: int = 0  # Turn-in point
    apex_idx: int = 0   # Minimum speed/radius point
    exit_idx: int = 0   # Track-out point

    # Braking zone (precedes corner)
    brake_start_idx: int = 0  # Where significant braking begins
    brake_end_idx: int = 0    # End of braking (usually near entry)

    # GPS coordinates
    entry_lat: float = 0.0
    entry_lon: float = 0.0
    apex_lat: float = 0.0
    apex_lon: float = 0.0
    exit_lat: float = 0.0
    exit_lon: float = 0.0

    # Characteristic values
    min_radius: float = 0.0      # Minimum GPS radius in the corner
    apex_speed_mph: float = 0.0  # Speed at apex
    direction: str = "left"      # "left" or "right" based on lateral acc sign
    corner_type: str = "normal"  # "hairpin", "sweeper", "chicane", "kink", "normal"
    severity: int = 3            # 1-5 rating based on speed loss percentage

    # Phase 6: Metrics storage (populated by CornerAnalyzer)
    _metrics: Optional[Dict] = field(default=None, repr=False)

    @property
    def metrics(self) -> Dict:
        """
        Access corner metrics as a dictionary.

        Returns dict with keys: entry_speed, apex_speed, exit_speed,
        max_lateral_g, max_braking_g, max_accel_g, brake_point_distance,
        throttle_on_distance, time_in_corner.

        Note: Metrics are only available after corner analysis.
        If not analyzed, returns basic metrics from detection.
        """
        if self._metrics is not None:
            return self._metrics

        # Return basic metrics from detection if full analysis not available
        return {
            "entry_speed": 0.0,
            "apex_speed": self.apex_speed_mph,
            "exit_speed": 0.0,
            "max_lateral_g": 0.0,
            "max_braking_g": 0.0,
            "max_accel_g": 0.0,
            "brake_point_distance": 0.0,
            "throttle_on_distance": 0.0,
            "time_in_corner": 0.0
        }

    def set_metrics(
        self,
        entry_speed: float,
        apex_speed: float,
        exit_speed: float,
        max_lateral_g: float,
        max_braking_g: float,
        max_accel_g: float,
        brake_point_distance: float,
        throttle_on_distance: float,
        time_in_corner: float
    ):
        """Set metrics for this corner (called by CornerAnalyzer)."""
        self._metrics = {
            "entry_speed": entry_speed,
            "apex_speed": apex_speed,
            "exit_speed": exit_speed,
            "max_lateral_g": max_lateral_g,
            "max_braking_g": max_braking_g,
            "max_accel_g": max_accel_g,
            "brake_point_distance": brake_point_distance,
            "throttle_on_distance": throttle_on_distance,
            "time_in_corner": time_in_corner
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "alias": self.alias,
            "entry_idx": self.entry_idx,
            "apex_idx": self.apex_idx,
            "exit_idx": self.exit_idx,
            "brake_start_idx": self.brake_start_idx,
            "brake_end_idx": self.brake_end_idx,
            "entry": {"lat": self.entry_lat, "lon": self.entry_lon},
            "apex": {"lat": self.apex_lat, "lon": self.apex_lon},
            "exit": {"lat": self.exit_lat, "lon": self.exit_lon},
            "min_radius": round(self.min_radius, 1),
            "apex_speed_mph": round(self.apex_speed_mph, 1),
            "direction": self.direction,
            "corner_type": self.corner_type,
            "severity": self.severity
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CornerZone":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            alias=data.get("alias"),
            entry_idx=data.get("entry_idx", 0),
            apex_idx=data.get("apex_idx", 0),
            exit_idx=data.get("exit_idx", 0),
            brake_start_idx=data.get("brake_start_idx", 0),
            brake_end_idx=data.get("brake_end_idx", 0),
            entry_lat=data.get("entry", {}).get("lat", 0.0),
            entry_lon=data.get("entry", {}).get("lon", 0.0),
            apex_lat=data.get("apex", {}).get("lat", 0.0),
            apex_lon=data.get("apex", {}).get("lon", 0.0),
            exit_lat=data.get("exit", {}).get("lat", 0.0),
            exit_lon=data.get("exit", {}).get("lon", 0.0),
            min_radius=data.get("min_radius", 0.0),
            apex_speed_mph=data.get("apex_speed_mph", 0.0),
            direction=data.get("direction", "left"),
            corner_type=data.get("corner_type", "normal"),
            severity=data.get("severity", 3)
        )


@dataclass
class CornerDetectionResult:
    """Result of corner detection for a lap."""
    lap_number: int
    corners: List[CornerZone]
    detection_params: Dict
    confidence: float  # 0-1 confidence in detection quality

    def to_dict(self) -> dict:
        return {
            "lap_number": self.lap_number,
            "corners": [c.to_dict() for c in self.corners],
            "detection_params": self.detection_params,
            "confidence": round(self.confidence, 2),
            "corner_count": len(self.corners)
        }


class CornerDetector:
    """
    Detects corners from telemetry data.

    Uses multiple signals:
    - GPS Radius: tight radius indicates corner
    - Speed: speed drop indicates braking zone and corner
    - Lateral acceleration: confirms cornering
    - Longitudinal acceleration: identifies braking zones
    """

    # Detection thresholds
    DEFAULT_RADIUS_THRESHOLD = 200.0    # Meters - below this is a corner
    DEFAULT_SPEED_DROP_THRESHOLD = 5.0  # MPH drop from local max
    DEFAULT_LAT_ACC_THRESHOLD = 0.3     # G - minimum lateral acc for corner
    DEFAULT_LON_ACC_BRAKE_THRESHOLD = -0.3  # G - indicates braking
    DEFAULT_MIN_CORNER_DURATION = 0.5   # Seconds
    DEFAULT_MIN_CORNER_GAP = 1.0        # Seconds between separate corners

    def __init__(
        self,
        radius_threshold: float = DEFAULT_RADIUS_THRESHOLD,
        speed_drop_threshold: float = DEFAULT_SPEED_DROP_THRESHOLD,
        lat_acc_threshold: float = DEFAULT_LAT_ACC_THRESHOLD,
        lon_acc_brake_threshold: float = DEFAULT_LON_ACC_BRAKE_THRESHOLD,
        min_corner_duration: float = DEFAULT_MIN_CORNER_DURATION,
        min_corner_gap: float = DEFAULT_MIN_CORNER_GAP
    ):
        self.radius_threshold = radius_threshold
        self.speed_drop_threshold = speed_drop_threshold
        self.lat_acc_threshold = lat_acc_threshold
        self.lon_acc_brake_threshold = lon_acc_brake_threshold
        self.min_corner_duration = min_corner_duration
        self.min_corner_gap = min_corner_gap

    def detect_from_arrays(
        self,
        time_data: np.ndarray,
        lat_data: np.ndarray,
        lon_data: np.ndarray,
        speed_data: np.ndarray,
        radius_data: np.ndarray = None,
        lat_acc_data: np.ndarray = None,
        lon_acc_data: np.ndarray = None,
        lap_number: int = 1
    ) -> CornerDetectionResult:
        """
        Detect corners from raw data arrays.

        Args:
            time_data: Time in seconds
            lat_data: GPS latitude
            lon_data: GPS longitude
            speed_data: Speed in mph
            radius_data: GPS radius in meters (optional, will compute if missing)
            lat_acc_data: Lateral acceleration in g (optional)
            lon_acc_data: Longitudinal acceleration in g (optional)
            lap_number: Lap number for labeling

        Returns:
            CornerDetectionResult with detected corners
        """
        n = len(time_data)
        if n < 10:
            return CornerDetectionResult(
                lap_number=lap_number,
                corners=[],
                detection_params=self._get_params(),
                confidence=0.0
            )

        # Compute radius from GPS if not provided
        if radius_data is None:
            radius_data = self._compute_radius(lat_data, lon_data, time_data)

        # Compute accelerations from speed if not provided
        if lon_acc_data is None:
            lon_acc_data = self._compute_lon_acc(speed_data, time_data)

        if lat_acc_data is None:
            lat_acc_data = self._compute_lat_acc(lat_data, lon_data, speed_data, time_data)

        # Find corner candidates based on radius
        corner_mask = self._find_corner_mask(radius_data, lat_acc_data, speed_data)

        # Group consecutive points into corner zones
        corners = self._group_corners(
            corner_mask, time_data, lat_data, lon_data,
            speed_data, radius_data, lat_acc_data, lon_acc_data
        )

        # Find braking zones for each corner
        self._find_braking_zones(corners, time_data, lon_acc_data, speed_data)

        # Calculate confidence based on signal quality
        confidence = self._calculate_confidence(corners, radius_data, lat_acc_data)

        return CornerDetectionResult(
            lap_number=lap_number,
            corners=corners,
            detection_params=self._get_params(),
            confidence=confidence
        )

    def detect_from_parquet(
        self,
        parquet_path: str,
        lap_number: int = 1
    ) -> CornerDetectionResult:
        """
        Detect corners from a Parquet file.

        Args:
            parquet_path: Path to parquet file
            lap_number: Lap number for labeling

        Returns:
            CornerDetectionResult
        """
        df = pd.read_parquet(parquet_path)

        time_data = df.index.values
        lat_data = find_column(df, ['GPS Latitude', 'gps_lat', 'latitude'])
        lon_data = find_column(df, ['GPS Longitude', 'gps_lon', 'longitude'])
        speed_data = find_column(df, ['GPS Speed', 'gps_speed', 'speed'])
        radius_data = find_column(df, ['GPS Radius', 'radius'])
        lat_acc_data = find_column(df, ['GPS LatAcc', 'lat_acc', 'lateral_acc'])
        lon_acc_data = find_column(df, ['GPS LonAcc', 'lon_acc', 'longitudinal_acc'])

        if lat_data is None or lon_data is None:
            raise ValueError("Parquet file missing GPS latitude/longitude columns")

        if speed_data is None:
            speed_data = np.zeros(len(time_data))
        elif speed_data.max() < 100:
            speed_data = speed_data * SPEED_MS_TO_MPH  # Convert m/s to mph

        return self.detect_from_arrays(
            time_data, lat_data, lon_data, speed_data,
            radius_data, lat_acc_data, lon_acc_data,
            lap_number
        )

    def _compute_radius(
        self,
        lat_data: np.ndarray,
        lon_data: np.ndarray,
        time_data: np.ndarray
    ) -> np.ndarray:
        """
        Compute instantaneous radius from GPS path.
        Uses three-point circle method.
        """
        n = len(lat_data)
        radius = np.full(n, 10000.0)  # Default to large radius (straight)

        # Convert to local coordinates (meters)
        lat_m = lat_data * 111000  # ~111km per degree
        lon_m = lon_data * 111000 * np.cos(np.radians(np.mean(lat_data)))

        for i in range(1, n - 1):
            # Three points: i-1, i, i+1
            x1, y1 = lon_m[i-1], lat_m[i-1]
            x2, y2 = lon_m[i], lat_m[i]
            x3, y3 = lon_m[i+1], lat_m[i+1]

            # Calculate circumradius of triangle
            a = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            b = np.sqrt((x3-x2)**2 + (y3-y2)**2)
            c = np.sqrt((x3-x1)**2 + (y3-y1)**2)

            if a < 0.1 or b < 0.1:  # Points too close
                continue

            # Area using cross product
            area = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))

            if area < 0.01:  # Nearly collinear (straight)
                radius[i] = 10000.0
            else:
                # Circumradius = abc / (4 * area)
                r = (a * b * c) / (4 * area)
                radius[i] = min(r, 10000.0)

        # Smooth the radius data
        window = 5
        radius_smooth = np.convolve(radius, np.ones(window)/window, mode='same')

        return radius_smooth

    def _compute_lon_acc(
        self,
        speed_data: np.ndarray,
        time_data: np.ndarray
    ) -> np.ndarray:
        """Compute longitudinal acceleration from speed."""
        n = len(speed_data)
        lon_acc = np.zeros(n)

        # Convert speed from mph to m/s
        speed_ms = speed_data * 0.44704

        for i in range(1, n):
            dt = time_data[i] - time_data[i-1]
            if dt > 0:
                dv = speed_ms[i] - speed_ms[i-1]
                lon_acc[i] = dv / dt / 9.81  # Convert to g

        return lon_acc

    def _compute_lat_acc(
        self,
        lat_data: np.ndarray,
        lon_data: np.ndarray,
        speed_data: np.ndarray,
        time_data: np.ndarray
    ) -> np.ndarray:
        """
        Compute lateral acceleration from path curvature.
        lat_acc = v^2 / r
        """
        radius = self._compute_radius(lat_data, lon_data, time_data)
        speed_ms = speed_data * 0.44704

        # Lateral acceleration = v^2 / r (in g)
        lat_acc = (speed_ms ** 2) / radius / 9.81

        # Limit to reasonable values
        lat_acc = np.clip(lat_acc, -3.0, 3.0)

        return lat_acc

    def _find_corner_mask(
        self,
        radius_data: np.ndarray,
        lat_acc_data: np.ndarray,
        speed_data: np.ndarray
    ) -> np.ndarray:
        """
        Create a mask of points that are in corners.

        Uses combination of:
        - Tight radius (primary signal)
        - Significant lateral acceleration (confirmation)
        - Speed below straightaway speeds
        """
        # Primary: radius below threshold
        radius_mask = radius_data < self.radius_threshold

        # Confirmation: lateral acceleration above threshold
        lat_acc_mask = np.abs(lat_acc_data) > self.lat_acc_threshold

        # Speed sanity check: not at max straightaway speed
        max_speed = np.percentile(speed_data[~np.isnan(speed_data)], 95) if len(speed_data) > 0 else 150
        speed_mask = speed_data < (max_speed - 5)

        # Combine: require radius AND (lat_acc OR reduced speed)
        corner_mask = radius_mask & (lat_acc_mask | speed_mask)

        return corner_mask

    def _group_corners(
        self,
        corner_mask: np.ndarray,
        time_data: np.ndarray,
        lat_data: np.ndarray,
        lon_data: np.ndarray,
        speed_data: np.ndarray,
        radius_data: np.ndarray,
        lat_acc_data: np.ndarray,
        lon_acc_data: np.ndarray
    ) -> List[CornerZone]:
        """Group consecutive corner points into CornerZone objects."""
        corners = []
        n = len(corner_mask)

        # Find start and end of each corner zone
        in_corner = False
        start_idx = 0
        corner_num = 1

        for i in range(n):
            if corner_mask[i] and not in_corner:
                # Start of new corner
                in_corner = True
                start_idx = i
            elif not corner_mask[i] and in_corner:
                # End of corner
                in_corner = False
                end_idx = i - 1

                # Check minimum duration
                duration = time_data[end_idx] - time_data[start_idx]
                if duration >= self.min_corner_duration:
                    # Check gap from previous corner
                    if corners and (time_data[start_idx] - time_data[corners[-1].exit_idx]) < self.min_corner_gap:
                        # Merge with previous corner
                        corners[-1].exit_idx = end_idx
                        corners[-1].exit_lat = lat_data[end_idx]
                        corners[-1].exit_lon = lon_data[end_idx]
                    else:
                        # Create new corner
                        corner = self._create_corner_zone(
                            corner_num, start_idx, end_idx,
                            time_data, lat_data, lon_data,
                            speed_data, radius_data, lat_acc_data
                        )
                        corners.append(corner)
                        corner_num += 1

        # Handle corner at end of data
        if in_corner:
            end_idx = n - 1
            duration = time_data[end_idx] - time_data[start_idx]
            if duration >= self.min_corner_duration:
                corner = self._create_corner_zone(
                    corner_num, start_idx, end_idx,
                    time_data, lat_data, lon_data,
                    speed_data, radius_data, lat_acc_data
                )
                corners.append(corner)

        return corners

    def _create_corner_zone(
        self,
        corner_num: int,
        start_idx: int,
        end_idx: int,
        time_data: np.ndarray,
        lat_data: np.ndarray,
        lon_data: np.ndarray,
        speed_data: np.ndarray,
        radius_data: np.ndarray,
        lat_acc_data: np.ndarray
    ) -> CornerZone:
        """Create a CornerZone from indices."""
        # Phase 4: Find apex as point of MAXIMUM lateral G within corner
        corner_lat_acc = np.abs(lat_acc_data[start_idx:end_idx+1])
        apex_local_idx = np.argmax(corner_lat_acc)
        apex_idx = start_idx + apex_local_idx

        # Determine direction from lateral acceleration at apex
        lat_acc_at_apex = lat_acc_data[apex_idx] if apex_idx < len(lat_acc_data) else 0
        direction = "right" if lat_acc_at_apex > 0 else "left"

        # Get speeds and calculate duration
        apex_speed = speed_data[apex_idx]
        min_radius = np.min(radius_data[start_idx:end_idx+1])
        duration = float(time_data[end_idx] - time_data[start_idx])

        # Find maximum speed before corner (lookback up to 30 samples / 3 seconds)
        lookback_start = max(0, start_idx - 30)
        pre_corner_speed = np.max(speed_data[lookback_start:start_idx+1]) if lookback_start < start_idx else speed_data[start_idx]
        entry_speed = max(pre_corner_speed, speed_data[start_idx])

        # Phase 5: Classify corner type and calculate severity
        corner_type, severity = classify_corner(
            apex_speed_mph=float(apex_speed),
            entry_speed_mph=float(entry_speed),
            duration_seconds=duration,
            min_radius=float(min_radius),
            lat_acc_data=lat_acc_data,
            entry_idx=start_idx,
            exit_idx=end_idx
        )

        return CornerZone(
            name=f"T{corner_num}",
            entry_idx=start_idx,
            apex_idx=apex_idx,
            exit_idx=end_idx,
            entry_lat=float(lat_data[start_idx]),
            entry_lon=float(lon_data[start_idx]),
            apex_lat=float(lat_data[apex_idx]),
            apex_lon=float(lon_data[apex_idx]),
            exit_lat=float(lat_data[end_idx]),
            exit_lon=float(lon_data[end_idx]),
            min_radius=float(min_radius),
            apex_speed_mph=float(apex_speed),
            direction=direction,
            corner_type=corner_type,
            severity=severity
        )

    def _find_braking_zones(
        self,
        corners: List[CornerZone],
        time_data: np.ndarray,
        lon_acc_data: np.ndarray,
        speed_data: np.ndarray
    ) -> None:
        """Find braking zone for each corner (modifies corners in place)."""
        for corner in corners:
            # Look backwards from entry to find where braking started
            brake_end_idx = corner.entry_idx
            brake_start_idx = brake_end_idx

            # Search backwards for start of braking
            for i in range(corner.entry_idx - 1, max(0, corner.entry_idx - 100), -1):
                if lon_acc_data[i] < self.lon_acc_brake_threshold:
                    brake_start_idx = i
                elif brake_start_idx < brake_end_idx:
                    # Found the start of braking zone
                    break

            corner.brake_start_idx = brake_start_idx
            corner.brake_end_idx = brake_end_idx

    def _calculate_confidence(
        self,
        corners: List[CornerZone],
        radius_data: np.ndarray,
        lat_acc_data: np.ndarray
    ) -> float:
        """Calculate confidence in detection quality."""
        if not corners:
            return 0.0

        # Check for valid data
        valid_radius = np.sum(~np.isnan(radius_data) & (radius_data < 10000)) / len(radius_data)
        valid_lat_acc = np.sum(~np.isnan(lat_acc_data) & (np.abs(lat_acc_data) > 0.05)) / len(lat_acc_data)

        # Check for reasonable corner count (most tracks have 8-17 corners)
        corner_count_score = 1.0 if 5 <= len(corners) <= 20 else 0.5

        confidence = (valid_radius * 0.4 + valid_lat_acc * 0.4 + corner_count_score * 0.2)
        return min(1.0, confidence)

    def _get_params(self) -> Dict:
        """Return current detection parameters."""
        return {
            "radius_threshold": self.radius_threshold,
            "speed_drop_threshold": self.speed_drop_threshold,
            "lat_acc_threshold": self.lat_acc_threshold,
            "lon_acc_brake_threshold": self.lon_acc_brake_threshold,
            "min_corner_duration": self.min_corner_duration,
            "min_corner_gap": self.min_corner_gap
        }


@dataclass
class Corner:
    """Simple corner representation for Phase 3 detection."""
    start_idx: int
    end_idx: int
    direction: str  # "left" or "right"
    apex_idx: int = 0  # Point of maximum lateral G (Phase 4)
    min_speed_idx: int = 0  # Point of minimum speed (secondary apex, Phase 4)


def detect_corners(
    df: pd.DataFrame,
    curvature_threshold: float = 0.005,
    lateral_g_threshold: float = 0.3,
    min_corner_duration: float = 0.5,
    merge_gap: float = 0.5,
    sample_rate: float = 10.0
) -> List[Corner]:
    """
    Detect corners using both GPS curvature and lateral G.

    Phase 3 implementation: Combines curvature analysis with lateral G confirmation
    for robust corner boundary detection.

    Phase 4 enhancement: Apex detection using maximum lateral G with secondary
    minimum speed point for comparison.

    Args:
        df: DataFrame with GPS Latitude, GPS Longitude, and GPS LatAcc columns
            Optional: GPS Speed for secondary apex detection
        curvature_threshold: Minimum curvature (1/m) to consider a corner (default 0.005)
        lateral_g_threshold: Minimum absolute lateral G to confirm corner (default 0.3g)
        min_corner_duration: Minimum corner duration in seconds (default 0.5s)
        merge_gap: Time gap threshold for merging nearby corners (default 0.5s)
        sample_rate: Sample rate in Hz if time data not in index (default 10.0)

    Returns:
        List of Corner objects with start_idx, end_idx, direction, apex_idx, and min_speed_idx

    Algorithm:
        1. Calculate GPS curvature from lat/lon coordinates
        2. Create curvature mask (curvature > threshold)
        3. Create lateral G mask (|lateral_g| > threshold)
        4. Corner starts when BOTH curvature AND lateral G exceed thresholds
        5. Corner ends when lateral G drops below threshold
        6. Filter corners by minimum duration
        7. Merge nearby corners separated by < merge_gap
        8. Determine direction from sign of lateral G
        9. Find apex as point of maximum absolute lateral G (Phase 4)
        10. Find secondary apex as point of minimum speed (Phase 4)

    Example:
        >>> df = pd.read_parquet('session.parquet')
        >>> corners = detect_corners(df)
        >>> print(f"Detected {len(corners)} corners")
        >>> for corner in corners:
        ...     print(f"Corner at {corner.start_idx}-{corner.end_idx}, "
        ...           f"apex: {corner.apex_idx}, direction: {corner.direction}")
    """
    # Extract required columns
    try:
        lat = df['GPS Latitude'].values if 'GPS Latitude' in df.columns else df['gps_lat'].values
        lon = df['GPS Longitude'].values if 'GPS Longitude' in df.columns else df['gps_lon'].values
        lateral_g = df['GPS LatAcc'].values if 'GPS LatAcc' in df.columns else df['lat_acc'].values
    except KeyError as e:
        raise ValueError(f"Missing required column: {e}")

    # Get speed data if available (for secondary apex detection)
    try:
        speed = df['GPS Speed'].values if 'GPS Speed' in df.columns else df['gps_speed'].values
        if speed.max() < 100:  # Likely in m/s, convert to mph
            speed = speed * SPEED_MS_TO_MPH
    except (KeyError, AttributeError):
        speed = None

    # Get time data from index or create synthetic time array
    if hasattr(df.index, 'values'):
        time_data = df.index.values
    else:
        time_data = np.arange(len(df)) / sample_rate

    n = len(lat)

    # Step 1: Calculate curvature from GPS coordinates
    curvature = calc_curvature(lat, lon, window_size=3)

    # Step 2: Create masks for corner detection
    curvature_mask = curvature > curvature_threshold
    lateral_g_mask = np.abs(lateral_g) > lateral_g_threshold

    # Step 3: Corners must have BOTH high curvature AND high lateral G
    # Entry requires both, but we use lateral G for exit detection
    corner_active_mask = lateral_g_mask  # Corner is active when lateral G exceeds threshold
    corner_entry_mask = curvature_mask & lateral_g_mask  # Entry requires both conditions

    # Step 4: Find corner boundaries
    corners = []
    in_corner = False
    start_idx = 0
    corner_lateral_g_values = []

    for i in range(n):
        if corner_entry_mask[i] and not in_corner:
            # Corner entry: both curvature AND lateral G exceed threshold
            in_corner = True
            start_idx = i
            corner_lateral_g_values = [lateral_g[i]]
        elif in_corner:
            # Track lateral G values within corner
            corner_lateral_g_values.append(lateral_g[i])

            if not corner_active_mask[i]:
                # Corner exit: lateral G dropped below threshold
                in_corner = False
                end_idx = i - 1

                # Check minimum duration
                if time_data is not None and isinstance(time_data[0], (int, float)):
                    duration = time_data[end_idx] - time_data[start_idx]
                else:
                    duration = (end_idx - start_idx + 1) / sample_rate

                if duration >= min_corner_duration:
                    # Determine direction from average lateral G
                    avg_lat_g = np.mean(corner_lateral_g_values)
                    direction = "right" if avg_lat_g > 0 else "left"

                    # Phase 4: Find apex as point of maximum lateral G
                    corner_lat_g_abs = np.abs(lateral_g[start_idx:end_idx+1])
                    apex_local_idx = np.argmax(corner_lat_g_abs)
                    apex_idx = start_idx + apex_local_idx

                    # Secondary apex: minimum speed point (if speed data available)
                    min_speed_idx = apex_idx  # Default to same as apex
                    if speed is not None:
                        corner_speeds = speed[start_idx:end_idx+1]
                        min_speed_local_idx = np.argmin(corner_speeds)
                        min_speed_idx = start_idx + min_speed_local_idx

                    corners.append(Corner(
                        start_idx=start_idx,
                        end_idx=end_idx,
                        direction=direction,
                        apex_idx=apex_idx,
                        min_speed_idx=min_speed_idx
                    ))

                corner_lateral_g_values = []

    # Handle corner at end of data
    if in_corner:
        end_idx = n - 1
        if time_data is not None and isinstance(time_data[0], (int, float)):
            duration = time_data[end_idx] - time_data[start_idx]
        else:
            duration = (end_idx - start_idx + 1) / sample_rate

        if duration >= min_corner_duration:
            avg_lat_g = np.mean(corner_lateral_g_values)
            direction = "right" if avg_lat_g > 0 else "left"

            # Phase 4: Find apex as point of maximum lateral G
            corner_lat_g_abs = np.abs(lateral_g[start_idx:end_idx+1])
            apex_local_idx = np.argmax(corner_lat_g_abs)
            apex_idx = start_idx + apex_local_idx

            # Secondary apex: minimum speed point (if speed data available)
            min_speed_idx = apex_idx  # Default to same as apex
            if speed is not None:
                corner_speeds = speed[start_idx:end_idx+1]
                min_speed_local_idx = np.argmin(corner_speeds)
                min_speed_idx = start_idx + min_speed_local_idx

            corners.append(Corner(
                start_idx=start_idx,
                end_idx=end_idx,
                direction=direction,
                apex_idx=apex_idx,
                min_speed_idx=min_speed_idx
            ))

    # Step 5: Merge nearby corners (chicanes)
    if len(corners) > 1:
        merged_corners = []
        i = 0
        while i < len(corners):
            current = corners[i]

            # Look ahead to see if next corner is close
            if i + 1 < len(corners):
                next_corner = corners[i + 1]
                gap = time_data[next_corner.start_idx] - time_data[current.end_idx]

                if gap < merge_gap:
                    # Merge corners: extend end to include next corner
                    # Direction determined by which side had more lateral G
                    combined_lat_g = np.concatenate([
                        lateral_g[current.start_idx:current.end_idx+1],
                        lateral_g[next_corner.start_idx:next_corner.end_idx+1]
                    ])
                    avg_lat_g = np.mean(combined_lat_g)
                    direction = "right" if avg_lat_g > 0 else "left"

                    # Recalculate apex for merged corner
                    merged_start = current.start_idx
                    merged_end = next_corner.end_idx
                    merged_lat_g_abs = np.abs(lateral_g[merged_start:merged_end+1])
                    apex_local_idx = np.argmax(merged_lat_g_abs)
                    apex_idx = merged_start + apex_local_idx

                    # Recalculate min speed for merged corner
                    min_speed_idx = apex_idx  # Default
                    if speed is not None:
                        merged_speeds = speed[merged_start:merged_end+1]
                        min_speed_local_idx = np.argmin(merged_speeds)
                        min_speed_idx = merged_start + min_speed_local_idx

                    merged = Corner(
                        start_idx=merged_start,
                        end_idx=merged_end,
                        direction=direction,
                        apex_idx=apex_idx,
                        min_speed_idx=min_speed_idx
                    )
                    merged_corners.append(merged)
                    i += 2  # Skip next corner since we merged it
                else:
                    merged_corners.append(current)
                    i += 1
            else:
                merged_corners.append(current)
                i += 1

        corners = merged_corners

    return corners


def detect_corners_from_parquet(parquet_path: str, lap_number: int = 1) -> CornerDetectionResult:
    """
    Detect corners from a parquet file using CornerDetector.

    Args:
        parquet_path: Path to parquet file
        lap_number: Lap number for labeling

    Returns:
        CornerDetectionResult

    Note:
        This is the old API that uses CornerDetector class.
        For the new Phase 3 API using curvature + lateral G, use detect_corners(df).
    """
    detector = CornerDetector()
    return detector.detect_from_parquet(parquet_path, lap_number)


# Phase 7: Corner Numbering and Persistence

def calculate_gps_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two GPS points in meters using Haversine formula.

    Args:
        lat1, lon1: First point (degrees)
        lat2, lon2: Second point (degrees)

    Returns:
        Distance in meters
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    # Haversine formula
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    # Earth radius in meters
    radius_earth = 6371000

    return radius_earth * c


def load_track_corners(track_name: str, corners_file: str = "data/track_corners.json") -> Optional[List[Dict]]:
    """
    Load corner definitions for a specific track.

    Args:
        track_name: Name of the track
        corners_file: Path to track corners JSON file

    Returns:
        List of corner definitions, or None if track not found
    """
    corners_path = Path(corners_file)

    if not corners_path.exists():
        return None

    try:
        with open(corners_path, 'r') as f:
            data = json.load(f)

        # Normalize track name for lookup
        track_key = track_name.lower().replace(' ', '_')

        if track_key in data:
            return data[track_key]

        return None
    except (json.JSONDecodeError, IOError):
        return None


def save_track_corners(
    track_name: str,
    corners: List[CornerZone],
    corners_file: str = "data/track_corners.json"
) -> None:
    """
    Save corner definitions for a specific track.

    Args:
        track_name: Name of the track
        corners: List of CornerZone objects to save
        corners_file: Path to track corners JSON file
    """
    corners_path = Path(corners_file)

    # Load existing data or create new
    if corners_path.exists():
        try:
            with open(corners_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            data = {}
    else:
        # Create directory if it doesn't exist
        corners_path.parent.mkdir(parents=True, exist_ok=True)
        data = {}

    # Normalize track name
    track_key = track_name.lower().replace(' ', '_')

    # Convert corners to serializable format
    corners_data = []
    for corner in corners:
        corner_dict = {
            "name": corner.name,
            "alias": corner.alias,
            "apex_lat": corner.apex_lat,
            "apex_lon": corner.apex_lon,
            "entry_lat": corner.entry_lat,
            "entry_lon": corner.entry_lon,
            "exit_lat": corner.exit_lat,
            "exit_lon": corner.exit_lon,
            "corner_type": corner.corner_type,
            "direction": corner.direction
        }
        corners_data.append(corner_dict)

    # Update data
    data[track_key] = corners_data

    # Save to file
    with open(corners_path, 'w') as f:
        json.dump(data, f, indent=2)


def match_corner_to_definition(
    corner: CornerZone,
    definitions: List[Dict],
    proximity_threshold: float = 50.0
) -> Optional[Dict]:
    """
    Match a detected corner to a stored corner definition by GPS proximity.

    Args:
        corner: CornerZone to match
        definitions: List of stored corner definitions
        proximity_threshold: Maximum distance in meters for a match (default 50m)

    Returns:
        Matching definition dict, or None if no match found
    """
    best_match = None
    best_distance = float('inf')

    for definition in definitions:
        # Calculate distance between apex points
        distance = calculate_gps_distance(
            corner.apex_lat, corner.apex_lon,
            definition["apex_lat"], definition["apex_lon"]
        )

        if distance < best_distance and distance < proximity_threshold:
            best_distance = distance
            best_match = definition

    return best_match


def number_corners_by_track_position(
    corners: List[CornerZone],
    start_lat: Optional[float] = None,
    start_lon: Optional[float] = None
) -> List[CornerZone]:
    """
    Number corners T1, T2, T3... based on track position.

    Args:
        corners: List of CornerZone objects
        start_lat: Optional start/finish line latitude
        start_lon: Optional start/finish line longitude

    Returns:
        List of CornerZone objects with updated names
    """
    if not corners:
        return corners

    # If start position provided, find nearest corner and start numbering from there
    if start_lat is not None and start_lon is not None:
        distances = []
        for i, corner in enumerate(corners):
            dist = calculate_gps_distance(start_lat, start_lon, corner.entry_lat, corner.entry_lon)
            distances.append((i, dist))

        # Sort by distance to find first corner
        distances.sort(key=lambda x: x[1])
        first_corner_idx = distances[0][0]

        # Reorder corners starting from first corner
        reordered = corners[first_corner_idx:] + corners[:first_corner_idx]
    else:
        # Use corners as-is (already in order from detection)
        reordered = corners

    # Number corners T1, T2, T3...
    for i, corner in enumerate(reordered, start=1):
        corner.name = f"T{i}"

    return reordered


def apply_corner_definitions(
    corners: List[CornerZone],
    track_name: str,
    corners_file: str = "data/track_corners.json",
    proximity_threshold: float = 50.0,
    auto_save_new: bool = True
) -> List[CornerZone]:
    """
    Apply stored corner definitions to detected corners, or save new definitions.

    This is the main Phase 7 API: loads stored corner definitions for a track,
    matches detected corners to stored definitions by GPS proximity, applies
    stored names/aliases, or creates new corner definitions if track is unknown.

    Args:
        corners: List of detected CornerZone objects
        track_name: Name of the track
        corners_file: Path to track corners JSON file
        proximity_threshold: Maximum distance in meters for GPS matching
        auto_save_new: If True, automatically save corner definitions for new tracks

    Returns:
        List of CornerZone objects with names/aliases applied
    """
    # Load stored definitions
    definitions = load_track_corners(track_name, corners_file)

    if definitions is None:
        # New track - number corners and optionally save
        numbered_corners = number_corners_by_track_position(corners)

        if auto_save_new and numbered_corners:
            save_track_corners(track_name, numbered_corners, corners_file)

        return numbered_corners

    # Match corners to definitions
    matched_corners = []
    unmatched_corners = []

    for corner in corners:
        match = match_corner_to_definition(corner, definitions, proximity_threshold)

        if match:
            # Apply stored name and alias
            corner.name = match.get("name", corner.name)
            corner.alias = match.get("alias")
            matched_corners.append(corner)
        else:
            unmatched_corners.append(corner)

    # Number any unmatched corners
    if unmatched_corners:
        # Find highest T-number in both matched corners AND stored definitions
        max_t_num = 0
        for corner in matched_corners:
            if corner.name.startswith("T") and corner.name[1:].isdigit():
                num = int(corner.name[1:])
                max_t_num = max(max_t_num, num)

        # Also check stored definitions for max T-number
        for defn in definitions:
            name = defn.get("name", "")
            if name.startswith("T") and name[1:].isdigit():
                num = int(name[1:])
                max_t_num = max(max_t_num, num)

        # Number unmatched corners starting after max
        for i, corner in enumerate(unmatched_corners, start=max_t_num + 1):
            corner.name = f"T{i}"

    # Combine and sort by entry position (assuming they're already in order)
    all_corners = matched_corners + unmatched_corners

    # Update stored definitions if we have new corners
    if unmatched_corners and auto_save_new:
        save_track_corners(track_name, all_corners, corners_file)

    return all_corners


def rename_corner(
    track_name: str,
    corner_name: str,
    new_name: Optional[str] = None,
    new_alias: Optional[str] = None,
    corners_file: str = "data/track_corners.json"
) -> bool:
    """
    Rename a corner or set its alias in the stored track definition.

    Args:
        track_name: Name of the track
        corner_name: Current name of the corner (e.g., "T1")
        new_name: New name for the corner (optional)
        new_alias: Alias for the corner (optional, e.g., "Carousel")
        corners_file: Path to track corners JSON file

    Returns:
        True if successful, False otherwise
    """
    corners_path = Path(corners_file)

    if not corners_path.exists():
        return False

    try:
        with open(corners_path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return False

    # Normalize track name
    track_key = track_name.lower().replace(' ', '_')

    if track_key not in data:
        return False

    # Find and update corner
    corners = data[track_key]
    found = False

    for corner in corners:
        if corner["name"] == corner_name:
            if new_name is not None:
                corner["name"] = new_name
            if new_alias is not None:
                corner["alias"] = new_alias
            found = True
            break

    if not found:
        return False

    # Save updated data
    try:
        with open(corners_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except IOError:
        return False


# Module-level functions for different use cases

def detect_corners_simple(parquet_path: str, lap_number: int = 1) -> CornerDetectionResult:
    """
    Convenience wrapper for CornerDetector.detect_from_parquet().
    Maintains backward compatibility with existing code.

    For the new Phase 3 API that returns Corner objects, use:
        df = pd.read_parquet(path)
        corners = detect_corners(df)
    """
    return detect_corners_from_parquet(parquet_path, lap_number)
