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
    corner_type: str = "normal"  # "normal", "hairpin", "kink", "chicane"

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
            "corner_type": self.corner_type
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
            corner_type=data.get("corner_type", "normal")
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
        lat_data = self._find_column(df, ['GPS Latitude', 'gps_lat', 'latitude'])
        lon_data = self._find_column(df, ['GPS Longitude', 'gps_lon', 'longitude'])
        speed_data = self._find_column(df, ['GPS Speed', 'gps_speed', 'speed'])
        radius_data = self._find_column(df, ['GPS Radius', 'radius'])
        lat_acc_data = self._find_column(df, ['GPS LatAcc', 'lat_acc', 'lateral_acc'])
        lon_acc_data = self._find_column(df, ['GPS LonAcc', 'lon_acc', 'longitudinal_acc'])

        if lat_data is None or lon_data is None:
            raise ValueError("Parquet file missing GPS latitude/longitude columns")

        if speed_data is None:
            speed_data = np.zeros(len(time_data))
        elif speed_data.max() < 100:
            speed_data = speed_data * 2.237  # Convert m/s to mph

        return self.detect_from_arrays(
            time_data, lat_data, lon_data, speed_data,
            radius_data, lat_acc_data, lon_acc_data,
            lap_number
        )

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[np.ndarray]:
        """Find a column by trying multiple names."""
        for col in candidates:
            if col in df.columns:
                return df[col].values
            for actual_col in df.columns:
                if actual_col.lower() == col.lower():
                    return df[actual_col].values
        return None

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
        # Find apex (minimum speed point within corner)
        corner_speeds = speed_data[start_idx:end_idx+1]
        apex_local_idx = np.argmin(corner_speeds)
        apex_idx = start_idx + apex_local_idx

        # Determine direction from lateral acceleration at apex
        lat_acc_at_apex = lat_acc_data[apex_idx] if apex_idx < len(lat_acc_data) else 0
        direction = "right" if lat_acc_at_apex > 0 else "left"

        # Classify corner type based on apex speed and radius
        apex_speed = speed_data[apex_idx]
        min_radius = np.min(radius_data[start_idx:end_idx+1])

        if apex_speed < 40 and min_radius < 30:
            corner_type = "hairpin"
        elif apex_speed > 100 and min_radius > 150:
            corner_type = "kink"
        else:
            corner_type = "normal"

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
            corner_type=corner_type
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


def detect_corners(parquet_path: str, lap_number: int = 1) -> CornerDetectionResult:
    """
    Convenience function to detect corners from a parquet file.

    Args:
        parquet_path: Path to parquet file
        lap_number: Lap number for labeling

    Returns:
        CornerDetectionResult
    """
    detector = CornerDetector()
    return detector.detect_from_parquet(parquet_path, lap_number)
