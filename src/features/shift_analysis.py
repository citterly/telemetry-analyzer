"""
Shift Point Analysis Feature
Detects shift points, calculates shift RPM by gear, identifies early/late shifts.

Uses gear_calculator.find_shift_points() as the foundation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
import json

from ..analysis.gear_calculator import GearCalculator, GearInfo
from ..config.vehicle_config import CURRENT_SETUP, TRANSMISSION_SCENARIOS


@dataclass
class ShiftEvent:
    """Individual gear shift event"""
    time: float
    from_gear: int
    to_gear: int
    shift_type: str  # 'upshift' or 'downshift'
    rpm_at_shift: float
    speed_mph: float
    rpm_delta: float = 0.0  # RPM change after shift
    shift_quality: str = "normal"  # 'early', 'optimal', 'late', 'over-rev'


@dataclass
class GearShiftStats:
    """Statistics for shifts involving a specific gear"""
    gear: int
    upshift_count: int = 0
    downshift_count: int = 0
    avg_upshift_rpm: float = 0.0
    min_upshift_rpm: float = 0.0
    max_upshift_rpm: float = 0.0
    avg_downshift_rpm: float = 0.0
    optimal_upshift_rpm: float = 6500.0  # Default target
    upshift_rpms: List[float] = field(default_factory=list)
    downshift_rpms: List[float] = field(default_factory=list)


@dataclass
class ShiftReport:
    """Complete shift analysis report"""
    session_id: str
    analysis_timestamp: str
    total_shifts: int
    total_upshifts: int
    total_downshifts: int
    shifts: List[ShiftEvent]
    gear_stats: Dict[int, GearShiftStats]
    early_shifts: int
    late_shifts: int
    over_rev_shifts: int
    recommendations: List[str]
    summary: Dict

    def to_dict(self) -> dict:
        """Convert report to dictionary for JSON serialization"""
        return {
            "session_id": self.session_id,
            "analysis_timestamp": self.analysis_timestamp,
            "total_shifts": self.total_shifts,
            "total_upshifts": self.total_upshifts,
            "total_downshifts": self.total_downshifts,
            "shifts": [
                {
                    "time": s.time,
                    "from_gear": s.from_gear,
                    "to_gear": s.to_gear,
                    "shift_type": s.shift_type,
                    "rpm_at_shift": s.rpm_at_shift,
                    "speed_mph": s.speed_mph,
                    "rpm_delta": s.rpm_delta,
                    "shift_quality": s.shift_quality
                }
                for s in self.shifts
            ],
            "gear_stats": {
                gear: {
                    "gear": stats.gear,
                    "upshift_count": stats.upshift_count,
                    "downshift_count": stats.downshift_count,
                    "avg_upshift_rpm": round(stats.avg_upshift_rpm, 0),
                    "min_upshift_rpm": round(stats.min_upshift_rpm, 0),
                    "max_upshift_rpm": round(stats.max_upshift_rpm, 0),
                    "avg_downshift_rpm": round(stats.avg_downshift_rpm, 0),
                    "optimal_upshift_rpm": stats.optimal_upshift_rpm
                }
                for gear, stats in self.gear_stats.items()
            },
            "early_shifts": self.early_shifts,
            "late_shifts": self.late_shifts,
            "over_rev_shifts": self.over_rev_shifts,
            "recommendations": self.recommendations,
            "summary": self.summary
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)


class ShiftAnalyzer:
    """
    Analyzes gear shift patterns in telemetry data.

    Detects shift points, calculates statistics, and identifies
    opportunities for improvement.
    """

    # RPM thresholds for shift quality assessment
    OPTIMAL_SHIFT_RPM_MIN = 6000
    OPTIMAL_SHIFT_RPM_MAX = 6800
    EARLY_SHIFT_RPM = 5500
    LATE_SHIFT_RPM = 7000
    OVER_REV_RPM = 7200

    def __init__(
        self,
        transmission_ratios: Optional[List[float]] = None,
        final_drive: Optional[float] = None
    ):
        """
        Initialize the shift analyzer.

        Args:
            transmission_ratios: Gear ratios (default from CURRENT_SETUP)
            final_drive: Final drive ratio (default from CURRENT_SETUP)
        """
        if transmission_ratios is None:
            transmission_ratios = CURRENT_SETUP['transmission_ratios']
        if final_drive is None:
            final_drive = CURRENT_SETUP['final_drive']

        self.gear_calculator = GearCalculator(transmission_ratios, final_drive)
        self.transmission_ratios = transmission_ratios
        self.final_drive = final_drive

    def analyze_session(
        self,
        rpm_data: np.ndarray,
        speed_data: np.ndarray,
        time_data: np.ndarray,
        session_id: str = "unknown"
    ) -> ShiftReport:
        """
        Analyze shift patterns for a complete session.

        Args:
            rpm_data: Array of RPM values
            speed_data: Array of speed values (mph)
            time_data: Array of time values (seconds)
            session_id: Identifier for the session

        Returns:
            ShiftReport with complete analysis
        """
        # Calculate gear trace
        gear_trace = self.gear_calculator.calculate_gear_trace(rpm_data, speed_data)

        # Find shift points
        raw_shifts = self.gear_calculator.find_shift_points(gear_trace, time_data)

        # Convert to ShiftEvent objects with quality assessment
        shifts = self._process_shifts(raw_shifts, gear_trace, time_data)

        # Calculate gear statistics
        gear_stats = self._calculate_gear_stats(shifts)

        # Count shift quality categories
        early_shifts = sum(1 for s in shifts if s.shift_quality == "early")
        late_shifts = sum(1 for s in shifts if s.shift_quality == "late")
        over_rev_shifts = sum(1 for s in shifts if s.shift_quality == "over-rev")

        # Generate recommendations
        recommendations = self._generate_recommendations(shifts, gear_stats)

        # Build summary
        upshifts = [s for s in shifts if s.shift_type == "upshift"]
        downshifts = [s for s in shifts if s.shift_type == "downshift"]

        summary = {
            "avg_upshift_rpm": np.mean([s.rpm_at_shift for s in upshifts]) if upshifts else 0,
            "avg_downshift_rpm": np.mean([s.rpm_at_shift for s in downshifts]) if downshifts else 0,
            "shift_quality_breakdown": {
                "optimal": sum(1 for s in shifts if s.shift_quality == "optimal"),
                "early": early_shifts,
                "late": late_shifts,
                "over-rev": over_rev_shifts,
                "normal": sum(1 for s in shifts if s.shift_quality == "normal")
            },
            "most_common_upshift_rpm_range": self._find_rpm_range(upshifts),
            "session_duration": float(time_data[-1] - time_data[0]) if len(time_data) > 0 else 0
        }

        return ShiftReport(
            session_id=session_id,
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            total_shifts=len(shifts),
            total_upshifts=len(upshifts),
            total_downshifts=len(downshifts),
            shifts=shifts,
            gear_stats=gear_stats,
            early_shifts=early_shifts,
            late_shifts=late_shifts,
            over_rev_shifts=over_rev_shifts,
            recommendations=recommendations,
            summary=summary
        )

    def analyze_from_parquet(
        self,
        parquet_path: str,
        rpm_column: str = "RPM",
        speed_column: str = "GPS Speed",
        session_id: Optional[str] = None
    ) -> ShiftReport:
        """
        Analyze shifts from a Parquet file.

        Args:
            parquet_path: Path to Parquet file
            rpm_column: Column name for RPM data
            speed_column: Column name for speed data (mph)
            session_id: Session identifier (defaults to filename)

        Returns:
            ShiftReport with complete analysis
        """
        df = pd.read_parquet(parquet_path)

        if session_id is None:
            from pathlib import Path
            session_id = Path(parquet_path).stem

        # Find RPM column (may have different names)
        rpm_col = None
        for col in df.columns:
            if 'rpm' in col.lower():
                rpm_col = col
                break
        if rpm_col is None:
            rpm_col = rpm_column

        # Find speed column
        speed_col = None
        for col in df.columns:
            if 'speed' in col.lower():
                speed_col = col
                break
        if speed_col is None:
            speed_col = speed_column

        rpm_data = df[rpm_col].values
        speed_data = df[speed_col].values
        time_data = df.index.values

        # Convert speed to mph if needed (GPS Speed is often in m/s)
        if speed_data.max() < 100:  # Likely in m/s
            speed_data = speed_data * 2.237

        return self.analyze_session(rpm_data, speed_data, time_data, session_id)

    def _process_shifts(
        self,
        raw_shifts: List[Dict],
        gear_trace: List[GearInfo],
        time_data: np.ndarray
    ) -> List[ShiftEvent]:
        """Process raw shifts into ShiftEvent objects with quality assessment"""
        shifts = []

        for shift in raw_shifts:
            rpm = shift['rpm']

            # Assess shift quality
            if shift['type'] == 'upshift':
                if rpm > self.OVER_REV_RPM:
                    quality = "over-rev"
                elif rpm > self.LATE_SHIFT_RPM:
                    quality = "late"
                elif rpm < self.EARLY_SHIFT_RPM:
                    quality = "early"
                elif self.OPTIMAL_SHIFT_RPM_MIN <= rpm <= self.OPTIMAL_SHIFT_RPM_MAX:
                    quality = "optimal"
                else:
                    quality = "normal"
            else:
                quality = "normal"  # Downshifts don't have the same quality criteria

            shift_event = ShiftEvent(
                time=shift['time'],
                from_gear=shift['from_gear'],
                to_gear=shift['to_gear'],
                shift_type=shift['type'],
                rpm_at_shift=rpm,
                speed_mph=shift['speed_mph'],
                shift_quality=quality
            )
            shifts.append(shift_event)

        return shifts

    def _calculate_gear_stats(self, shifts: List[ShiftEvent]) -> Dict[int, GearShiftStats]:
        """Calculate statistics for each gear"""
        stats = {}

        # Initialize stats for each gear
        for i in range(1, 7):  # Gears 1-6
            stats[i] = GearShiftStats(gear=i)

        # Process each shift
        for shift in shifts:
            if shift.shift_type == "upshift":
                gear = shift.from_gear
                if gear in stats:
                    stats[gear].upshift_count += 1
                    stats[gear].upshift_rpms.append(shift.rpm_at_shift)
            else:  # downshift
                gear = shift.to_gear
                if gear in stats:
                    stats[gear].downshift_count += 1
                    stats[gear].downshift_rpms.append(shift.rpm_at_shift)

        # Calculate averages
        for gear, gear_stats in stats.items():
            if gear_stats.upshift_rpms:
                gear_stats.avg_upshift_rpm = np.mean(gear_stats.upshift_rpms)
                gear_stats.min_upshift_rpm = np.min(gear_stats.upshift_rpms)
                gear_stats.max_upshift_rpm = np.max(gear_stats.upshift_rpms)
            if gear_stats.downshift_rpms:
                gear_stats.avg_downshift_rpm = np.mean(gear_stats.downshift_rpms)

        return stats

    def _generate_recommendations(
        self,
        shifts: List[ShiftEvent],
        gear_stats: Dict[int, GearShiftStats]
    ) -> List[str]:
        """Generate shift improvement recommendations"""
        recommendations = []

        # Check for consistent early shifts
        early_count = sum(1 for s in shifts if s.shift_quality == "early")
        if early_count > len(shifts) * 0.3 and len(shifts) > 5:
            recommendations.append(
                f"Consider holding gears longer - {early_count} shifts ({100*early_count/len(shifts):.0f}%) "
                "were early (below 5500 RPM). Target 6000-6500 RPM for upshifts."
            )

        # Check for over-revs
        over_rev_count = sum(1 for s in shifts if s.shift_quality == "over-rev")
        if over_rev_count > 0:
            recommendations.append(
                f"WARNING: {over_rev_count} shift(s) above 7200 RPM (over-rev zone). "
                "This risks engine damage. Shift earlier."
            )

        # Check for late shifts
        late_count = sum(1 for s in shifts if s.shift_quality == "late")
        if late_count > len(shifts) * 0.2 and len(shifts) > 5:
            recommendations.append(
                f"{late_count} shifts were late (above 7000 RPM). "
                "Consider shifting slightly earlier to stay in optimal power band."
            )

        # Check for gear-specific issues
        for gear, stats in gear_stats.items():
            if stats.upshift_count > 3:
                if stats.max_upshift_rpm - stats.min_upshift_rpm > 1000:
                    recommendations.append(
                        f"Gear {gear} upshift RPM varies significantly "
                        f"({stats.min_upshift_rpm:.0f}-{stats.max_upshift_rpm:.0f} RPM). "
                        "Work on consistent shift points."
                    )

        if not recommendations:
            recommendations.append("Shift patterns look good! Keep up the consistent driving.")

        return recommendations

    def _find_rpm_range(self, upshifts: List[ShiftEvent]) -> str:
        """Find the most common RPM range for upshifts"""
        if not upshifts:
            return "N/A"

        rpms = [s.rpm_at_shift for s in upshifts]
        avg = np.mean(rpms)
        std = np.std(rpms)

        return f"{avg - std:.0f}-{avg + std:.0f} RPM"

    def get_shift_timing_by_gear(
        self,
        report: ShiftReport
    ) -> Dict[int, Dict[str, float]]:
        """
        Get detailed shift timing statistics by gear.

        Returns:
            Dict mapping gear number to timing statistics
        """
        result = {}

        for gear, stats in report.gear_stats.items():
            if stats.upshift_count > 0:
                result[gear] = {
                    "count": stats.upshift_count,
                    "avg_rpm": stats.avg_upshift_rpm,
                    "min_rpm": stats.min_upshift_rpm,
                    "max_rpm": stats.max_upshift_rpm,
                    "optimal_rpm": stats.optimal_upshift_rpm,
                    "deviation_from_optimal": abs(stats.avg_upshift_rpm - stats.optimal_upshift_rpm)
                }

        return result
