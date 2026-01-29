"""
Gear Usage Analysis Feature
Time spent in each gear, gear usage by track section, optimal gear recommendations.

Wraps the gear_calculator module with additional analysis features.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json

from ..analysis.gear_calculator import GearCalculator, GearInfo, analyze_lap_gearing
from ..config.vehicle_config import (
    CURRENT_SETUP,
    TRANSMISSION_SCENARIOS,
    TRACK_CONFIG,
    ENGINE_SPECS
)


@dataclass
class GearUsageStats:
    """Statistics for usage of a single gear"""
    gear_number: int
    time_seconds: float
    usage_percent: float
    sample_count: int
    speed_min_mph: float
    speed_max_mph: float
    speed_avg_mph: float
    rpm_min: float
    rpm_max: float
    rpm_avg: float
    shift_in_count: int  # Times shifted into this gear
    shift_out_count: int  # Times shifted out of this gear


@dataclass
class TrackSectionStats:
    """Gear usage statistics for a track section"""
    section_name: str
    dominant_gear: int
    gear_distribution: Dict[int, float]  # gear -> percent
    avg_speed_mph: float
    avg_rpm: float
    sample_count: int


@dataclass
class GearAnalysisReport:
    """Complete gear usage analysis report"""
    session_id: str
    track_name: str
    analysis_timestamp: str
    total_duration_seconds: float
    gear_usage: List[GearUsageStats]
    track_sections: List[TrackSectionStats]
    shift_summary: Dict
    rpm_analysis: Dict
    recommendations: List[str]
    summary: Dict

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "session_id": self.session_id,
            "track_name": self.track_name,
            "analysis_timestamp": self.analysis_timestamp,
            "total_duration_seconds": round(self.total_duration_seconds, 1),
            "gear_usage": [
                {
                    "gear_number": gu.gear_number,
                    "time_seconds": round(gu.time_seconds, 1),
                    "usage_percent": round(gu.usage_percent, 1),
                    "speed_range_mph": {
                        "min": round(gu.speed_min_mph, 1),
                        "max": round(gu.speed_max_mph, 1),
                        "avg": round(gu.speed_avg_mph, 1)
                    },
                    "rpm_range": {
                        "min": round(gu.rpm_min, 0),
                        "max": round(gu.rpm_max, 0),
                        "avg": round(gu.rpm_avg, 0)
                    },
                    "shifts_in": gu.shift_in_count,
                    "shifts_out": gu.shift_out_count
                }
                for gu in self.gear_usage
            ],
            "track_sections": [
                {
                    "section_name": ts.section_name,
                    "dominant_gear": ts.dominant_gear,
                    "gear_distribution": {
                        str(k): round(v, 1) for k, v in ts.gear_distribution.items()
                    },
                    "avg_speed_mph": round(ts.avg_speed_mph, 1),
                    "avg_rpm": round(ts.avg_rpm, 0)
                }
                for ts in self.track_sections
            ],
            "shift_summary": self.shift_summary,
            "rpm_analysis": self.rpm_analysis,
            "recommendations": self.recommendations,
            "summary": self.summary
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)


class GearAnalysis:
    """
    Analyzes gear usage from telemetry sessions.

    Calculates time in each gear, analyzes usage by track section,
    generates optimization recommendations.
    """

    def __init__(
        self,
        track_name: str = None,
        scenario_name: str = "Current Setup"
    ):
        """
        Initialize gear analyzer.

        Args:
            track_name: Name of the track (default from config)
            scenario_name: Transmission scenario to use
        """
        self.track_name = track_name or TRACK_CONFIG['name']
        self.scenario_name = scenario_name
        self.scenario = self._get_scenario(scenario_name)
        self.calculator = GearCalculator(
            self.scenario['transmission_ratios'],
            self.scenario['final_drive']
        )
        self.turn_coordinates = TRACK_CONFIG.get('turn_coordinates', {})

    def _get_scenario(self, name: str) -> Dict:
        """Get transmission scenario by name"""
        for scenario in TRANSMISSION_SCENARIOS:
            if scenario['name'] == name:
                return scenario
        return CURRENT_SETUP

    def analyze_from_arrays(
        self,
        time_data: np.ndarray,
        rpm_data: np.ndarray,
        speed_data: np.ndarray,
        latitude_data: np.ndarray = None,
        longitude_data: np.ndarray = None,
        session_id: str = "unknown"
    ) -> GearAnalysisReport:
        """
        Analyze gear usage from raw data arrays.

        Args:
            time_data: Array of timestamps (seconds)
            rpm_data: Engine RPM values
            speed_data: Speed values (mph)
            latitude_data: GPS latitude values (optional, for section analysis)
            longitude_data: GPS longitude values (optional, for section analysis)
            session_id: Session identifier

        Returns:
            GearAnalysisReport with complete analysis
        """
        # Calculate gear trace
        gear_trace = self.calculator.calculate_gear_trace(rpm_data, speed_data)

        # Calculate total duration
        total_duration = float(time_data[-1] - time_data[0]) if len(time_data) > 1 else 0

        # Calculate sample rate (for time calculations)
        if len(time_data) > 1:
            sample_rate = len(time_data) / total_duration
        else:
            sample_rate = 10  # Default assumption

        # Calculate gear usage statistics
        gear_usage = self._calculate_gear_usage(
            gear_trace, rpm_data, speed_data, sample_rate
        )

        # Find shift points
        shift_points = self.calculator.find_shift_points(gear_trace, time_data)

        # Update gear usage with shift counts
        self._add_shift_counts(gear_usage, shift_points)

        # Analyze by track section if GPS data available
        track_sections = []
        if latitude_data is not None and longitude_data is not None:
            track_sections = self._analyze_by_section(
                gear_trace, rpm_data, speed_data, latitude_data, longitude_data
            )

        # Analyze RPM usage
        rpm_analysis = self._analyze_rpm_usage(gear_trace, rpm_data)

        # Build shift summary
        shift_summary = self._build_shift_summary(shift_points)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            gear_usage, shift_points, rpm_analysis
        )

        # Build summary
        summary = self._build_summary(
            gear_usage, shift_points, rpm_analysis, total_duration
        )

        return GearAnalysisReport(
            session_id=session_id,
            track_name=self.track_name,
            analysis_timestamp=datetime.utcnow().isoformat(),
            total_duration_seconds=total_duration,
            gear_usage=gear_usage,
            track_sections=track_sections,
            shift_summary=shift_summary,
            rpm_analysis=rpm_analysis,
            recommendations=recommendations,
            summary=summary
        )

    def analyze_from_parquet(
        self,
        parquet_path: str,
        session_id: Optional[str] = None
    ) -> GearAnalysisReport:
        """
        Analyze gear usage from a Parquet file.

        Args:
            parquet_path: Path to Parquet file
            session_id: Session identifier (defaults to filename)

        Returns:
            GearAnalysisReport with complete analysis
        """
        df = pd.read_parquet(parquet_path)

        if session_id is None:
            from pathlib import Path
            session_id = Path(parquet_path).stem

        # Find required columns
        time_data = df.index.values
        rpm_data = self._find_column(df, ['RPM', 'rpm', 'RPM dup 3'])
        speed_data = self._find_column(df, ['GPS Speed', 'speed', 'Speed'])
        lat_data = self._find_column(df, ['GPS Latitude', 'latitude', 'Latitude'])
        lon_data = self._find_column(df, ['GPS Longitude', 'longitude', 'Longitude'])

        # Convert speed to mph if needed
        if speed_data is not None and speed_data.max() < 100:
            speed_data = speed_data * 2.237

        if rpm_data is None:
            raise ValueError("Parquet file missing RPM column")

        if speed_data is None:
            raise ValueError("Parquet file missing speed column")

        return self.analyze_from_arrays(
            time_data, rpm_data, speed_data, lat_data, lon_data, session_id
        )

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[np.ndarray]:
        """Find a column by trying multiple names"""
        for col in candidates:
            if col in df.columns:
                return df[col].values
            # Try case-insensitive match
            for actual_col in df.columns:
                if actual_col.lower() == col.lower():
                    return df[actual_col].values
        return None

    def _calculate_gear_usage(
        self,
        gear_trace: List[GearInfo],
        rpm_data: np.ndarray,
        speed_data: np.ndarray,
        sample_rate: float
    ) -> List[GearUsageStats]:
        """Calculate usage statistics for each gear"""
        gear_stats = {}

        for i, gear_info in enumerate(gear_trace):
            gear = gear_info.gear
            if gear <= 0:  # Skip neutral/unknown
                continue

            if gear not in gear_stats:
                gear_stats[gear] = {
                    'count': 0,
                    'speeds': [],
                    'rpms': []
                }

            gear_stats[gear]['count'] += 1
            gear_stats[gear]['speeds'].append(speed_data[i])
            gear_stats[gear]['rpms'].append(rpm_data[i])

        total_samples = len(gear_trace)
        usage_list = []

        for gear in sorted(gear_stats.keys()):
            stats = gear_stats[gear]
            count = stats['count']
            speeds = stats['speeds']
            rpms = stats['rpms']

            usage_list.append(GearUsageStats(
                gear_number=gear,
                time_seconds=count / sample_rate,
                usage_percent=(count / total_samples) * 100,
                sample_count=count,
                speed_min_mph=float(min(speeds)),
                speed_max_mph=float(max(speeds)),
                speed_avg_mph=float(np.mean(speeds)),
                rpm_min=float(min(rpms)),
                rpm_max=float(max(rpms)),
                rpm_avg=float(np.mean(rpms)),
                shift_in_count=0,
                shift_out_count=0
            ))

        return usage_list

    def _add_shift_counts(
        self,
        gear_usage: List[GearUsageStats],
        shift_points: List[Dict]
    ):
        """Add shift in/out counts to gear usage stats"""
        for shift in shift_points:
            from_gear = shift['from_gear']
            to_gear = shift['to_gear']

            for gu in gear_usage:
                if gu.gear_number == from_gear:
                    gu.shift_out_count += 1
                if gu.gear_number == to_gear:
                    gu.shift_in_count += 1

    def _analyze_by_section(
        self,
        gear_trace: List[GearInfo],
        rpm_data: np.ndarray,
        speed_data: np.ndarray,
        latitude_data: np.ndarray,
        longitude_data: np.ndarray
    ) -> List[TrackSectionStats]:
        """Analyze gear usage by track section"""
        if not self.turn_coordinates:
            return []

        sections = []

        for section_name, (turn_lat, turn_lon) in self.turn_coordinates.items():
            # Find points near this turn (within ~50m)
            threshold = 0.0005  # Roughly 50m at mid-latitudes
            section_indices = []

            for i, (lat, lon) in enumerate(zip(latitude_data, longitude_data)):
                if (abs(lat - turn_lat) < threshold and
                    abs(lon - turn_lon) < threshold):
                    section_indices.append(i)

            if not section_indices:
                continue

            # Calculate gear distribution for this section
            gear_counts = {}
            total = 0
            speeds = []
            rpms = []

            for idx in section_indices:
                gear = gear_trace[idx].gear
                if gear > 0:
                    gear_counts[gear] = gear_counts.get(gear, 0) + 1
                    total += 1
                    speeds.append(speed_data[idx])
                    rpms.append(rpm_data[idx])

            if total == 0:
                continue

            # Calculate distribution percentages
            distribution = {g: (c / total) * 100 for g, c in gear_counts.items()}

            # Find dominant gear
            dominant = max(gear_counts, key=gear_counts.get)

            sections.append(TrackSectionStats(
                section_name=section_name,
                dominant_gear=dominant,
                gear_distribution=distribution,
                avg_speed_mph=float(np.mean(speeds)),
                avg_rpm=float(np.mean(rpms)),
                sample_count=total
            ))

        return sections

    def _analyze_rpm_usage(
        self,
        gear_trace: List[GearInfo],
        rpm_data: np.ndarray
    ) -> Dict:
        """Analyze RPM usage patterns"""
        valid_rpms = [rpm for gear, rpm in zip(gear_trace, rpm_data)
                      if gear.gear > 0]

        if not valid_rpms:
            return {
                "avg_rpm": 0,
                "max_rpm": 0,
                "min_rpm": 0,
                "time_over_safe_limit_pct": 0,
                "time_in_power_band_pct": 0
            }

        safe_limit = ENGINE_SPECS['safe_rpm_limit']
        power_band_min = ENGINE_SPECS['power_band_min']
        power_band_max = ENGINE_SPECS['power_band_max']

        over_limit = sum(1 for rpm in valid_rpms if rpm > safe_limit)
        in_power_band = sum(1 for rpm in valid_rpms
                           if power_band_min <= rpm <= power_band_max)

        return {
            "avg_rpm": float(np.mean(valid_rpms)),
            "max_rpm": float(max(valid_rpms)),
            "min_rpm": float(min(valid_rpms)),
            "time_over_safe_limit_pct": (over_limit / len(valid_rpms)) * 100,
            "time_in_power_band_pct": (in_power_band / len(valid_rpms)) * 100
        }

    def _build_shift_summary(self, shift_points: List[Dict]) -> Dict:
        """Build summary of shifting patterns"""
        if not shift_points:
            return {
                "total_shifts": 0,
                "upshifts": 0,
                "downshifts": 0,
                "avg_upshift_rpm": 0,
                "avg_downshift_rpm": 0,
                "shifts_by_transition": {}
            }

        upshifts = [s for s in shift_points if s['type'] == 'upshift']
        downshifts = [s for s in shift_points if s['type'] == 'downshift']

        # Count shifts by gear transition
        transitions = {}
        for shift in shift_points:
            key = f"{shift['from_gear']}->{shift['to_gear']}"
            transitions[key] = transitions.get(key, 0) + 1

        return {
            "total_shifts": len(shift_points),
            "upshifts": len(upshifts),
            "downshifts": len(downshifts),
            "avg_upshift_rpm": float(np.mean([s['rpm'] for s in upshifts])) if upshifts else 0,
            "avg_downshift_rpm": float(np.mean([s['rpm'] for s in downshifts])) if downshifts else 0,
            "shifts_by_transition": transitions
        }

    def _generate_recommendations(
        self,
        gear_usage: List[GearUsageStats],
        shift_points: List[Dict],
        rpm_analysis: Dict
    ) -> List[str]:
        """Generate recommendations based on gear analysis"""
        recommendations = []

        if not gear_usage:
            return ["No gear data available for analysis"]

        # Check RPM over-rev
        if rpm_analysis['time_over_safe_limit_pct'] > 5:
            recommendations.append(
                f"Spending {rpm_analysis['time_over_safe_limit_pct']:.1f}% of time over "
                f"{ENGINE_SPECS['safe_rpm_limit']} RPM. Consider earlier upshifts."
            )

        # Check power band usage
        if rpm_analysis['time_in_power_band_pct'] < 30:
            recommendations.append(
                f"Only {rpm_analysis['time_in_power_band_pct']:.1f}% time in power band "
                f"({ENGINE_SPECS['power_band_min']}-{ENGINE_SPECS['power_band_max']} RPM). "
                "Review gear selection for better power delivery."
            )

        # Check for underutilized gears
        for gu in gear_usage:
            if gu.usage_percent < 5 and gu.shift_in_count > 3:
                recommendations.append(
                    f"Gear {gu.gear_number} used only {gu.usage_percent:.1f}% "
                    f"but shifted into {gu.shift_in_count} times. "
                    "Consider skipping this gear in some situations."
                )

        # Check upshift RPM consistency
        upshifts = [s for s in shift_points if s['type'] == 'upshift']
        if len(upshifts) >= 3:
            rpms = [s['rpm'] for s in upshifts]
            rpm_std = float(np.std(rpms))
            if rpm_std > 500:
                recommendations.append(
                    f"Upshift RPM varies by {rpm_std:.0f} RPM. "
                    "More consistent shift points may improve lap times."
                )

        # Check for high RPM on downshifts (potential for lockup)
        downshifts = [s for s in shift_points if s['type'] == 'downshift']
        high_rpm_downshifts = [s for s in downshifts if s['rpm'] > 6000]
        if high_rpm_downshifts:
            recommendations.append(
                f"{len(high_rpm_downshifts)} downshifts at >6000 RPM. "
                "Ensure proper rev-matching to prevent wheel lockup."
            )

        if not recommendations:
            recommendations.append(
                "Gear usage looks efficient. "
                f"Average {rpm_analysis['time_in_power_band_pct']:.1f}% time in power band."
            )

        return recommendations

    def _build_summary(
        self,
        gear_usage: List[GearUsageStats],
        shift_points: List[Dict],
        rpm_analysis: Dict,
        total_duration: float
    ) -> Dict:
        """Build analysis summary"""
        most_used = max(gear_usage, key=lambda x: x.usage_percent) if gear_usage else None

        return {
            "gears_used": len(gear_usage),
            "most_used_gear": most_used.gear_number if most_used else 0,
            "most_used_gear_pct": most_used.usage_percent if most_used else 0,
            "total_shifts": len(shift_points),
            "shifts_per_minute": len(shift_points) / (total_duration / 60) if total_duration > 0 else 0,
            "avg_rpm": rpm_analysis['avg_rpm'],
            "max_rpm": rpm_analysis['max_rpm'],
            "pct_over_safe_rpm": rpm_analysis['time_over_safe_limit_pct'],
            "pct_in_power_band": rpm_analysis['time_in_power_band_pct']
        }

    def get_gear_comparison(
        self,
        report: GearAnalysisReport,
        gear_a: int,
        gear_b: int
    ) -> Dict:
        """
        Compare usage of two gears.

        Args:
            report: GearAnalysisReport from analyze
            gear_a: First gear number
            gear_b: Second gear number

        Returns:
            Dictionary with comparison data
        """
        gear_a_stats = None
        gear_b_stats = None

        for gu in report.gear_usage:
            if gu.gear_number == gear_a:
                gear_a_stats = gu
            if gu.gear_number == gear_b:
                gear_b_stats = gu

        if not gear_a_stats or not gear_b_stats:
            return {"error": "Gear not found in analysis"}

        return {
            "gear_a": gear_a,
            "gear_b": gear_b,
            "usage_difference_pct": gear_b_stats.usage_percent - gear_a_stats.usage_percent,
            "avg_speed_difference": gear_b_stats.speed_avg_mph - gear_a_stats.speed_avg_mph,
            "avg_rpm_difference": gear_b_stats.rpm_avg - gear_a_stats.rpm_avg,
            "shift_activity_difference": (
                (gear_b_stats.shift_in_count + gear_b_stats.shift_out_count) -
                (gear_a_stats.shift_in_count + gear_a_stats.shift_out_count)
            )
        }

    def get_optimal_gear_at_speed(self, speed_mph: float) -> Dict:
        """
        Determine optimal gear for a given speed.

        Args:
            speed_mph: Target speed in mph

        Returns:
            Dictionary with recommended gear and RPM info
        """
        power_band_target = (ENGINE_SPECS['power_band_min'] +
                            ENGINE_SPECS['power_band_max']) / 2

        # Allow RPM from idle to safe limit for gear selection
        min_usable_rpm = 2000  # Reasonable minimum for driving
        max_usable_rpm = ENGINE_SPECS['safe_rpm_limit']

        best_gear = None
        best_rpm_diff = float('inf')

        for gear_num, ratio in enumerate(self.scenario['transmission_ratios'], 1):
            # Calculate RPM at this speed in this gear
            speed_ms = speed_mph / 2.237
            from ..config.vehicle_config import theoretical_rpm_at_speed
            rpm = theoretical_rpm_at_speed(
                speed_ms, ratio, self.scenario['final_drive']
            )

            # Check if RPM is in usable range
            if min_usable_rpm <= rpm <= max_usable_rpm:
                rpm_diff = abs(rpm - power_band_target)
                if rpm_diff < best_rpm_diff:
                    best_rpm_diff = rpm_diff
                    best_gear = {
                        'gear': gear_num,
                        'rpm': rpm,
                        'in_power_band': ENGINE_SPECS['power_band_min'] <= rpm <= ENGINE_SPECS['power_band_max'],
                        'rpm_headroom': ENGINE_SPECS['safe_rpm_limit'] - rpm
                    }

        if best_gear is None:
            return {
                "error": f"No suitable gear for {speed_mph} mph",
                "speed_mph": speed_mph
            }

        return {
            "speed_mph": speed_mph,
            "recommended_gear": best_gear['gear'],
            "rpm_at_speed": best_gear['rpm'],
            "in_power_band": best_gear['in_power_band'],
            "rpm_headroom": best_gear['rpm_headroom']
        }
