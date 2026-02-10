"""
Transmission Comparison Feature
Compare current vs proposed transmission setups using real session data.

Shows RPM/speed differences, theoretical performance, and recommendations.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
import json

from ..config.vehicle_config import (
    TRANSMISSION_SCENARIOS,
    CURRENT_SETUP,
    ENGINE_SPECS,
    TIRE_CIRCUMFERENCE_METERS,
    theoretical_speed_at_rpm,
    theoretical_rpm_at_speed,
    get_scenario_by_name
)
from ..analysis.gear_calculator import GearCalculator


@dataclass
class GearComparison:
    """Comparison data for a single gear"""
    gear_number: int
    current_ratio: float
    proposed_ratio: float
    ratio_difference_pct: float
    current_top_speed_mph: float
    proposed_top_speed_mph: float
    speed_difference_mph: float
    rpm_difference_at_60mph: float


@dataclass
class ScenarioPerformance:
    """Performance metrics for a transmission scenario"""
    name: str
    transmission_ratios: List[float]
    final_drive: float
    weight_lbs: int
    gear_top_speeds_mph: List[float]
    gear_top_speeds_at_redline: List[float]
    overlap_analysis: Dict[int, Dict]
    power_band_coverage: Dict[int, Tuple[float, float]]  # Speed range in power band


@dataclass
class TransmissionComparisonReport:
    """Complete transmission comparison report"""
    analysis_timestamp: str
    current_setup: ScenarioPerformance
    proposed_setup: ScenarioPerformance
    gear_comparisons: List[GearComparison]
    session_based_analysis: Optional[Dict] = None
    recommendations: List[str] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert report to dictionary"""
        return {
            "analysis_timestamp": self.analysis_timestamp,
            "current_setup": {
                "name": self.current_setup.name,
                "transmission_ratios": self.current_setup.transmission_ratios,
                "final_drive": self.current_setup.final_drive,
                "weight_lbs": self.current_setup.weight_lbs,
                "gear_top_speeds_mph": [round(s, 1) for s in self.current_setup.gear_top_speeds_mph],
                "power_band_coverage": {
                    k: [round(v[0], 1), round(v[1], 1)]
                    for k, v in self.current_setup.power_band_coverage.items()
                }
            },
            "proposed_setup": {
                "name": self.proposed_setup.name,
                "transmission_ratios": self.proposed_setup.transmission_ratios,
                "final_drive": self.proposed_setup.final_drive,
                "weight_lbs": self.proposed_setup.weight_lbs,
                "gear_top_speeds_mph": [round(s, 1) for s in self.proposed_setup.gear_top_speeds_mph],
                "power_band_coverage": {
                    k: [round(v[0], 1), round(v[1], 1)]
                    for k, v in self.proposed_setup.power_band_coverage.items()
                }
            },
            "gear_comparisons": [
                {
                    "gear": gc.gear_number,
                    "current_ratio": gc.current_ratio,
                    "proposed_ratio": gc.proposed_ratio,
                    "ratio_difference_pct": round(gc.ratio_difference_pct, 1),
                    "current_top_speed_mph": round(gc.current_top_speed_mph, 1),
                    "proposed_top_speed_mph": round(gc.proposed_top_speed_mph, 1),
                    "speed_difference_mph": round(gc.speed_difference_mph, 1)
                }
                for gc in self.gear_comparisons
            ],
            "session_based_analysis": self.session_based_analysis,
            "recommendations": self.recommendations,
            "summary": self.summary
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)


class TransmissionComparison:
    """
    Compare transmission setups and analyze performance differences.

    Uses theoretical calculations and real session data.
    """

    def __init__(
        self,
        tire_circumference: float = TIRE_CIRCUMFERENCE_METERS,
        safe_rpm: int = ENGINE_SPECS['safe_rpm_limit'],
        redline_rpm: int = ENGINE_SPECS['max_rpm'],
        power_band: Tuple[int, int] = (
            ENGINE_SPECS['power_band_min'],
            ENGINE_SPECS['power_band_max']
        )
    ):
        """
        Initialize the comparison analyzer.

        Args:
            tire_circumference: Tire circumference in meters
            safe_rpm: Safe operating RPM limit
            redline_rpm: Absolute redline RPM
            power_band: Tuple of (min_rpm, max_rpm) for power band
        """
        self.tire_circumference = tire_circumference
        self.safe_rpm = safe_rpm
        self.redline_rpm = redline_rpm
        self.power_band = power_band

    def compare(
        self,
        current_name: str = "Current Setup",
        proposed_name: str = "New Trans + Current Final"
    ) -> TransmissionComparisonReport:
        """
        Compare two transmission scenarios.

        Args:
            current_name: Name of current setup (from TRANSMISSION_SCENARIOS)
            proposed_name: Name of proposed setup

        Returns:
            TransmissionComparisonReport with complete analysis
        """
        current = get_scenario_by_name(current_name)
        proposed = get_scenario_by_name(proposed_name)

        # Calculate performance for each scenario
        current_perf = self._calculate_performance(current)
        proposed_perf = self._calculate_performance(proposed)

        # Generate gear-by-gear comparison
        gear_comparisons = self._compare_gears(current, proposed)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            current_perf, proposed_perf, gear_comparisons
        )

        # Build summary
        summary = self._build_summary(current_perf, proposed_perf)

        return TransmissionComparisonReport(
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            current_setup=current_perf,
            proposed_setup=proposed_perf,
            gear_comparisons=gear_comparisons,
            recommendations=recommendations,
            summary=summary
        )

    def compare_with_session_data(
        self,
        rpm_data: np.ndarray,
        speed_data: np.ndarray,
        current_name: str = "Current Setup",
        proposed_name: str = "New Trans + Current Final"
    ) -> TransmissionComparisonReport:
        """
        Compare setups using actual session data.

        Analyzes how RPM would differ at the same speeds with proposed setup.

        Args:
            rpm_data: Array of actual RPM values from session
            speed_data: Array of speed values (mph)
            current_name: Current setup name
            proposed_name: Proposed setup name

        Returns:
            TransmissionComparisonReport with session-based analysis
        """
        # Get basic comparison
        report = self.compare(current_name, proposed_name)

        current = get_scenario_by_name(current_name)
        proposed = get_scenario_by_name(proposed_name)

        # Create gear calculators
        current_calc = GearCalculator(
            current['transmission_ratios'],
            current['final_drive']
        )
        proposed_calc = GearCalculator(
            proposed['transmission_ratios'],
            proposed['final_drive']
        )

        # Calculate gear traces for both setups
        current_trace = current_calc.calculate_gear_trace(rpm_data, speed_data)
        proposed_trace = proposed_calc.calculate_gear_trace(rpm_data, speed_data)

        # Analyze differences
        session_analysis = self._analyze_session_differences(
            rpm_data, speed_data, current_trace, proposed_trace,
            current, proposed
        )

        report.session_based_analysis = session_analysis

        # Add session-specific recommendations
        session_recommendations = self._generate_session_recommendations(
            session_analysis
        )
        report.recommendations.extend(session_recommendations)

        return report

    def _calculate_performance(self, scenario: Dict) -> ScenarioPerformance:
        """Calculate performance metrics for a scenario"""
        ratios = scenario['transmission_ratios']
        final_drive = scenario['final_drive']

        # Calculate top speeds at safe RPM
        top_speeds = []
        top_speeds_redline = []
        power_band_coverage = {}

        for i, ratio in enumerate(ratios):
            # Speed at safe RPM
            speed_ms = theoretical_speed_at_rpm(
                self.safe_rpm, ratio, final_drive, self.tire_circumference
            )
            top_speeds.append(speed_ms * 2.237)  # Convert to mph

            # Speed at redline
            speed_redline = theoretical_speed_at_rpm(
                self.redline_rpm, ratio, final_drive, self.tire_circumference
            )
            top_speeds_redline.append(speed_redline * 2.237)

            # Power band speed range
            pb_low = theoretical_speed_at_rpm(
                self.power_band[0], ratio, final_drive, self.tire_circumference
            ) * 2.237
            pb_high = theoretical_speed_at_rpm(
                self.power_band[1], ratio, final_drive, self.tire_circumference
            ) * 2.237
            power_band_coverage[i + 1] = (pb_low, pb_high)

        # Analyze gear overlap
        overlap_analysis = self._analyze_overlap(ratios, final_drive)

        return ScenarioPerformance(
            name=scenario['name'],
            transmission_ratios=ratios,
            final_drive=final_drive,
            weight_lbs=scenario['weight_lbs'],
            gear_top_speeds_mph=top_speeds,
            gear_top_speeds_at_redline=top_speeds_redline,
            overlap_analysis=overlap_analysis,
            power_band_coverage=power_band_coverage
        )

    def _analyze_overlap(
        self,
        ratios: List[float],
        final_drive: float
    ) -> Dict[int, Dict]:
        """Analyze gear overlap (speed range where two gears share power band)"""
        overlap = {}

        for i in range(len(ratios) - 1):
            gear = i + 1
            next_gear = i + 2

            # Get power band speeds for each gear
            gear_pb_high = theoretical_speed_at_rpm(
                self.power_band[1], ratios[i], final_drive, self.tire_circumference
            ) * 2.237

            next_gear_pb_low = theoretical_speed_at_rpm(
                self.power_band[0], ratios[i + 1], final_drive, self.tire_circumference
            ) * 2.237

            overlap_amount = gear_pb_high - next_gear_pb_low

            overlap[gear] = {
                "to_gear": next_gear,
                "overlap_mph": overlap_amount,
                "has_gap": overlap_amount < 0,
                "gap_mph": abs(overlap_amount) if overlap_amount < 0 else 0
            }

        return overlap

    def _compare_gears(
        self,
        current: Dict,
        proposed: Dict
    ) -> List[GearComparison]:
        """Generate gear-by-gear comparison"""
        comparisons = []

        current_ratios = current['transmission_ratios']
        proposed_ratios = proposed['transmission_ratios']

        # Handle different number of gears
        max_gears = max(len(current_ratios), len(proposed_ratios))

        for i in range(max_gears):
            current_ratio = current_ratios[i] if i < len(current_ratios) else 0
            proposed_ratio = proposed_ratios[i] if i < len(proposed_ratios) else 0

            # Calculate ratio difference
            if current_ratio > 0:
                ratio_diff_pct = ((proposed_ratio - current_ratio) / current_ratio) * 100
            else:
                ratio_diff_pct = 0

            # Calculate top speeds
            current_speed = theoretical_speed_at_rpm(
                self.safe_rpm, current_ratio, current['final_drive'],
                self.tire_circumference
            ) * 2.237 if current_ratio > 0 else 0

            proposed_speed = theoretical_speed_at_rpm(
                self.safe_rpm, proposed_ratio, proposed['final_drive'],
                self.tire_circumference
            ) * 2.237 if proposed_ratio > 0 else 0

            # Calculate RPM difference at a reference speed (60 mph)
            reference_speed_ms = 60 / 2.237
            current_rpm = theoretical_rpm_at_speed(
                reference_speed_ms, current_ratio, current['final_drive'],
                self.tire_circumference
            ) if current_ratio > 0 else 0

            proposed_rpm = theoretical_rpm_at_speed(
                reference_speed_ms, proposed_ratio, proposed['final_drive'],
                self.tire_circumference
            ) if proposed_ratio > 0 else 0

            comparisons.append(GearComparison(
                gear_number=i + 1,
                current_ratio=current_ratio,
                proposed_ratio=proposed_ratio,
                ratio_difference_pct=ratio_diff_pct,
                current_top_speed_mph=current_speed,
                proposed_top_speed_mph=proposed_speed,
                speed_difference_mph=proposed_speed - current_speed,
                rpm_difference_at_60mph=proposed_rpm - current_rpm
            ))

        return comparisons

    def _analyze_session_differences(
        self,
        rpm_data: np.ndarray,
        speed_data: np.ndarray,
        current_trace,
        proposed_trace,
        current: Dict,
        proposed: Dict
    ) -> Dict:
        """Analyze how proposed setup would affect a real session"""
        # Calculate what RPM would be at each point with proposed setup
        proposed_rpm_estimate = []

        for i, (speed, current_gear) in enumerate(zip(speed_data, current_trace)):
            gear = current_gear.gear
            if gear > 0 and gear <= len(proposed['transmission_ratios']):
                speed_ms = speed / 2.237
                new_rpm = theoretical_rpm_at_speed(
                    speed_ms,
                    proposed['transmission_ratios'][gear - 1],
                    proposed['final_drive'],
                    self.tire_circumference
                )
                proposed_rpm_estimate.append(new_rpm)
            else:
                proposed_rpm_estimate.append(rpm_data[i])

        proposed_rpm_array = np.array(proposed_rpm_estimate)

        # Calculate statistics
        rpm_difference = proposed_rpm_array - rpm_data
        valid_mask = rpm_data > 1000  # Only consider meaningful RPM values

        return {
            "avg_rpm_difference": float(np.mean(rpm_difference[valid_mask])),
            "max_rpm_difference": float(np.max(rpm_difference[valid_mask])),
            "min_rpm_difference": float(np.min(rpm_difference[valid_mask])),
            "pct_time_higher_rpm": float(
                np.sum(rpm_difference[valid_mask] > 0) / np.sum(valid_mask) * 100
            ),
            "proposed_max_rpm": float(np.max(proposed_rpm_array[valid_mask])),
            "current_max_rpm": float(np.max(rpm_data[valid_mask])),
            "sample_points": int(np.sum(valid_mask))
        }

    def _generate_recommendations(
        self,
        current: ScenarioPerformance,
        proposed: ScenarioPerformance,
        comparisons: List[GearComparison]
    ) -> List[str]:
        """Generate recommendations based on comparison"""
        recommendations = []

        # Check for gear gaps
        for gear, overlap_info in proposed.overlap_analysis.items():
            if overlap_info['has_gap']:
                recommendations.append(
                    f"WARNING: Gear {gear} to {overlap_info['to_gear']} has a "
                    f"{overlap_info['gap_mph']:.1f} mph gap outside power band. "
                    "May need to over-rev or under-rev during shift."
                )

        # Check top speed differences
        current_top = max(current.gear_top_speeds_mph)
        proposed_top = max(proposed.gear_top_speeds_mph)

        if proposed_top < current_top - 5:
            recommendations.append(
                f"Top speed reduced by {current_top - proposed_top:.1f} mph. "
                "Consider if this affects track performance."
            )
        elif proposed_top > current_top + 5:
            recommendations.append(
                f"Top speed increased by {proposed_top - current_top:.1f} mph. "
                "Verify engine can handle sustained high-speed running."
            )

        # Check weight difference
        weight_diff = proposed.weight_lbs - current.weight_lbs
        if weight_diff != 0:
            recommendations.append(
                f"Weight {'reduced' if weight_diff < 0 else 'increased'} by "
                f"{abs(weight_diff)} lbs, affecting acceleration."
            )

        # Check first gear ratio
        if comparisons and comparisons[0].ratio_difference_pct > 20:
            recommendations.append(
                f"First gear {comparisons[0].ratio_difference_pct:.0f}% numerically higher. "
                "Better acceleration but may spin tires more easily."
            )

        if not recommendations:
            recommendations.append(
                "Both setups appear comparable. Test on track to evaluate feel."
            )

        return recommendations

    def _generate_session_recommendations(self, session_analysis: Dict) -> List[str]:
        """Generate recommendations from session analysis"""
        recommendations = []

        avg_diff = session_analysis['avg_rpm_difference']
        max_proposed = session_analysis['proposed_max_rpm']

        if avg_diff > 500:
            recommendations.append(
                f"Session data shows proposed setup runs ~{abs(avg_diff):.0f} RPM higher. "
                "Shift points will need adjustment."
            )
        elif avg_diff < -500:
            recommendations.append(
                f"Session data shows proposed setup runs ~{abs(avg_diff):.0f} RPM lower. "
                "May feel less responsive but more relaxed."
            )

        if max_proposed > self.redline_rpm:
            recommendations.append(
                f"WARNING: Proposed setup would exceed redline ({max_proposed:.0f} RPM) "
                "at current driving speeds. Shift earlier or reconsider ratios."
            )

        return recommendations

    def _build_summary(
        self,
        current: ScenarioPerformance,
        proposed: ScenarioPerformance
    ) -> Dict:
        """Build comparison summary"""
        return {
            "current_name": current.name,
            "proposed_name": proposed.name,
            "top_speed_change_mph": max(proposed.gear_top_speeds_mph) - max(current.gear_top_speeds_mph),
            "weight_change_lbs": proposed.weight_lbs - current.weight_lbs,
            "gear_count_current": len(current.transmission_ratios),
            "gear_count_proposed": len(proposed.transmission_ratios),
            "final_drive_change": proposed.final_drive - current.final_drive
        }

    def list_available_scenarios(self) -> List[str]:
        """List all available transmission scenarios"""
        return [s['name'] for s in TRANSMISSION_SCENARIOS]

    def get_scenario_details(self, name: str) -> Optional[Dict]:
        """Get details of a specific scenario"""
        return get_scenario_by_name(name)
