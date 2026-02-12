"""
Lap Detection and Analysis Feature
GPS-based lap detection, fastest lap identification, per-lap statistics.

Wraps the lap_analyzer module with additional analysis features.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
import json

from ..analysis.lap_analyzer import LapAnalyzer, LapInfo, analyze_session_laps
from .base_analyzer import BaseAnalyzer, BaseAnalysisReport
from .registry import analyzer_registry
from ..config.tracks import get_track_config as _get_track_config
from ..config.vehicles import get_processing_config as _get_processing_config
from ..session.models import LapClassification
from ..utils.dataframe_helpers import find_column, SPEED_MS_TO_MPH, safe_float as _safe_float


@dataclass
class LapStatistics:
    """Statistics for a single lap"""
    lap_number: int
    lap_time: float
    gap_to_fastest: float
    gap_to_previous: float
    max_speed_mph: float
    avg_speed_mph: float
    max_rpm: float
    avg_rpm: float
    distance_meters: float
    sector_times: List[float] = field(default_factory=list)
    classification: str = "race_pace"
    classification_reason: str = ""


@dataclass
class LapRecommendation:
    """Recommended laps for analysis based on detected patterns"""
    recommended_laps: List[int]  # Lap numbers to analyze
    reason: str  # Why these laps are recommended
    pattern_detected: str  # "warming", "consistent", "full_session", "degrading"
    best_representative_lap: int  # Most representative hot lap
    best_3_lap_average: float  # Best 3 consecutive lap average
    warm_up_detected: bool
    cool_down_detected: bool

    def to_dict(self) -> dict:
        return {
            "recommended_laps": self.recommended_laps,
            "reason": self.reason,
            "pattern_detected": self.pattern_detected,
            "best_representative_lap": self.best_representative_lap,
            "best_3_lap_average": round(_safe_float(self.best_3_lap_average), 2),
            "warm_up_detected": self.warm_up_detected,
            "cool_down_detected": self.cool_down_detected
        }


@dataclass
class LapAnalysisReport(BaseAnalysisReport):
    """Complete lap analysis report"""
    session_id: str
    track_name: str
    analysis_timestamp: str
    total_laps: int
    fastest_lap_number: int
    fastest_lap_time: float
    average_lap_time: float
    lap_time_consistency: float  # Standard deviation
    laps: List[LapStatistics]
    improvement_trend: str
    recommendations: List[str]
    summary: Dict
    lap_recommendations: Optional[LapRecommendation] = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        # Sanitize summary dict for NaN/inf values
        safe_summary = {}
        for key, value in self.summary.items():
            if isinstance(value, float):
                safe_summary[key] = _safe_float(value)
            else:
                safe_summary[key] = value

        result = {
            "session_id": self.session_id,
            "track_name": self.track_name,
            "analysis_timestamp": self.analysis_timestamp,
            "total_laps": self.total_laps,
            "fastest_lap": {
                "lap_number": self.fastest_lap_number,
                "lap_time": round(_safe_float(self.fastest_lap_time), 2)
            },
            "average_lap_time": round(_safe_float(self.average_lap_time), 2),
            "lap_time_consistency": round(_safe_float(self.lap_time_consistency), 2),
            "laps": [
                {
                    "lap_number": lap.lap_number,
                    "lap_time": round(_safe_float(lap.lap_time), 2),
                    "gap_to_fastest": round(_safe_float(lap.gap_to_fastest), 2),
                    "gap_to_previous": round(_safe_float(lap.gap_to_previous), 2),
                    "max_speed_mph": round(_safe_float(lap.max_speed_mph), 1),
                    "avg_speed_mph": round(_safe_float(lap.avg_speed_mph), 1),
                    "max_rpm": round(_safe_float(lap.max_rpm), 0),
                    "avg_rpm": round(_safe_float(lap.avg_rpm), 0),
                    "classification": lap.classification,
                    "classification_reason": lap.classification_reason
                }
                for lap in self.laps
            ],
            "improvement_trend": self.improvement_trend,
            "recommendations": self.recommendations,
            "summary": safe_summary
        }
        if self.lap_recommendations:
            result["lap_recommendations"] = self.lap_recommendations.to_dict()
        result.update(self._trace_dict())
        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)


class LapAnalysis(BaseAnalyzer):
    """
    Analyzes lap data from telemetry sessions.

    Detects laps, calculates statistics, identifies improvement opportunities.
    """

    # Registry metadata
    registry_key = "laps"
    required_channels = ["latitude", "longitude", "speed"]
    optional_channels = ["rpm"]
    config_params = ["track_name"]

    def __init__(self, track_name: str = None):
        """
        Initialize lap analyzer.

        Args:
            track_name: Name of the track (default from config)
        """
        _track_cfg = _get_track_config()
        self.track_name = track_name or _track_cfg['name']
        self.start_finish_gps = _track_cfg.get('start_finish_gps')

    def analyze_from_arrays(
        self,
        time_data: np.ndarray,
        latitude_data: np.ndarray,
        longitude_data: np.ndarray,
        rpm_data: np.ndarray,
        speed_data: np.ndarray,
        session_id: str = "unknown"
    ) -> LapAnalysisReport:
        """
        Analyze laps from raw data arrays.

        Args:
            time_data: Array of timestamps (seconds)
            latitude_data: GPS latitude values
            longitude_data: GPS longitude values
            rpm_data: Engine RPM values
            speed_data: Speed values (mph)
            session_id: Session identifier

        Returns:
            LapAnalysisReport with complete analysis
        """
        # Build session data dict for lap analyzer
        session_data = {
            'time': time_data,
            'latitude': latitude_data,
            'longitude': longitude_data,
            'rpm': rpm_data,
            'speed_mph': speed_data,
            'speed_ms': speed_data / SPEED_MS_TO_MPH
        }

        # Use existing lap analyzer
        analyzer = LapAnalyzer(session_data)
        laps = analyzer.detect_laps()
        analyzer.find_fastest_lap()

        # Build report
        return self._build_report(
            analyzer.laps,
            analyzer.fastest_lap,
            session_data,
            session_id
        )

    def analyze_from_parquet(
        self,
        parquet_path: str,
        session_id: Optional[str] = None,
        include_trace: bool = False,
        **kwargs,
    ) -> LapAnalysisReport:
        """
        Analyze laps from a Parquet file.

        Args:
            parquet_path: Path to Parquet file
            session_id: Session identifier (defaults to filename)
            include_trace: If True, attach CalculationTrace to report.

        Returns:
            LapAnalysisReport with complete analysis
        """
        trace = self._create_trace("LapAnalysis") if include_trace else None

        df = pd.read_parquet(parquet_path)

        if session_id is None:
            from pathlib import Path
            session_id = Path(parquet_path).stem

        # Find required columns
        from ..utils.dataframe_helpers import find_column_name
        time_data = df.index.values
        lat_col = find_column_name(df, ['GPS Latitude', 'latitude', 'Latitude'])
        lon_col = find_column_name(df, ['GPS Longitude', 'longitude', 'Longitude'])
        speed_col = find_column_name(df, ['GPS Speed', 'speed', 'Speed'])

        lat_data = find_column(df, ['GPS Latitude', 'latitude', 'Latitude'])
        lon_data = find_column(df, ['GPS Longitude', 'longitude', 'Longitude'])
        rpm_data = find_column(df, ['RPM', 'rpm', 'RPM dup 3'])
        speed_data = find_column(df, ['GPS Speed', 'speed', 'Speed'])

        speed_unit_detected = "mph"
        # Convert speed to mph if needed
        if speed_data is not None and speed_data.max() < 100:
            speed_data = speed_data * SPEED_MS_TO_MPH
            speed_unit_detected = "m/s"

        if lat_data is None or lon_data is None:
            raise ValueError("Parquet file missing GPS latitude/longitude columns")

        if rpm_data is None:
            rpm_data = np.zeros(len(time_data))

        if speed_data is None:
            speed_data = np.zeros(len(time_data))

        report = self.analyze_from_arrays(
            time_data, lat_data, lon_data, rpm_data, speed_data, session_id
        )

        if trace:
            trace.record_input("latitude_column", lat_col)
            trace.record_input("longitude_column", lon_col)
            trace.record_input("speed_column", speed_col)
            trace.record_input("speed_unit_detected", speed_unit_detected)
            trace.record_input("sample_count", len(time_data))
            trace.record_input("track_detected", self.track_name)

            trace.record_config("track_name", self.track_name)
            if self.start_finish_gps:
                trace.record_config("start_finish_gps", self.start_finish_gps)

            trace.record_intermediate("laps_detected", report.total_laps)
            trace.record_intermediate("fastest_lap_time", report.fastest_lap_time)
            if report.laps:
                trace.record_intermediate("slowest_lap_time", max(l.lap_time for l in report.laps))
            trace.record_intermediate("avg_lap_time", report.average_lap_time)
            if report.laps:
                total_dist = sum(l.distance_meters for l in report.laps)
                avg_dist_miles = (total_dist / len(report.laps)) / 1609.34
                trace.record_intermediate("estimated_lap_distance_miles", round(avg_dist_miles, 2))

            self._run_sanity_checks(trace, report, lat_data, lon_data, speed_unit_detected)
            report.trace = trace

        return report

    def _run_sanity_checks(self, trace, report: LapAnalysisReport,
                           lat_data: np.ndarray, lon_data: np.ndarray,
                           speed_unit_detected: str) -> None:
        """Run sanity checks on lap analysis results."""
        # Check 5.1: speed_unit_consistent
        if speed_unit_detected == "m/s" and report.laps:
            max_speed = max(l.max_speed_mph for l in report.laps)
            if 30 <= max_speed <= 200:
                trace.add_check(
                    "speed_unit_consistent", "pass",
                    f"Speed converted from m/s, max after conversion {max_speed:.0f} mph is reasonable",
                    expected="30-200 mph", actual=f"{max_speed:.0f} mph",
                    impact="Lap analysis uses speed for time validation and distance estimation. A wrong speed unit makes lap distances off by ~2.2x and may cause incorrect lap classifications.",
                )
            else:
                trace.add_check(
                    "speed_unit_consistent", "warn",
                    f"Speed converted from m/s but max {max_speed:.0f} mph is unusual",
                    expected="30-200 mph", actual=f"{max_speed:.0f} mph",
                    impact="Lap analysis uses speed for time validation and distance estimation. A wrong speed unit makes lap distances off by ~2.2x and may cause incorrect lap classifications.",
                )
        else:
            trace.add_check(
                "speed_unit_consistent", "pass",
                f"Speed unit detected as {speed_unit_detected}",
                impact="Lap analysis uses speed for time validation and distance estimation. A wrong speed unit makes lap distances off by ~2.2x and may cause incorrect lap classifications.",
            )

        # Check 5.2: lap_distance_plausible
        if report.laps:
            avg_dist_miles = trace.intermediates.get("estimated_lap_distance_miles", 0)
            # Road America is ~4.0 miles, most tracks 1.5-4.5 miles
            if 1.0 <= avg_dist_miles <= 6.0:
                trace.add_check(
                    "lap_distance_plausible", "pass",
                    f"Estimated lap distance {avg_dist_miles:.1f} miles is reasonable",
                    expected="1.0-6.0 miles", actual=f"{avg_dist_miles:.1f} miles",
                    impact="Estimated lap distance is checked against known track lengths. An implausible distance means GPS is unreliable or start/finish detection is wrong, affecting all per-lap statistics.",
                )
            elif avg_dist_miles > 0:
                trace.add_check(
                    "lap_distance_plausible", "warn",
                    f"Estimated lap distance {avg_dist_miles:.1f} miles is unusual for a road course",
                    expected="1.0-6.0 miles", actual=f"{avg_dist_miles:.1f} miles",
                    impact="Estimated lap distance is checked against known track lengths. An implausible distance means GPS is unreliable or start/finish detection is wrong, affecting all per-lap statistics.",
                )
            else:
                trace.add_check(
                    "lap_distance_plausible", "warn",
                    "Could not estimate lap distance",
                    impact="Estimated lap distance is checked against known track lengths. An implausible distance means GPS is unreliable or start/finish detection is wrong, affecting all per-lap statistics.",
                )
        else:
            trace.add_check(
                "lap_distance_plausible", "warn", "No laps detected, cannot check distance",
                impact="Estimated lap distance is checked against known track lengths. An implausible distance means GPS is unreliable or start/finish detection is wrong, affecting all per-lap statistics.",
            )

        # Check 5.3: lap_time_plausible
        if report.laps:
            fastest = report.fastest_lap_time
            slowest = max(l.lap_time for l in report.laps)
            if fastest > 30 and slowest < 600:
                trace.add_check(
                    "lap_time_plausible", "pass",
                    f"Lap times {fastest:.1f}s - {slowest:.1f}s are plausible",
                    expected="30-600s", actual=f"{fastest:.1f}-{slowest:.1f}s",
                    impact="Lap times outside 30-600s indicate GPS detection issues. This affects fastest lap identification, lap classification, and all per-lap metrics.",
                )
            else:
                trace.add_check(
                    "lap_time_plausible", "warn",
                    f"Lap times {fastest:.1f}s - {slowest:.1f}s outside expected range",
                    expected="30-600s", actual=f"{fastest:.1f}-{slowest:.1f}s",
                    impact="Lap times outside 30-600s indicate GPS detection issues. This affects fastest lap identification, lap classification, and all per-lap metrics.",
                )
        else:
            trace.add_check(
                "lap_time_plausible", "warn", "No laps detected",
                impact="Lap times outside 30-600s indicate GPS detection issues. This affects fastest lap identification, lap classification, and all per-lap metrics.",
            )

        # Check 5.4: gps_coordinates_valid
        valid_lat = lat_data[~np.isnan(lat_data)]
        valid_lon = lon_data[~np.isnan(lon_data)]
        if (len(valid_lat) > 0 and len(valid_lon) > 0
                and -90 <= np.min(valid_lat) and np.max(valid_lat) <= 90
                and -180 <= np.min(valid_lon) and np.max(valid_lon) <= 180
                and not (np.all(valid_lat == 0) and np.all(valid_lon == 0))):
            trace.add_check(
                "gps_coordinates_valid", "pass",
                f"GPS coordinates in valid range (lat {np.min(valid_lat):.4f}-{np.max(valid_lat):.4f})",
                impact="All lap detection depends on GPS for start/finish line crossing. Invalid GPS means no laps can be detected and all per-lap metrics are unavailable.",
            )
        else:
            trace.add_check(
                "gps_coordinates_valid", "fail",
                "GPS coordinates are invalid (out of range or all zeros)",
                severity="error",
                impact="All lap detection depends on GPS for start/finish line crossing. Invalid GPS means no laps can be detected and all per-lap metrics are unavailable.",
            )

    def _build_report(
        self,
        laps: List[LapInfo],
        fastest_lap: Optional[LapInfo],
        session_data: Dict,
        session_id: str
    ) -> LapAnalysisReport:
        """Build the analysis report"""
        if not laps:
            return LapAnalysisReport(
                session_id=session_id,
                track_name=self.track_name,
                analysis_timestamp=datetime.now(timezone.utc).isoformat(),
                total_laps=0,
                fastest_lap_number=0,
                fastest_lap_time=0,
                average_lap_time=0,
                lap_time_consistency=0,
                laps=[],
                improvement_trend="N/A",
                recommendations=["No laps detected in session"],
                summary={}
            )

        fastest_time = fastest_lap.lap_time if fastest_lap else min(lap.lap_time for lap in laps)
        lap_times = [lap.lap_time for lap in laps]

        # Build lap statistics
        lap_stats = []
        for i, lap in enumerate(laps):
            prev_time = laps[i - 1].lap_time if i > 0 else lap.lap_time

            # Calculate average speed for this lap
            start_idx = lap.start_index
            end_idx = lap.end_index
            lap_speed = session_data['speed_mph'][start_idx:end_idx + 1]
            avg_speed = float(np.mean(lap_speed)) if len(lap_speed) > 0 else 0

            # Estimate distance
            time_diff = lap.lap_time
            distance = avg_speed * 0.44704 * time_diff  # Convert mph to m/s, multiply by time

            lap_stats.append(LapStatistics(
                lap_number=lap.lap_number,
                lap_time=lap.lap_time,
                gap_to_fastest=lap.lap_time - fastest_time,
                gap_to_previous=lap.lap_time - prev_time,
                max_speed_mph=lap.max_speed_mph,
                avg_speed_mph=avg_speed,
                max_rpm=lap.max_rpm,
                avg_rpm=lap.avg_rpm,
                distance_meters=distance
            ))

        # Calculate improvement trend
        trend = self._calculate_trend(lap_times)

        # Generate recommendations
        recommendations = self._generate_recommendations(lap_stats, fastest_time)

        # Classify laps and get lap recommendations
        self.classify_laps(lap_stats, fastest_time)
        lap_recs = self.get_lap_recommendations(lap_stats)

        # Build summary
        summary = {
            "session_duration_seconds": float(session_data['time'][-1] - session_data['time'][0]),
            "total_distance_meters": sum(ls.distance_meters for ls in lap_stats),
            "best_3_lap_avg": np.mean(sorted(lap_times)[:3]) if len(lap_times) >= 3 else np.mean(lap_times),
            "worst_lap_time": max(lap_times),
            "lap_time_range": max(lap_times) - min(lap_times)
        }

        return LapAnalysisReport(
            session_id=session_id,
            track_name=self.track_name,
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            total_laps=len(laps),
            fastest_lap_number=fastest_lap.lap_number if fastest_lap else laps[0].lap_number,
            fastest_lap_time=fastest_time,
            average_lap_time=np.mean(lap_times),
            lap_time_consistency=np.std(lap_times),
            laps=lap_stats,
            improvement_trend=trend,
            recommendations=recommendations,
            summary=summary,
            lap_recommendations=lap_recs
        )

    def _calculate_trend(self, lap_times: List[float]) -> str:
        """Calculate improvement trend from lap times"""
        if len(lap_times) < 3:
            return "insufficient_data"

        # Compare first half to second half
        mid = len(lap_times) // 2
        first_half_avg = np.mean(lap_times[:mid])
        second_half_avg = np.mean(lap_times[mid:])

        diff = second_half_avg - first_half_avg

        if diff < -2:
            return "improving_significantly"
        elif diff < -0.5:
            return "improving"
        elif diff > 2:
            return "degrading_significantly"
        elif diff > 0.5:
            return "degrading"
        else:
            return "consistent"

    def _generate_recommendations(
        self,
        lap_stats: List[LapStatistics],
        fastest_time: float
    ) -> List[str]:
        """Generate recommendations based on lap analysis"""
        recommendations = []

        if not lap_stats:
            return ["No lap data available"]

        # Check consistency
        gaps = [ls.gap_to_fastest for ls in lap_stats]
        avg_gap = np.mean(gaps)
        max_gap = max(gaps)

        if max_gap > 5:
            recommendations.append(
                f"Large lap time variation detected ({max_gap:.1f}s gap to fastest). "
                "Focus on consistency through corner entry speeds."
            )

        if avg_gap > 3:
            recommendations.append(
                f"Average lap {avg_gap:.1f}s off pace. "
                "Review fastest lap telemetry for improvement areas."
            )

        # Check for improvement trend
        if len(lap_stats) >= 4:
            first_3_avg = np.mean([ls.lap_time for ls in lap_stats[:3]])
            last_3_avg = np.mean([ls.lap_time for ls in lap_stats[-3:]])

            if last_3_avg < first_3_avg - 2:
                recommendations.append(
                    "Good improvement trend! Last 3 laps significantly faster than first 3."
                )
            elif last_3_avg > first_3_avg + 2:
                recommendations.append(
                    "Lap times degrading toward end of session. "
                    "Consider tire wear, fuel load, or driver fatigue."
                )

        # Check RPM usage
        max_rpms = [ls.max_rpm for ls in lap_stats]
        if max(max_rpms) > 7200:
            recommendations.append(
                f"Max RPM reached {max(max_rpms):.0f} (over-rev zone). "
                "Consider earlier shift points."
            )

        if not recommendations:
            recommendations.append(
                f"Solid session! Fastest lap {fastest_time:.2f}s with good consistency."
            )

        return recommendations

    def classify_laps(self, lap_stats: List[LapStatistics], fastest_time: float) -> None:
        """
        Classify each lap based on timing patterns.

        Modifies lap_stats in place, setting classification and classification_reason.

        Classification rules:
        - Hot lap: gap_to_fastest <= 1.5s
        - Warm-up: first 2-3 laps with improving times
        - Cool-down: last 2 laps with degrading times
        - Out lap: first lap or gap_to_previous > 5s (came out of pits)
        - Race pace: everything else
        """
        if not lap_stats:
            return

        n = len(lap_stats)

        # First pass: detect warm-up pattern (first laps with improving times)
        warm_up_laps = set()
        if n >= 3:
            # Check if first 2-3 laps show improvement pattern
            improving = True
            for i in range(min(3, n - 1)):
                if i > 0 and lap_stats[i].lap_time >= lap_stats[i - 1].lap_time:
                    # Times not consistently improving
                    if i == 1:
                        improving = False
                    break
                if lap_stats[i].gap_to_fastest > 3.0:  # More than 3s off pace
                    warm_up_laps.add(lap_stats[i].lap_number)

            # If first lap is significantly slower, it's warm-up
            if lap_stats[0].gap_to_fastest > 2.0:
                warm_up_laps.add(lap_stats[0].lap_number)
                if n > 1 and lap_stats[1].gap_to_fastest > 1.5:
                    warm_up_laps.add(lap_stats[1].lap_number)

        # Second pass: detect cool-down (last 2 laps with degrading times)
        cool_down_laps = set()
        if n >= 4:
            # Check if last 2 laps are slower than the middle laps
            middle_avg = np.mean([ls.lap_time for ls in lap_stats[2:-2]]) if n > 4 else fastest_time
            last_2_avg = np.mean([ls.lap_time for ls in lap_stats[-2:]])

            if last_2_avg > middle_avg + 1.5:
                for ls in lap_stats[-2:]:
                    cool_down_laps.add(ls.lap_number)

        # Third pass: classify each lap
        for i, lap in enumerate(lap_stats):
            # Out lap: first lap or big gap from previous (pit exit)
            if i == 0 or lap.gap_to_previous > 5.0:
                lap.classification = LapClassification.OUT_LAP.value
                if i == 0:
                    lap.classification_reason = "First lap of session"
                else:
                    lap.classification_reason = f"Large gap from previous (+{lap.gap_to_previous:.1f}s)"
                continue

            # Warm-up
            if lap.lap_number in warm_up_laps:
                lap.classification = LapClassification.WARM_UP.value
                lap.classification_reason = f"Early session, {lap.gap_to_fastest:.1f}s off pace"
                continue

            # Cool-down
            if lap.lap_number in cool_down_laps:
                lap.classification = LapClassification.COOL_DOWN.value
                lap.classification_reason = "End of session, times degrading"
                continue

            # Hot lap: within 1.5s of fastest
            if lap.gap_to_fastest <= 1.5:
                lap.classification = LapClassification.HOT_LAP.value
                if lap.gap_to_fastest == 0:
                    lap.classification_reason = "Fastest lap"
                else:
                    lap.classification_reason = f"Within {lap.gap_to_fastest:.1f}s of fastest"
                continue

            # Race pace: everything else
            lap.classification = LapClassification.RACE_PACE.value
            lap.classification_reason = f"{lap.gap_to_fastest:.1f}s off pace"

    def get_lap_recommendations(self, lap_stats: List[LapStatistics]) -> Optional[LapRecommendation]:
        """
        Generate lap recommendations for analysis.

        Identifies the best laps to analyze based on patterns in the data.

        Returns:
            LapRecommendation with suggested laps and reasoning
        """
        if not lap_stats or len(lap_stats) < 2:
            return None

        # Get hot laps (top 3 by time)
        sorted_laps = sorted(lap_stats, key=lambda x: x.lap_time)
        hot_laps = [ls.lap_number for ls in sorted_laps[:min(3, len(sorted_laps))]]

        # Find best 3 consecutive lap average
        best_3_avg = float('inf')
        best_3_start = 0
        if len(lap_stats) >= 3:
            for i in range(len(lap_stats) - 2):
                avg = np.mean([lap_stats[i + j].lap_time for j in range(3)])
                if avg < best_3_avg:
                    best_3_avg = avg
                    best_3_start = i
        else:
            best_3_avg = np.mean([ls.lap_time for ls in lap_stats])

        # Detect patterns
        warm_up_detected = any(ls.classification == LapClassification.WARM_UP.value for ls in lap_stats)
        cool_down_detected = any(ls.classification == LapClassification.COOL_DOWN.value for ls in lap_stats)

        # Determine pattern type
        hot_lap_count = sum(1 for ls in lap_stats if ls.classification == LapClassification.HOT_LAP.value)
        race_pace_count = sum(1 for ls in lap_stats if ls.classification == LapClassification.RACE_PACE.value)

        if hot_lap_count >= 3:
            pattern = "consistent"
            reason = f"Consistent session with {hot_lap_count} hot laps. Focus on laps {', '.join(map(str, hot_laps))} for detailed analysis."
        elif warm_up_detected and hot_lap_count >= 1:
            pattern = "warming"
            reason = f"Warm-up pattern detected. Best performance in laps {', '.join(map(str, hot_laps))}."
        elif cool_down_detected:
            pattern = "degrading"
            reason = f"Performance degraded toward end. Analyze hot laps {', '.join(map(str, hot_laps))} from early-mid session."
        else:
            pattern = "full_session"
            reason = f"Mixed session. Recommended laps for analysis: {', '.join(map(str, hot_laps))}."

        # Best representative lap: the hot lap closest to the average of hot laps
        best_rep = hot_laps[0] if hot_laps else lap_stats[0].lap_number
        if len(hot_laps) >= 2:
            hot_lap_stats = [ls for ls in lap_stats if ls.lap_number in hot_laps]
            avg_hot = np.mean([ls.lap_time for ls in hot_lap_stats])
            best_rep = min(hot_lap_stats, key=lambda x: abs(x.lap_time - avg_hot)).lap_number

        return LapRecommendation(
            recommended_laps=hot_laps,
            reason=reason,
            pattern_detected=pattern,
            best_representative_lap=best_rep,
            best_3_lap_average=best_3_avg,
            warm_up_detected=warm_up_detected,
            cool_down_detected=cool_down_detected
        )

    def get_lap_comparison(
        self,
        report: LapAnalysisReport,
        lap_a: int,
        lap_b: int
    ) -> Dict:
        """
        Compare two laps from the same session.

        Args:
            report: LapAnalysisReport from analyze
            lap_a: First lap number
            lap_b: Second lap number

        Returns:
            Dictionary with comparison data
        """
        lap_a_stats = None
        lap_b_stats = None

        for ls in report.laps:
            if ls.lap_number == lap_a:
                lap_a_stats = ls
            if ls.lap_number == lap_b:
                lap_b_stats = ls

        if not lap_a_stats or not lap_b_stats:
            return {"error": "Lap not found"}

        return {
            "lap_a": lap_a,
            "lap_b": lap_b,
            "time_difference": lap_b_stats.lap_time - lap_a_stats.lap_time,
            "speed_difference": {
                "max": lap_b_stats.max_speed_mph - lap_a_stats.max_speed_mph,
                "avg": lap_b_stats.avg_speed_mph - lap_a_stats.avg_speed_mph
            },
            "rpm_difference": {
                "max": lap_b_stats.max_rpm - lap_a_stats.max_rpm,
                "avg": lap_b_stats.avg_rpm - lap_a_stats.avg_rpm
            },
            "faster_lap": lap_a if lap_a_stats.lap_time < lap_b_stats.lap_time else lap_b
        }

    def get_fastest_segments(
        self,
        report: LapAnalysisReport,
        top_n: int = 5
    ) -> List[Dict]:
        """
        Get the fastest lap segments.

        Args:
            report: LapAnalysisReport from analyze
            top_n: Number of fastest laps to return

        Returns:
            List of fastest laps with stats
        """
        sorted_laps = sorted(report.laps, key=lambda x: x.lap_time)

        return [
            {
                "rank": i + 1,
                "lap_number": lap.lap_number,
                "lap_time": lap.lap_time,
                "gap_to_fastest": lap.gap_to_fastest,
                "max_speed_mph": lap.max_speed_mph
            }
            for i, lap in enumerate(sorted_laps[:top_n])
        ]

    def analyze_from_channels(self, channels, session_id="unknown",
                              include_trace=False, **kwargs):
        """Analyze from pre-loaded SessionChannels."""
        trace = self._create_trace("LapAnalysis") if include_trace else None

        rpm = channels.rpm if channels.rpm is not None else np.zeros(channels.sample_count)
        report = self.analyze_from_arrays(
            time_data=channels.time,
            latitude_data=channels.latitude,
            longitude_data=channels.longitude,
            rpm_data=rpm,
            speed_data=channels.speed_mph,
            session_id=session_id,
        )

        if trace:
            trace.record_input("latitude_channel", "latitude")
            trace.record_input("longitude_channel", "longitude")
            trace.record_input("speed_channel", "speed_mph")
            trace.record_input("speed_unit_detected", channels.speed_unit_detected)
            trace.record_input("sample_count", channels.sample_count)
            trace.record_input("track_detected", self.track_name)

            trace.record_config("track_name", self.track_name)
            if self.start_finish_gps:
                trace.record_config("start_finish_gps", self.start_finish_gps)

            trace.record_intermediate("laps_detected", report.total_laps)
            trace.record_intermediate("fastest_lap_time", report.fastest_lap_time)
            if report.laps:
                trace.record_intermediate("slowest_lap_time", max(l.lap_time for l in report.laps))
            trace.record_intermediate("avg_lap_time", report.average_lap_time)
            if report.laps:
                total_dist = sum(l.distance_meters for l in report.laps)
                avg_dist_miles = (total_dist / len(report.laps)) / 1609.34
                trace.record_intermediate("estimated_lap_distance_miles", round(avg_dist_miles, 2))

            self._run_sanity_checks(trace, report, channels.latitude, channels.longitude, channels.speed_unit_detected)
            report.trace = trace

        return report


analyzer_registry.register(LapAnalysis)


@dataclass
class LapComparisonResult:
    """Result of comparing two laps"""
    lap_a: int
    lap_b: int
    time_delta: float  # Positive = lap_b slower
    speed_deltas: List[Dict]  # Speed difference at distance points
    segments: List[Dict]  # Segment-by-segment comparison
    summary: Dict

    def to_dict(self) -> dict:
        return {
            "lap_a": self.lap_a,
            "lap_b": self.lap_b,
            "time_delta": round(self.time_delta, 3),
            "faster_lap": self.lap_a if self.time_delta > 0 else self.lap_b,
            "speed_deltas": self.speed_deltas,
            "segments": self.segments,
            "summary": self.summary
        }


def compare_laps_detailed(
    parquet_path: str,
    lap_a: int,
    lap_b: int,
    num_segments: int = 10
) -> Optional[LapComparisonResult]:
    """
    Compare two laps in detail, showing where time/speed was gained or lost.

    Args:
        parquet_path: Path to the parquet file
        lap_a: First lap number (reference)
        lap_b: Second lap number (comparison)
        num_segments: Number of segments to divide the lap into

    Returns:
        LapComparisonResult with detailed comparison data
    """
    df = pd.read_parquet(parquet_path)

    # Find required columns
    time_data = df.index.values
    lat_col = None
    lon_col = None
    speed_col = None

    for col in df.columns:
        col_lower = col.lower()
        if 'latitude' in col_lower:
            lat_col = col
        elif 'longitude' in col_lower:
            lon_col = col
        elif 'speed' in col_lower and speed_col is None:
            speed_col = col

    if lat_col is None or lon_col is None:
        return None

    lat_data = df[lat_col].values
    lon_data = df[lon_col].values
    speed_data = df[speed_col].values if speed_col else np.zeros(len(time_data))

    # Convert speed to mph if needed
    if speed_data.max() < 100:
        speed_data = speed_data * SPEED_MS_TO_MPH

    # Build session data for lap analyzer
    session_data = {
        'time': time_data,
        'latitude': lat_data,
        'longitude': lon_data,
        'rpm': np.zeros(len(time_data)),
        'speed_mph': speed_data,
        'speed_ms': speed_data / SPEED_MS_TO_MPH
    }

    # Detect laps
    analyzer = LapAnalyzer(session_data)
    laps = analyzer.detect_laps()

    if not laps:
        return None

    # Find the requested laps
    lap_a_info = None
    lap_b_info = None
    for lap in laps:
        if lap.lap_number == lap_a:
            lap_a_info = lap
        if lap.lap_number == lap_b:
            lap_b_info = lap

    if lap_a_info is None or lap_b_info is None:
        return None

    # Get lap data
    lap_a_data = analyzer.get_lap_data(lap_a_info)
    lap_b_data = analyzer.get_lap_data(lap_b_info)

    # Calculate cumulative distance for each lap
    def calc_distance(lat, lon):
        """Calculate cumulative distance in meters"""
        dist = [0.0]
        for i in range(1, len(lat)):
            # Haversine approximation (simple for small distances)
            dlat = (lat[i] - lat[i-1]) * 111000  # ~111km per degree
            dlon = (lon[i] - lon[i-1]) * 111000 * np.cos(np.radians(lat[i]))
            d = np.sqrt(dlat**2 + dlon**2)
            dist.append(dist[-1] + d)
        return np.array(dist)

    dist_a = calc_distance(lap_a_data['latitude'], lap_a_data['longitude'])
    dist_b = calc_distance(lap_b_data['latitude'], lap_b_data['longitude'])

    # Normalize distance to percentage of lap
    dist_a_pct = dist_a / dist_a[-1] * 100 if dist_a[-1] > 0 else dist_a
    dist_b_pct = dist_b / dist_b[-1] * 100 if dist_b[-1] > 0 else dist_b

    # Interpolate lap B data to match lap A's distance points
    speed_a = lap_a_data.get('speed_mph', np.zeros(len(lap_a_data['time'])))
    speed_b = lap_b_data.get('speed_mph', np.zeros(len(lap_b_data['time'])))
    time_a = lap_a_data['time']
    time_b = lap_b_data['time']

    # Sample at regular distance intervals
    sample_points = np.linspace(0, 100, 50)  # 50 points along lap

    speed_deltas = []
    for pct in sample_points:
        # Find speed at this percentage for each lap
        idx_a = np.searchsorted(dist_a_pct, pct)
        idx_b = np.searchsorted(dist_b_pct, pct)

        idx_a = min(idx_a, len(speed_a) - 1)
        idx_b = min(idx_b, len(speed_b) - 1)

        spd_a = float(speed_a[idx_a]) if idx_a < len(speed_a) else 0
        spd_b = float(speed_b[idx_b]) if idx_b < len(speed_b) else 0

        speed_deltas.append({
            "distance_pct": round(pct, 1),
            "speed_a": round(spd_a, 1),
            "speed_b": round(spd_b, 1),
            "delta": round(spd_a - spd_b, 1)  # Positive = lap_a faster
        })

    # Segment analysis
    segments = []
    segment_size = 100 / num_segments

    for i in range(num_segments):
        seg_start = i * segment_size
        seg_end = (i + 1) * segment_size

        # Find time spent in this segment for each lap
        mask_a = (dist_a_pct >= seg_start) & (dist_a_pct < seg_end)
        mask_b = (dist_b_pct >= seg_start) & (dist_b_pct < seg_end)

        # Calculate time using first/last indices (more accurate than np.diff on masked arrays)
        if np.any(mask_a):
            indices_a = np.where(mask_a)[0]
            time_in_seg_a = float(time_a[indices_a[-1]] - time_a[indices_a[0]])
        else:
            time_in_seg_a = 0

        if np.any(mask_b):
            indices_b = np.where(mask_b)[0]
            time_in_seg_b = float(time_b[indices_b[-1]] - time_b[indices_b[0]])
        else:
            time_in_seg_b = 0

        # Average speed in segment
        avg_speed_a = float(np.mean(speed_a[mask_a])) if np.any(mask_a) else 0
        avg_speed_b = float(np.mean(speed_b[mask_b])) if np.any(mask_b) else 0

        time_delta = time_in_seg_b - time_in_seg_a  # Positive = lap_a faster

        segments.append({
            "segment": i + 1,
            "start_pct": round(seg_start, 1),
            "end_pct": round(seg_end, 1),
            "time_a": round(float(time_in_seg_a), 3),
            "time_b": round(float(time_in_seg_b), 3),
            "time_delta": round(float(time_delta), 3),
            "avg_speed_a": round(avg_speed_a, 1),
            "avg_speed_b": round(avg_speed_b, 1),
            "faster": "A" if time_delta > 0.05 else ("B" if time_delta < -0.05 else "=")
        })

    # Summary
    time_delta = lap_b_info.lap_time - lap_a_info.lap_time
    segments_a_faster = sum(1 for s in segments if s["faster"] == "A")
    segments_b_faster = sum(1 for s in segments if s["faster"] == "B")

    avg_speed_a = float(np.mean(speed_a))
    avg_speed_b = float(np.mean(speed_b))

    summary = {
        "lap_a_time": round(lap_a_info.lap_time, 2),
        "lap_b_time": round(lap_b_info.lap_time, 2),
        "lap_a_max_speed": round(lap_a_info.max_speed_mph, 1),
        "lap_b_max_speed": round(lap_b_info.max_speed_mph, 1),
        "lap_a_avg_speed": round(avg_speed_a, 1),
        "lap_b_avg_speed": round(avg_speed_b, 1),
        "segments_a_faster": segments_a_faster,
        "segments_b_faster": segments_b_faster,
        "segments_equal": num_segments - segments_a_faster - segments_b_faster,
        "biggest_gain_segment": max(segments, key=lambda s: s["time_delta"])["segment"] if segments else 0,
        "biggest_loss_segment": min(segments, key=lambda s: s["time_delta"])["segment"] if segments else 0
    }

    return LapComparisonResult(
        lap_a=lap_a,
        lap_b=lap_b,
        time_delta=time_delta,
        speed_deltas=speed_deltas,
        segments=segments,
        summary=summary
    )
