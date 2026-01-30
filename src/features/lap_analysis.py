"""
Lap Detection and Analysis Feature
GPS-based lap detection, fastest lap identification, per-lap statistics.

Wraps the lap_analyzer module with additional analysis features.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from enum import Enum
import json

from ..analysis.lap_analyzer import LapAnalyzer, LapInfo, analyze_session_laps
from ..config.vehicle_config import TRACK_CONFIG, PROCESSING_CONFIG


def _safe_float(value: float, default: float = 0.0) -> float:
    """Convert a float to a JSON-safe value, replacing NaN/inf with default."""
    if np.isnan(value) or np.isinf(value):
        return default
    return float(value)


class LapClassification(str, Enum):
    """Classification of lap types based on timing patterns"""
    HOT_LAP = "hot_lap"
    RACE_PACE = "race_pace"
    WARM_UP = "warm_up"
    COOL_DOWN = "cool_down"
    OUT_LAP = "out_lap"
    INCOMPLETE = "incomplete"


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
class LapAnalysisReport:
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
        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)


class LapAnalysis:
    """
    Analyzes lap data from telemetry sessions.

    Detects laps, calculates statistics, identifies improvement opportunities.
    """

    def __init__(self, track_name: str = None):
        """
        Initialize lap analyzer.

        Args:
            track_name: Name of the track (default from config)
        """
        self.track_name = track_name or TRACK_CONFIG['name']
        self.start_finish_gps = TRACK_CONFIG.get('start_finish_gps')

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
            'speed_ms': speed_data / 2.237
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
        session_id: Optional[str] = None
    ) -> LapAnalysisReport:
        """
        Analyze laps from a Parquet file.

        Args:
            parquet_path: Path to Parquet file
            session_id: Session identifier (defaults to filename)

        Returns:
            LapAnalysisReport with complete analysis
        """
        df = pd.read_parquet(parquet_path)

        if session_id is None:
            from pathlib import Path
            session_id = Path(parquet_path).stem

        # Find required columns
        time_data = df.index.values
        lat_data = self._find_column(df, ['GPS Latitude', 'latitude', 'Latitude'])
        lon_data = self._find_column(df, ['GPS Longitude', 'longitude', 'Longitude'])
        rpm_data = self._find_column(df, ['RPM', 'rpm', 'RPM dup 3'])
        speed_data = self._find_column(df, ['GPS Speed', 'speed', 'Speed'])

        # Convert speed to mph if needed
        if speed_data is not None and speed_data.max() < 100:
            speed_data = speed_data * 2.237

        if lat_data is None or lon_data is None:
            raise ValueError("Parquet file missing GPS latitude/longitude columns")

        if rpm_data is None:
            rpm_data = np.zeros(len(time_data))

        if speed_data is None:
            speed_data = np.zeros(len(time_data))

        return self.analyze_from_arrays(
            time_data, lat_data, lon_data, rpm_data, speed_data, session_id
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
                analysis_timestamp=datetime.utcnow().isoformat(),
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
            analysis_timestamp=datetime.utcnow().isoformat(),
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
