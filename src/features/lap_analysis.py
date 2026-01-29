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
import json

from ..analysis.lap_analyzer import LapAnalyzer, LapInfo, analyze_session_laps
from ..config.vehicle_config import TRACK_CONFIG, PROCESSING_CONFIG


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

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "session_id": self.session_id,
            "track_name": self.track_name,
            "analysis_timestamp": self.analysis_timestamp,
            "total_laps": self.total_laps,
            "fastest_lap": {
                "lap_number": self.fastest_lap_number,
                "lap_time": round(self.fastest_lap_time, 2)
            },
            "average_lap_time": round(self.average_lap_time, 2),
            "lap_time_consistency": round(self.lap_time_consistency, 2),
            "laps": [
                {
                    "lap_number": lap.lap_number,
                    "lap_time": round(lap.lap_time, 2),
                    "gap_to_fastest": round(lap.gap_to_fastest, 2),
                    "gap_to_previous": round(lap.gap_to_previous, 2),
                    "max_speed_mph": round(lap.max_speed_mph, 1),
                    "avg_speed_mph": round(lap.avg_speed_mph, 1),
                    "max_rpm": round(lap.max_rpm, 0),
                    "avg_rpm": round(lap.avg_rpm, 0)
                }
                for lap in self.laps
            ],
            "improvement_trend": self.improvement_trend,
            "recommendations": self.recommendations,
            "summary": self.summary
        }

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
            summary=summary
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
