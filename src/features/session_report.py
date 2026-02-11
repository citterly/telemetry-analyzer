"""
Session Report Generator Feature
Full session report combining laps, shifts, gear usage, and recommendations.

Aggregates all analysis features into comprehensive JSON and HTML reports.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
import json
from pathlib import Path

from .lap_analysis import LapAnalysis, LapAnalysisReport
from .shift_analysis import ShiftAnalyzer, ShiftReport
from .gear_analysis import GearAnalysis, GearAnalysisReport
from .power_analysis import PowerAnalysis, PowerAnalysisReport
from ..config.tracks import get_track_config as _get_track_config
from ..config.vehicles import get_current_setup as _get_current_setup
from ..utils.dataframe_helpers import find_column, SPEED_MS_TO_MPH
from .base_analyzer import BaseAnalyzer, BaseAnalysisReport


@dataclass
class SessionMetadata:
    """Metadata about the analyzed session"""
    session_id: str
    track_name: str
    vehicle_setup: str
    analysis_timestamp: str
    data_source: str
    total_duration_seconds: float
    sample_count: int


@dataclass
class SessionSummary:
    """High-level session summary"""
    total_laps: int
    fastest_lap_time: float
    fastest_lap_number: int
    average_lap_time: float
    total_shifts: int
    max_speed_mph: float
    max_rpm: float
    max_power_hp: float
    max_braking_g: float
    improvement_trend: str


@dataclass
class SessionReport(BaseAnalysisReport):
    """Complete session analysis report"""
    metadata: SessionMetadata
    summary: SessionSummary
    lap_analysis: Optional[LapAnalysisReport]
    shift_analysis: Optional[ShiftReport]
    gear_analysis: Optional[GearAnalysisReport]
    power_analysis: Optional[PowerAnalysisReport]
    combined_recommendations: List[str]
    warnings: List[str]

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        result = {
            "metadata": {
                "session_id": self.metadata.session_id,
                "track_name": self.metadata.track_name,
                "vehicle_setup": self.metadata.vehicle_setup,
                "analysis_timestamp": self.metadata.analysis_timestamp,
                "data_source": self.metadata.data_source,
                "total_duration_seconds": round(self.metadata.total_duration_seconds, 1),
                "sample_count": self.metadata.sample_count
            },
            "summary": {
                "total_laps": self.summary.total_laps,
                "fastest_lap": {
                    "lap_number": self.summary.fastest_lap_number,
                    "lap_time": round(self.summary.fastest_lap_time, 2)
                },
                "average_lap_time": round(self.summary.average_lap_time, 2),
                "total_shifts": self.summary.total_shifts,
                "max_speed_mph": round(self.summary.max_speed_mph, 1),
                "max_rpm": round(self.summary.max_rpm, 0),
                "max_power_hp": round(self.summary.max_power_hp, 1),
                "max_braking_g": round(self.summary.max_braking_g, 2),
                "improvement_trend": self.summary.improvement_trend
            },
            "lap_analysis": self.lap_analysis.to_dict() if self.lap_analysis else None,
            "shift_analysis": self.shift_analysis.to_dict() if self.shift_analysis else None,
            "gear_analysis": self.gear_analysis.to_dict() if self.gear_analysis else None,
            "power_analysis": self.power_analysis.to_dict() if self.power_analysis else None,
            "combined_recommendations": self.combined_recommendations,
            "warnings": self.warnings
        }
        result.update(self._trace_dict())
        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)

    def to_html(self) -> str:
        """Convert to HTML report"""
        return SessionReportGenerator._generate_html(self)


class SessionReportGenerator(BaseAnalyzer):
    """
    Generates comprehensive session analysis reports.

    Combines lap analysis, shift analysis, gear usage, and power analysis
    into a unified report with cross-referenced recommendations.
    """

    def __init__(
        self,
        track_name: str = None,
        vehicle_setup: str = None,
        vehicle_mass_kg: float = 1565
    ):
        """
        Initialize report generator.

        Args:
            track_name: Name of the track (default from config)
            vehicle_setup: Name of vehicle setup (default from config)
            vehicle_mass_kg: Vehicle mass for power calculations
        """
        self.track_name = track_name or _get_track_config()['name']
        self.vehicle_setup = vehicle_setup or _get_current_setup()['name']
        self.vehicle_mass_kg = vehicle_mass_kg

        # Initialize analyzers
        self.lap_analyzer = LapAnalysis(track_name=self.track_name)
        self.shift_analyzer = ShiftAnalyzer()
        self.gear_analyzer = GearAnalysis(track_name=self.track_name)
        self.power_analyzer = PowerAnalysis(vehicle_mass_kg=vehicle_mass_kg)

    def generate_from_arrays(
        self,
        time_data: np.ndarray,
        latitude_data: np.ndarray,
        longitude_data: np.ndarray,
        rpm_data: np.ndarray,
        speed_data: np.ndarray,
        session_id: str = "unknown"
    ) -> SessionReport:
        """
        Generate full session report from raw data arrays.

        Args:
            time_data: Array of timestamps (seconds)
            latitude_data: GPS latitude values
            longitude_data: GPS longitude values
            rpm_data: Engine RPM values
            speed_data: Speed values (mph)
            session_id: Session identifier

        Returns:
            SessionReport with complete analysis
        """
        warnings = []

        # Run all analyses
        lap_report = self._run_lap_analysis(
            time_data, latitude_data, longitude_data, rpm_data, speed_data, session_id, warnings
        )

        shift_report = self._run_shift_analysis(
            time_data, rpm_data, speed_data, session_id, warnings
        )

        gear_report = self._run_gear_analysis(
            time_data, rpm_data, speed_data, latitude_data, longitude_data, session_id, warnings
        )

        power_report = self._run_power_analysis(
            time_data, speed_data, rpm_data, session_id, warnings
        )

        # Build metadata
        metadata = SessionMetadata(
            session_id=session_id,
            track_name=self.track_name,
            vehicle_setup=self.vehicle_setup,
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            data_source="array",
            total_duration_seconds=float(time_data[-1] - time_data[0]) if len(time_data) > 1 else 0,
            sample_count=len(time_data)
        )

        # Build summary
        summary = self._build_summary(
            lap_report, shift_report, gear_report, power_report, speed_data, rpm_data
        )

        # Combine and prioritize recommendations
        combined_recs = self._combine_recommendations(
            lap_report, shift_report, gear_report, power_report
        )

        return SessionReport(
            metadata=metadata,
            summary=summary,
            lap_analysis=lap_report,
            shift_analysis=shift_report,
            gear_analysis=gear_report,
            power_analysis=power_report,
            combined_recommendations=combined_recs,
            warnings=warnings
        )

    def generate_from_parquet(
        self,
        parquet_path: str,
        session_id: Optional[str] = None,
        include_trace: bool = False,
    ) -> SessionReport:
        """
        Generate full session report from a Parquet file.

        Args:
            parquet_path: Path to Parquet file
            session_id: Session identifier (defaults to filename)
            include_trace: If True, attach CalculationTrace with cross-validation checks.

        Returns:
            SessionReport with complete analysis
        """
        trace = self._create_trace("SessionReport") if include_trace else None

        df = pd.read_parquet(parquet_path)

        if session_id is None:
            session_id = Path(parquet_path).stem

        # Find required columns
        from ..utils.dataframe_helpers import find_column_name
        time_data = df.index.values
        lat_data = find_column(df, ['GPS Latitude', 'latitude', 'Latitude'])
        lon_data = find_column(df, ['GPS Longitude', 'longitude', 'Longitude'])
        rpm_data = find_column(df, ['RPM', 'rpm', 'RPM dup 3'])
        speed_data = find_column(df, ['GPS Speed', 'speed', 'Speed'])

        speed_unit_detected = "mph"
        # Convert speed to mph if needed
        if speed_data is not None and speed_data.max() < 100:
            speed_data = speed_data * SPEED_MS_TO_MPH
            speed_unit_detected = "m/s"
        elif speed_data is None:
            speed_unit_detected = "none"

        # Use zeros for missing GPS data
        has_gps = lat_data is not None and lon_data is not None
        has_rpm = rpm_data is not None
        has_speed = speed_data is not None
        if lat_data is None:
            lat_data = np.zeros(len(time_data))
        if lon_data is None:
            lon_data = np.zeros(len(time_data))
        if rpm_data is None:
            rpm_data = np.zeros(len(time_data))
        if speed_data is None:
            speed_data = np.zeros(len(time_data))

        report = self.generate_from_arrays(
            time_data, lat_data, lon_data, rpm_data, speed_data, session_id
        )

        if trace:
            trace.record_input("sample_count", len(time_data))
            trace.record_input("speed_unit_detected", speed_unit_detected)
            trace.record_input("has_gps", has_gps)
            trace.record_input("has_rpm", has_rpm)
            trace.record_input("has_speed", has_speed)
            trace.record_input("speed_column", find_column_name(df, ['GPS Speed', 'speed', 'Speed']))
            trace.record_input("rpm_column", find_column_name(df, ['RPM', 'rpm', 'RPM dup 3']))

            trace.record_config("track_name", self.track_name)
            trace.record_config("vehicle_setup", self.vehicle_setup)
            trace.record_config("vehicle_mass_kg", self.vehicle_mass_kg)

            trace.record_intermediate("sub_analyzers_run", 4)
            trace.record_intermediate("lap_analysis_ok", report.lap_analysis is not None)
            trace.record_intermediate("shift_analysis_ok", report.shift_analysis is not None)
            trace.record_intermediate("gear_analysis_ok", report.gear_analysis is not None)
            trace.record_intermediate("power_analysis_ok", report.power_analysis is not None)
            trace.record_intermediate("warnings_count", len(report.warnings))

            self._run_cross_validation_checks(trace, report, speed_unit_detected, len(time_data))
            report.trace = trace

        return report

    def _run_cross_validation_checks(self, trace, report: SessionReport,
                                     speed_unit_detected: str,
                                     sample_count: int) -> None:
        """Run cross-validation sanity checks across sub-analyzers."""
        # Check 7.1: speed_unit_consensus
        # SessionReportGenerator converts speed once and passes to all sub-analyzers,
        # so by construction they all use the same unit. Record for documentation.
        trace.add_check(
            "speed_unit_consensus", "pass",
            f"Speed detected as '{speed_unit_detected}', converted to mph for all sub-analyzers",
            expected="consistent", actual=speed_unit_detected,
        )

        # Check 7.2: sample_count_consistent
        # All sub-analyzers receive the same arrays, so sample count is consistent.
        # Verify none of the sub-reports show drastically different counts.
        sub_counts = []
        if report.lap_analysis is not None:
            sub_counts.append(("lap", getattr(report.lap_analysis, 'sample_count', sample_count)))
        if report.shift_analysis is not None:
            sub_counts.append(("shift", getattr(report.shift_analysis, 'sample_count', sample_count)))
        if report.gear_analysis is not None:
            sub_counts.append(("gear", sample_count))  # GearAnalysis doesn't expose sample_count
        if report.power_analysis is not None:
            sub_counts.append(("power", getattr(report.power_analysis, 'sample_count', sample_count)))

        if sub_counts:
            counts = [c for _, c in sub_counts]
            max_diff_pct = (max(counts) - min(counts)) / max(counts) * 100 if max(counts) > 0 else 0
            if max_diff_pct <= 5:
                trace.add_check(
                    "sample_count_consistent", "pass",
                    f"All sub-analyzers processed ~{sample_count} samples",
                    expected=f"within 5%", actual=f"{max_diff_pct:.1f}% difference",
                )
            else:
                trace.add_check(
                    "sample_count_consistent", "warn",
                    f"Sub-analyzer sample counts differ by {max_diff_pct:.1f}%",
                    expected=f"within 5%", actual=f"{max_diff_pct:.1f}% difference",
                )
        else:
            trace.add_check(
                "sample_count_consistent", "warn",
                "No sub-analyzers produced results to compare",
                expected="at least 1 sub-analyzer", actual="0",
            )

        # Check 7.3: config_consistent
        # All sub-analyzers are created by this generator with same config.
        # Verify track_name matches across sub-reports.
        config_issues = []
        if report.lap_analysis is not None:
            lap_track = getattr(report.lap_analysis, 'track_name', None)
            if lap_track and lap_track != self.track_name:
                config_issues.append(f"lap track={lap_track}")
        if report.gear_analysis is not None:
            gear_track = getattr(report.gear_analysis, 'track_name', None)
            if gear_track and gear_track != self.track_name:
                config_issues.append(f"gear track={gear_track}")

        if not config_issues:
            trace.add_check(
                "config_consistent", "pass",
                f"All sub-analyzers using consistent config (track={self.track_name})",
                expected="consistent", actual="consistent",
            )
        else:
            trace.add_check(
                "config_consistent", "fail",
                f"Config mismatch: {', '.join(config_issues)}",
                expected=f"track={self.track_name}", actual=str(config_issues),
                severity="error",
            )

    def analyze_from_parquet(
        self,
        parquet_path: str,
        session_id: Optional[str] = None,
        include_trace: bool = False,
        **kwargs,
    ) -> SessionReport:
        """BaseAnalyzer interface - delegates to generate_from_parquet."""
        return self.generate_from_parquet(parquet_path, session_id, include_trace=include_trace)

    def _run_lap_analysis(
        self,
        time_data, lat_data, lon_data, rpm_data, speed_data, session_id, warnings
    ) -> Optional[LapAnalysisReport]:
        """Run lap analysis with error handling"""
        try:
            return self.lap_analyzer.analyze_from_arrays(
                time_data, lat_data, lon_data, rpm_data, speed_data, session_id
            )
        except Exception as e:
            warnings.append(f"Lap analysis failed: {str(e)}")
            return None

    def _run_shift_analysis(
        self,
        time_data, rpm_data, speed_data, session_id, warnings
    ) -> Optional[ShiftReport]:
        """Run shift analysis with error handling"""
        try:
            return self.shift_analyzer.analyze_from_arrays(
                time_data, rpm_data, speed_data, session_id
            )
        except Exception as e:
            warnings.append(f"Shift analysis failed: {str(e)}")
            return None

    def _run_gear_analysis(
        self,
        time_data, rpm_data, speed_data, lat_data, lon_data, session_id, warnings
    ) -> Optional[GearAnalysisReport]:
        """Run gear analysis with error handling"""
        try:
            return self.gear_analyzer.analyze_from_arrays(
                time_data, rpm_data, speed_data, lat_data, lon_data, session_id
            )
        except Exception as e:
            warnings.append(f"Gear analysis failed: {str(e)}")
            return None

    def _run_power_analysis(
        self,
        time_data, speed_data, rpm_data, session_id, warnings
    ) -> Optional[PowerAnalysisReport]:
        """Run power analysis with error handling"""
        try:
            return self.power_analyzer.analyze_from_arrays(
                time_data, speed_data, rpm_data, session_id
            )
        except Exception as e:
            warnings.append(f"Power analysis failed: {str(e)}")
            return None

    def _build_summary(
        self,
        lap_report: Optional[LapAnalysisReport],
        shift_report: Optional[ShiftReport],
        gear_report: Optional[GearAnalysisReport],
        power_report: Optional[PowerAnalysisReport],
        speed_data: np.ndarray,
        rpm_data: np.ndarray
    ) -> SessionSummary:
        """Build high-level session summary"""
        # Lap stats
        total_laps = lap_report.total_laps if lap_report else 0
        fastest_time = lap_report.fastest_lap_time if lap_report else 0
        fastest_num = lap_report.fastest_lap_number if lap_report else 0
        avg_time = lap_report.average_lap_time if lap_report else 0
        trend = lap_report.improvement_trend if lap_report else "N/A"

        # Shift stats
        total_shifts = shift_report.total_shifts if shift_report else 0

        # Speed/RPM stats
        max_speed = float(np.max(speed_data)) if len(speed_data) > 0 else 0
        max_rpm = float(np.max(rpm_data)) if len(rpm_data) > 0 else 0

        # Power stats
        max_power = power_report.max_power_hp if power_report else 0
        max_brake = power_report.max_braking_g if power_report else 0

        return SessionSummary(
            total_laps=total_laps,
            fastest_lap_time=fastest_time,
            fastest_lap_number=fastest_num,
            average_lap_time=avg_time,
            total_shifts=total_shifts,
            max_speed_mph=max_speed,
            max_rpm=max_rpm,
            max_power_hp=max_power,
            max_braking_g=max_brake,
            improvement_trend=trend
        )

    def _combine_recommendations(
        self,
        lap_report: Optional[LapAnalysisReport],
        shift_report: Optional[ShiftReport],
        gear_report: Optional[GearAnalysisReport],
        power_report: Optional[PowerAnalysisReport]
    ) -> List[str]:
        """Combine and prioritize recommendations from all analyses"""
        all_recs = []

        # Collect all recommendations with source
        if lap_report and lap_report.recommendations:
            for rec in lap_report.recommendations:
                all_recs.append(("Lap", rec))

        if shift_report and shift_report.recommendations:
            for rec in shift_report.recommendations:
                all_recs.append(("Shift", rec))

        if gear_report and gear_report.recommendations:
            for rec in gear_report.recommendations:
                all_recs.append(("Gear", rec))

        if power_report and power_report.recommendations:
            for rec in power_report.recommendations:
                all_recs.append(("Power", rec))

        # Remove duplicates and format
        seen = set()
        combined = []

        for source, rec in all_recs:
            # Simple deduplication - check for similar content
            rec_key = rec.lower()[:50]
            if rec_key not in seen:
                seen.add(rec_key)
                combined.append(f"[{source}] {rec}")

        # Prioritize safety-related recommendations
        safety_keywords = ['rpm', 'limit', 'over-rev', 'danger', 'safe']
        prioritized = []
        others = []

        for rec in combined:
            if any(kw in rec.lower() for kw in safety_keywords):
                prioritized.append(rec)
            else:
                others.append(rec)

        return prioritized + others

    @staticmethod
    def _generate_html(report: SessionReport) -> str:
        """Generate HTML report"""
        summary = report.summary
        metadata = report.metadata

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Session Report - {metadata.session_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: #1a1a2e; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .card h2 {{ margin-top: 0; color: #1a1a2e; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; }}
        .stat {{ background: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #1a1a2e; }}
        .stat-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .rec-list {{ list-style: none; padding: 0; }}
        .rec-list li {{ padding: 10px; margin: 5px 0; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px; }}
        .rec-list li.safety {{ background: #f8d7da; border-left-color: #dc3545; }}
        .warning {{ background: #f8d7da; color: #721c24; padding: 10px; border-radius: 4px; margin-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{metadata.track_name} Session Report</h1>
            <p>Session: {metadata.session_id} | Setup: {metadata.vehicle_setup} | Generated: {metadata.analysis_timestamp[:19]}</p>
        </div>

        <div class="card">
            <h2>Session Summary</h2>
            <div class="stat-grid">
                <div class="stat">
                    <div class="stat-value">{summary.total_laps}</div>
                    <div class="stat-label">Total Laps</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{summary.fastest_lap_time:.2f}s</div>
                    <div class="stat-label">Fastest Lap (#{summary.fastest_lap_number})</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{summary.average_lap_time:.2f}s</div>
                    <div class="stat-label">Average Lap</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{summary.max_speed_mph:.1f}</div>
                    <div class="stat-label">Max Speed (mph)</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{summary.max_rpm:.0f}</div>
                    <div class="stat-label">Max RPM</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{summary.max_power_hp:.0f}</div>
                    <div class="stat-label">Est. Max Power (HP)</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{summary.max_braking_g:.2f}g</div>
                    <div class="stat-label">Max Braking</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{summary.total_shifts}</div>
                    <div class="stat-label">Total Shifts</div>
                </div>
            </div>
            <p style="margin-top: 15px;"><strong>Improvement Trend:</strong> {summary.improvement_trend}</p>
        </div>
"""

        # Warnings section
        if report.warnings:
            html += """
        <div class="card">
            <h2>Warnings</h2>
"""
            for warning in report.warnings:
                html += f'            <div class="warning">{warning}</div>\n'
            html += "        </div>\n"

        # Recommendations section
        html += """
        <div class="card">
            <h2>Recommendations</h2>
            <ul class="rec-list">
"""
        for rec in report.combined_recommendations:
            css_class = "safety" if any(kw in rec.lower() for kw in ['rpm', 'limit', 'danger']) else ""
            html += f'                <li class="{css_class}">{rec}</li>\n'

        html += """            </ul>
        </div>
"""

        # Lap times table
        if report.lap_analysis and report.lap_analysis.laps:
            html += """
        <div class="card">
            <h2>Lap Times</h2>
            <table>
                <tr>
                    <th>Lap</th>
                    <th>Time</th>
                    <th>Gap to Fastest</th>
                    <th>Max Speed</th>
                    <th>Max RPM</th>
                </tr>
"""
            for lap in report.lap_analysis.laps:
                html += f"""                <tr>
                    <td>{lap.lap_number}</td>
                    <td>{lap.lap_time:.2f}s</td>
                    <td>+{lap.gap_to_fastest:.2f}s</td>
                    <td>{lap.max_speed_mph:.1f} mph</td>
                    <td>{lap.max_rpm:.0f}</td>
                </tr>
"""
            html += """            </table>
        </div>
"""

        # Gear usage table
        if report.gear_analysis and report.gear_analysis.gear_usage:
            html += """
        <div class="card">
            <h2>Gear Usage</h2>
            <table>
                <tr>
                    <th>Gear</th>
                    <th>Time</th>
                    <th>Usage %</th>
                    <th>Speed Range</th>
                    <th>RPM Range</th>
                </tr>
"""
            for gu in report.gear_analysis.gear_usage:
                html += f"""                <tr>
                    <td>{gu.gear_number}</td>
                    <td>{gu.time_seconds:.1f}s</td>
                    <td>{gu.usage_percent:.1f}%</td>
                    <td>{gu.speed_min_mph:.0f} - {gu.speed_max_mph:.0f} mph</td>
                    <td>{gu.rpm_min:.0f} - {gu.rpm_max:.0f}</td>
                </tr>
"""
            html += """            </table>
        </div>
"""

        html += """
    </div>
</body>
</html>"""

        return html

    def save_report(
        self,
        report: SessionReport,
        output_dir: str,
        formats: List[str] = None
    ) -> Dict[str, str]:
        """
        Save report to files.

        Args:
            report: SessionReport to save
            output_dir: Directory to save files
            formats: List of formats ('json', 'html'). Defaults to both.

        Returns:
            Dictionary mapping format to output file path
        """
        if formats is None:
            formats = ['json', 'html']

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}
        session_id = report.metadata.session_id

        if 'json' in formats:
            json_path = output_path / f"{session_id}_report.json"
            with open(json_path, 'w') as f:
                f.write(report.to_json())
            saved_files['json'] = str(json_path)

        if 'html' in formats:
            html_path = output_path / f"{session_id}_report.html"
            with open(html_path, 'w') as f:
                f.write(report.to_html())
            saved_files['html'] = str(html_path)

        return saved_files
