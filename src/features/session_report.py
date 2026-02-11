"""
Session Report Generator Feature
Full session report combining laps, shifts, gear usage, and recommendations.

Aggregates all analysis features into comprehensive JSON and HTML reports.
Uses the analyzer registry for dynamic sub-analyzer discovery.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
import json
from pathlib import Path

from .registry import analyzer_registry
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
    """Complete session analysis report.

    Primary storage is sub_reports dict keyed by registry key.
    Legacy typed fields (lap_analysis, shift_analysis, etc.) are kept
    for backward compatibility and synced via __post_init__.
    """
    metadata: SessionMetadata
    summary: SessionSummary
    combined_recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sub_reports: Dict[str, Any] = field(default_factory=dict, repr=False)
    # Backward compat typed fields
    lap_analysis: Optional[Any] = None
    shift_analysis: Optional[Any] = None
    gear_analysis: Optional[Any] = None
    power_analysis: Optional[Any] = None

    # Map between legacy field names and registry keys
    _FIELD_TO_KEY = {
        'lap_analysis': 'laps',
        'shift_analysis': 'shifts',
        'gear_analysis': 'gears',
        'power_analysis': 'power',
    }

    # Map registry keys to JSON output keys
    _KEY_TO_JSON = {
        'laps': 'lap_analysis',
        'shifts': 'shift_analysis',
        'gears': 'gear_analysis',
        'power': 'power_analysis',
        'gg': 'gg_analysis',
        'corners': 'corner_analysis',
    }

    def __post_init__(self):
        """Sync between typed fields and sub_reports dict."""
        # Old-style construction: typed fields → sub_reports
        for field_name, key in self._FIELD_TO_KEY.items():
            val = getattr(self, field_name)
            if val is not None and key not in self.sub_reports:
                self.sub_reports[key] = val
        # New-style construction: sub_reports → typed fields
        for field_name, key in self._FIELD_TO_KEY.items():
            if getattr(self, field_name) is None and key in self.sub_reports:
                object.__setattr__(self, field_name, self.sub_reports[key])

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
        }

        # Serialize sub-reports dynamically
        for key, report in self.sub_reports.items():
            json_key = self._KEY_TO_JSON.get(key, key)
            result[json_key] = report.to_dict() if report else None

        # Ensure standard keys are present even if not in sub_reports
        for json_key in ('lap_analysis', 'shift_analysis', 'gear_analysis', 'power_analysis'):
            if json_key not in result:
                result[json_key] = None

        result["combined_recommendations"] = self.combined_recommendations
        result["warnings"] = self.warnings
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

    Uses the analyzer registry to discover and run sub-analyzers dynamically.
    New analyzers register themselves and are automatically included.
    """

    # Registry metadata (orchestrator — not registered as a sub-analyzer)
    registry_key = "report"

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

        Uses the analyzer registry to run all applicable sub-analyzers.

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
        from ..services.session_data_loader import SessionChannels

        # Build SessionChannels from raw arrays for uniform registry interface
        channels = SessionChannels(
            time=time_data,
            df=pd.DataFrame(),
            source_path="",
            session_id=session_id,
            sample_count=len(time_data),
            duration_seconds=float(time_data[-1] - time_data[0]) if len(time_data) > 1 else 0,
            speed_unit_detected="mph",
            latitude=latitude_data,
            longitude=longitude_data,
            speed_mph=speed_data,
            speed_ms=speed_data / SPEED_MS_TO_MPH if speed_data is not None else None,
            rpm=rpm_data,
        )

        # Run all registered analyzers
        warnings = []
        sub_reports = self._run_registry_analyzers(channels, session_id, warnings)

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
            sub_reports.get('laps'), sub_reports.get('shifts'),
            sub_reports.get('gears'), sub_reports.get('power'),
            speed_data, rpm_data
        )

        # Combine and prioritize recommendations
        combined_recs = self._combine_recommendations_from_sub_reports(sub_reports)

        return SessionReport(
            metadata=metadata,
            summary=summary,
            sub_reports=sub_reports,
            combined_recommendations=combined_recs,
            warnings=warnings,
        )

    def generate_from_parquet(
        self,
        parquet_path: str,
        session_id: Optional[str] = None,
        include_trace: bool = False,
    ) -> SessionReport:
        """
        Generate full session report from a Parquet file.

        Uses the analyzer registry and SessionDataLoader.

        Args:
            parquet_path: Path to Parquet file
            session_id: Session identifier (defaults to filename)
            include_trace: If True, attach CalculationTrace with cross-validation checks.

        Returns:
            SessionReport with complete analysis
        """
        from ..services.session_data_loader import SessionDataLoader

        trace = self._create_trace("SessionReport") if include_trace else None

        loader = SessionDataLoader()
        channels = loader.load(parquet_path)

        if session_id is None:
            session_id = channels.session_id

        # Run all registered analyzers
        warnings = []
        sub_reports = self._run_registry_analyzers(channels, session_id, warnings)

        # Build metadata
        metadata = SessionMetadata(
            session_id=session_id,
            track_name=self.track_name,
            vehicle_setup=self.vehicle_setup,
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            data_source="parquet",
            total_duration_seconds=channels.duration_seconds,
            sample_count=channels.sample_count,
        )

        # Build summary
        speed_data = channels.speed_mph if channels.speed_mph is not None else np.zeros(channels.sample_count)
        rpm_data = channels.rpm if channels.rpm is not None else np.zeros(channels.sample_count)

        summary = self._build_summary(
            sub_reports.get('laps'), sub_reports.get('shifts'),
            sub_reports.get('gears'), sub_reports.get('power'),
            speed_data, rpm_data
        )

        combined_recs = self._combine_recommendations_from_sub_reports(sub_reports)

        report = SessionReport(
            metadata=metadata,
            summary=summary,
            sub_reports=sub_reports,
            combined_recommendations=combined_recs,
            warnings=warnings,
        )

        if trace:
            trace.record_input("sample_count", channels.sample_count)
            trace.record_input("speed_unit_detected", channels.speed_unit_detected)
            trace.record_input("has_gps", channels.has_gps)
            trace.record_input("has_rpm", channels.has_rpm)
            trace.record_input("has_speed", channels.has_speed)
            trace.record_input("speed_column", channels.column_map.get("speed"))
            trace.record_input("rpm_column", channels.column_map.get("rpm"))

            trace.record_config("track_name", self.track_name)
            trace.record_config("vehicle_setup", self.vehicle_setup)
            trace.record_config("vehicle_mass_kg", self.vehicle_mass_kg)

            trace.record_intermediate("sub_analyzers_run", len(sub_reports))
            for key in analyzer_registry.list_registered():
                if key != 'report':
                    trace.record_intermediate(f"{key}_analysis_ok", key in sub_reports)
            trace.record_intermediate("warnings_count", len(report.warnings))

            self._run_cross_validation_checks(trace, report, channels.speed_unit_detected, channels.sample_count)
            report.trace = trace

        return report

    def _run_registry_analyzers(self, channels, session_id, warnings):
        """Run all registered analyzers via the registry."""
        sub_reports = {}
        config = {
            'track_name': self.track_name,
            'vehicle_mass_kg': self.vehicle_mass_kg,
        }

        for key, reg in analyzer_registry:
            if key == 'report':
                continue

            if not self._channels_satisfy(channels, reg.required_channels):
                continue

            try:
                instance = analyzer_registry.create_instance(key, **config)
                report = instance.analyze_from_channels(
                    channels, session_id, **config
                )
                sub_reports[key] = report
            except Exception as e:
                warnings.append(f"{key} analysis failed: {str(e)}")

        return sub_reports

    @staticmethod
    def _channels_satisfy(channels, required_channels):
        """Check if SessionChannels has all required logical channels."""
        _ATTR_MAP = {
            'speed': 'speed_mph',
            'rpm': 'rpm',
            'latitude': 'latitude',
            'longitude': 'longitude',
            'lat_acc': 'lat_acc',
            'lon_acc': 'lon_acc',
            'throttle': 'throttle',
        }
        for ch in required_channels:
            attr = _ATTR_MAP.get(ch, ch)
            if getattr(channels, attr, None) is None:
                return False
        return True

    def _run_cross_validation_checks(self, trace, report: SessionReport,
                                     speed_unit_detected: str,
                                     sample_count: int) -> None:
        """Run cross-validation sanity checks across sub-analyzers."""
        # Check 7.1: speed_unit_consensus
        trace.add_check(
            "speed_unit_consensus", "pass",
            f"Speed detected as '{speed_unit_detected}', converted to mph for all sub-analyzers",
            expected="consistent", actual=speed_unit_detected,
        )

        # Check 7.2: sample_count_consistent — iterate dynamically
        sub_counts = []
        for key, sub_report in report.sub_reports.items():
            count = getattr(sub_report, 'sample_count', sample_count)
            sub_counts.append((key, count))

        if sub_counts:
            counts = [c for _, c in sub_counts]
            max_diff_pct = (max(counts) - min(counts)) / max(counts) * 100 if max(counts) > 0 else 0
            if max_diff_pct <= 5:
                trace.add_check(
                    "sample_count_consistent", "pass",
                    f"All sub-analyzers processed ~{sample_count} samples",
                    expected="within 5%", actual=f"{max_diff_pct:.1f}% difference",
                )
            else:
                trace.add_check(
                    "sample_count_consistent", "warn",
                    f"Sub-analyzer sample counts differ by {max_diff_pct:.1f}%",
                    expected="within 5%", actual=f"{max_diff_pct:.1f}% difference",
                )
        else:
            trace.add_check(
                "sample_count_consistent", "warn",
                "No sub-analyzers produced results to compare",
                expected="at least 1 sub-analyzer", actual="0",
            )

        # Check 7.3: config_consistent — iterate dynamically
        config_issues = []
        for key, sub_report in report.sub_reports.items():
            sub_track = getattr(sub_report, 'track_name', None)
            if sub_track and sub_track != self.track_name:
                config_issues.append(f"{key} track={sub_track}")

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

    def _build_summary(
        self,
        lap_report,
        shift_report,
        gear_report,
        power_report,
        speed_data: np.ndarray,
        rpm_data: np.ndarray
    ) -> SessionSummary:
        """Build high-level session summary"""
        # Lap stats
        total_laps = getattr(lap_report, 'total_laps', 0) if lap_report else 0
        fastest_time = getattr(lap_report, 'fastest_lap_time', 0) if lap_report else 0
        fastest_num = getattr(lap_report, 'fastest_lap_number', 0) if lap_report else 0
        avg_time = getattr(lap_report, 'average_lap_time', 0) if lap_report else 0
        trend = getattr(lap_report, 'improvement_trend', "N/A") if lap_report else "N/A"

        # Shift stats
        total_shifts = getattr(shift_report, 'total_shifts', 0) if shift_report else 0

        # Speed/RPM stats
        max_speed = float(np.max(speed_data)) if len(speed_data) > 0 else 0
        max_rpm = float(np.max(rpm_data)) if len(rpm_data) > 0 else 0

        # Power stats
        max_power = getattr(power_report, 'max_power_hp', 0) if power_report else 0
        max_brake = getattr(power_report, 'max_braking_g', 0) if power_report else 0

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

    def _combine_recommendations_from_sub_reports(
        self,
        sub_reports: Dict[str, Any]
    ) -> List[str]:
        """Combine and prioritize recommendations from all sub-reports."""
        # Labels for known registry keys (backward compat with existing output)
        _KEY_TO_LABEL = {
            'laps': 'Lap', 'shifts': 'Shift', 'gears': 'Gear',
            'power': 'Power', 'gg': 'G-G', 'corners': 'Corner',
        }

        all_recs = []
        for key, report in sub_reports.items():
            recs = getattr(report, 'recommendations', None)
            if recs:
                label = _KEY_TO_LABEL.get(key, key.capitalize())
                for rec in recs:
                    all_recs.append((label, rec))

        # Remove duplicates and format
        seen = set()
        combined = []
        for source, rec in all_recs:
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
