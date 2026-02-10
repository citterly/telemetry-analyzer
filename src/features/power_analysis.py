"""
Acceleration and Power Analysis Feature
Calculate acceleration from speed data, estimate power output, braking analysis.

Provides physics-based power estimation using F=ma*v.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
import json
from scipy import signal
from .base_analyzer import BaseAnalyzer, BaseAnalysisReport

from ..config.vehicle_config import ENGINE_SPECS
from ..utils.dataframe_helpers import find_column, SPEED_MS_TO_MPH


@dataclass
class AccelerationEvent:
    """Represents a single acceleration or braking event"""
    event_type: str  # 'acceleration' or 'braking'
    start_time: float
    end_time: float
    duration_seconds: float
    start_speed_mph: float
    end_speed_mph: float
    speed_change_mph: float
    peak_acceleration_g: float
    avg_acceleration_g: float
    peak_power_hp: float  # Only for acceleration events
    avg_power_hp: float  # Only for acceleration events


@dataclass
class PowerEstimate:
    """Power estimate at a specific point"""
    time: float
    speed_mph: float
    rpm: float
    acceleration_g: float
    power_hp: float
    in_power_band: bool


@dataclass
class PowerAnalysisReport(BaseAnalysisReport):
    """Complete power and acceleration analysis report"""
    session_id: str
    analysis_timestamp: str
    vehicle_mass_kg: float
    total_duration_seconds: float

    # Power statistics
    max_power_hp: float
    avg_power_hp: float
    power_at_peak_rpm: float

    # Acceleration statistics
    max_acceleration_g: float
    avg_acceleration_g: float
    acceleration_events: List[AccelerationEvent]

    # Braking statistics
    max_braking_g: float
    avg_braking_g: float
    braking_events: List[AccelerationEvent]

    # RPM analysis
    rpm_analysis: Dict

    # Recommendations
    recommendations: List[str]
    summary: Dict

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        result = {
            "session_id": self.session_id,
            "analysis_timestamp": self.analysis_timestamp,
            "vehicle_mass_kg": self.vehicle_mass_kg,
            "total_duration_seconds": round(self.total_duration_seconds, 1),
            "power": {
                "max_hp": round(self.max_power_hp, 1),
                "avg_hp": round(self.avg_power_hp, 1),
                "at_peak_rpm_hp": round(self.power_at_peak_rpm, 1)
            },
            "acceleration": {
                "max_g": round(self.max_acceleration_g, 3),
                "avg_g": round(self.avg_acceleration_g, 3),
                "events_count": len(self.acceleration_events),
                "events": [
                    {
                        "start_time": round(e.start_time, 1),
                        "duration_s": round(e.duration_seconds, 1),
                        "speed_change_mph": round(e.speed_change_mph, 1),
                        "peak_g": round(e.peak_acceleration_g, 3),
                        "peak_power_hp": round(e.peak_power_hp, 1)
                    }
                    for e in self.acceleration_events[:10]  # Limit to top 10
                ]
            },
            "braking": {
                "max_g": round(self.max_braking_g, 3),
                "avg_g": round(self.avg_braking_g, 3),
                "events_count": len(self.braking_events),
                "events": [
                    {
                        "start_time": round(e.start_time, 1),
                        "duration_s": round(e.duration_seconds, 1),
                        "speed_change_mph": round(e.speed_change_mph, 1),
                        "peak_g": round(e.peak_acceleration_g, 3)
                    }
                    for e in self.braking_events[:10]  # Limit to top 10
                ]
            },
            "rpm_analysis": {
                k: round(v, 2) if isinstance(v, float) else v
                for k, v in self.rpm_analysis.items()
            },
            "recommendations": self.recommendations,
            "summary": self.summary
        }
        result.update(self._trace_dict())
        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)


class PowerAnalysis(BaseAnalyzer):
    """
    Analyzes acceleration, power output, and braking from telemetry data.

    Uses physics-based calculations: P = F * v = m * a * v
    """

    def __init__(
        self,
        vehicle_mass_kg: float = 1565,  # Default ~3450 lbs
        min_accel_threshold_g: float = 0.1,
        min_brake_threshold_g: float = 0.15,
        smoothing_window: int = 11
    ):
        """
        Initialize power analyzer.

        Args:
            vehicle_mass_kg: Vehicle mass in kg (default 3450 lbs = 1565 kg)
            min_accel_threshold_g: Minimum g-force to count as acceleration event
            min_brake_threshold_g: Minimum g-force to count as braking event
            smoothing_window: Window size for Savitzky-Golay filter
        """
        self.vehicle_mass_kg = vehicle_mass_kg
        self.min_accel_threshold = min_accel_threshold_g
        self.min_brake_threshold = min_brake_threshold_g
        self.smoothing_window = smoothing_window

    def analyze_from_arrays(
        self,
        time_data: np.ndarray,
        speed_data: np.ndarray,
        rpm_data: np.ndarray = None,
        session_id: str = "unknown"
    ) -> PowerAnalysisReport:
        """
        Analyze power and acceleration from raw data arrays.

        Args:
            time_data: Array of timestamps (seconds)
            speed_data: Speed values (mph)
            rpm_data: Engine RPM values (optional)
            session_id: Session identifier

        Returns:
            PowerAnalysisReport with complete analysis
        """
        # Convert speed to m/s for calculations
        speed_ms = speed_data / SPEED_MS_TO_MPH

        # Calculate acceleration
        accel_ms2, accel_g = self._calculate_acceleration(time_data, speed_ms)

        # Calculate power estimates
        power_estimates = self._calculate_power(
            time_data, speed_data, speed_ms, accel_ms2, rpm_data
        )

        # Find acceleration events
        accel_events = self._find_acceleration_events(
            time_data, speed_data, accel_g, power_estimates
        )

        # Find braking events
        brake_events = self._find_braking_events(
            time_data, speed_data, accel_g
        )

        # Analyze RPM usage
        rpm_analysis = self._analyze_rpm(rpm_data, speed_data, accel_g) if rpm_data is not None else {}

        # Calculate statistics
        total_duration = float(time_data[-1] - time_data[0]) if len(time_data) > 1 else 0

        # Power stats (only from positive acceleration periods)
        accel_power = [p.power_hp for p in power_estimates if p.power_hp > 0]
        max_power = max(accel_power) if accel_power else 0
        avg_power = float(np.mean(accel_power)) if accel_power else 0

        # Power at peak RPM
        if rpm_data is not None and len(power_estimates) > 0:
            peak_rpm_idx = np.argmax(rpm_data)
            power_at_peak = power_estimates[min(peak_rpm_idx, len(power_estimates)-1)].power_hp
        else:
            power_at_peak = 0

        # Acceleration stats
        positive_accel = accel_g[accel_g > 0]
        max_accel = float(max(positive_accel)) if len(positive_accel) > 0 else 0
        avg_accel = float(np.mean(positive_accel)) if len(positive_accel) > 0 else 0

        # Braking stats
        negative_accel = accel_g[accel_g < 0]
        max_brake = float(abs(min(negative_accel))) if len(negative_accel) > 0 else 0
        avg_brake = float(abs(np.mean(negative_accel))) if len(negative_accel) > 0 else 0

        # Generate recommendations
        recommendations = self._generate_recommendations(
            max_power, accel_events, brake_events, rpm_analysis
        )

        # Build summary
        summary = self._build_summary(
            power_estimates, accel_events, brake_events, total_duration
        )

        return PowerAnalysisReport(
            session_id=session_id,
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            vehicle_mass_kg=self.vehicle_mass_kg,
            total_duration_seconds=total_duration,
            max_power_hp=max_power,
            avg_power_hp=avg_power,
            power_at_peak_rpm=power_at_peak,
            max_acceleration_g=max_accel,
            avg_acceleration_g=avg_accel,
            acceleration_events=accel_events,
            max_braking_g=max_brake,
            avg_braking_g=avg_brake,
            braking_events=brake_events,
            rpm_analysis=rpm_analysis,
            recommendations=recommendations,
            summary=summary
        )

    def analyze_from_parquet(
        self,
        parquet_path: str,
        session_id: Optional[str] = None,
        include_trace: bool = False,
        **kwargs,
    ) -> PowerAnalysisReport:
        """
        Analyze power and acceleration from a Parquet file.

        Args:
            parquet_path: Path to Parquet file
            session_id: Session identifier (defaults to filename)
            include_trace: If True, attach CalculationTrace to report.

        Returns:
            PowerAnalysisReport with complete analysis
        """
        trace = self._create_trace("PowerAnalysis") if include_trace else None

        df = pd.read_parquet(parquet_path)

        if session_id is None:
            from pathlib import Path
            session_id = Path(parquet_path).stem

        # Find required columns
        time_data = df.index.values
        speed_col_name = None
        for col_name in ['GPS Speed', 'speed', 'Speed']:
            if col_name in df.columns:
                speed_col_name = col_name
                break
        speed_data = find_column(df, ['GPS Speed', 'speed', 'Speed'])
        rpm_col_name = None
        for col_name in ['RPM', 'rpm', 'RPM dup 3']:
            if col_name in df.columns:
                rpm_col_name = col_name
                break
        rpm_data = find_column(df, ['RPM', 'rpm', 'RPM dup 3'])

        # Record trace inputs
        speed_unit_detected = "mph"
        speed_max_raw = float(speed_data.max()) if speed_data is not None else 0.0

        # Convert speed to mph if needed (likely in m/s or km/h)
        if speed_data is not None and speed_data.max() < 100:
            speed_data = speed_data * SPEED_MS_TO_MPH  # m/s to mph
            speed_unit_detected = "m/s"

        if speed_data is None:
            raise ValueError("Parquet file missing speed column")

        if trace:
            trace.record_input("speed_column", speed_col_name)
            trace.record_input("speed_unit_detected", speed_unit_detected)
            trace.record_input("speed_max_raw", speed_max_raw)
            trace.record_input("rpm_column", rpm_col_name)
            trace.record_input("sample_count", len(time_data))
            dt = np.diff(time_data)
            trace.record_input("dt_mean", float(np.mean(dt)) if len(dt) > 0 else 0.0)

            trace.record_config("vehicle_mass_kg", self.vehicle_mass_kg)
            trace.record_config("smoothing_window", self.smoothing_window)
            trace.record_config("smoothing_polyorder", 3)
            trace.record_config("power_band_min_rpm", ENGINE_SPECS.get('power_band_min', 5500))
            trace.record_config("power_band_max_rpm", ENGINE_SPECS.get('power_band_max', 7000))
            trace.record_config("safe_rpm_limit", ENGINE_SPECS.get('safe_rpm_limit', 7000))

        report = self.analyze_from_arrays(time_data, speed_data, rpm_data, session_id)

        if trace:
            # Record key intermediates
            trace.record_intermediate("max_raw_power_hp", report.max_power_hp)
            trace.record_intermediate("max_raw_accel_g", report.max_acceleration_g)
            trace.record_intermediate("accel_event_count", len(report.acceleration_events))
            trace.record_intermediate("braking_event_count", len(report.braking_events))
            if report.rpm_analysis and "pct_in_power_band" in report.rpm_analysis:
                trace.record_intermediate("pct_in_power_band", report.rpm_analysis["pct_in_power_band"])

            if rpm_data is None:
                trace.warnings.append("RPM column missing, RPM analysis skipped")

            # Run sanity checks
            self._run_sanity_checks(trace, report)

            report.trace = trace

        return report

    def _run_sanity_checks(self, trace, report: PowerAnalysisReport) -> None:
        """Run sanity checks on power analysis results."""
        # Check 2.1: mass_matches_config
        try:
            from ..config.vehicles import get_active_vehicle
            vehicle = get_active_vehicle()
            if vehicle:
                config_mass_kg = vehicle.weight_lbs / 2.205
                pct_diff = abs(self.vehicle_mass_kg - config_mass_kg) / config_mass_kg
                if pct_diff < 0.10:
                    trace.add_check(
                        "mass_matches_config", "pass",
                        f"Analyzer mass {self.vehicle_mass_kg:.0f} kg matches vehicle config {config_mass_kg:.0f} kg",
                        expected=round(config_mass_kg, 1), actual=self.vehicle_mass_kg,
                    )
                else:
                    trace.add_check(
                        "mass_matches_config", "warn",
                        f"Analyzer mass {self.vehicle_mass_kg:.0f} kg differs from vehicle config {config_mass_kg:.0f} kg by {pct_diff*100:.0f}%",
                        expected=round(config_mass_kg, 1), actual=self.vehicle_mass_kg,
                    )
        except Exception:
            trace.add_check(
                "mass_matches_config", "warn",
                "Could not load active vehicle to compare mass",
            )

        # Check 2.2: max_power_plausible
        if report.max_power_hp < 800:
            trace.add_check(
                "max_power_plausible", "pass",
                f"Max power {report.max_power_hp:.0f} HP is plausible",
                expected="< 800", actual=round(report.max_power_hp, 1),
            )
        else:
            trace.add_check(
                "max_power_plausible", "fail",
                f"Max power {report.max_power_hp:.0f} HP exceeds 800 HP limit for street cars",
                expected="< 800", actual=round(report.max_power_hp, 1),
                severity="error",
            )

        # Check 2.3: power_weight_ratio
        if self.vehicle_mass_kg > 0:
            pwr = report.max_power_hp / self.vehicle_mass_kg
            if pwr < 0.5:
                trace.add_check(
                    "power_weight_ratio", "pass",
                    f"Power/weight ratio {pwr:.3f} HP/kg is reasonable",
                    expected="< 0.5", actual=round(pwr, 3),
                )
            else:
                trace.add_check(
                    "power_weight_ratio", "warn",
                    f"Power/weight ratio {pwr:.3f} HP/kg exceeds supercar territory (0.5 HP/kg)",
                    expected="< 0.5", actual=round(pwr, 3),
                )

        # Check 2.4: speed_unit_confidence
        speed_max_raw = trace.inputs.get("speed_max_raw", 0)
        if 45 <= speed_max_raw <= 110:
            trace.add_check(
                "speed_unit_confidence", "warn",
                f"Max raw speed {speed_max_raw:.1f} is in ambiguous zone (45-110). Could be m/s or mph.",
                expected="outside 45-110", actual=round(speed_max_raw, 1),
            )
        else:
            trace.add_check(
                "speed_unit_confidence", "pass",
                f"Max raw speed {speed_max_raw:.1f} is unambiguous",
                expected="outside 45-110", actual=round(speed_max_raw, 1),
            )

        # Check 2.5: sufficient_data
        sample_count = trace.inputs.get("sample_count", 0)
        dt_mean = trace.inputs.get("dt_mean", 0)
        duration = sample_count * dt_mean if dt_mean > 0 else 0
        if sample_count > 100 and duration > 10:
            trace.add_check(
                "sufficient_data", "pass",
                f"{sample_count} samples over {duration:.1f}s is sufficient",
                expected="> 100 samples, > 10s", actual=f"{sample_count} samples, {duration:.1f}s",
            )
        else:
            trace.add_check(
                "sufficient_data", "fail",
                f"Insufficient data: {sample_count} samples, {duration:.1f}s (need >100 samples, >10s)",
                expected="> 100 samples, > 10s", actual=f"{sample_count} samples, {duration:.1f}s",
                severity="error",
            )

    def _calculate_acceleration(
        self,
        time_data: np.ndarray,
        speed_ms: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate acceleration in m/s² and g-force"""
        # Calculate time differences
        dt = np.diff(time_data)
        dt[dt < 0.001] = 0.001  # Prevent division by zero

        # Calculate velocity differences
        dv = np.diff(speed_ms)

        # Acceleration = dv/dt
        accel_ms2 = dv / dt

        # Pad to match original array length
        accel_ms2 = np.append(accel_ms2, accel_ms2[-1])

        # Apply smoothing if enough data points
        if len(accel_ms2) > self.smoothing_window:
            window = min(self.smoothing_window, len(accel_ms2) // 2)
            if window % 2 == 0:
                window += 1  # Must be odd
            if window >= 3:
                accel_ms2 = signal.savgol_filter(accel_ms2, window, 3)

        # Convert to g-force (1g = 9.81 m/s²)
        accel_g = accel_ms2 / 9.81

        return accel_ms2, accel_g

    def _calculate_power(
        self,
        time_data: np.ndarray,
        speed_mph: np.ndarray,
        speed_ms: np.ndarray,
        accel_ms2: np.ndarray,
        rpm_data: np.ndarray
    ) -> List[PowerEstimate]:
        """Calculate power estimates using P = m * a * v"""
        estimates = []

        power_band_min = ENGINE_SPECS.get('power_band_min', 5500)
        power_band_max = ENGINE_SPECS.get('power_band_max', 7000)

        for i in range(len(time_data)):
            v = speed_ms[i]
            a = accel_ms2[i]
            rpm = rpm_data[i] if rpm_data is not None else 0

            # Power calculation: P = F * v = m * a * v
            if v > 2 and a > 0:  # Only when moving forward and accelerating
                force = self.vehicle_mass_kg * a
                power_watts = force * v
                power_hp = power_watts / 745.7
            else:
                power_hp = 0

            in_power_band = power_band_min <= rpm <= power_band_max if rpm > 0 else False

            estimates.append(PowerEstimate(
                time=float(time_data[i]),
                speed_mph=float(speed_mph[i]),
                rpm=float(rpm),
                acceleration_g=float(a / 9.81),
                power_hp=float(power_hp),
                in_power_band=in_power_band
            ))

        return estimates

    def _find_acceleration_events(
        self,
        time_data: np.ndarray,
        speed_data: np.ndarray,
        accel_g: np.ndarray,
        power_estimates: List[PowerEstimate]
    ) -> List[AccelerationEvent]:
        """Find significant acceleration events"""
        events = []
        in_event = False
        event_start = 0

        for i in range(len(accel_g)):
            if accel_g[i] > self.min_accel_threshold:
                if not in_event:
                    in_event = True
                    event_start = i
            else:
                if in_event and i - event_start >= 5:  # Minimum 5 samples
                    # End of event
                    event_indices = range(event_start, i)
                    event_accels = accel_g[event_start:i]
                    event_powers = [power_estimates[j].power_hp for j in event_indices
                                   if j < len(power_estimates)]

                    events.append(AccelerationEvent(
                        event_type='acceleration',
                        start_time=float(time_data[event_start]),
                        end_time=float(time_data[i-1]),
                        duration_seconds=float(time_data[i-1] - time_data[event_start]),
                        start_speed_mph=float(speed_data[event_start]),
                        end_speed_mph=float(speed_data[i-1]),
                        speed_change_mph=float(speed_data[i-1] - speed_data[event_start]),
                        peak_acceleration_g=float(max(event_accels)),
                        avg_acceleration_g=float(np.mean(event_accels)),
                        peak_power_hp=float(max(event_powers)) if event_powers else 0,
                        avg_power_hp=float(np.mean(event_powers)) if event_powers else 0
                    ))
                in_event = False

        # Sort by peak power
        events.sort(key=lambda x: x.peak_power_hp, reverse=True)
        return events

    def _find_braking_events(
        self,
        time_data: np.ndarray,
        speed_data: np.ndarray,
        accel_g: np.ndarray
    ) -> List[AccelerationEvent]:
        """Find significant braking events"""
        events = []
        in_event = False
        event_start = 0

        for i in range(len(accel_g)):
            if accel_g[i] < -self.min_brake_threshold:
                if not in_event:
                    in_event = True
                    event_start = i
            else:
                if in_event and i - event_start >= 5:  # Minimum 5 samples
                    # End of event
                    event_accels = accel_g[event_start:i]

                    events.append(AccelerationEvent(
                        event_type='braking',
                        start_time=float(time_data[event_start]),
                        end_time=float(time_data[i-1]),
                        duration_seconds=float(time_data[i-1] - time_data[event_start]),
                        start_speed_mph=float(speed_data[event_start]),
                        end_speed_mph=float(speed_data[i-1]),
                        speed_change_mph=float(speed_data[i-1] - speed_data[event_start]),
                        peak_acceleration_g=float(abs(min(event_accels))),
                        avg_acceleration_g=float(abs(np.mean(event_accels))),
                        peak_power_hp=0,
                        avg_power_hp=0
                    ))
                in_event = False

        # Sort by peak braking
        events.sort(key=lambda x: x.peak_acceleration_g, reverse=True)
        return events

    def _analyze_rpm(
        self,
        rpm_data: np.ndarray,
        speed_data: np.ndarray,
        accel_g: np.ndarray
    ) -> Dict:
        """Analyze RPM usage patterns"""
        safe_limit = ENGINE_SPECS.get('safe_rpm_limit', 7000)
        power_band_min = ENGINE_SPECS.get('power_band_min', 5500)
        power_band_max = ENGINE_SPECS.get('power_band_max', 7000)

        valid_mask = rpm_data > 1000  # Filter out idle/stopped
        valid_rpms = rpm_data[valid_mask]

        if len(valid_rpms) == 0:
            return {"error": "No valid RPM data"}

        over_limit = np.sum(valid_rpms > safe_limit)
        in_power_band = np.sum((valid_rpms >= power_band_min) & (valid_rpms <= power_band_max))

        # RPM during acceleration
        accel_mask = (accel_g > 0.1) & valid_mask
        accel_rpms = rpm_data[accel_mask] if np.any(accel_mask) else np.array([])

        return {
            "avg_rpm": float(np.mean(valid_rpms)),
            "max_rpm": float(np.max(valid_rpms)),
            "min_rpm": float(np.min(valid_rpms)),
            "pct_over_safe_limit": float(over_limit / len(valid_rpms) * 100),
            "pct_in_power_band": float(in_power_band / len(valid_rpms) * 100),
            "avg_rpm_during_accel": float(np.mean(accel_rpms)) if len(accel_rpms) > 0 else 0,
            "max_rpm_during_accel": float(np.max(accel_rpms)) if len(accel_rpms) > 0 else 0
        }

    def _generate_recommendations(
        self,
        max_power: float,
        accel_events: List[AccelerationEvent],
        brake_events: List[AccelerationEvent],
        rpm_analysis: Dict
    ) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        # Power recommendations
        if max_power < 200:
            recommendations.append(
                f"Peak power estimate of {max_power:.0f} HP seems low. "
                "May indicate conservative driving or measurement issues."
            )
        elif max_power > 400:
            recommendations.append(
                f"Peak power estimate of {max_power:.0f} HP. "
                "Verify vehicle weight for accuracy."
            )

        # RPM recommendations
        if rpm_analysis and 'pct_over_safe_limit' in rpm_analysis:
            if rpm_analysis['pct_over_safe_limit'] > 5:
                recommendations.append(
                    f"{rpm_analysis['pct_over_safe_limit']:.1f}% of time over safe RPM limit. "
                    "Consider earlier upshifts or taller gearing."
                )

            if rpm_analysis['pct_in_power_band'] < 40:
                recommendations.append(
                    f"Only {rpm_analysis['pct_in_power_band']:.1f}% of time in power band. "
                    "May benefit from more aggressive downshifts before corners."
                )

        # Braking recommendations
        if brake_events:
            max_brake = brake_events[0].peak_acceleration_g
            if max_brake < 0.8:
                recommendations.append(
                    f"Max braking of {max_brake:.2f}g. "
                    "There may be room for later braking points."
                )
            elif max_brake > 1.2:
                recommendations.append(
                    f"Max braking of {max_brake:.2f}g indicates strong brake capability. "
                    "Ensure consistent brake application."
                )

        # Acceleration recommendations
        if accel_events:
            # Check for consistency
            if len(accel_events) >= 3:
                peak_powers = [e.peak_power_hp for e in accel_events[:5]]
                power_std = np.std(peak_powers)
                if power_std > 50:
                    recommendations.append(
                        f"Acceleration power varies by {power_std:.0f} HP. "
                        "Focus on consistent throttle application."
                    )

        if not recommendations:
            recommendations.append(
                "Good overall performance. "
                "Review individual events for fine-tuning opportunities."
            )

        return recommendations

    def _build_summary(
        self,
        power_estimates: List[PowerEstimate],
        accel_events: List[AccelerationEvent],
        brake_events: List[AccelerationEvent],
        total_duration: float
    ) -> Dict:
        """Build analysis summary"""
        # Time in different states
        accel_samples = sum(1 for p in power_estimates if p.acceleration_g > 0.1)
        brake_samples = sum(1 for p in power_estimates if p.acceleration_g < -0.1)
        coast_samples = len(power_estimates) - accel_samples - brake_samples

        total_samples = len(power_estimates) if power_estimates else 1

        return {
            "pct_accelerating": float(accel_samples / total_samples * 100),
            "pct_braking": float(brake_samples / total_samples * 100),
            "pct_coasting": float(coast_samples / total_samples * 100),
            "total_accel_events": len(accel_events),
            "total_brake_events": len(brake_events),
            "events_per_minute": (
                (len(accel_events) + len(brake_events)) / (total_duration / 60)
                if total_duration > 0 else 0
            )
        }

    def get_power_curve(
        self,
        rpm_data: np.ndarray,
        power_estimates: List[PowerEstimate],
        rpm_bins: int = 20
    ) -> Dict:
        """
        Generate estimated power curve by RPM.

        Args:
            rpm_data: RPM values
            power_estimates: Power estimates from analysis
            rpm_bins: Number of RPM bins

        Returns:
            Dictionary with RPM bins and average power
        """
        if not power_estimates or rpm_data is None:
            return {"error": "Insufficient data for power curve"}

        # Create RPM bins
        min_rpm = max(2000, np.min(rpm_data))
        max_rpm = min(8000, np.max(rpm_data))
        bins = np.linspace(min_rpm, max_rpm, rpm_bins + 1)

        rpm_centers = []
        avg_powers = []
        sample_counts = []

        for i in range(len(bins) - 1):
            bin_min, bin_max = bins[i], bins[i+1]

            # Find power estimates in this RPM range
            bin_powers = [
                p.power_hp for p in power_estimates
                if bin_min <= p.rpm < bin_max and p.power_hp > 0
            ]

            rpm_centers.append((bin_min + bin_max) / 2)
            avg_powers.append(float(np.mean(bin_powers)) if bin_powers else 0)
            sample_counts.append(len(bin_powers))

        return {
            "rpm_centers": rpm_centers,
            "avg_power_hp": avg_powers,
            "sample_counts": sample_counts,
            "peak_power_rpm": rpm_centers[np.argmax(avg_powers)] if avg_powers else 0,
            "peak_power_hp": max(avg_powers) if avg_powers else 0
        }
