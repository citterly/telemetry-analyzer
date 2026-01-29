"""
Tests for acceleration and power analysis feature
"""

import os
import sys
import numpy as np
import pytest
import json
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.power_analysis import (
    PowerAnalysis,
    PowerAnalysisReport,
    AccelerationEvent,
    PowerEstimate
)


class TestAccelerationEvent:
    """Tests for AccelerationEvent dataclass"""

    def test_acceleration_event_creation(self):
        """Test creating acceleration event"""
        event = AccelerationEvent(
            event_type='acceleration',
            start_time=10.0,
            end_time=15.0,
            duration_seconds=5.0,
            start_speed_mph=30.0,
            end_speed_mph=60.0,
            speed_change_mph=30.0,
            peak_acceleration_g=0.5,
            avg_acceleration_g=0.4,
            peak_power_hp=250.0,
            avg_power_hp=200.0
        )
        assert event.event_type == 'acceleration'
        assert event.duration_seconds == 5.0
        assert event.speed_change_mph == 30.0

    def test_braking_event_creation(self):
        """Test creating braking event"""
        event = AccelerationEvent(
            event_type='braking',
            start_time=20.0,
            end_time=23.0,
            duration_seconds=3.0,
            start_speed_mph=100.0,
            end_speed_mph=40.0,
            speed_change_mph=-60.0,
            peak_acceleration_g=1.0,
            avg_acceleration_g=0.8,
            peak_power_hp=0,
            avg_power_hp=0
        )
        assert event.event_type == 'braking'
        assert event.peak_acceleration_g == 1.0


class TestPowerEstimate:
    """Tests for PowerEstimate dataclass"""

    def test_power_estimate_creation(self):
        """Test creating power estimate"""
        est = PowerEstimate(
            time=5.0,
            speed_mph=80.0,
            rpm=6000,
            acceleration_g=0.3,
            power_hp=220.0,
            in_power_band=True
        )
        assert est.speed_mph == 80.0
        assert est.power_hp == 220.0
        assert est.in_power_band is True


class TestPowerAnalysisReport:
    """Tests for PowerAnalysisReport"""

    @pytest.fixture
    def sample_report(self):
        """Create sample report"""
        accel_events = [
            AccelerationEvent(
                event_type='acceleration', start_time=10.0, end_time=15.0,
                duration_seconds=5.0, start_speed_mph=30.0, end_speed_mph=60.0,
                speed_change_mph=30.0, peak_acceleration_g=0.5, avg_acceleration_g=0.4,
                peak_power_hp=250.0, avg_power_hp=200.0
            )
        ]

        brake_events = [
            AccelerationEvent(
                event_type='braking', start_time=20.0, end_time=23.0,
                duration_seconds=3.0, start_speed_mph=100.0, end_speed_mph=40.0,
                speed_change_mph=-60.0, peak_acceleration_g=1.0, avg_acceleration_g=0.8,
                peak_power_hp=0, avg_power_hp=0
            )
        ]

        return PowerAnalysisReport(
            session_id="test_session",
            analysis_timestamp="2026-01-29T02:30:00",
            vehicle_mass_kg=1565,
            total_duration_seconds=180.0,
            max_power_hp=280.0,
            avg_power_hp=150.0,
            power_at_peak_rpm=260.0,
            max_acceleration_g=0.55,
            avg_acceleration_g=0.25,
            acceleration_events=accel_events,
            max_braking_g=1.1,
            avg_braking_g=0.7,
            braking_events=brake_events,
            rpm_analysis={
                "avg_rpm": 5500,
                "max_rpm": 6800,
                "pct_over_safe_limit": 2.5
            },
            recommendations=["Good session"],
            summary={"pct_accelerating": 40.0}
        )

    def test_report_to_dict(self, sample_report):
        """Test report serialization to dict"""
        data = sample_report.to_dict()
        assert data["session_id"] == "test_session"
        assert data["power"]["max_hp"] == 280.0
        assert data["acceleration"]["max_g"] == 0.55
        assert data["braking"]["max_g"] == 1.1

    def test_report_to_json(self, sample_report):
        """Test report serialization to JSON"""
        json_str = sample_report.to_json()
        parsed = json.loads(json_str)
        assert parsed["session_id"] == "test_session"


class TestPowerAnalysis:
    """Tests for PowerAnalysis"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer"""
        return PowerAnalysis()

    @pytest.fixture
    def sample_session(self):
        """Generate sample session with acceleration and braking"""
        total_time = 60  # 60 seconds
        sample_rate = 10
        n_samples = total_time * sample_rate

        time_data = np.linspace(0, total_time, n_samples)

        # Create speed profile with acceleration and braking phases
        speed_data = np.zeros(n_samples)

        # Phase 1: Acceleration (0-20s): 0 to 100 mph
        phase1 = slice(0, 200)
        speed_data[phase1] = np.linspace(0, 100, 200)

        # Phase 2: Braking (20-25s): 100 to 50 mph
        phase2 = slice(200, 250)
        speed_data[phase2] = np.linspace(100, 50, 50)

        # Phase 3: Acceleration (25-40s): 50 to 120 mph
        phase3 = slice(250, 400)
        speed_data[phase3] = np.linspace(50, 120, 150)

        # Phase 4: Hard braking (40-45s): 120 to 30 mph
        phase4 = slice(400, 450)
        speed_data[phase4] = np.linspace(120, 30, 50)

        # Phase 5: Gentle acceleration (45-60s): 30 to 80 mph
        phase5 = slice(450, 600)
        speed_data[phase5] = np.linspace(30, 80, 150)

        # RPM correlates roughly with speed
        rpm_data = 2000 + (speed_data / 130) * 5000

        return {
            "time": time_data,
            "speed": speed_data,
            "rpm": rpm_data
        }

    def test_analyzer_init_default(self, analyzer):
        """Test analyzer initialization with defaults"""
        assert analyzer.vehicle_mass_kg == 1565
        assert analyzer.min_accel_threshold > 0
        assert analyzer.min_brake_threshold > 0

    def test_analyzer_init_custom(self):
        """Test custom initialization"""
        analyzer = PowerAnalysis(
            vehicle_mass_kg=1400,
            min_accel_threshold_g=0.2,
            min_brake_threshold_g=0.25
        )
        assert analyzer.vehicle_mass_kg == 1400
        assert analyzer.min_accel_threshold == 0.2

    def test_analyze_from_arrays(self, analyzer, sample_session):
        """Test analysis from raw arrays"""
        report = analyzer.analyze_from_arrays(
            time_data=sample_session["time"],
            speed_data=sample_session["speed"],
            rpm_data=sample_session["rpm"],
            session_id="test"
        )

        assert isinstance(report, PowerAnalysisReport)
        assert report.session_id == "test"

    def test_analyze_returns_power_stats(self, analyzer, sample_session):
        """Test that analysis returns power statistics"""
        report = analyzer.analyze_from_arrays(
            time_data=sample_session["time"],
            speed_data=sample_session["speed"],
            rpm_data=sample_session["rpm"]
        )

        assert report.max_power_hp >= 0
        assert report.avg_power_hp >= 0

    def test_analyze_returns_acceleration_stats(self, analyzer, sample_session):
        """Test that analysis returns acceleration statistics"""
        report = analyzer.analyze_from_arrays(
            time_data=sample_session["time"],
            speed_data=sample_session["speed"],
            rpm_data=sample_session["rpm"]
        )

        assert report.max_acceleration_g >= 0
        assert report.avg_acceleration_g >= 0

    def test_analyze_returns_braking_stats(self, analyzer, sample_session):
        """Test that analysis returns braking statistics"""
        report = analyzer.analyze_from_arrays(
            time_data=sample_session["time"],
            speed_data=sample_session["speed"],
            rpm_data=sample_session["rpm"]
        )

        assert report.max_braking_g >= 0
        assert report.avg_braking_g >= 0

    def test_analyze_finds_acceleration_events(self, analyzer, sample_session):
        """Test that analysis finds acceleration events"""
        report = analyzer.analyze_from_arrays(
            time_data=sample_session["time"],
            speed_data=sample_session["speed"],
            rpm_data=sample_session["rpm"]
        )

        assert len(report.acceleration_events) > 0
        assert all(e.event_type == 'acceleration' for e in report.acceleration_events)

    def test_analyze_finds_braking_events(self, analyzer, sample_session):
        """Test that analysis finds braking events"""
        report = analyzer.analyze_from_arrays(
            time_data=sample_session["time"],
            speed_data=sample_session["speed"],
            rpm_data=sample_session["rpm"]
        )

        assert len(report.braking_events) > 0
        assert all(e.event_type == 'braking' for e in report.braking_events)

    def test_analyze_has_rpm_analysis(self, analyzer, sample_session):
        """Test that analysis has RPM analysis"""
        report = analyzer.analyze_from_arrays(
            time_data=sample_session["time"],
            speed_data=sample_session["speed"],
            rpm_data=sample_session["rpm"]
        )

        assert "avg_rpm" in report.rpm_analysis
        assert "max_rpm" in report.rpm_analysis

    def test_analyze_has_recommendations(self, analyzer, sample_session):
        """Test that analysis generates recommendations"""
        report = analyzer.analyze_from_arrays(
            time_data=sample_session["time"],
            speed_data=sample_session["speed"],
            rpm_data=sample_session["rpm"]
        )

        assert isinstance(report.recommendations, list)
        assert len(report.recommendations) >= 1

    def test_analyze_has_summary(self, analyzer, sample_session):
        """Test that analysis has summary"""
        report = analyzer.analyze_from_arrays(
            time_data=sample_session["time"],
            speed_data=sample_session["speed"],
            rpm_data=sample_session["rpm"]
        )

        assert "pct_accelerating" in report.summary
        assert "pct_braking" in report.summary
        assert "pct_coasting" in report.summary

    def test_analyze_timestamp(self, analyzer, sample_session):
        """Test that report has timestamp"""
        report = analyzer.analyze_from_arrays(
            time_data=sample_session["time"],
            speed_data=sample_session["speed"],
            rpm_data=sample_session["rpm"]
        )

        assert report.analysis_timestamp is not None
        from datetime import datetime
        datetime.fromisoformat(report.analysis_timestamp)


class TestPowerAnalysisParquet:
    """Tests for Parquet file analysis"""

    @pytest.fixture
    def sample_parquet(self):
        """Create sample Parquet file"""
        import pandas as pd

        n_samples = 500
        time = np.linspace(0, 50, n_samples)

        # Create speed and RPM data
        speed = 40 + 30 * np.sin(time / 10)  # Oscillating speed
        rpm = 4000 + 2000 * np.sin(time / 10)

        df = pd.DataFrame({
            "GPS Speed": speed,
            "RPM": rpm
        }, index=time)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name)
            yield f.name

        try:
            os.unlink(f.name)
        except:
            pass

    def test_analyze_from_parquet(self, sample_parquet):
        """Test analyzing from Parquet file"""
        analyzer = PowerAnalysis()
        report = analyzer.analyze_from_parquet(sample_parquet)

        assert isinstance(report, PowerAnalysisReport)

    def test_parquet_custom_session_id(self, sample_parquet):
        """Test custom session ID"""
        analyzer = PowerAnalysis()
        report = analyzer.analyze_from_parquet(
            sample_parquet,
            session_id="my_session"
        )

        assert report.session_id == "my_session"


class TestPowerCurve:
    """Tests for power curve generation"""

    @pytest.fixture
    def analyzer_with_data(self):
        """Create analyzer with sample power estimates"""
        analyzer = PowerAnalysis()

        # Generate some sample data
        n = 100
        time = np.linspace(0, 10, n)
        speed = np.linspace(40, 120, n)
        rpm = np.linspace(4000, 7000, n)

        report = analyzer.analyze_from_arrays(time, speed, rpm)

        return analyzer, rpm, report

    def test_power_curve_generation(self, analyzer_with_data):
        """Test power curve generation"""
        analyzer, rpm_data, report = analyzer_with_data

        # Access power estimates through re-analyzing
        # (since _calculate_power is internal)
        speed_ms = np.linspace(40, 120, 100) / 2.237
        time_data = np.linspace(0, 10, 100)
        speed_mph = np.linspace(40, 120, 100)

        dt = np.diff(time_data)
        dt[dt < 0.001] = 0.001
        dv = np.diff(speed_ms)
        accel_ms2 = np.append(dv / dt, 0)

        # Create mock power estimates
        power_estimates = [
            PowerEstimate(
                time=t, speed_mph=s, rpm=r,
                acceleration_g=a/9.81,
                power_hp=max(0, 1565 * a * (s/2.237) / 745.7),
                in_power_band=5500 <= r <= 7000
            )
            for t, s, r, a in zip(time_data, speed_mph, rpm_data, accel_ms2)
        ]

        curve = analyzer.get_power_curve(rpm_data, power_estimates)

        assert "rpm_centers" in curve
        assert "avg_power_hp" in curve
        assert len(curve["rpm_centers"]) > 0


class TestEdgeCases:
    """Tests for edge cases"""

    def test_minimal_data(self):
        """Test handling of minimal data"""
        analyzer = PowerAnalysis()

        report = analyzer.analyze_from_arrays(
            time_data=np.array([0, 1, 2]),
            speed_data=np.array([50, 50, 50]),
            rpm_data=np.array([4000, 4000, 4000])
        )

        assert report is not None
        assert report.total_duration_seconds == 2.0

    def test_no_rpm_data(self):
        """Test analysis without RPM data"""
        analyzer = PowerAnalysis()

        report = analyzer.analyze_from_arrays(
            time_data=np.linspace(0, 10, 100),
            speed_data=np.linspace(30, 100, 100),
            rpm_data=None
        )

        assert report is not None
        assert report.rpm_analysis == {}

    def test_constant_speed(self):
        """Test session with constant speed (no acceleration)"""
        analyzer = PowerAnalysis()

        time = np.linspace(0, 30, 300)
        speed = np.full(300, 80)  # Constant 80 mph
        rpm = np.full(300, 5000)

        report = analyzer.analyze_from_arrays(time, speed, rpm)

        # Should have minimal acceleration events
        assert report.max_acceleration_g < 0.2

    def test_high_g_forces(self):
        """Test detection of high g-force events"""
        analyzer = PowerAnalysis()

        time = np.linspace(0, 5, 50)
        # Very rapid speed change (simulating hard braking)
        speed = np.linspace(120, 30, 50)
        rpm = np.linspace(6500, 3000, 50)

        report = analyzer.analyze_from_arrays(time, speed, rpm)

        # Should detect high braking g
        assert report.max_braking_g > 0.5


class TestRecommendations:
    """Tests for recommendation generation"""

    def test_low_power_recommendation(self):
        """Test recommendation for low power output"""
        analyzer = PowerAnalysis()

        # Low acceleration = low power
        time = np.linspace(0, 30, 300)
        speed = np.linspace(40, 60, 300)  # Very gentle acceleration
        rpm = np.linspace(3000, 4000, 300)

        report = analyzer.analyze_from_arrays(time, speed, rpm)

        # Should have some recommendation
        assert len(report.recommendations) >= 1

    def test_over_rev_recommendation(self):
        """Test recommendation for over-revving"""
        analyzer = PowerAnalysis()

        time = np.linspace(0, 10, 100)
        speed = np.linspace(50, 130, 100)
        rpm = np.full(100, 7500)  # Consistently over safe limit

        report = analyzer.analyze_from_arrays(time, speed, rpm)

        # Should recommend earlier upshifts
        any_rpm_rec = any("rpm" in r.lower() or "upshift" in r.lower()
                        for r in report.recommendations)
        assert any_rpm_rec or report.rpm_analysis.get('pct_over_safe_limit', 0) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
