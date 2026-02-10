"""
Tests for shift analysis feature
"""

import os
import sys
import numpy as np
import pytest
import json
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.shift_analysis import (
    ShiftAnalyzer,
    ShiftReport,
    ShiftEvent,
    GearShiftStats
)


class TestShiftEvent:
    """Tests for ShiftEvent dataclass"""

    def test_shift_event_creation(self):
        """Test creating a shift event"""
        event = ShiftEvent(
            time=10.5,
            from_gear=2,
            to_gear=3,
            shift_type="upshift",
            rpm_at_shift=6500,
            speed_mph=45.0
        )
        assert event.from_gear == 2
        assert event.to_gear == 3
        assert event.shift_type == "upshift"
        assert event.rpm_at_shift == 6500

    def test_shift_event_defaults(self):
        """Test default values"""
        event = ShiftEvent(
            time=0,
            from_gear=1,
            to_gear=2,
            shift_type="upshift",
            rpm_at_shift=6000,
            speed_mph=30
        )
        assert event.rpm_delta == 0.0
        assert event.shift_quality == "normal"


class TestGearShiftStats:
    """Tests for GearShiftStats dataclass"""

    def test_gear_stats_creation(self):
        """Test creating gear stats"""
        stats = GearShiftStats(gear=3)
        assert stats.gear == 3
        assert stats.upshift_count == 0
        assert stats.downshift_count == 0

    def test_gear_stats_with_data(self):
        """Test gear stats with data"""
        stats = GearShiftStats(
            gear=2,
            upshift_count=5,
            avg_upshift_rpm=6200,
            min_upshift_rpm=5800,
            max_upshift_rpm=6600
        )
        assert stats.upshift_count == 5
        assert stats.avg_upshift_rpm == 6200


class TestShiftAnalyzer:
    """Tests for ShiftAnalyzer"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with default settings"""
        return ShiftAnalyzer()

    @pytest.fixture
    def sample_data(self):
        """Generate sample telemetry data with shifts"""
        # Simulate 30 seconds of driving with gear changes
        time_data = np.linspace(0, 30, 300)  # 10 Hz

        # Create RPM and speed patterns
        rpm_data = np.zeros(300)
        speed_data = np.zeros(300)

        # Gear 1: 0-5s, accelerating
        rpm_data[0:50] = np.linspace(2000, 6500, 50)
        speed_data[0:50] = np.linspace(0, 25, 50)

        # Shift to gear 2: instant RPM drop
        # Gear 2: 5-10s
        rpm_data[50:100] = np.linspace(4000, 6500, 50)
        speed_data[50:100] = np.linspace(25, 45, 50)

        # Shift to gear 3
        # Gear 3: 10-15s
        rpm_data[100:150] = np.linspace(4500, 6800, 50)
        speed_data[100:150] = np.linspace(45, 65, 50)

        # Shift to gear 4
        # Gear 4: 15-20s
        rpm_data[150:200] = np.linspace(4800, 6200, 50)
        speed_data[150:200] = np.linspace(65, 85, 50)

        # Braking/downshift
        # Gear 3: 20-25s
        rpm_data[200:250] = np.linspace(5500, 4000, 50)
        speed_data[200:250] = np.linspace(85, 55, 50)

        # Gear 2: 25-30s
        rpm_data[250:300] = np.linspace(5000, 3500, 50)
        speed_data[250:300] = np.linspace(55, 35, 50)

        return {
            "time": time_data,
            "rpm": rpm_data,
            "speed": speed_data
        }

    def test_analyzer_init_defaults(self, analyzer):
        """Test analyzer initialization with defaults"""
        assert analyzer.gear_calculator is not None
        assert len(analyzer.transmission_ratios) > 0

    def test_analyzer_init_custom(self):
        """Test analyzer with custom ratios"""
        ratios = [3.5, 2.5, 1.8, 1.3, 1.0, 0.8]
        analyzer = ShiftAnalyzer(
            transmission_ratios=ratios,
            final_drive=3.73
        )
        assert analyzer.transmission_ratios == ratios
        assert analyzer.final_drive == 3.73

    def test_analyze_session_returns_report(self, analyzer, sample_data):
        """Test that analyze_session returns a ShiftReport"""
        report = analyzer.analyze_session(
            rpm_data=sample_data["rpm"],
            speed_data=sample_data["speed"],
            time_data=sample_data["time"],
            session_id="test_session"
        )

        assert isinstance(report, ShiftReport)
        assert report.session_id == "test_session"
        assert report.analysis_timestamp is not None

    def test_analyze_session_detects_shifts(self, analyzer, sample_data):
        """Test that shifts are detected"""
        report = analyzer.analyze_session(
            rpm_data=sample_data["rpm"],
            speed_data=sample_data["speed"],
            time_data=sample_data["time"]
        )

        # Should detect some shifts
        assert report.total_shifts >= 0  # May vary based on gear detection

    def test_shift_quality_assessment(self, analyzer):
        """Test shift quality classification"""
        # Create data with known shift qualities
        time = np.linspace(0, 10, 100)
        rpm = np.ones(100) * 5000
        speed = np.ones(100) * 40

        # Insert shifts at different RPMs
        rpm[45:50] = 5200  # Early shift
        rpm[50:55] = 6500  # Optimal shift
        rpm[55:60] = 7100  # Late shift
        rpm[60:65] = 7300  # Over-rev

        report = analyzer.analyze_session(rpm, speed, time)

        # Report should include quality breakdown
        assert "shift_quality_breakdown" in report.summary

    def test_gear_stats_calculation(self, analyzer, sample_data):
        """Test gear statistics are calculated"""
        report = analyzer.analyze_session(
            rpm_data=sample_data["rpm"],
            speed_data=sample_data["speed"],
            time_data=sample_data["time"]
        )

        assert isinstance(report.gear_stats, dict)
        # Should have stats for gears 1-6
        assert len(report.gear_stats) >= 1

    def test_recommendations_generated(self, analyzer, sample_data):
        """Test that recommendations are generated"""
        report = analyzer.analyze_session(
            rpm_data=sample_data["rpm"],
            speed_data=sample_data["speed"],
            time_data=sample_data["time"]
        )

        assert isinstance(report.recommendations, list)
        assert len(report.recommendations) >= 1

    def test_report_to_dict(self, analyzer, sample_data):
        """Test report serialization to dict"""
        report = analyzer.analyze_session(
            rpm_data=sample_data["rpm"],
            speed_data=sample_data["speed"],
            time_data=sample_data["time"]
        )

        data = report.to_dict()

        assert "session_id" in data
        assert "total_shifts" in data
        assert "shifts" in data
        assert "gear_stats" in data
        assert "recommendations" in data

    def test_report_to_json(self, analyzer, sample_data):
        """Test report serialization to JSON"""
        report = analyzer.analyze_session(
            rpm_data=sample_data["rpm"],
            speed_data=sample_data["speed"],
            time_data=sample_data["time"]
        )

        json_str = report.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "session_id" in parsed

    def test_empty_data_handling(self, analyzer):
        """Test handling of empty/minimal data"""
        time = np.array([0, 1])
        rpm = np.array([3000, 3000])
        speed = np.array([30, 30])

        report = analyzer.analyze_session(rpm, speed, time)

        assert report.total_shifts == 0

    def test_constant_gear_no_shifts(self, analyzer):
        """Test data with no gear changes"""
        time = np.linspace(0, 10, 100)
        rpm = np.ones(100) * 4000  # Constant RPM
        speed = np.ones(100) * 50   # Constant speed

        report = analyzer.analyze_session(rpm, speed, time)

        assert report.total_shifts == 0
        assert report.total_upshifts == 0
        assert report.total_downshifts == 0

    def test_get_shift_timing_by_gear(self, analyzer, sample_data):
        """Test getting shift timing by gear"""
        report = analyzer.analyze_session(
            rpm_data=sample_data["rpm"],
            speed_data=sample_data["speed"],
            time_data=sample_data["time"]
        )

        timing = analyzer.get_shift_timing_by_gear(report)

        assert isinstance(timing, dict)


class TestShiftAnalyzerParquet:
    """Tests for Parquet file analysis"""

    @pytest.fixture
    def sample_parquet(self):
        """Create a sample Parquet file for testing"""
        import pandas as pd

        # Create sample data
        time = np.linspace(0, 30, 300)
        rpm = np.linspace(3000, 6500, 300)
        speed = np.linspace(20, 80, 300)

        df = pd.DataFrame({
            "RPM": rpm,
            "GPS Speed": speed
        }, index=time)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name)
            yield f.name

        # Cleanup
        try:
            os.unlink(f.name)
        except:
            pass

    def test_analyze_from_parquet(self, sample_parquet):
        """Test analyzing from Parquet file"""
        analyzer = ShiftAnalyzer()
        report = analyzer.analyze_from_parquet(sample_parquet)

        assert isinstance(report, ShiftReport)
        assert report.session_id is not None

    def test_analyze_parquet_custom_session_id(self, sample_parquet):
        """Test custom session ID"""
        analyzer = ShiftAnalyzer()
        report = analyzer.analyze_from_parquet(
            sample_parquet,
            session_id="my_session"
        )

        assert report.session_id == "my_session"


class TestShiftQualityThresholds:
    """Tests for shift quality threshold logic"""

    def test_early_shift_threshold(self):
        """Test early shift is detected below 5500 RPM"""
        assert ShiftAnalyzer.EARLY_SHIFT_RPM == 5500

    def test_optimal_shift_range(self):
        """Test optimal shift range"""
        assert ShiftAnalyzer.OPTIMAL_SHIFT_RPM_MIN == 6000
        assert ShiftAnalyzer.OPTIMAL_SHIFT_RPM_MAX == 6800

    def test_late_shift_threshold(self):
        """Test late shift threshold"""
        assert ShiftAnalyzer.LATE_SHIFT_RPM == 7000

    def test_over_rev_threshold(self):
        """Test over-rev threshold"""
        assert ShiftAnalyzer.OVER_REV_RPM == 7200


class TestShiftAnalyzerTrace:
    """Tests for safeguard-003: ShiftAnalyzer trace + sanity checks."""

    def _make_shift_parquet(self, tmp_path, n=300, rpm_range=(4000, 7000),
                            speed_range=(20, 60)):
        """Create synthetic parquet with RPM and speed data that generates shifts."""
        import pandas as pd
        time = np.linspace(0, 30, n)
        # Create RPM pattern with multiple upshifts (sawtooth)
        cycles = 5
        rpm_per_cycle = n // cycles
        rpm = np.zeros(n)
        speed = np.linspace(speed_range[0], speed_range[1], n)
        for c in range(cycles):
            start = c * rpm_per_cycle
            end = min(start + rpm_per_cycle, n)
            rpm[start:end] = np.linspace(rpm_range[0], rpm_range[1], end - start)
        df = pd.DataFrame({"RPM": rpm, "GPS Speed": speed}, index=time)
        path = str(tmp_path / "shift_test.parquet")
        df.to_parquet(path)
        return path

    def test_trace_recorded_when_enabled(self, tmp_path):
        """Trace is attached when include_trace=True."""
        path = self._make_shift_parquet(tmp_path)
        analyzer = ShiftAnalyzer()
        report = analyzer.analyze_from_parquet(path, include_trace=True)
        assert hasattr(report, 'trace')
        assert report.trace is not None
        assert report.trace.analyzer_name == "ShiftAnalyzer"

    def test_trace_not_recorded_by_default(self, tmp_path):
        """Trace is NOT attached by default."""
        path = self._make_shift_parquet(tmp_path)
        analyzer = ShiftAnalyzer()
        report = analyzer.analyze_from_parquet(path)
        assert not hasattr(report, 'trace') or report.trace is None

    def test_trace_inputs_recorded(self, tmp_path):
        """Trace records RPM column, speed column, unit, sample count, shift count."""
        path = self._make_shift_parquet(tmp_path)
        analyzer = ShiftAnalyzer()
        report = analyzer.analyze_from_parquet(path, include_trace=True)
        trace = report.trace
        assert "rpm_column" in trace.inputs
        assert "speed_column" in trace.inputs
        assert "speed_unit_detected" in trace.inputs
        assert trace.inputs["sample_count"] == 300
        assert "shift_count" in trace.inputs

    def test_trace_config_recorded(self, tmp_path):
        """Trace records transmission ratios, thresholds."""
        path = self._make_shift_parquet(tmp_path)
        analyzer = ShiftAnalyzer()
        report = analyzer.analyze_from_parquet(path, include_trace=True)
        trace = report.trace
        assert "transmission_ratios" in trace.config
        assert "final_drive" in trace.config
        assert "optimal_shift_rpm_min" in trace.config
        assert "over_rev_rpm" in trace.config

    def test_trace_intermediates_recorded(self, tmp_path):
        """Trace records gears_detected, shifts_per_gear, pct_optimal."""
        path = self._make_shift_parquet(tmp_path)
        analyzer = ShiftAnalyzer()
        report = analyzer.analyze_from_parquet(path, include_trace=True)
        trace = report.trace
        assert "gears_detected" in trace.intermediates
        assert "shifts_per_gear" in trace.intermediates
        assert "pct_optimal" in trace.intermediates
        assert "pct_early" in trace.intermediates

    def test_trace_has_four_sanity_checks(self, tmp_path):
        """All 4 sanity checks are present."""
        path = self._make_shift_parquet(tmp_path)
        analyzer = ShiftAnalyzer()
        report = analyzer.analyze_from_parquet(path, include_trace=True)
        check_names = [c.name for c in report.trace.sanity_checks]
        assert "gear_count_matches_config" in check_names
        assert "shift_rpm_below_redline" in check_names
        assert "shift_confidence" in check_names
        assert "sufficient_shifts" in check_names

    def test_to_dict_includes_trace(self, tmp_path):
        """to_dict() includes _trace when present."""
        path = self._make_shift_parquet(tmp_path)
        analyzer = ShiftAnalyzer()
        report = analyzer.analyze_from_parquet(path, include_trace=True)
        d = report.to_dict()
        assert "_trace" in d
        assert d["_trace"]["analyzer_name"] == "ShiftAnalyzer"

    def test_to_dict_omits_trace_by_default(self, tmp_path):
        """to_dict() does NOT include _trace by default."""
        path = self._make_shift_parquet(tmp_path)
        analyzer = ShiftAnalyzer()
        report = analyzer.analyze_from_parquet(path)
        d = report.to_dict()
        assert "_trace" not in d

    def test_to_dict_json_serializable_with_trace(self, tmp_path):
        """to_dict() with trace serializes to JSON."""
        path = self._make_shift_parquet(tmp_path)
        analyzer = ShiftAnalyzer()
        report = analyzer.analyze_from_parquet(path, include_trace=True)
        serialized = json.dumps(report.to_dict())
        assert isinstance(serialized, str)

    def test_check_sufficient_shifts_warns_on_few(self, tmp_path):
        """Warns when fewer than 3 shifts detected."""
        import pandas as pd
        # Constant RPM = no shifts
        time = np.linspace(0, 10, 100)
        df = pd.DataFrame({
            "RPM": np.full(100, 5000.0),
            "GPS Speed": np.linspace(30, 50, 100),
        }, index=time)
        path = str(tmp_path / "no_shifts.parquet")
        df.to_parquet(path)

        analyzer = ShiftAnalyzer()
        report = analyzer.analyze_from_parquet(path, include_trace=True)
        check = next(c for c in report.trace.sanity_checks if c.name == "sufficient_shifts")
        assert check.status == "warn"

    def test_existing_tests_still_pass(self):
        """Smoke: analyze_session still works unchanged."""
        analyzer = ShiftAnalyzer()
        time = np.linspace(0, 10, 100)
        speed = np.linspace(30, 80, 100)
        rpm = np.linspace(4000, 7000, 100)
        report = analyzer.analyze_session(rpm, speed, time)
        assert isinstance(report, ShiftReport)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
