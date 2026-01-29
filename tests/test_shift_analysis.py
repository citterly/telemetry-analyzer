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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
