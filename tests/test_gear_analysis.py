"""
Tests for gear usage analysis feature
"""

import os
import sys
import numpy as np
import pytest
import json
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.gear_analysis import (
    GearAnalysis,
    GearAnalysisReport,
    GearUsageStats,
    TrackSectionStats
)


class TestGearUsageStats:
    """Tests for GearUsageStats dataclass"""

    def test_gear_usage_stats_creation(self):
        """Test creating gear usage statistics"""
        stats = GearUsageStats(
            gear_number=2,
            time_seconds=45.0,
            usage_percent=25.0,
            sample_count=450,
            speed_min_mph=35.0,
            speed_max_mph=65.0,
            speed_avg_mph=50.0,
            rpm_min=4000,
            rpm_max=6500,
            rpm_avg=5200,
            shift_in_count=12,
            shift_out_count=12
        )
        assert stats.gear_number == 2
        assert stats.time_seconds == 45.0
        assert stats.usage_percent == 25.0
        assert stats.shift_in_count == 12


class TestTrackSectionStats:
    """Tests for TrackSectionStats dataclass"""

    def test_track_section_stats_creation(self):
        """Test creating track section statistics"""
        stats = TrackSectionStats(
            section_name="T1",
            dominant_gear=3,
            gear_distribution={2: 20.0, 3: 60.0, 4: 20.0},
            avg_speed_mph=75.0,
            avg_rpm=5500,
            sample_count=100
        )
        assert stats.section_name == "T1"
        assert stats.dominant_gear == 3
        assert stats.gear_distribution[3] == 60.0


class TestGearAnalysisReport:
    """Tests for GearAnalysisReport"""

    @pytest.fixture
    def sample_report(self):
        """Create sample report"""
        gear_usage = [
            GearUsageStats(
                gear_number=2, time_seconds=30.0, usage_percent=20.0,
                sample_count=300, speed_min_mph=30, speed_max_mph=60,
                speed_avg_mph=45, rpm_min=4000, rpm_max=6500, rpm_avg=5200,
                shift_in_count=8, shift_out_count=8
            ),
            GearUsageStats(
                gear_number=3, time_seconds=60.0, usage_percent=40.0,
                sample_count=600, speed_min_mph=50, speed_max_mph=90,
                speed_avg_mph=70, rpm_min=4500, rpm_max=6800, rpm_avg=5500,
                shift_in_count=10, shift_out_count=10
            ),
        ]

        return GearAnalysisReport(
            session_id="test_session",
            track_name="Road America",
            analysis_timestamp="2026-01-29T02:00:00",
            total_duration_seconds=150.0,
            gear_usage=gear_usage,
            track_sections=[],
            shift_summary={
                "total_shifts": 20,
                "upshifts": 10,
                "downshifts": 10
            },
            rpm_analysis={
                "avg_rpm": 5400,
                "max_rpm": 6800,
                "time_over_safe_limit_pct": 2.5,
                "time_in_power_band_pct": 45.0
            },
            recommendations=["Good session"],
            summary={"gears_used": 2}
        )

    def test_report_to_dict(self, sample_report):
        """Test report serialization to dict"""
        data = sample_report.to_dict()
        assert data["session_id"] == "test_session"
        assert data["total_duration_seconds"] == 150.0
        assert len(data["gear_usage"]) == 2
        assert data["gear_usage"][0]["gear_number"] == 2

    def test_report_to_json(self, sample_report):
        """Test report serialization to JSON"""
        json_str = sample_report.to_json()
        parsed = json.loads(json_str)
        assert parsed["session_id"] == "test_session"


class TestGearAnalysis:
    """Tests for GearAnalysis"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer"""
        return GearAnalysis()

    @pytest.fixture
    def sample_session(self):
        """Generate sample session with gear changes"""
        total_time = 180  # 3 minutes
        sample_rate = 10
        n_samples = total_time * sample_rate

        time_data = np.linspace(0, total_time, n_samples)

        # Simulate acceleration run through gears with varying patterns
        rpm_data = np.zeros(n_samples)
        speed_data = np.zeros(n_samples)

        # Gear 1: 0-15s (accelerating)
        rpm_data[0:150] = np.linspace(2000, 6500, 150)
        speed_data[0:150] = np.linspace(0, 35, 150)

        # Gear 2: 15-35s
        rpm_data[150:350] = np.linspace(4500, 6500, 200)
        speed_data[150:350] = np.linspace(35, 60, 200)

        # Gear 3: 35-70s
        rpm_data[350:700] = np.linspace(4800, 6800, 350)
        speed_data[350:700] = np.linspace(60, 100, 350)

        # Gear 4: 70-100s
        rpm_data[700:1000] = np.linspace(5000, 6500, 300)
        speed_data[700:1000] = np.linspace(100, 130, 300)

        # Braking: 100-120s (downshifts)
        rpm_data[1000:1200] = np.linspace(6000, 4000, 200)
        speed_data[1000:1200] = np.linspace(130, 50, 200)

        # Another acceleration: 120-180s
        rpm_data[1200:1400] = np.linspace(5000, 6500, 200)
        speed_data[1200:1400] = np.linspace(50, 90, 200)

        rpm_data[1400:1800] = np.linspace(5500, 6200, 400)
        speed_data[1400:1800] = np.linspace(90, 120, 400)

        return {
            "time": time_data,
            "rpm": rpm_data,
            "speed": speed_data
        }

    def test_analyzer_init_default(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.track_name == "Road America"
        assert analyzer.scenario_name == "Current Setup"

    def test_analyzer_init_custom(self):
        """Test custom initialization"""
        analyzer = GearAnalysis(
            track_name="Watkins Glen",
            scenario_name="New Trans + Current Final"
        )
        assert analyzer.track_name == "Watkins Glen"
        assert analyzer.scenario_name == "New Trans + Current Final"

    def test_analyze_from_arrays(self, analyzer, sample_session):
        """Test analysis from raw arrays"""
        report = analyzer.analyze_from_arrays(
            time_data=sample_session["time"],
            rpm_data=sample_session["rpm"],
            speed_data=sample_session["speed"],
            session_id="test"
        )

        assert isinstance(report, GearAnalysisReport)
        assert report.session_id == "test"
        assert report.track_name == "Road America"

    def test_analyze_returns_gear_usage(self, analyzer, sample_session):
        """Test that analysis returns gear usage statistics"""
        report = analyzer.analyze_from_arrays(
            time_data=sample_session["time"],
            rpm_data=sample_session["rpm"],
            speed_data=sample_session["speed"]
        )

        assert len(report.gear_usage) > 0
        assert all(isinstance(gu, GearUsageStats) for gu in report.gear_usage)

    def test_gear_usage_has_all_fields(self, analyzer, sample_session):
        """Test that gear usage has required fields"""
        report = analyzer.analyze_from_arrays(
            time_data=sample_session["time"],
            rpm_data=sample_session["rpm"],
            speed_data=sample_session["speed"]
        )

        for gu in report.gear_usage:
            assert gu.gear_number > 0
            assert gu.time_seconds >= 0
            assert 0 <= gu.usage_percent <= 100
            assert gu.speed_min_mph <= gu.speed_max_mph
            assert gu.rpm_min <= gu.rpm_max

    def test_analyze_has_shift_summary(self, analyzer, sample_session):
        """Test that analysis has shift summary"""
        report = analyzer.analyze_from_arrays(
            time_data=sample_session["time"],
            rpm_data=sample_session["rpm"],
            speed_data=sample_session["speed"]
        )

        assert "total_shifts" in report.shift_summary
        assert "upshifts" in report.shift_summary
        assert "downshifts" in report.shift_summary

    def test_analyze_has_rpm_analysis(self, analyzer, sample_session):
        """Test that analysis has RPM analysis"""
        report = analyzer.analyze_from_arrays(
            time_data=sample_session["time"],
            rpm_data=sample_session["rpm"],
            speed_data=sample_session["speed"]
        )

        assert "avg_rpm" in report.rpm_analysis
        assert "max_rpm" in report.rpm_analysis
        assert "time_in_power_band_pct" in report.rpm_analysis

    def test_analyze_has_recommendations(self, analyzer, sample_session):
        """Test that analysis generates recommendations"""
        report = analyzer.analyze_from_arrays(
            time_data=sample_session["time"],
            rpm_data=sample_session["rpm"],
            speed_data=sample_session["speed"]
        )

        assert isinstance(report.recommendations, list)
        assert len(report.recommendations) >= 1

    def test_analyze_timestamp(self, analyzer, sample_session):
        """Test that report has timestamp"""
        report = analyzer.analyze_from_arrays(
            time_data=sample_session["time"],
            rpm_data=sample_session["rpm"],
            speed_data=sample_session["speed"]
        )

        assert report.analysis_timestamp is not None
        from datetime import datetime
        datetime.fromisoformat(report.analysis_timestamp)

    def test_total_duration(self, analyzer, sample_session):
        """Test total duration calculation"""
        report = analyzer.analyze_from_arrays(
            time_data=sample_session["time"],
            rpm_data=sample_session["rpm"],
            speed_data=sample_session["speed"]
        )

        # Session is 180 seconds
        assert abs(report.total_duration_seconds - 180.0) < 1.0


class TestGearAnalysisParquet:
    """Tests for Parquet file analysis"""

    @pytest.fixture
    def sample_parquet(self):
        """Create sample Parquet file"""
        import pandas as pd

        n_samples = 1000
        time = np.linspace(0, 100, n_samples)

        # Create varying RPM and speed data
        rpm = 4000 + 2000 * np.sin(time / 20)
        speed = 50 + 30 * np.sin(time / 15)

        df = pd.DataFrame({
            "RPM": rpm,
            "GPS Speed": speed
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
        analyzer = GearAnalysis()
        report = analyzer.analyze_from_parquet(sample_parquet)

        assert isinstance(report, GearAnalysisReport)

    def test_parquet_custom_session_id(self, sample_parquet):
        """Test custom session ID"""
        analyzer = GearAnalysis()
        report = analyzer.analyze_from_parquet(
            sample_parquet,
            session_id="my_session"
        )

        assert report.session_id == "my_session"


class TestGearComparison:
    """Tests for gear comparison functionality"""

    @pytest.fixture
    def report_with_gears(self):
        """Create report with multiple gears"""
        gear_usage = [
            GearUsageStats(
                gear_number=2, time_seconds=30.0, usage_percent=20.0,
                sample_count=300, speed_min_mph=30, speed_max_mph=60,
                speed_avg_mph=45, rpm_min=4000, rpm_max=6500, rpm_avg=5200,
                shift_in_count=8, shift_out_count=8
            ),
            GearUsageStats(
                gear_number=3, time_seconds=60.0, usage_percent=40.0,
                sample_count=600, speed_min_mph=50, speed_max_mph=90,
                speed_avg_mph=70, rpm_min=4500, rpm_max=6800, rpm_avg=5500,
                shift_in_count=10, shift_out_count=10
            ),
            GearUsageStats(
                gear_number=4, time_seconds=45.0, usage_percent=30.0,
                sample_count=450, speed_min_mph=80, speed_max_mph=130,
                speed_avg_mph=105, rpm_min=5000, rpm_max=6500, rpm_avg=5800,
                shift_in_count=6, shift_out_count=6
            ),
        ]

        return GearAnalysisReport(
            session_id="test",
            track_name="Road America",
            analysis_timestamp="2026-01-29T02:00:00",
            total_duration_seconds=150.0,
            gear_usage=gear_usage,
            track_sections=[],
            shift_summary={},
            rpm_analysis={},
            recommendations=[],
            summary={}
        )

    def test_gear_comparison(self, report_with_gears):
        """Test comparing two gears"""
        analyzer = GearAnalysis()
        comparison = analyzer.get_gear_comparison(report_with_gears, 2, 3)

        assert comparison["gear_a"] == 2
        assert comparison["gear_b"] == 3
        assert comparison["usage_difference_pct"] == 20.0  # 40 - 20
        assert comparison["avg_speed_difference"] == 25.0  # 70 - 45

    def test_gear_comparison_invalid(self, report_with_gears):
        """Test comparing with invalid gear number"""
        analyzer = GearAnalysis()
        comparison = analyzer.get_gear_comparison(report_with_gears, 2, 99)

        assert "error" in comparison


class TestOptimalGear:
    """Tests for optimal gear calculation"""

    def test_optimal_gear_at_speed(self):
        """Test getting optimal gear at various speeds"""
        analyzer = GearAnalysis()

        # Low speed - may not have suitable gear in power band
        result = analyzer.get_optimal_gear_at_speed(40)
        # At 40mph may return error or low gear depending on ratios
        assert "recommended_gear" in result or "error" in result

        # Medium speed - should find a gear
        result = analyzer.get_optimal_gear_at_speed(70)
        assert "recommended_gear" in result

        # High speed - should be higher gear
        result = analyzer.get_optimal_gear_at_speed(110)
        assert "recommended_gear" in result
        assert result["recommended_gear"] >= 3

    def test_optimal_gear_includes_rpm_info(self):
        """Test that optimal gear includes RPM information"""
        analyzer = GearAnalysis()
        result = analyzer.get_optimal_gear_at_speed(80)

        assert "rpm_at_speed" in result
        assert "in_power_band" in result
        assert "rpm_headroom" in result


class TestRecommendationGeneration:
    """Tests for recommendation generation"""

    def test_over_rev_recommendation(self):
        """Test recommendations for over-revving"""
        analyzer = GearAnalysis()

        # Create session with high RPMs
        n_samples = 1000
        time = np.linspace(0, 100, n_samples)
        rpm = np.full(n_samples, 7200)  # Consistently high RPM
        speed = np.linspace(50, 130, n_samples)

        report = analyzer.analyze_from_arrays(time, rpm, speed)

        # Should recommend earlier upshifts
        any_overrev = any("over" in r.lower() or "earlier" in r.lower()
                         for r in report.recommendations)
        assert any_overrev or report.rpm_analysis['time_over_safe_limit_pct'] > 0

    def test_power_band_recommendation(self):
        """Test recommendations for poor power band usage"""
        analyzer = GearAnalysis()

        # Create session with low RPMs (outside power band)
        n_samples = 1000
        time = np.linspace(0, 100, n_samples)
        rpm = np.full(n_samples, 3500)  # Below power band
        speed = np.linspace(20, 80, n_samples)

        report = analyzer.analyze_from_arrays(time, rpm, speed)

        # Should have low power band percentage
        assert report.rpm_analysis['time_in_power_band_pct'] < 50


class TestMinimalData:
    """Tests for handling edge cases"""

    def test_empty_session_handling(self):
        """Test handling of minimal data"""
        analyzer = GearAnalysis()

        report = analyzer.analyze_from_arrays(
            time_data=np.array([0, 1]),
            rpm_data=np.array([3000, 3000]),
            speed_data=np.array([50, 50])
        )

        # Should handle gracefully
        assert report is not None
        assert len(report.gear_usage) >= 0

    def test_constant_gear_session(self):
        """Test session that stays in one gear"""
        analyzer = GearAnalysis()

        # Session at constant speed/rpm (one gear)
        n_samples = 500
        time = np.linspace(0, 50, n_samples)
        rpm = np.full(n_samples, 5500)
        speed = np.full(n_samples, 85)

        report = analyzer.analyze_from_arrays(time, rpm, speed)

        # Should detect single gear usage
        assert len(report.gear_usage) >= 1
        if len(report.gear_usage) == 1:
            assert report.gear_usage[0].usage_percent > 90


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
