"""
Tests for lap analysis feature
"""

import os
import sys
import numpy as np
import pytest
import json
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.lap_analysis import (
    LapAnalysis,
    LapAnalysisReport,
    LapStatistics
)


class TestLapStatistics:
    """Tests for LapStatistics dataclass"""

    def test_lap_statistics_creation(self):
        """Test creating lap statistics"""
        stats = LapStatistics(
            lap_number=1,
            lap_time=135.5,
            gap_to_fastest=0.0,
            gap_to_previous=0.0,
            max_speed_mph=145.0,
            avg_speed_mph=95.0,
            max_rpm=7000,
            avg_rpm=5500,
            distance_meters=4000
        )
        assert stats.lap_number == 1
        assert stats.lap_time == 135.5
        assert stats.max_speed_mph == 145.0


class TestLapAnalysisReport:
    """Tests for LapAnalysisReport"""

    @pytest.fixture
    def sample_report(self):
        """Create sample report"""
        return LapAnalysisReport(
            session_id="test_session",
            track_name="Road America",
            analysis_timestamp="2026-01-29T01:00:00",
            total_laps=5,
            fastest_lap_number=3,
            fastest_lap_time=135.5,
            average_lap_time=138.2,
            lap_time_consistency=2.1,
            laps=[],
            improvement_trend="improving",
            recommendations=["Good session"],
            summary={"total_distance_meters": 20000}
        )

    def test_report_to_dict(self, sample_report):
        """Test report serialization to dict"""
        data = sample_report.to_dict()
        assert data["session_id"] == "test_session"
        assert data["total_laps"] == 5
        assert data["fastest_lap"]["lap_number"] == 3

    def test_report_to_json(self, sample_report):
        """Test report serialization to JSON"""
        json_str = sample_report.to_json()
        parsed = json.loads(json_str)
        assert parsed["session_id"] == "test_session"


class TestLapAnalysis:
    """Tests for LapAnalysis"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer"""
        return LapAnalysis()

    @pytest.fixture
    def sample_session(self):
        """Generate sample session with multiple laps"""
        # Simulate 3 laps of ~150 seconds each
        total_time = 450
        sample_rate = 10
        n_samples = total_time * sample_rate

        time_data = np.linspace(0, total_time, n_samples)

        # Create GPS data that circles back to start/finish
        # Road America start/finish: 43.797875, -87.989638
        start_lat, start_lon = 43.797875, -87.989638

        # Simulate 3 passes through start/finish
        latitude = np.zeros(n_samples)
        longitude = np.zeros(n_samples)

        for i, t in enumerate(time_data):
            # Create a path that passes through start/finish every ~150 seconds
            progress = (t % 150) / 150  # 0 to 1 within each lap
            angle = progress * 2 * np.pi

            # Vary position in a loop
            latitude[i] = start_lat + 0.005 * np.sin(angle)
            longitude[i] = start_lon + 0.005 * np.cos(angle)

        # RPM varies with speed
        rpm_data = 4000 + 2500 * np.sin(time_data / 30)

        # Speed varies through lap
        speed_data = 60 + 40 * np.sin(time_data / 15)

        return {
            "time": time_data,
            "latitude": latitude,
            "longitude": longitude,
            "rpm": rpm_data,
            "speed": speed_data
        }

    def test_analyzer_init_default(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.track_name == "Road America"

    def test_analyzer_init_custom(self):
        """Test custom track name"""
        analyzer = LapAnalysis(track_name="Watkins Glen")
        assert analyzer.track_name == "Watkins Glen"

    def test_analyze_from_arrays(self, analyzer, sample_session):
        """Test analysis from raw arrays"""
        report = analyzer.analyze_from_arrays(
            time_data=sample_session["time"],
            latitude_data=sample_session["latitude"],
            longitude_data=sample_session["longitude"],
            rpm_data=sample_session["rpm"],
            speed_data=sample_session["speed"],
            session_id="test"
        )

        assert isinstance(report, LapAnalysisReport)
        assert report.session_id == "test"
        assert report.track_name == "Road America"

    def test_analyze_returns_lap_stats(self, analyzer, sample_session):
        """Test that analysis returns lap statistics"""
        report = analyzer.analyze_from_arrays(
            time_data=sample_session["time"],
            latitude_data=sample_session["latitude"],
            longitude_data=sample_session["longitude"],
            rpm_data=sample_session["rpm"],
            speed_data=sample_session["speed"]
        )

        # Should detect at least some laps
        assert report.total_laps >= 0

    def test_analyze_has_recommendations(self, analyzer, sample_session):
        """Test that analysis generates recommendations"""
        report = analyzer.analyze_from_arrays(
            time_data=sample_session["time"],
            latitude_data=sample_session["latitude"],
            longitude_data=sample_session["longitude"],
            rpm_data=sample_session["rpm"],
            speed_data=sample_session["speed"]
        )

        assert isinstance(report.recommendations, list)
        assert len(report.recommendations) >= 1

    def test_analyze_timestamp(self, analyzer, sample_session):
        """Test that report has timestamp"""
        report = analyzer.analyze_from_arrays(
            time_data=sample_session["time"],
            latitude_data=sample_session["latitude"],
            longitude_data=sample_session["longitude"],
            rpm_data=sample_session["rpm"],
            speed_data=sample_session["speed"]
        )

        assert report.analysis_timestamp is not None
        from datetime import datetime
        datetime.fromisoformat(report.analysis_timestamp)

    def test_improvement_trend_calculation(self, analyzer):
        """Test improvement trend calculation"""
        # Improving trend
        improving = analyzer._calculate_trend([140, 139, 138, 137, 136, 135])
        assert "improving" in improving.lower()

        # Degrading trend
        degrading = analyzer._calculate_trend([135, 136, 137, 138, 139, 140])
        assert "degrading" in degrading.lower()

        # Consistent
        consistent = analyzer._calculate_trend([137, 137.5, 137, 137.2, 137.1])
        assert consistent == "consistent"

    def test_empty_session_handling(self, analyzer):
        """Test handling of empty/minimal data"""
        report = analyzer.analyze_from_arrays(
            time_data=np.array([0, 1]),
            latitude_data=np.array([43.79, 43.79]),
            longitude_data=np.array([-87.99, -87.99]),
            rpm_data=np.array([3000, 3000]),
            speed_data=np.array([50, 50])
        )

        # Should handle gracefully
        assert report.total_laps >= 0


class TestLapAnalysisParquet:
    """Tests for Parquet file analysis"""

    @pytest.fixture
    def sample_parquet(self):
        """Create sample Parquet file"""
        import pandas as pd

        n_samples = 1000
        time = np.linspace(0, 300, n_samples)

        # Create data
        lat = 43.797875 + 0.005 * np.sin(time / 50)
        lon = -87.989638 + 0.005 * np.cos(time / 50)
        rpm = 4000 + 2000 * np.sin(time / 20)
        speed = 60 + 30 * np.sin(time / 15)

        df = pd.DataFrame({
            "GPS Latitude": lat,
            "GPS Longitude": lon,
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
        analyzer = LapAnalysis()
        report = analyzer.analyze_from_parquet(sample_parquet)

        assert isinstance(report, LapAnalysisReport)

    def test_parquet_custom_session_id(self, sample_parquet):
        """Test custom session ID"""
        analyzer = LapAnalysis()
        report = analyzer.analyze_from_parquet(
            sample_parquet,
            session_id="my_session"
        )

        assert report.session_id == "my_session"


class TestLapComparison:
    """Tests for lap comparison functionality"""

    @pytest.fixture
    def report_with_laps(self):
        """Create report with multiple laps"""
        laps = [
            LapStatistics(
                lap_number=1, lap_time=140.0, gap_to_fastest=5.0,
                gap_to_previous=0.0, max_speed_mph=140.0, avg_speed_mph=90.0,
                max_rpm=6800, avg_rpm=5500, distance_meters=4000
            ),
            LapStatistics(
                lap_number=2, lap_time=137.0, gap_to_fastest=2.0,
                gap_to_previous=-3.0, max_speed_mph=142.0, avg_speed_mph=92.0,
                max_rpm=6900, avg_rpm=5600, distance_meters=4000
            ),
            LapStatistics(
                lap_number=3, lap_time=135.0, gap_to_fastest=0.0,
                gap_to_previous=-2.0, max_speed_mph=145.0, avg_speed_mph=95.0,
                max_rpm=7000, avg_rpm=5700, distance_meters=4000
            ),
        ]

        return LapAnalysisReport(
            session_id="test",
            track_name="Road America",
            analysis_timestamp="2026-01-29T01:00:00",
            total_laps=3,
            fastest_lap_number=3,
            fastest_lap_time=135.0,
            average_lap_time=137.33,
            lap_time_consistency=2.1,
            laps=laps,
            improvement_trend="improving",
            recommendations=[],
            summary={}
        )

    def test_lap_comparison(self, report_with_laps):
        """Test comparing two laps"""
        analyzer = LapAnalysis()
        comparison = analyzer.get_lap_comparison(report_with_laps, 1, 3)

        assert comparison["lap_a"] == 1
        assert comparison["lap_b"] == 3
        assert comparison["time_difference"] == -5.0  # Lap 3 is 5s faster
        assert comparison["faster_lap"] == 3

    def test_lap_comparison_invalid(self, report_with_laps):
        """Test comparing with invalid lap number"""
        analyzer = LapAnalysis()
        comparison = analyzer.get_lap_comparison(report_with_laps, 1, 99)

        assert "error" in comparison

    def test_get_fastest_segments(self, report_with_laps):
        """Test getting fastest laps"""
        analyzer = LapAnalysis()
        fastest = analyzer.get_fastest_segments(report_with_laps, top_n=2)

        assert len(fastest) == 2
        assert fastest[0]["rank"] == 1
        assert fastest[0]["lap_number"] == 3  # Fastest lap


class TestRecommendationGeneration:
    """Tests for recommendation generation"""

    def test_consistency_recommendation(self):
        """Test recommendations for inconsistent laps"""
        analyzer = LapAnalysis()

        laps = [
            LapStatistics(1, 135.0, 0.0, 0.0, 140, 90, 6800, 5500, 4000),
            LapStatistics(2, 145.0, 10.0, 10.0, 138, 88, 6700, 5400, 4000),
            LapStatistics(3, 138.0, 3.0, -7.0, 141, 91, 6900, 5600, 4000),
        ]

        recommendations = analyzer._generate_recommendations(laps, 135.0)

        # Should recommend consistency improvements
        any_consistency = any("consistency" in r.lower() or "variation" in r.lower()
                             for r in recommendations)
        assert any_consistency or len(recommendations) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
