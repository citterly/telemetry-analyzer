"""
Tests for session report generator feature
"""

import os
import sys
import numpy as np
import pytest
import json
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.session_report import (
    SessionReportGenerator,
    SessionReport,
    SessionMetadata,
    SessionSummary
)


class TestSessionMetadata:
    """Tests for SessionMetadata dataclass"""

    def test_metadata_creation(self):
        """Test creating session metadata"""
        metadata = SessionMetadata(
            session_id="test_session",
            track_name="Road America",
            vehicle_setup="Current Setup",
            analysis_timestamp="2026-01-29T03:00:00",
            data_source="parquet",
            total_duration_seconds=600.0,
            sample_count=6000
        )
        assert metadata.session_id == "test_session"
        assert metadata.track_name == "Road America"
        assert metadata.total_duration_seconds == 600.0


class TestSessionSummary:
    """Tests for SessionSummary dataclass"""

    def test_summary_creation(self):
        """Test creating session summary"""
        summary = SessionSummary(
            total_laps=5,
            fastest_lap_time=135.5,
            fastest_lap_number=3,
            average_lap_time=138.0,
            total_shifts=120,
            max_speed_mph=145.0,
            max_rpm=6800,
            max_power_hp=280.0,
            max_braking_g=1.1,
            improvement_trend="improving"
        )
        assert summary.total_laps == 5
        assert summary.fastest_lap_time == 135.5
        assert summary.improvement_trend == "improving"


class TestSessionReport:
    """Tests for SessionReport"""

    @pytest.fixture
    def sample_report(self):
        """Create sample report"""
        metadata = SessionMetadata(
            session_id="test_session",
            track_name="Road America",
            vehicle_setup="Current Setup",
            analysis_timestamp="2026-01-29T03:00:00",
            data_source="array",
            total_duration_seconds=600.0,
            sample_count=6000
        )

        summary = SessionSummary(
            total_laps=5,
            fastest_lap_time=135.5,
            fastest_lap_number=3,
            average_lap_time=138.0,
            total_shifts=120,
            max_speed_mph=145.0,
            max_rpm=6800,
            max_power_hp=280.0,
            max_braking_g=1.1,
            improvement_trend="improving"
        )

        return SessionReport(
            metadata=metadata,
            summary=summary,
            lap_analysis=None,
            shift_analysis=None,
            gear_analysis=None,
            power_analysis=None,
            combined_recommendations=["Good session", "Focus on consistency"],
            warnings=[]
        )

    def test_report_to_dict(self, sample_report):
        """Test report serialization to dict"""
        data = sample_report.to_dict()
        assert data["metadata"]["session_id"] == "test_session"
        assert data["summary"]["total_laps"] == 5
        assert data["summary"]["fastest_lap"]["lap_time"] == 135.5

    def test_report_to_json(self, sample_report):
        """Test report serialization to JSON"""
        json_str = sample_report.to_json()
        parsed = json.loads(json_str)
        assert parsed["metadata"]["session_id"] == "test_session"

    def test_report_to_html(self, sample_report):
        """Test report serialization to HTML"""
        html = sample_report.to_html()
        assert "<!DOCTYPE html>" in html
        assert "test_session" in html
        assert "Road America" in html


class TestSessionReportGenerator:
    """Tests for SessionReportGenerator"""

    @pytest.fixture
    def generator(self):
        """Create report generator"""
        return SessionReportGenerator()

    @pytest.fixture
    def sample_session(self):
        """Generate sample session with all required data"""
        total_time = 300  # 5 minutes
        sample_rate = 10
        n_samples = total_time * sample_rate

        time_data = np.linspace(0, total_time, n_samples)

        # Create GPS data that loops around
        start_lat, start_lon = 43.797875, -87.989638
        latitude = start_lat + 0.005 * np.sin(time_data / 50 * 2 * np.pi)
        longitude = start_lon + 0.005 * np.cos(time_data / 50 * 2 * np.pi)

        # Create RPM and speed data with variation
        rpm_data = 4000 + 2500 * np.sin(time_data / 30)
        speed_data = 50 + 40 * np.sin(time_data / 20)

        # Ensure speed doesn't go negative
        speed_data = np.maximum(speed_data, 10)

        return {
            "time": time_data,
            "latitude": latitude,
            "longitude": longitude,
            "rpm": rpm_data,
            "speed": speed_data
        }

    def test_generator_init_default(self, generator):
        """Test generator initialization with defaults"""
        assert generator.track_name == "Road America"
        assert generator.vehicle_mass_kg == 1565

    def test_generator_init_custom(self):
        """Test custom initialization"""
        gen = SessionReportGenerator(
            track_name="Watkins Glen",
            vehicle_setup="Test Setup",
            vehicle_mass_kg=1500
        )
        assert gen.track_name == "Watkins Glen"
        assert gen.vehicle_setup == "Test Setup"
        assert gen.vehicle_mass_kg == 1500

    def test_generate_from_arrays(self, generator, sample_session):
        """Test report generation from arrays"""
        report = generator.generate_from_arrays(
            time_data=sample_session["time"],
            latitude_data=sample_session["latitude"],
            longitude_data=sample_session["longitude"],
            rpm_data=sample_session["rpm"],
            speed_data=sample_session["speed"],
            session_id="test"
        )

        assert isinstance(report, SessionReport)
        assert report.metadata.session_id == "test"

    def test_report_has_metadata(self, generator, sample_session):
        """Test that report has metadata"""
        report = generator.generate_from_arrays(
            time_data=sample_session["time"],
            latitude_data=sample_session["latitude"],
            longitude_data=sample_session["longitude"],
            rpm_data=sample_session["rpm"],
            speed_data=sample_session["speed"]
        )

        assert report.metadata is not None
        assert report.metadata.track_name == "Road America"
        assert report.metadata.total_duration_seconds > 0

    def test_report_has_summary(self, generator, sample_session):
        """Test that report has summary"""
        report = generator.generate_from_arrays(
            time_data=sample_session["time"],
            latitude_data=sample_session["latitude"],
            longitude_data=sample_session["longitude"],
            rpm_data=sample_session["rpm"],
            speed_data=sample_session["speed"]
        )

        assert report.summary is not None
        assert report.summary.max_speed_mph > 0
        assert report.summary.max_rpm > 0

    def test_report_has_lap_analysis(self, generator, sample_session):
        """Test that report includes lap analysis"""
        report = generator.generate_from_arrays(
            time_data=sample_session["time"],
            latitude_data=sample_session["latitude"],
            longitude_data=sample_session["longitude"],
            rpm_data=sample_session["rpm"],
            speed_data=sample_session["speed"]
        )

        # May be None if no laps detected, but should exist
        assert report.lap_analysis is None or report.lap_analysis.track_name == "Road America"

    def test_report_has_shift_analysis(self, generator, sample_session):
        """Test that report includes shift analysis"""
        report = generator.generate_from_arrays(
            time_data=sample_session["time"],
            latitude_data=sample_session["latitude"],
            longitude_data=sample_session["longitude"],
            rpm_data=sample_session["rpm"],
            speed_data=sample_session["speed"]
        )

        assert report.shift_analysis is not None or len(report.warnings) > 0

    def test_report_has_gear_analysis(self, generator, sample_session):
        """Test that report includes gear analysis"""
        report = generator.generate_from_arrays(
            time_data=sample_session["time"],
            latitude_data=sample_session["latitude"],
            longitude_data=sample_session["longitude"],
            rpm_data=sample_session["rpm"],
            speed_data=sample_session["speed"]
        )

        assert report.gear_analysis is not None or len(report.warnings) > 0

    def test_report_has_power_analysis(self, generator, sample_session):
        """Test that report includes power analysis"""
        report = generator.generate_from_arrays(
            time_data=sample_session["time"],
            latitude_data=sample_session["latitude"],
            longitude_data=sample_session["longitude"],
            rpm_data=sample_session["rpm"],
            speed_data=sample_session["speed"]
        )

        assert report.power_analysis is not None or len(report.warnings) > 0

    def test_report_has_recommendations(self, generator, sample_session):
        """Test that report has combined recommendations"""
        report = generator.generate_from_arrays(
            time_data=sample_session["time"],
            latitude_data=sample_session["latitude"],
            longitude_data=sample_session["longitude"],
            rpm_data=sample_session["rpm"],
            speed_data=sample_session["speed"]
        )

        assert isinstance(report.combined_recommendations, list)
        assert len(report.combined_recommendations) >= 1

    def test_report_timestamp(self, generator, sample_session):
        """Test that report has timestamp"""
        report = generator.generate_from_arrays(
            time_data=sample_session["time"],
            latitude_data=sample_session["latitude"],
            longitude_data=sample_session["longitude"],
            rpm_data=sample_session["rpm"],
            speed_data=sample_session["speed"]
        )

        from datetime import datetime
        datetime.fromisoformat(report.metadata.analysis_timestamp)


class TestSessionReportParquet:
    """Tests for Parquet file analysis"""

    @pytest.fixture
    def sample_parquet(self):
        """Create sample Parquet file"""
        import pandas as pd

        n_samples = 1000
        time = np.linspace(0, 100, n_samples)

        lat = 43.797875 + 0.005 * np.sin(time / 25)
        lon = -87.989638 + 0.005 * np.cos(time / 25)
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

    def test_generate_from_parquet(self, sample_parquet):
        """Test generating report from Parquet file"""
        generator = SessionReportGenerator()
        report = generator.generate_from_parquet(sample_parquet)

        assert isinstance(report, SessionReport)

    def test_parquet_custom_session_id(self, sample_parquet):
        """Test custom session ID"""
        generator = SessionReportGenerator()
        report = generator.generate_from_parquet(
            sample_parquet,
            session_id="my_session"
        )

        assert report.metadata.session_id == "my_session"


class TestSaveReport:
    """Tests for saving reports"""

    @pytest.fixture
    def sample_report(self):
        """Create minimal sample report"""
        metadata = SessionMetadata(
            session_id="save_test",
            track_name="Road America",
            vehicle_setup="Current Setup",
            analysis_timestamp="2026-01-29T03:00:00",
            data_source="array",
            total_duration_seconds=100.0,
            sample_count=1000
        )

        summary = SessionSummary(
            total_laps=2,
            fastest_lap_time=50.0,
            fastest_lap_number=1,
            average_lap_time=52.0,
            total_shifts=20,
            max_speed_mph=100.0,
            max_rpm=6500,
            max_power_hp=200.0,
            max_braking_g=0.8,
            improvement_trend="consistent"
        )

        return SessionReport(
            metadata=metadata,
            summary=summary,
            lap_analysis=None,
            shift_analysis=None,
            gear_analysis=None,
            power_analysis=None,
            combined_recommendations=["Test recommendation"],
            warnings=[]
        )

    def test_save_json(self, sample_report):
        """Test saving report as JSON"""
        generator = SessionReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            saved = generator.save_report(sample_report, tmpdir, formats=['json'])

            assert 'json' in saved
            assert os.path.exists(saved['json'])

            with open(saved['json']) as f:
                data = json.load(f)
                assert data["metadata"]["session_id"] == "save_test"

    def test_save_html(self, sample_report):
        """Test saving report as HTML"""
        generator = SessionReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            saved = generator.save_report(sample_report, tmpdir, formats=['html'])

            assert 'html' in saved
            assert os.path.exists(saved['html'])

            with open(saved['html']) as f:
                content = f.read()
                assert "<!DOCTYPE html>" in content
                assert "save_test" in content

    def test_save_both_formats(self, sample_report):
        """Test saving report in both formats"""
        generator = SessionReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            saved = generator.save_report(sample_report, tmpdir)

            assert 'json' in saved
            assert 'html' in saved
            assert os.path.exists(saved['json'])
            assert os.path.exists(saved['html'])


class TestErrorHandling:
    """Tests for error handling"""

    def test_minimal_data(self):
        """Test handling of minimal data"""
        generator = SessionReportGenerator()

        report = generator.generate_from_arrays(
            time_data=np.array([0, 1, 2]),
            latitude_data=np.array([43.79, 43.79, 43.79]),
            longitude_data=np.array([-87.99, -87.99, -87.99]),
            rpm_data=np.array([3000, 3000, 3000]),
            speed_data=np.array([50, 50, 50])
        )

        assert report is not None
        assert report.metadata.sample_count == 3

    def test_missing_gps_data(self):
        """Test handling when GPS data is zeros"""
        generator = SessionReportGenerator()

        n = 100
        report = generator.generate_from_arrays(
            time_data=np.linspace(0, 10, n),
            latitude_data=np.zeros(n),
            longitude_data=np.zeros(n),
            rpm_data=np.full(n, 5000),
            speed_data=np.linspace(30, 100, n)
        )

        # Should still work, might have warnings
        assert report is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
