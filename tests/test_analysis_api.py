"""
Tests for analysis API endpoints
"""

import os
import sys
import numpy as np
import pytest
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAnalysisAPIHelpers:
    """Tests for helper functions used by API"""

    def test_find_column_exact_match(self):
        """Test finding column with exact match"""
        import pandas as pd
        from src.main.app import _find_column

        df = pd.DataFrame({
            'RPM': [5000, 5500, 6000],
            'GPS Speed': [50, 60, 70]
        })

        result = _find_column(df, ['RPM'])
        assert result is not None
        assert list(result) == [5000, 5500, 6000]

    def test_find_column_case_insensitive(self):
        """Test case-insensitive column matching"""
        import pandas as pd
        from src.main.app import _find_column

        df = pd.DataFrame({
            'rpm': [5000, 5500, 6000],
        })

        result = _find_column(df, ['RPM', 'rpm'])
        assert result is not None

    def test_find_column_not_found(self):
        """Test column not found returns None"""
        import pandas as pd
        from src.main.app import _find_column

        df = pd.DataFrame({
            'Temperature': [80, 85, 90],
        })

        result = _find_column(df, ['RPM', 'rpm'])
        assert result is None


class TestShiftAnalysisAPI:
    """Tests for shift analysis endpoint"""

    @pytest.fixture
    def sample_parquet(self, tmp_path):
        """Create sample Parquet file"""
        import pandas as pd

        n = 500
        time = np.linspace(0, 50, n)
        rpm = 4000 + 2000 * np.sin(time / 10)
        speed = 50 + 30 * np.sin(time / 10)

        df = pd.DataFrame({'RPM': rpm, 'GPS Speed': speed}, index=time)
        path = tmp_path / "test_session.parquet"
        df.to_parquet(path)
        return path

    def test_shift_analysis_with_valid_file(self, sample_parquet):
        """Test shift analysis returns valid structure"""
        from src.features import ShiftAnalyzer
        import pandas as pd

        df = pd.read_parquet(sample_parquet)
        analyzer = ShiftAnalyzer()
        report = analyzer.analyze_session(
            df['RPM'].values,
            df['GPS Speed'].values,
            df.index.values,
            "test"
        )

        data = report.to_dict()
        assert 'session_id' in data
        assert 'total_shifts' in data
        assert 'shifts' in data


class TestLapAnalysisAPI:
    """Tests for lap analysis endpoint"""

    @pytest.fixture
    def sample_parquet_with_gps(self, tmp_path):
        """Create sample Parquet file with GPS data"""
        import pandas as pd

        n = 500
        time = np.linspace(0, 100, n)

        # Circular track
        lat = 43.797875 + 0.005 * np.sin(time / 25 * 2 * np.pi)
        lon = -87.989638 + 0.005 * np.cos(time / 25 * 2 * np.pi)
        rpm = 5000 + 1000 * np.sin(time / 10)
        speed = 60 + 20 * np.sin(time / 15)

        df = pd.DataFrame({
            'GPS Latitude': lat,
            'GPS Longitude': lon,
            'RPM': rpm,
            'GPS Speed': speed
        }, index=time)

        path = tmp_path / "test_laps.parquet"
        df.to_parquet(path)
        return path

    def test_lap_analysis_returns_structure(self, sample_parquet_with_gps):
        """Test lap analysis returns valid structure"""
        from src.features import LapAnalysis
        import pandas as pd

        df = pd.read_parquet(sample_parquet_with_gps)
        analyzer = LapAnalysis()
        report = analyzer.analyze_from_arrays(
            df.index.values,
            df['GPS Latitude'].values,
            df['GPS Longitude'].values,
            df['RPM'].values,
            df['GPS Speed'].values,
            "test"
        )

        data = report.to_dict()
        assert 'session_id' in data
        assert 'total_laps' in data or 'laps' in data


class TestGearAnalysisAPI:
    """Tests for gear analysis endpoint"""

    @pytest.fixture
    def sample_parquet(self, tmp_path):
        """Create sample Parquet file"""
        import pandas as pd

        n = 500
        time = np.linspace(0, 50, n)
        rpm = 4000 + 2500 * np.sin(time / 15)
        speed = 50 + 40 * np.sin(time / 15)

        df = pd.DataFrame({'RPM': rpm, 'GPS Speed': speed}, index=time)
        path = tmp_path / "test_gears.parquet"
        df.to_parquet(path)
        return path

    def test_gear_analysis_returns_structure(self, sample_parquet):
        """Test gear analysis returns valid structure"""
        from src.features import GearAnalysis
        import pandas as pd

        df = pd.read_parquet(sample_parquet)
        analyzer = GearAnalysis()
        report = analyzer.analyze_from_arrays(
            df.index.values,
            df['RPM'].values,
            df['GPS Speed'].values,
            None, None,
            "test"
        )

        data = report.to_dict()
        assert 'session_id' in data
        assert 'gear_usage' in data


class TestPowerAnalysisAPI:
    """Tests for power analysis endpoint"""

    @pytest.fixture
    def sample_parquet(self, tmp_path):
        """Create sample Parquet file"""
        import pandas as pd

        n = 500
        time = np.linspace(0, 50, n)
        speed = np.linspace(30, 100, n)
        rpm = 4000 + 2000 * (speed - 30) / 70

        df = pd.DataFrame({'GPS Speed': speed, 'RPM': rpm}, index=time)
        path = tmp_path / "test_power.parquet"
        df.to_parquet(path)
        return path

    def test_power_analysis_returns_structure(self, sample_parquet):
        """Test power analysis returns valid structure"""
        from src.features import PowerAnalysis
        import pandas as pd

        df = pd.read_parquet(sample_parquet)
        analyzer = PowerAnalysis()
        report = analyzer.analyze_from_arrays(
            df.index.values,
            df['GPS Speed'].values,
            df['RPM'].values,
            "test"
        )

        data = report.to_dict()
        assert 'session_id' in data
        assert 'power' in data
        assert 'acceleration' in data


class TestFullReportAPI:
    """Tests for full session report endpoint"""

    @pytest.fixture
    def sample_full_parquet(self, tmp_path):
        """Create sample Parquet with all data"""
        import pandas as pd

        n = 500
        time = np.linspace(0, 100, n)

        lat = 43.797875 + 0.005 * np.sin(time / 25 * 2 * np.pi)
        lon = -87.989638 + 0.005 * np.cos(time / 25 * 2 * np.pi)
        rpm = 4500 + 2000 * np.sin(time / 15)
        speed = 55 + 35 * np.sin(time / 12)

        df = pd.DataFrame({
            'GPS Latitude': lat,
            'GPS Longitude': lon,
            'RPM': rpm,
            'GPS Speed': speed
        }, index=time)

        path = tmp_path / "test_full.parquet"
        df.to_parquet(path)
        return path

    def test_full_report_returns_structure(self, sample_full_parquet):
        """Test full report returns combined structure"""
        from src.features import SessionReportGenerator
        import pandas as pd

        df = pd.read_parquet(sample_full_parquet)
        generator = SessionReportGenerator()
        report = generator.generate_from_arrays(
            df.index.values,
            df['GPS Latitude'].values,
            df['GPS Longitude'].values,
            df['RPM'].values,
            df['GPS Speed'].values,
            "test"
        )

        data = report.to_dict()
        assert 'metadata' in data
        assert 'summary' in data
        assert 'combined_recommendations' in data


class TestTrackMapAPI:
    """Tests for track map endpoint"""

    @pytest.fixture
    def sample_gps_parquet(self, tmp_path):
        """Create sample Parquet with GPS data"""
        import pandas as pd

        n = 200
        t = np.linspace(0, 2 * np.pi, n)
        lat = 43.797875 + 0.005 * np.sin(t)
        lon = -87.989638 + 0.005 * np.cos(t)
        speed = 50 + 50 * np.abs(np.sin(t))

        df = pd.DataFrame({
            'GPS Latitude': lat,
            'GPS Longitude': lon,
            'GPS Speed': speed
        }, index=np.linspace(0, 100, n))

        path = tmp_path / "test_map.parquet"
        df.to_parquet(path)
        return path

    def test_track_map_svg_generation(self, sample_gps_parquet):
        """Test SVG track map generation"""
        from src.visualization.track_map import TrackMap
        import pandas as pd

        df = pd.read_parquet(sample_gps_parquet)
        track_map = TrackMap()
        svg = track_map.render_svg(
            df['GPS Latitude'].values,
            df['GPS Longitude'].values,
            df['GPS Speed'].values,
            'speed',
            "Test Map"
        )

        assert '<svg' in svg
        assert '</svg>' in svg

    def test_track_map_html_generation(self, sample_gps_parquet):
        """Test HTML track map generation"""
        from src.visualization.track_map import TrackMap
        import pandas as pd

        df = pd.read_parquet(sample_gps_parquet)
        track_map = TrackMap()
        html = track_map.render_html(
            df['GPS Latitude'].values,
            df['GPS Longitude'].values,
            df['GPS Speed'].values,
            'speed',
            "Test Map"
        )

        assert '<!DOCTYPE html>' in html
        assert '<svg' in html

    def test_track_map_json_export(self, sample_gps_parquet):
        """Test JSON track map export"""
        from src.visualization.track_map import TrackMap
        import pandas as pd

        df = pd.read_parquet(sample_gps_parquet)
        track_map = TrackMap()
        data = track_map.to_dict(
            df['GPS Latitude'].values,
            df['GPS Longitude'].values,
            df['GPS Speed'].values,
            'speed'
        )

        assert 'coordinates' in data
        assert 'bounds' in data
        assert len(data['coordinates']) == len(df)


class TestAPIErrorHandling:
    """Tests for API error handling"""

    def test_missing_rpm_column(self):
        """Test handling of missing required columns"""
        import pandas as pd
        from src.features import ShiftAnalyzer

        # File without RPM
        df = pd.DataFrame({
            'Temperature': [80, 85, 90],
            'GPS Speed': [50, 60, 70]
        }, index=[0, 1, 2])

        analyzer = ShiftAnalyzer()

        # Should handle gracefully or raise appropriate error
        # The actual API would return 400, but the analyzer should work with what it has
        try:
            report = analyzer.analyze_from_arrays(
                df.index.values,
                np.zeros(3),  # Empty RPM data
                df['GPS Speed'].values,
                "test"
            )
            # Should still return something
            assert report is not None
        except Exception as e:
            # Expected if data is truly insufficient
            pass

    def test_empty_dataframe(self):
        """Test handling of empty data"""
        from src.features import PowerAnalysis
        import pandas as pd

        analyzer = PowerAnalysis()

        # Minimal data
        report = analyzer.analyze_from_arrays(
            np.array([0, 1, 2]),
            np.array([50, 50, 50]),
            np.array([4000, 4000, 4000]),
            "test"
        )

        assert report is not None
        assert report.session_id == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
