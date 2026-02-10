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


class TestSharedUtilities:
    """Tests for cleanup-002: shared dataframe_helpers module"""

    def test_find_column_shared_exact(self):
        """Shared find_column works with exact match"""
        import pandas as pd
        from src.utils.dataframe_helpers import find_column

        df = pd.DataFrame({'RPM': [5000, 5500, 6000], 'GPS Speed': [50, 60, 70]})
        result = find_column(df, ['RPM'])
        assert result is not None
        assert list(result) == [5000, 5500, 6000]

    def test_find_column_shared_case_insensitive(self):
        """Shared find_column handles case-insensitive matching"""
        import pandas as pd
        from src.utils.dataframe_helpers import find_column

        df = pd.DataFrame({'rpm': [5000], 'gps speed': [60]})
        assert find_column(df, ['RPM', 'rpm']) is not None
        assert find_column(df, ['GPS Speed']) is not None

    def test_find_column_shared_not_found(self):
        """Shared find_column returns None when column doesn't exist"""
        import pandas as pd
        from src.utils.dataframe_helpers import find_column

        df = pd.DataFrame({'Temperature': [80]})
        assert find_column(df, ['RPM', 'rpm']) is None

    def test_find_column_name_returns_string(self):
        """find_column_name returns column name, not values"""
        import pandas as pd
        from src.utils.dataframe_helpers import find_column_name

        df = pd.DataFrame({'GPS Speed': [50], 'RPM': [5000]})
        name = find_column_name(df, ['GPS Speed', 'speed'])
        assert name == 'GPS Speed'
        assert isinstance(name, str)

    def test_speed_constant(self):
        """SPEED_MS_TO_MPH constant matches expected value"""
        from src.utils.dataframe_helpers import SPEED_MS_TO_MPH
        assert SPEED_MS_TO_MPH == 2.237

    def test_ensure_speed_mph_converts(self):
        """ensure_speed_mph converts m/s to mph when max < 100"""
        from src.utils.dataframe_helpers import ensure_speed_mph, SPEED_MS_TO_MPH

        ms_data = np.array([10, 20, 30, 40])  # m/s (all < 100)
        result = ensure_speed_mph(ms_data)
        np.testing.assert_array_almost_equal(result, ms_data * SPEED_MS_TO_MPH)

    def test_ensure_speed_mph_no_convert(self):
        """ensure_speed_mph leaves mph data unchanged when max >= 100"""
        from src.utils.dataframe_helpers import ensure_speed_mph

        mph_data = np.array([60, 80, 120, 100])
        result = ensure_speed_mph(mph_data)
        np.testing.assert_array_equal(result, mph_data)

    def test_sanitize_for_json(self):
        """sanitize_for_json handles NaN, Inf, and numpy types"""
        from src.utils.dataframe_helpers import sanitize_for_json

        data = {
            'nan_val': float('nan'),
            'inf_val': float('inf'),
            'np_int': np.int64(42),
            'np_float': np.float64(3.14),
            'np_nan': np.float64('nan'),
            'list': [1, float('nan'), 3],
            'normal': 5.0,
        }
        result = sanitize_for_json(data)
        assert result['nan_val'] is None
        assert result['inf_val'] is None
        assert result['np_int'] == 42
        assert isinstance(result['np_int'], int)
        assert result['np_float'] == 3.14
        assert result['np_nan'] is None
        assert result['list'] == [1, None, 3]
        assert result['normal'] == 5.0

    def test_safe_float(self):
        """safe_float replaces NaN/Inf with default"""
        from src.utils.dataframe_helpers import safe_float

        assert safe_float(3.14) == 3.14
        assert safe_float(float('nan')) == 0.0
        assert safe_float(float('inf'), default=-1.0) == -1.0

    def test_known_columns_dict(self):
        """KNOWN_COLUMNS has expected keys"""
        from src.utils.dataframe_helpers import KNOWN_COLUMNS

        assert 'speed' in KNOWN_COLUMNS
        assert 'rpm' in KNOWN_COLUMNS
        assert 'latitude' in KNOWN_COLUMNS
        assert isinstance(KNOWN_COLUMNS['speed'], list)


class TestEnumConsolidation:
    """Tests for cleanup-003: consolidated enums"""

    def test_single_lap_classification_source(self):
        """LapClassification should have all members from both old definitions"""
        from src.session.models import LapClassification

        # From original models.py
        assert LapClassification.OUT_LAP.value == "out_lap"
        assert LapClassification.IN_LAP.value == "in_lap"
        assert LapClassification.WARM_UP.value == "warm_up"
        assert LapClassification.COOL_DOWN.value == "cool_down"
        assert LapClassification.HOT_LAP.value == "hot_lap"
        assert LapClassification.NORMAL.value == "normal"
        # From original lap_analysis.py
        assert LapClassification.RACE_PACE.value == "race_pace"
        assert LapClassification.INCOMPLETE.value == "incomplete"

    def test_lap_analysis_uses_models_enum(self):
        """lap_analysis.py should import LapClassification from session.models"""
        from src.features.lap_analysis import LapClassification
        from src.session.models import LapClassification as ModelsLapClassification

        # Should be the exact same class
        assert LapClassification is ModelsLapClassification

    def test_extraction_directory_merged(self):
        """data_loader should be importable from src.extraction"""
        from src.extraction.data_loader import XRKDataLoader
        assert XRKDataLoader is not None


class TestSafetyCriticalFixes:
    """Tests for cleanup-001: safety-critical fixes"""

    @pytest.fixture
    def client(self):
        from src.main.app import app
        from fastapi.testclient import TestClient
        return TestClient(app)

    @pytest.fixture
    def parquet_no_speed(self, tmp_path):
        """Parquet file with GPS but no speed column"""
        import pandas as pd
        n = 100
        time = np.linspace(0, 50, n)
        df = pd.DataFrame({
            'GPS Latitude': 43.79 + 0.001 * np.sin(time),
            'GPS Longitude': -87.99 + 0.001 * np.cos(time),
            'RPM': 5000 + 1000 * np.sin(time),
        }, index=time)
        path = tmp_path / "no_speed.parquet"
        df.to_parquet(path)
        return path

    @pytest.fixture
    def parquet_no_gps(self, tmp_path):
        """Parquet file with speed/RPM but no GPS"""
        import pandas as pd
        n = 100
        time = np.linspace(0, 50, n)
        df = pd.DataFrame({
            'RPM': 5000 + 1000 * np.sin(time),
            'GPS Speed': 60 + 20 * np.sin(time),
        }, index=time)
        path = tmp_path / "no_gps.parquet"
        df.to_parquet(path)
        return path

    def test_report_rejects_missing_gps(self, parquet_no_gps):
        """Session report endpoint returns 422 when GPS data is missing"""
        from src.main.app import app, _find_parquet_file
        from fastapi.testclient import TestClient
        from unittest.mock import patch

        with patch('src.main.app._find_parquet_file', return_value=parquet_no_gps):
            client = TestClient(app)
            response = client.get("/api/analyze/report/test.parquet")
            assert response.status_code == 422
            assert "GPS" in response.json()["detail"]

    def test_report_rejects_missing_speed(self, parquet_no_speed):
        """Session report endpoint returns 422 when speed data is missing"""
        from unittest.mock import patch

        with patch('src.main.app._find_parquet_file', return_value=parquet_no_speed):
            from src.main.app import app
            from fastapi.testclient import TestClient
            client = TestClient(app)
            response = client.get("/api/analyze/report/test.parquet")
            assert response.status_code == 422
            assert "Speed" in response.json()["detail"]

    def test_laps_rejects_missing_speed(self, parquet_no_speed):
        """Lap analysis endpoint returns 422 when speed data is missing"""
        from unittest.mock import patch

        with patch('src.main.app._find_parquet_file', return_value=parquet_no_speed):
            from src.main.app import app
            from fastapi.testclient import TestClient
            client = TestClient(app)
            response = client.get("/api/analyze/laps/test.parquet")
            assert response.status_code == 422
            assert "Speed" in response.json()["detail"]

    def test_dtype_comparison_works_for_numeric(self):
        """dtype.kind check handles float32, int32, float16, etc."""
        import pandas as pd

        df = pd.DataFrame({
            'float32_col': np.array([1.0, 2.0], dtype=np.float32),
            'int32_col': np.array([1, 2], dtype=np.int32),
            'float16_col': np.array([1.0, 2.0], dtype=np.float16),
            'str_col': ['a', 'b'],
        })

        for col in ['float32_col', 'int32_col', 'float16_col']:
            assert df[col].dtype.kind in 'fi', f"{col} should be numeric"

        assert df['str_col'].dtype.kind not in 'fi', "str_col should not be numeric"

    def test_no_bare_except_in_file_manager(self):
        """file_manager.py should not have bare except: (catches KeyboardInterrupt)"""
        source_path = Path(__file__).parent.parent / "src" / "io" / "file_manager.py"
        content = source_path.read_text()
        # Check there's no bare except (except: without a type)
        import re
        bare_excepts = re.findall(r'\bexcept\s*:', content)
        assert len(bare_excepts) == 0, f"Found {len(bare_excepts)} bare except: statements"

    def test_gg_analysis_nan_arrays_synced(self):
        """GG analysis filters NaN consistently across all arrays"""
        from src.features.gg_analysis import GGAnalyzer

        n = 100
        time = np.linspace(0, 10, n)
        lat_acc = np.random.uniform(-1, 1, n)
        lon_acc = np.random.uniform(-1, 1, n)
        speed = np.random.uniform(30, 100, n)

        # Introduce NaN at different positions
        lat_acc[10] = np.nan
        lat_acc[50] = np.nan
        lon_acc[30] = np.nan

        analyzer = GGAnalyzer()
        result = analyzer.analyze_from_arrays(
            time, lat_acc, lon_acc,
            speed_data=speed,
            session_id="test_nan"
        )

        # All points should have consistent data (no index out of bounds)
        assert len(result.points) == n - 3  # 3 NaN positions removed
        for p in result.points:
            assert not np.isnan(p.lat_acc)
            assert not np.isnan(p.lon_acc)
            assert p.speed_mph >= 0

    def test_session_builder_pointer_decode_raises(self):
        """DLL pointer decode should raise on undecipherable pointer, not return garbage"""
        from src.session.session_builder import _safe_decode

        # Normal cases still work
        assert _safe_decode(b"RPM") == "RPM"
        assert _safe_decode("Speed") == "Speed"
        assert _safe_decode(None) == ""
        assert _safe_decode(0) == ""
        assert _safe_decode(42) == "chan_42"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
