"""
Tests for G-G diagram analysis and visualization.
"""

import pytest
import numpy as np
import tempfile
import os

from src.features.gg_analysis import (
    GGAnalyzer, GGPoint, GGStats, GGAnalysisResult, analyze_gg_diagram
)
from src.visualization.gg_diagram import GGDiagram, GGDiagramConfig


class TestGGPoint:
    """Tests for GGPoint dataclass"""

    def test_gg_point_creation(self):
        """Test creating a GGPoint"""
        point = GGPoint(
            time=10.5,
            lat_acc=0.5,
            lon_acc=-0.3,
            total_g=0.58,
            speed_mph=65.0,
            throttle_pct=80.0,
            gear=3,
            lap_number=2
        )
        assert point.time == 10.5
        assert point.lat_acc == 0.5
        assert point.lon_acc == -0.3
        assert point.total_g == 0.58
        assert point.speed_mph == 65.0

    def test_gg_point_defaults(self):
        """Test GGPoint default values"""
        point = GGPoint(time=0, lat_acc=0, lon_acc=0, total_g=0)
        assert point.speed_mph == 0.0
        assert point.throttle_pct == 0.0
        assert point.gear == 0


class TestGGStats:
    """Tests for GGStats dataclass"""

    def test_gg_stats_creation(self):
        """Test creating GGStats"""
        stats = GGStats(
            max_lateral_g=1.5,
            max_braking_g=1.2,
            max_acceleration_g=0.6,
            max_combined_g=1.6,
            avg_utilized_g=0.8,
            utilization_pct=65.0,
            data_derived_max_g=1.4,
            points_count=1000
        )
        assert stats.max_lateral_g == 1.5
        assert stats.max_braking_g == 1.2
        assert stats.utilization_pct == 65.0


class TestGGAnalyzer:
    """Tests for GGAnalyzer class"""

    @pytest.fixture
    def sample_data(self):
        """Sample G-G data"""
        n = 200
        np.random.seed(42)
        time = np.linspace(0, 100, n)
        # Simulate track driving: lateral in corners, longitudinal for accel/brake
        lat_acc = np.sin(time / 10) * 0.8 + np.random.randn(n) * 0.1
        lon_acc = np.cos(time / 10) * 0.5 + np.random.randn(n) * 0.1 - 0.2
        speed = 50 + 30 * np.sin(time / 20) + np.random.randn(n) * 5
        return time, lat_acc, lon_acc, speed

    def test_analyzer_creation(self):
        """Test creating GGAnalyzer"""
        analyzer = GGAnalyzer(max_g_reference=1.3)
        assert analyzer.max_g_reference == 1.3

    def test_analyze_from_arrays(self, sample_data):
        """Test analyzing from arrays"""
        time, lat_acc, lon_acc, speed = sample_data
        analyzer = GGAnalyzer(max_g_reference=1.3)

        result = analyzer.analyze_from_arrays(
            time, lat_acc, lon_acc,
            speed_data=speed,
            session_id="test_session"
        )

        assert isinstance(result, GGAnalysisResult)
        assert result.session_id == "test_session"
        assert len(result.points) == len(time)
        assert result.stats.points_count == len(time)

    def test_analyze_calculates_total_g(self, sample_data):
        """Test that total_g is calculated correctly"""
        time, lat_acc, lon_acc, _ = sample_data
        analyzer = GGAnalyzer()

        result = analyzer.analyze_from_arrays(time, lat_acc, lon_acc)

        # Check total_g calculation for first point
        expected_total_g = np.sqrt(lat_acc[0]**2 + lon_acc[0]**2)
        assert abs(result.points[0].total_g - expected_total_g) < 0.001

    def test_analyze_max_lateral_g(self, sample_data):
        """Test max lateral g calculation"""
        time, lat_acc, lon_acc, _ = sample_data
        analyzer = GGAnalyzer()

        result = analyzer.analyze_from_arrays(time, lat_acc, lon_acc)

        expected_max_lat = float(np.max(np.abs(lat_acc)))
        assert abs(result.stats.max_lateral_g - expected_max_lat) < 0.001

    def test_analyze_max_braking_g(self, sample_data):
        """Test max braking g calculation"""
        time, lat_acc, lon_acc, _ = sample_data
        analyzer = GGAnalyzer()

        result = analyzer.analyze_from_arrays(time, lat_acc, lon_acc)

        expected_max_brake = float(np.abs(np.min(lon_acc)))
        assert abs(result.stats.max_braking_g - expected_max_brake) < 0.001

    def test_analyze_utilization_percentage(self, sample_data):
        """Test utilization percentage calculation"""
        time, lat_acc, lon_acc, _ = sample_data
        analyzer = GGAnalyzer(max_g_reference=1.5)

        result = analyzer.analyze_from_arrays(time, lat_acc, lon_acc)

        # Utilization should be between 0 and 100
        assert 0 <= result.stats.utilization_pct <= 100

    def test_analyze_reference_max_g_from_config(self):
        """Test that reference max g uses config value when higher"""
        time = np.array([0, 1, 2])
        lat_acc = np.array([0.2, 0.3, 0.2])  # Low g values
        lon_acc = np.array([0.1, 0.1, 0.1])

        analyzer = GGAnalyzer(max_g_reference=1.5)
        result = analyzer.analyze_from_arrays(time, lat_acc, lon_acc)

        # Reference should be config value since data is low
        assert result.reference_max_g == 1.5

    def test_to_dict(self, sample_data):
        """Test result to_dict serialization"""
        time, lat_acc, lon_acc, speed = sample_data
        analyzer = GGAnalyzer()

        result = analyzer.analyze_from_arrays(time, lat_acc, lon_acc, speed_data=speed)
        data = result.to_dict()

        assert "session_id" in data
        assert "stats" in data
        assert "points" in data
        assert "reference_max_g" in data
        assert isinstance(data["stats"]["max_lateral_g"], float)

    def test_to_json(self, sample_data):
        """Test result to_json serialization"""
        time, lat_acc, lon_acc, _ = sample_data
        analyzer = GGAnalyzer()

        result = analyzer.analyze_from_arrays(time, lat_acc, lon_acc)
        json_str = result.to_json()

        assert isinstance(json_str, str)
        assert "max_lateral_g" in json_str

    def test_low_utilization_zones(self):
        """Test detection of low utilization zones"""
        n = 100
        time = np.linspace(0, 10, n)
        # Create a section with high lateral g but low total g (coasting in turn)
        lat_acc = np.zeros(n)
        lon_acc = np.zeros(n)

        # Add a turn with low utilization
        lat_acc[30:50] = 0.5  # In a turn
        lon_acc[30:50] = 0.0  # But not accelerating or braking

        analyzer = GGAnalyzer(max_g_reference=1.5)
        result = analyzer.analyze_from_arrays(time, lat_acc, lon_acc)

        # Should detect the low utilization zone
        assert isinstance(result.low_utilization_zones, list)


class TestGGDiagram:
    """Tests for GGDiagram class"""

    @pytest.fixture
    def sample_gg_data(self):
        """Sample G-G data for visualization"""
        n = 100
        np.random.seed(42)
        lat_acc = np.random.randn(n) * 0.6
        lon_acc = np.random.randn(n) * 0.4 - 0.2
        speed = np.random.uniform(40, 120, n)
        return lat_acc, lon_acc, speed

    def test_diagram_creation(self):
        """Test creating GGDiagram"""
        diagram = GGDiagram()
        assert diagram.config.width == 600
        assert diagram.config.height == 600

    def test_diagram_custom_config(self):
        """Test diagram with custom config"""
        config = GGDiagramConfig(width=800, height=800)
        diagram = GGDiagram(config)
        assert diagram.config.width == 800

    def test_render_svg_basic(self, sample_gg_data):
        """Test basic SVG rendering"""
        lat_acc, lon_acc, _ = sample_gg_data
        diagram = GGDiagram()

        svg = diagram.render_svg(lat_acc, lon_acc, reference_max_g=1.3)

        assert '<svg' in svg
        assert '</svg>' in svg
        assert 'data-points' in svg

    def test_render_svg_has_grid(self, sample_gg_data):
        """Test SVG has grid"""
        lat_acc, lon_acc, _ = sample_gg_data
        diagram = GGDiagram()

        svg = diagram.render_svg(lat_acc, lon_acc)

        assert 'grid' in svg

    def test_render_svg_has_axes(self, sample_gg_data):
        """Test SVG has axes"""
        lat_acc, lon_acc, _ = sample_gg_data
        diagram = GGDiagram()

        svg = diagram.render_svg(lat_acc, lon_acc)

        assert 'axes' in svg
        assert 'Right' in svg or 'Left' in svg

    def test_render_svg_has_reference_circle(self, sample_gg_data):
        """Test SVG has reference circle"""
        lat_acc, lon_acc, _ = sample_gg_data
        diagram = GGDiagram()

        svg = diagram.render_svg(lat_acc, lon_acc, reference_max_g=1.3)

        assert 'Max G' in svg

    def test_render_svg_has_data_circle(self, sample_gg_data):
        """Test SVG has data-derived circle"""
        lat_acc, lon_acc, _ = sample_gg_data
        diagram = GGDiagram()

        svg = diagram.render_svg(lat_acc, lon_acc, reference_max_g=1.3, data_max_g=1.1)

        assert '95th' in svg

    def test_render_svg_has_title(self, sample_gg_data):
        """Test SVG has title"""
        lat_acc, lon_acc, _ = sample_gg_data
        diagram = GGDiagram()

        svg = diagram.render_svg(lat_acc, lon_acc, title="Test Diagram")

        assert 'Test Diagram' in svg

    def test_render_svg_with_color_data(self, sample_gg_data):
        """Test SVG with color data"""
        lat_acc, lon_acc, speed = sample_gg_data
        diagram = GGDiagram()

        svg = diagram.render_svg(lat_acc, lon_acc, color_data=speed, color_scheme='speed')

        assert '<svg' in svg
        assert 'data-points' in svg

    def test_to_chartjs_data(self, sample_gg_data):
        """Test Chart.js data format"""
        lat_acc, lon_acc, speed = sample_gg_data
        diagram = GGDiagram()

        data = diagram.to_chartjs_data(
            lat_acc, lon_acc,
            color_data=speed,
            reference_max_g=1.3,
            downsample=5
        )

        assert 'datasets' in data
        assert 'reference_circles' in data
        assert 'bounds' in data
        assert len(data['datasets'][0]['data']) > 0

    def test_save_svg(self, sample_gg_data):
        """Test saving SVG to file"""
        lat_acc, lon_acc, _ = sample_gg_data
        diagram = GGDiagram()

        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as f:
            diagram.save_svg(f.name, lat_acc, lon_acc, title="Test Save")
            assert os.path.exists(f.name)

            with open(f.name) as svg_file:
                content = svg_file.read()
                assert '<svg' in content

        os.unlink(f.name)


class TestConvenienceFunction:
    """Tests for module convenience functions"""

    def test_analyze_gg_diagram_function(self):
        """Test analyze_gg_diagram convenience function with real data"""
        from pathlib import Path

        parquet_path = Path('data/exports/processed')
        if not parquet_path.exists():
            pytest.skip("Test data not available")

        files = list(parquet_path.glob('*.parquet'))
        if not files:
            pytest.skip("No parquet files available")

        # Find a file with GPS acceleration columns
        import pandas as pd
        target_file = None
        for f in files:
            cols = pd.read_parquet(f, columns=[]).columns.tolist()
            # Re-read just column names by reading 0 rows
            cols = pd.read_parquet(f).columns.tolist()
            if any('LatAcc' in c for c in cols):
                target_file = f
                break
        if target_file is None:
            pytest.skip("No parquet files with GPS LatAcc/LonAcc columns")

        result = analyze_gg_diagram(str(target_file), max_g_reference=1.3)

        assert isinstance(result, GGAnalysisResult)
        assert result.stats.points_count > 0


class TestGGAnalyzerTrace:
    """Tests for safeguard-004: GGAnalyzer trace + sanity checks."""

    def _make_gg_parquet(self, tmp_path, n=200, max_g=1.0, nan_pct=0.0):
        """Create synthetic parquet with acceleration data."""
        import pandas as pd
        time = np.linspace(0, 30, n)
        # Circular pattern for lat/lon acceleration
        angles = np.linspace(0, 4 * np.pi, n)
        lat_acc = np.sin(angles) * max_g
        lon_acc = np.cos(angles) * max_g * 0.8

        # Inject NaN if requested
        if nan_pct > 0:
            nan_count = int(n * nan_pct / 100)
            nan_indices = np.random.choice(n, nan_count, replace=False)
            lat_acc[nan_indices] = np.nan

        speed = np.linspace(20, 80, n)
        df = pd.DataFrame({
            "GPS LatAcc": lat_acc,
            "GPS LonAcc": lon_acc,
            "GPS Speed": speed,
        }, index=time)
        path = str(tmp_path / "gg_test.parquet")
        df.to_parquet(path)
        return path

    def test_trace_recorded_when_enabled(self, tmp_path):
        """Trace is attached when include_trace=True."""
        path = self._make_gg_parquet(tmp_path)
        analyzer = GGAnalyzer(max_g_reference=1.3)
        result = analyzer.analyze_from_parquet(path, include_trace=True)
        assert hasattr(result, 'trace')
        assert result.trace is not None
        assert result.trace.analyzer_name == "GGAnalyzer"

    def test_trace_not_recorded_by_default(self, tmp_path):
        """Trace is NOT attached by default."""
        path = self._make_gg_parquet(tmp_path)
        analyzer = GGAnalyzer()
        result = analyzer.analyze_from_parquet(path)
        assert not hasattr(result, 'trace') or result.trace is None

    def test_trace_inputs_recorded(self, tmp_path):
        """Trace records column names, sample count, nan_pct."""
        path = self._make_gg_parquet(tmp_path)
        analyzer = GGAnalyzer()
        result = analyzer.analyze_from_parquet(path, include_trace=True)
        trace = result.trace
        assert trace.inputs["lat_acc_column"] == "GPS LatAcc"
        assert trace.inputs["lon_acc_column"] == "GPS LonAcc"
        assert trace.inputs["speed_column"] == "GPS Speed"
        assert "sample_count" in trace.inputs
        assert "nan_pct" in trace.inputs

    def test_trace_config_recorded(self, tmp_path):
        """Trace records G-force reference values."""
        path = self._make_gg_parquet(tmp_path)
        analyzer = GGAnalyzer(max_g_reference=1.5, max_braking_g=1.6)
        result = analyzer.analyze_from_parquet(path, include_trace=True)
        trace = result.trace
        assert trace.config["max_g_reference"] == 1.5
        assert trace.config["max_braking_g"] == 1.6
        assert "vehicle_name" in trace.config

    def test_trace_intermediates_recorded(self, tmp_path):
        """Trace records key analysis intermediates."""
        path = self._make_gg_parquet(tmp_path)
        analyzer = GGAnalyzer()
        result = analyzer.analyze_from_parquet(path, include_trace=True)
        trace = result.trace
        assert "max_combined_g" in trace.intermediates
        assert "p95_combined_g" in trace.intermediates
        assert "utilization_pct" in trace.intermediates

    def test_trace_has_four_sanity_checks(self, tmp_path):
        """All 4 sanity checks are present."""
        path = self._make_gg_parquet(tmp_path)
        analyzer = GGAnalyzer()
        result = analyzer.analyze_from_parquet(path, include_trace=True)
        check_names = [c.name for c in result.trace.sanity_checks]
        assert "config_matches_vehicle" in check_names
        assert "data_quality" in check_names
        assert "g_force_plausible" in check_names
        assert "utilization_plausible" in check_names

    def test_to_dict_includes_trace(self, tmp_path):
        """to_dict() includes _trace when present."""
        path = self._make_gg_parquet(tmp_path)
        analyzer = GGAnalyzer()
        result = analyzer.analyze_from_parquet(path, include_trace=True)
        d = result.to_dict()
        assert "_trace" in d
        assert d["_trace"]["analyzer_name"] == "GGAnalyzer"

    def test_to_dict_omits_trace_by_default(self, tmp_path):
        """to_dict() does NOT include _trace by default."""
        path = self._make_gg_parquet(tmp_path)
        analyzer = GGAnalyzer()
        result = analyzer.analyze_from_parquet(path)
        d = result.to_dict()
        assert "_trace" not in d

    def test_check_data_quality_warns_high_nan(self, tmp_path):
        """High NaN percentage triggers data quality warning."""
        path = self._make_gg_parquet(tmp_path, nan_pct=15)
        analyzer = GGAnalyzer()
        result = analyzer.analyze_from_parquet(path, include_trace=True)
        check = next(c for c in result.trace.sanity_checks if c.name == "data_quality")
        assert check.status == "warn"

    def test_check_g_force_plausible_passes(self, tmp_path):
        """Normal G-force values pass plausibility check."""
        path = self._make_gg_parquet(tmp_path, max_g=1.0)
        analyzer = GGAnalyzer()
        result = analyzer.analyze_from_parquet(path, include_trace=True)
        check = next(c for c in result.trace.sanity_checks if c.name == "g_force_plausible")
        assert check.status == "pass"

    def test_check_g_force_plausible_fails_extreme(self, tmp_path):
        """Extreme G-force values fail plausibility check."""
        path = self._make_gg_parquet(tmp_path, max_g=3.5)
        analyzer = GGAnalyzer()
        result = analyzer.analyze_from_parquet(path, include_trace=True)
        check = next(c for c in result.trace.sanity_checks if c.name == "g_force_plausible")
        assert check.status == "fail"

    def test_existing_tests_still_pass(self):
        """Smoke: analyze_from_arrays still works."""
        analyzer = GGAnalyzer()
        time = np.linspace(0, 10, 100)
        lat_acc = np.sin(np.linspace(0, 4, 100))
        lon_acc = np.cos(np.linspace(0, 4, 100)) * 0.8
        result = analyzer.analyze_from_arrays(time, lat_acc, lon_acc)
        assert isinstance(result, GGAnalysisResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
