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

        result = analyze_gg_diagram(str(files[0]), max_g_reference=1.3)

        assert isinstance(result, GGAnalysisResult)
        assert result.stats.points_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
