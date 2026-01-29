"""
Tests for track map visualization feature
"""

import os
import sys
import numpy as np
import pytest
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization.track_map import (
    TrackMap,
    TrackMapConfig,
    ColorScale
)


class TestColorScale:
    """Tests for ColorScale"""

    def test_color_scale_creation(self):
        """Test creating color scale"""
        scale = ColorScale(
            name='Test',
            min_val=0,
            max_val=100
        )
        assert scale.name == 'Test'
        assert scale.min_val == 0
        assert scale.max_val == 100
        assert len(scale.colors) > 0

    def test_color_scale_custom_colors(self):
        """Test custom colors"""
        scale = ColorScale(
            name='Custom',
            min_val=0,
            max_val=100,
            colors=['#ff0000', '#00ff00', '#0000ff']
        )
        assert len(scale.colors) == 3
        assert scale.colors[0] == '#ff0000'

    def test_get_color_min(self):
        """Test getting color at minimum value"""
        scale = ColorScale(
            name='Test',
            min_val=0,
            max_val=100,
            colors=['#000000', '#ffffff']
        )
        color = scale.get_color(0)
        assert color == '#000000'

    def test_get_color_max(self):
        """Test getting color at maximum value"""
        scale = ColorScale(
            name='Test',
            min_val=0,
            max_val=100,
            colors=['#000000', '#ffffff']
        )
        color = scale.get_color(100)
        assert color == '#ffffff'

    def test_get_color_middle(self):
        """Test getting interpolated color"""
        scale = ColorScale(
            name='Test',
            min_val=0,
            max_val=100,
            colors=['#000000', '#ffffff']
        )
        color = scale.get_color(50)
        # Should be around gray
        assert color.startswith('#')
        assert len(color) == 7

    def test_get_color_below_min(self):
        """Test color for value below minimum"""
        scale = ColorScale(name='Test', min_val=50, max_val=100)
        color = scale.get_color(0)
        assert color == scale.colors[0]

    def test_get_color_above_max(self):
        """Test color for value above maximum"""
        scale = ColorScale(name='Test', min_val=0, max_val=50)
        color = scale.get_color(100)
        assert color == scale.colors[-1]


class TestTrackMapConfig:
    """Tests for TrackMapConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = TrackMapConfig()
        assert config.width == 800
        assert config.height == 600
        assert config.padding == 40

    def test_custom_config(self):
        """Test custom configuration"""
        config = TrackMapConfig(
            width=1200,
            height=800,
            padding=60,
            background_color='#ffffff'
        )
        assert config.width == 1200
        assert config.background_color == '#ffffff'


class TestTrackMap:
    """Tests for TrackMap"""

    @pytest.fixture
    def track_map(self):
        """Create track map instance"""
        return TrackMap()

    @pytest.fixture
    def sample_gps_data(self):
        """Generate sample GPS data (circular track)"""
        n_points = 200
        t = np.linspace(0, 2 * np.pi, n_points)

        # Road America start/finish area
        center_lat, center_lon = 43.797875, -87.989638
        radius = 0.005  # Roughly 500m

        latitude = center_lat + radius * np.sin(t)
        longitude = center_lon + radius * np.cos(t)

        return latitude, longitude

    @pytest.fixture
    def sample_speed_data(self):
        """Generate sample speed data"""
        n_points = 200
        # Varying speed pattern
        return 50 + 50 * np.sin(np.linspace(0, 4 * np.pi, n_points))

    @pytest.fixture
    def sample_rpm_data(self):
        """Generate sample RPM data"""
        n_points = 200
        return 4000 + 2500 * np.sin(np.linspace(0, 4 * np.pi, n_points))

    def test_render_svg_basic(self, track_map, sample_gps_data):
        """Test basic SVG rendering"""
        lat, lon = sample_gps_data
        svg = track_map.render_svg(lat, lon)

        assert svg.startswith('<svg')
        assert '</svg>' in svg
        assert 'viewBox' in svg

    def test_render_svg_with_speed(self, track_map, sample_gps_data, sample_speed_data):
        """Test SVG rendering with speed coloring"""
        lat, lon = sample_gps_data
        svg = track_map.render_svg(lat, lon, sample_speed_data, color_scheme='speed')

        assert '<line' in svg or '<polyline' in svg
        assert 'stroke=' in svg

    def test_render_svg_with_rpm(self, track_map, sample_gps_data, sample_rpm_data):
        """Test SVG rendering with RPM coloring"""
        lat, lon = sample_gps_data
        svg = track_map.render_svg(lat, lon, sample_rpm_data, color_scheme='rpm')

        assert '<svg' in svg

    def test_render_svg_with_title(self, track_map, sample_gps_data):
        """Test SVG has title"""
        lat, lon = sample_gps_data
        svg = track_map.render_svg(lat, lon, title="Test Track")

        assert 'Test Track' in svg

    def test_render_svg_has_legend(self, track_map, sample_gps_data, sample_speed_data):
        """Test SVG has legend"""
        lat, lon = sample_gps_data
        svg = track_map.render_svg(lat, lon, sample_speed_data, color_scheme='speed')

        assert 'legend' in svg.lower() or 'linearGradient' in svg

    def test_render_svg_has_start_finish(self, track_map, sample_gps_data):
        """Test SVG has start/finish marker"""
        lat, lon = sample_gps_data
        svg = track_map.render_svg(lat, lon)

        assert 'S/F' in svg or 'start' in svg.lower()

    def test_render_html(self, track_map, sample_gps_data, sample_speed_data):
        """Test HTML rendering"""
        lat, lon = sample_gps_data
        html = track_map.render_html(lat, lon, sample_speed_data, color_scheme='speed')

        assert '<!DOCTYPE html>' in html
        assert '<svg' in html
        assert '</html>' in html

    def test_render_html_has_controls(self, track_map, sample_gps_data):
        """Test HTML has controls when enabled"""
        lat, lon = sample_gps_data
        html = track_map.render_html(lat, lon, include_controls=True)

        assert 'select' in html
        assert 'colorScheme' in html or 'Color by' in html

    def test_render_html_no_controls(self, track_map, sample_gps_data):
        """Test HTML without controls"""
        lat, lon = sample_gps_data
        html = track_map.render_html(lat, lon, include_controls=False)

        assert '<!DOCTYPE html>' in html

    def test_to_dict(self, track_map, sample_gps_data, sample_speed_data):
        """Test dictionary export"""
        lat, lon = sample_gps_data
        data = track_map.to_dict(lat, lon, sample_speed_data, color_scheme='speed')

        assert 'coordinates' in data
        assert 'bounds' in data
        assert 'color_scheme' in data
        assert len(data['coordinates']) == len(lat)

    def test_to_dict_has_bounds(self, track_map, sample_gps_data):
        """Test dictionary has correct bounds"""
        lat, lon = sample_gps_data
        data = track_map.to_dict(lat, lon)

        assert data['bounds']['lat_min'] <= data['bounds']['lat_max']
        assert data['bounds']['lon_min'] <= data['bounds']['lon_max']


class TestTrackMapColorSchemes:
    """Tests for predefined color schemes"""

    def test_speed_scheme_exists(self):
        """Test speed color scheme exists"""
        assert 'speed' in TrackMap.COLOR_SCHEMES

    def test_rpm_scheme_exists(self):
        """Test RPM color scheme exists"""
        assert 'rpm' in TrackMap.COLOR_SCHEMES

    def test_gear_scheme_exists(self):
        """Test gear color scheme exists"""
        assert 'gear' in TrackMap.COLOR_SCHEMES

    def test_throttle_scheme_exists(self):
        """Test throttle color scheme exists"""
        assert 'throttle' in TrackMap.COLOR_SCHEMES

    def test_brake_scheme_exists(self):
        """Test brake color scheme exists"""
        assert 'brake' in TrackMap.COLOR_SCHEMES


class TestTrackMapSaveFiles:
    """Tests for file saving"""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data"""
        n = 100
        t = np.linspace(0, 2 * np.pi, n)
        lat = 43.8 + 0.005 * np.sin(t)
        lon = -88.0 + 0.005 * np.cos(t)
        speed = 50 + 50 * np.abs(np.sin(t))
        return lat, lon, speed

    def test_save_svg(self, sample_data):
        """Test saving SVG file"""
        track_map = TrackMap()
        lat, lon, speed = sample_data

        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as f:
            track_map.save_svg(f.name, lat, lon, speed)
            assert os.path.exists(f.name)

            with open(f.name) as svg_file:
                content = svg_file.read()
                assert '<svg' in content

        os.unlink(f.name)

    def test_save_html(self, sample_data):
        """Test saving HTML file"""
        track_map = TrackMap()
        lat, lon, speed = sample_data

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            track_map.save_html(f.name, lat, lon, speed, title="Test Map")
            assert os.path.exists(f.name)

            with open(f.name) as html_file:
                content = html_file.read()
                assert '<!DOCTYPE html>' in content
                assert 'Test Map' in content

        os.unlink(f.name)


class TestTrackMapEdgeCases:
    """Tests for edge cases"""

    def test_empty_data(self):
        """Test handling of empty data"""
        track_map = TrackMap()
        svg = track_map.render_svg(np.array([]), np.array([]))

        assert '<svg' in svg
        assert '</svg>' in svg

    def test_single_point(self):
        """Test handling of single point"""
        track_map = TrackMap()
        svg = track_map.render_svg(np.array([43.8]), np.array([-88.0]))

        assert '<svg' in svg

    def test_two_points(self):
        """Test handling of two points"""
        track_map = TrackMap()
        lat = np.array([43.8, 43.81])
        lon = np.array([-88.0, -88.01])
        svg = track_map.render_svg(lat, lon)

        assert '<svg' in svg

    def test_large_dataset(self):
        """Test handling of large dataset"""
        track_map = TrackMap()
        n = 10000
        t = np.linspace(0, 10 * np.pi, n)
        lat = 43.8 + 0.005 * np.sin(t)
        lon = -88.0 + 0.005 * np.cos(t)
        speed = np.random.uniform(30, 150, n)

        svg = track_map.render_svg(lat, lon, speed)
        assert '<svg' in svg

    def test_negative_coordinates(self):
        """Test handling of various coordinate ranges"""
        track_map = TrackMap()
        lat = np.array([-33.8, -33.9, -33.85])
        lon = np.array([151.2, 151.3, 151.25])

        svg = track_map.render_svg(lat, lon)
        assert '<svg' in svg


class TestTrackMapCustomConfig:
    """Tests for custom configuration"""

    def test_custom_dimensions(self):
        """Test custom width and height"""
        config = TrackMapConfig(width=1200, height=900)
        track_map = TrackMap(config)

        lat = np.array([43.8, 43.81])
        lon = np.array([-88.0, -88.01])
        svg = track_map.render_svg(lat, lon)

        assert 'width="1200"' in svg
        assert 'height="900"' in svg

    def test_custom_colors(self):
        """Test custom background color"""
        config = TrackMapConfig(background_color='#ff0000')
        track_map = TrackMap(config)

        lat = np.array([43.8, 43.81])
        lon = np.array([-88.0, -88.01])
        svg = track_map.render_svg(lat, lon)

        assert '#ff0000' in svg

    def test_no_legend(self):
        """Test rendering without legend"""
        config = TrackMapConfig(show_legend=False)
        track_map = TrackMap(config)

        lat = np.array([43.8, 43.81, 43.82])
        lon = np.array([-88.0, -88.01, -88.02])
        svg = track_map.render_svg(lat, lon)

        # Should not have gradient definition if no legend
        assert svg.count('linearGradient') == 0 or 'legend' not in svg.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
