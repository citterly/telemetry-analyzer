"""
Test Corner Map Overlay (Phase 8)

Tests for the track map corner overlay visualization feature.
"""

import pytest
import numpy as np
from src.visualization.track_map import TrackMap, TrackMapConfig


def test_render_corner_overlay_svg_basic():
    """Test basic corner overlay rendering."""
    # Create sample GPS data
    lat_data = np.array([43.05, 43.051, 43.052, 43.053, 43.054])
    lon_data = np.array([-88.15, -88.151, -88.152, -88.153, -88.154])

    # Create sample corners
    corners = [
        {
            'name': 'T1',
            'alias': None,
            'apex_lat': 43.051,
            'apex_lon': -88.151,
            'entry_idx': 0,
            'apex_idx': 1,
            'exit_idx': 2,
            'corner_type': 'hairpin',
            'direction': 'left'
        },
        {
            'name': 'T2',
            'alias': 'Carousel',
            'apex_lat': 43.053,
            'apex_lon': -88.153,
            'entry_idx': 2,
            'apex_idx': 3,
            'exit_idx': 4,
            'corner_type': 'sweeper',
            'direction': 'right'
        }
    ]

    track_map = TrackMap()
    svg = track_map.render_corner_overlay_svg(
        lat_data, lon_data, corners,
        title="Test Track"
    )

    # Verify SVG is generated
    assert svg is not None
    assert len(svg) > 0
    assert '<svg' in svg
    assert '</svg>' in svg

    # Verify corner markers are present
    assert 'class="corner-markers"' in svg
    assert 'class="corner-marker"' in svg

    # Verify corner names appear
    assert 'T1' in svg
    assert 'T2' in svg

    # Verify corner boundaries are present
    assert 'class="corner-boundaries"' in svg


def test_corner_marker_with_tooltip():
    """Test that corner markers include tooltip with corner type."""
    lat_data = np.array([43.05, 43.051, 43.052])
    lon_data = np.array([-88.15, -88.151, -88.152])

    corners = [{
        'name': 'T1',
        'alias': 'The Kink',
        'apex_lat': 43.051,
        'apex_lon': -88.151,
        'entry_idx': 0,
        'apex_idx': 1,
        'exit_idx': 2,
        'corner_type': 'kink',
        'direction': 'right'
    }]

    track_map = TrackMap()
    svg = track_map.render_corner_overlay_svg(lat_data, lon_data, corners)

    # Verify tooltip is present
    assert '<title>' in svg
    assert 'T1' in svg
    assert 'The Kink' in svg
    assert 'kink' in svg.lower()
    assert 'right' in svg.lower()


def test_selected_corner_highlighting():
    """Test that selected corner is highlighted differently."""
    lat_data = np.array([43.05, 43.051, 43.052, 43.053])
    lon_data = np.array([-88.15, -88.151, -88.152, -88.153])

    corners = [
        {
            'name': 'T1',
            'alias': None,
            'apex_lat': 43.051,
            'apex_lon': -88.151,
            'entry_idx': 0,
            'apex_idx': 1,
            'exit_idx': 2,
            'corner_type': 'hairpin',
            'direction': 'left'
        },
        {
            'name': 'T2',
            'alias': None,
            'apex_lat': 43.052,
            'apex_lon': -88.152,
            'entry_idx': 1,
            'apex_idx': 2,
            'exit_idx': 3,
            'corner_type': 'normal',
            'direction': 'right'
        }
    ]

    track_map = TrackMap()

    # Render without selection
    svg_unselected = track_map.render_corner_overlay_svg(lat_data, lon_data, corners)

    # Render with selection
    svg_selected = track_map.render_corner_overlay_svg(
        lat_data, lon_data, corners, selected_corner='T1'
    )

    # Selected version should be different (has highlighting)
    assert svg_selected != svg_unselected

    # Both should have corner markers
    assert 'corner-marker' in svg_unselected
    assert 'corner-marker' in svg_selected


def test_corner_boundary_segments():
    """Test that corner boundaries are drawn as highlighted segments."""
    lat_data = np.array([43.05, 43.051, 43.052, 43.053, 43.054])
    lon_data = np.array([-88.15, -88.151, -88.152, -88.153, -88.154])

    corners = [{
        'name': 'T1',
        'alias': None,
        'apex_lat': 43.052,
        'apex_lon': -88.152,
        'entry_idx': 1,
        'apex_idx': 2,
        'exit_idx': 3,
        'corner_type': 'normal',
        'direction': 'left'
    }]

    track_map = TrackMap()
    svg = track_map.render_corner_overlay_svg(lat_data, lon_data, corners)

    # Verify boundary segment is present
    assert 'corner-boundary' in svg
    assert 'polyline' in svg
    assert 'data-corner="T1"' in svg


def test_multiple_corner_types_color_coded():
    """Test that different corner types are color-coded."""
    lat_data = np.array([43.05 + i*0.001 for i in range(20)])
    lon_data = np.array([-88.15 + i*0.001 for i in range(20)])

    corners = [
        {'name': 'T1', 'alias': None, 'apex_lat': lat_data[2], 'apex_lon': lon_data[2],
         'entry_idx': 0, 'apex_idx': 2, 'exit_idx': 4, 'corner_type': 'hairpin', 'direction': 'left'},
        {'name': 'T2', 'alias': None, 'apex_lat': lat_data[7], 'apex_lon': lon_data[7],
         'entry_idx': 5, 'apex_idx': 7, 'exit_idx': 9, 'corner_type': 'sweeper', 'direction': 'right'},
        {'name': 'T3', 'alias': None, 'apex_lat': lat_data[12], 'apex_lon': lon_data[12],
         'entry_idx': 10, 'apex_idx': 12, 'exit_idx': 14, 'corner_type': 'kink', 'direction': 'left'},
        {'name': 'T4', 'alias': None, 'apex_lat': lat_data[17], 'apex_lon': lon_data[17],
         'entry_idx': 15, 'apex_idx': 17, 'exit_idx': 19, 'corner_type': 'chicane', 'direction': 'right'}
    ]

    track_map = TrackMap()
    svg = track_map.render_corner_overlay_svg(lat_data, lon_data, corners)

    # All corner types should be present
    assert 'T1' in svg and 'T2' in svg and 'T3' in svg and 'T4' in svg

    # Should have multiple corner markers
    assert svg.count('corner-marker') >= 4


def test_render_with_speed_coloring():
    """Test corner overlay with speed-based track coloring."""
    lat_data = np.array([43.05, 43.051, 43.052, 43.053, 43.054])
    lon_data = np.array([-88.15, -88.151, -88.152, -88.153, -88.154])
    speed_data = np.array([80, 60, 40, 50, 70])

    corners = [{
        'name': 'T1',
        'alias': None,
        'apex_lat': 43.052,
        'apex_lon': -88.152,
        'entry_idx': 1,
        'apex_idx': 2,
        'exit_idx': 3,
        'corner_type': 'normal',
        'direction': 'left'
    }]

    track_map = TrackMap()
    svg = track_map.render_corner_overlay_svg(
        lat_data, lon_data, corners,
        color_data=speed_data,
        color_scheme='speed'
    )

    # Should have both track coloring and corner overlay
    assert 'corner-marker' in svg
    assert 'track-trace' in svg or 'line' in svg  # Colored track segments


def test_empty_corners_list():
    """Test rendering with no corners (should still show track)."""
    lat_data = np.array([43.05, 43.051, 43.052])
    lon_data = np.array([-88.15, -88.151, -88.152])

    track_map = TrackMap()
    svg = track_map.render_corner_overlay_svg(lat_data, lon_data, [])

    # Should still render track
    assert '<svg' in svg
    assert '</svg>' in svg

    # But no corner markers
    assert 'corner-marker' not in svg


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
