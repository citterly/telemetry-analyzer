"""
Track Map Visualization
GPS trace visualization color-coded by speed, gear, or RPM.

Generates SVG and HTML visualizations of telemetry data on track outline.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Literal
from pathlib import Path
import json


@dataclass
class TrackMapConfig:
    """Configuration for track map rendering"""
    width: int = 800
    height: int = 600
    padding: int = 40
    point_size: int = 3
    line_width: int = 2
    background_color: str = "#1a1a2e"
    track_outline_color: str = "#333344"
    title_color: str = "#ffffff"
    legend_position: str = "bottom-right"
    show_start_finish: bool = True
    show_legend: bool = True
    show_title: bool = True


@dataclass
class ColorScale:
    """Color scale for value mapping"""
    name: str
    min_val: float
    max_val: float
    colors: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    # Discrete 3-color mode: thresholds as percentages (0-100)
    discrete_mode: bool = False
    low_threshold: float = 33.0  # Below this % = green
    high_threshold: float = 66.0  # Above this % = red, between = yellow

    # Discrete colors: green (low), yellow (mid), red (high)
    DISCRETE_COLORS = ["#2ecc71", "#f1c40f", "#e74c3c"]

    def __post_init__(self):
        if not self.colors:
            # Default blue -> green -> yellow -> red scale
            self.colors = ["#3498db", "#2ecc71", "#f1c40f", "#e74c3c"]
        if not self.labels:
            step = (self.max_val - self.min_val) / (len(self.colors) - 1)
            self.labels = [f"{self.min_val + i*step:.0f}" for i in range(len(self.colors))]

    def get_color(self, value: float) -> str:
        """Get color for a value (interpolated or discrete based on mode)"""
        if self.discrete_mode:
            return self._get_discrete_color(value)
        return self._get_gradient_color(value)

    def _get_discrete_color(self, value: float) -> str:
        """Get discrete 3-color value (green/yellow/red based on thresholds)"""
        if self.max_val == self.min_val:
            return self.DISCRETE_COLORS[1]  # Yellow for no range

        # Normalize to percentage
        pct = (value - self.min_val) / (self.max_val - self.min_val) * 100

        if pct <= self.low_threshold:
            return self.DISCRETE_COLORS[0]  # Green
        elif pct >= self.high_threshold:
            return self.DISCRETE_COLORS[2]  # Red
        else:
            return self.DISCRETE_COLORS[1]  # Yellow

    def _get_gradient_color(self, value: float) -> str:
        """Get interpolated gradient color for a value"""
        if value <= self.min_val:
            return self.colors[0]
        if value >= self.max_val:
            return self.colors[-1]

        # Normalize value to 0-1
        normalized = (value - self.min_val) / (self.max_val - self.min_val)

        # Find which color segment we're in
        segments = len(self.colors) - 1
        segment_idx = int(normalized * segments)
        segment_idx = min(segment_idx, segments - 1)

        # Interpolate within segment
        segment_pos = (normalized * segments) - segment_idx

        color1 = self.colors[segment_idx]
        color2 = self.colors[segment_idx + 1]

        return self._interpolate_colors(color1, color2, segment_pos)

    def _interpolate_colors(self, color1: str, color2: str, t: float) -> str:
        """Interpolate between two hex colors"""
        r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
        r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)

        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)

        return f"#{r:02x}{g:02x}{b:02x}"


class TrackMap:
    """
    Generates track map visualizations from GPS telemetry data.

    Supports coloring by speed, RPM, gear, or custom values.
    Outputs SVG and HTML formats.
    """

    # Predefined color schemes
    COLOR_SCHEMES = {
        'speed': ColorScale(
            name='Speed (mph)',
            min_val=0,
            max_val=150,
            colors=['#3498db', '#2ecc71', '#f1c40f', '#e74c3c'],
            labels=['0', '50', '100', '150']
        ),
        'rpm': ColorScale(
            name='RPM',
            min_val=2000,
            max_val=7500,
            colors=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'],
            labels=['2000', '4000', '6000', '7500']
        ),
        'gear': ColorScale(
            name='Gear',
            min_val=1,
            max_val=4,
            colors=['#e74c3c', '#f39c12', '#3498db', '#9b59b6'],
            labels=['1', '2', '3', '4']
        ),
        'throttle': ColorScale(
            name='Throttle %',
            min_val=0,
            max_val=100,
            colors=['#e74c3c', '#f1c40f', '#2ecc71'],
            labels=['0', '50', '100']
        ),
        'brake': ColorScale(
            name='Brake',
            min_val=0,
            max_val=1,
            colors=['#2ecc71', '#f1c40f', '#e74c3c'],
            labels=['Off', '', 'Full']
        )
    }

    def __init__(self, config: TrackMapConfig = None):
        """
        Initialize track map generator.

        Args:
            config: TrackMapConfig for rendering options
        """
        self.config = config or TrackMapConfig()

    def render_svg(
        self,
        latitude_data: np.ndarray,
        longitude_data: np.ndarray,
        color_data: np.ndarray = None,
        color_scheme: str = 'speed',
        title: str = "Track Map",
        custom_scale: ColorScale = None,
        discrete_mode: bool = True,
        low_threshold: float = 33.0,
        high_threshold: float = 66.0
    ) -> str:
        """
        Render track map as SVG.

        Args:
            latitude_data: GPS latitude values
            longitude_data: GPS longitude values
            color_data: Values for color coding (e.g., speed, rpm)
            color_scheme: Predefined scheme ('speed', 'rpm', 'gear', 'throttle', 'brake')
            title: Map title
            custom_scale: Custom ColorScale to use instead of predefined
            discrete_mode: Use 3-color discrete mode (green/yellow/red)
            low_threshold: Percentage threshold for green (0-100)
            high_threshold: Percentage threshold for red (0-100)

        Returns:
            SVG string
        """
        # Get color scale
        if custom_scale:
            scale = custom_scale
        elif color_scheme in self.COLOR_SCHEMES:
            scale = self.COLOR_SCHEMES[color_scheme]
        else:
            scale = self.COLOR_SCHEMES['speed']

        # Update scale based on actual data if provided
        if color_data is not None and len(color_data) > 0:
            scale = ColorScale(
                name=scale.name,
                min_val=float(np.nanmin(color_data)),
                max_val=float(np.nanmax(color_data)),
                colors=scale.colors,
                discrete_mode=discrete_mode,
                low_threshold=low_threshold,
                high_threshold=high_threshold
            )

        # Transform GPS to SVG coordinates
        coords = self._transform_coordinates(latitude_data, longitude_data)

        # Build SVG
        svg = self._build_svg_header()

        # Background
        svg += f'<rect width="{self.config.width}" height="{self.config.height}" fill="{self.config.background_color}"/>\n'

        # Track outline (faint)
        svg += self._build_track_outline(coords)

        # Colored track trace
        svg += self._build_colored_trace(coords, color_data, scale)

        # Start/finish marker
        if self.config.show_start_finish and len(coords) > 0:
            svg += self._build_start_finish_marker(coords[0])

        # Legend
        if self.config.show_legend:
            svg += self._build_legend(scale)

        # Title
        if self.config.show_title:
            svg += self._build_title(title)

        svg += '</svg>'
        return svg

    def render_html(
        self,
        latitude_data: np.ndarray,
        longitude_data: np.ndarray,
        color_data: np.ndarray = None,
        color_scheme: str = 'speed',
        title: str = "Track Map",
        custom_scale: ColorScale = None,
        include_controls: bool = True
    ) -> str:
        """
        Render track map as standalone HTML.

        Args:
            latitude_data: GPS latitude values
            longitude_data: GPS longitude values
            color_data: Values for color coding
            color_scheme: Predefined scheme name
            title: Map title
            custom_scale: Custom color scale
            include_controls: Include interactive controls

        Returns:
            HTML string
        """
        svg = self.render_svg(
            latitude_data, longitude_data, color_data, color_scheme, title, custom_scale
        )

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background: {self.config.background_color};
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #fff;
        }}
        .container {{
            max-width: {self.config.width + 40}px;
            margin: 0 auto;
        }}
        h1 {{
            margin-bottom: 20px;
        }}
        .map-container {{
            background: #16213e;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        svg {{
            display: block;
            margin: 0 auto;
        }}
"""

        if include_controls:
            html += """        .controls {{
            margin-top: 20px;
            padding: 15px;
            background: #16213e;
            border-radius: 8px;
        }}
        .controls label {{
            margin-right: 10px;
        }}
        .controls select {{
            padding: 5px 10px;
            background: #1a1a2e;
            color: #fff;
            border: 1px solid #333;
            border-radius: 4px;
        }}
"""

        html += """    </style>
</head>
<body>
    <div class="container">
        <div class="map-container">
"""
        html += svg
        html += """        </div>
"""

        if include_controls:
            html += f"""        <div class="controls">
            <label for="colorScheme">Color by:</label>
            <select id="colorScheme" disabled>
                <option value="speed" {'selected' if color_scheme == 'speed' else ''}>Speed</option>
                <option value="rpm" {'selected' if color_scheme == 'rpm' else ''}>RPM</option>
                <option value="gear" {'selected' if color_scheme == 'gear' else ''}>Gear</option>
            </select>
            <small style="color: #666; margin-left: 10px;">(Interactive controls require JavaScript)</small>
        </div>
"""

        html += """    </div>
</body>
</html>"""
        return html

    def _transform_coordinates(
        self,
        latitude_data: np.ndarray,
        longitude_data: np.ndarray
    ) -> List[Tuple[float, float]]:
        """Transform GPS coordinates to SVG coordinates"""
        if len(latitude_data) == 0:
            return []

        # Get bounds
        lat_min, lat_max = np.min(latitude_data), np.max(latitude_data)
        lon_min, lon_max = np.min(longitude_data), np.max(longitude_data)

        # Calculate scale to fit in viewport
        lat_range = lat_max - lat_min or 1
        lon_range = lon_max - lon_min or 1

        available_width = self.config.width - 2 * self.config.padding
        available_height = self.config.height - 2 * self.config.padding

        # Maintain aspect ratio
        scale_x = available_width / lon_range
        scale_y = available_height / lat_range
        scale = min(scale_x, scale_y)

        # Center the track
        center_x = self.config.width / 2
        center_y = self.config.height / 2
        data_center_lon = (lon_min + lon_max) / 2
        data_center_lat = (lat_min + lat_max) / 2

        coords = []
        for lat, lon in zip(latitude_data, longitude_data):
            x = center_x + (lon - data_center_lon) * scale
            y = center_y - (lat - data_center_lat) * scale  # Flip Y axis
            coords.append((x, y))

        return coords

    def _build_svg_header(self) -> str:
        """Build SVG header"""
        return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.config.width} {self.config.height}" width="{self.config.width}" height="{self.config.height}">
'''

    def _build_track_outline(self, coords: List[Tuple[float, float]]) -> str:
        """Build faint track outline"""
        if len(coords) < 2:
            return ""

        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        return f'<polyline points="{points}" fill="none" stroke="{self.config.track_outline_color}" stroke-width="{self.config.line_width + 4}" stroke-linecap="round" stroke-linejoin="round"/>\n'

    def _build_colored_trace(
        self,
        coords: List[Tuple[float, float]],
        color_data: np.ndarray,
        scale: ColorScale
    ) -> str:
        """Build colored track trace"""
        if len(coords) < 2:
            return ""

        svg = '<g class="track-trace">\n'

        # If no color data, use single color
        if color_data is None or len(color_data) == 0:
            points = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
            svg += f'<polyline points="{points}" fill="none" stroke="#3498db" stroke-width="{self.config.line_width}" stroke-linecap="round" stroke-linejoin="round"/>\n'
        else:
            # Draw colored segments
            for i in range(len(coords) - 1):
                x1, y1 = coords[i]
                x2, y2 = coords[i + 1]
                color = scale.get_color(color_data[i])
                svg += f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{self.config.line_width}" stroke-linecap="round"/>\n'

        svg += '</g>\n'
        return svg

    def _build_start_finish_marker(self, coord: Tuple[float, float]) -> str:
        """Build start/finish marker"""
        x, y = coord
        return f'''<g class="start-finish">
    <circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="#ffffff" stroke="#1a1a2e" stroke-width="2"/>
    <text x="{x:.1f}" y="{y + 20:.1f}" text-anchor="middle" fill="#ffffff" font-size="10" font-family="sans-serif">S/F</text>
</g>
'''

    def _build_legend(self, scale: ColorScale) -> str:
        """Build color legend"""
        legend_width = 150
        legend_height = 20

        if self.config.legend_position == "bottom-right":
            x = self.config.width - legend_width - self.config.padding
            y = self.config.height - 50
        elif self.config.legend_position == "top-right":
            x = self.config.width - legend_width - self.config.padding
            y = self.config.padding + 30
        else:  # bottom-left
            x = self.config.padding
            y = self.config.height - 50

        svg = f'<g class="legend" transform="translate({x}, {y})">\n'

        # Legend title
        svg += f'<text x="0" y="-5" fill="{self.config.title_color}" font-size="10" font-family="sans-serif">{scale.name}</text>\n'

        if scale.discrete_mode:
            # Discrete 3-band legend
            low_pct = scale.low_threshold / 100
            high_pct = scale.high_threshold / 100

            # Green band (0 to low_threshold)
            green_width = low_pct * legend_width
            svg += f'<rect x="0" y="0" width="{green_width:.1f}" height="{legend_height}" fill="{scale.DISCRETE_COLORS[0]}" rx="3" ry="3"/>\n'

            # Yellow band (low_threshold to high_threshold)
            yellow_width = (high_pct - low_pct) * legend_width
            svg += f'<rect x="{green_width:.1f}" y="0" width="{yellow_width:.1f}" height="{legend_height}" fill="{scale.DISCRETE_COLORS[1]}"/>\n'

            # Red band (high_threshold to 100)
            red_width = (1 - high_pct) * legend_width
            svg += f'<rect x="{green_width + yellow_width:.1f}" y="0" width="{red_width:.1f}" height="{legend_height}" fill="{scale.DISCRETE_COLORS[2]}" rx="3" ry="3"/>\n'

            # Labels for discrete mode
            val_range = scale.max_val - scale.min_val
            low_val = scale.min_val + val_range * low_pct
            high_val = scale.min_val + val_range * high_pct
            labels = [f"{scale.min_val:.0f}", f"{low_val:.0f}", f"{high_val:.0f}", f"{scale.max_val:.0f}"]
            positions = [0, green_width, green_width + yellow_width, legend_width]

            for pos, label in zip(positions, labels):
                svg += f'<text x="{pos:.1f}" y="{legend_height + 12}" fill="{self.config.title_color}" font-size="9" text-anchor="middle" font-family="sans-serif">{label}</text>\n'
        else:
            # Gradient bar
            gradient_id = f"legend-gradient-{id(scale)}"
            svg += f'<defs><linearGradient id="{gradient_id}">\n'
            for i, color in enumerate(scale.colors):
                offset = i / (len(scale.colors) - 1) * 100
                svg += f'<stop offset="{offset}%" stop-color="{color}"/>\n'
            svg += '</linearGradient></defs>\n'

            svg += f'<rect x="0" y="0" width="{legend_width}" height="{legend_height}" fill="url(#{gradient_id})" rx="3"/>\n'

            # Labels
            for i, label in enumerate(scale.labels):
                label_x = i / (len(scale.labels) - 1) * legend_width
                svg += f'<text x="{label_x}" y="{legend_height + 12}" fill="{self.config.title_color}" font-size="9" text-anchor="middle" font-family="sans-serif">{label}</text>\n'

        svg += '</g>\n'
        return svg

    def _build_title(self, title: str) -> str:
        """Build title"""
        return f'<text x="{self.config.padding}" y="{self.config.padding - 10}" fill="{self.config.title_color}" font-size="16" font-weight="bold" font-family="sans-serif">{title}</text>\n'

    def to_dict(
        self,
        latitude_data: np.ndarray,
        longitude_data: np.ndarray,
        color_data: np.ndarray = None,
        color_scheme: str = 'speed'
    ) -> Dict:
        """
        Convert track map data to dictionary format.

        Useful for JSON serialization or passing to frontend.

        Args:
            latitude_data: GPS latitude values
            longitude_data: GPS longitude values
            color_data: Values for color coding
            color_scheme: Color scheme name

        Returns:
            Dictionary with track data
        """
        scale = self.COLOR_SCHEMES.get(color_scheme, self.COLOR_SCHEMES['speed'])

        if color_data is not None and len(color_data) > 0:
            colors = [scale.get_color(v) for v in color_data]
        else:
            colors = ['#3498db'] * len(latitude_data)

        return {
            "coordinates": [
                {"lat": float(lat), "lon": float(lon), "color": color}
                for lat, lon, color in zip(latitude_data, longitude_data, colors)
            ],
            "bounds": {
                "lat_min": float(np.min(latitude_data)),
                "lat_max": float(np.max(latitude_data)),
                "lon_min": float(np.min(longitude_data)),
                "lon_max": float(np.max(longitude_data))
            },
            "color_scheme": color_scheme,
            "config": {
                "width": self.config.width,
                "height": self.config.height
            }
        }

    def save_svg(
        self,
        filepath: str,
        latitude_data: np.ndarray,
        longitude_data: np.ndarray,
        color_data: np.ndarray = None,
        color_scheme: str = 'speed',
        title: str = "Track Map"
    ):
        """Save track map as SVG file"""
        svg = self.render_svg(
            latitude_data, longitude_data, color_data, color_scheme, title
        )
        Path(filepath).write_text(svg)

    def save_html(
        self,
        filepath: str,
        latitude_data: np.ndarray,
        longitude_data: np.ndarray,
        color_data: np.ndarray = None,
        color_scheme: str = 'speed',
        title: str = "Track Map"
    ):
        """Save track map as HTML file"""
        html = self.render_html(
            latitude_data, longitude_data, color_data, color_scheme, title
        )
        Path(filepath).write_text(html)

    def render_delta_svg(
        self,
        ref_lat: np.ndarray,
        ref_lon: np.ndarray,
        comp_lat: np.ndarray,
        comp_lon: np.ndarray,
        segment_deltas: List[Dict],
        title: str = "Delta Track Map",
        ref_label: str = "Lap A",
        comp_label: str = "Lap B"
    ) -> str:
        """
        Render track map colored by time delta between two laps.

        Green sections = reference lap faster (gaining time)
        Red sections = comparison lap faster (losing time)

        Args:
            ref_lat: GPS latitude for reference lap
            ref_lon: GPS longitude for reference lap
            comp_lat: GPS latitude for comparison lap
            comp_lon: GPS longitude for comparison lap
            segment_deltas: List of dicts with 'start_pct', 'end_pct', 'time_delta', 'faster'
            title: Map title
            ref_label: Label for reference lap
            comp_label: Label for comparison lap

        Returns:
            SVG string
        """
        # Use reference lap for track outline
        coords = self._transform_coordinates(ref_lat, ref_lon)

        if len(coords) < 2:
            return self._build_svg_header() + '</svg>'

        # Build color scale for delta
        max_delta = max(abs(s.get('time_delta', 0)) for s in segment_deltas) if segment_deltas else 0.5
        max_delta = max(max_delta, 0.1)  # Minimum range

        delta_scale = ColorScale(
            name=f'Time Delta ({ref_label} vs {comp_label})',
            min_val=-max_delta,
            max_val=max_delta,
            colors=['#e74c3c', '#f1c40f', '#2ecc71'],  # Red -> Yellow -> Green
            labels=[f'-{max_delta:.1f}s', '0', f'+{max_delta:.1f}s']
        )

        # Assign delta values to each point based on which segment it falls into
        num_points = len(coords)
        point_deltas = np.zeros(num_points)

        for i in range(num_points):
            pct = (i / (num_points - 1)) * 100 if num_points > 1 else 0
            # Find which segment this point belongs to
            for seg in segment_deltas:
                if seg['start_pct'] <= pct < seg['end_pct'] or (seg['end_pct'] == 100 and pct == 100):
                    # Positive delta means ref lap is slower (time_delta = comparison - reference)
                    # We want green when ref is faster, so we negate
                    point_deltas[i] = -seg.get('time_delta', 0)
                    break

        # Build SVG
        svg = self._build_svg_header()

        # Background
        svg += f'<rect width="{self.config.width}" height="{self.config.height}" fill="{self.config.background_color}"/>\n'

        # Track outline (faint)
        svg += self._build_track_outline(coords)

        # Colored delta trace
        svg += self._build_delta_trace(coords, point_deltas, delta_scale)

        # Start/finish marker
        if self.config.show_start_finish and len(coords) > 0:
            svg += self._build_start_finish_marker(coords[0])

        # Delta legend
        if self.config.show_legend:
            svg += self._build_delta_legend(delta_scale, ref_label, comp_label)

        # Title
        if self.config.show_title:
            svg += self._build_title(title)

        svg += '</svg>'
        return svg

    def _build_delta_trace(
        self,
        coords: List[Tuple[float, float]],
        delta_data: np.ndarray,
        scale: ColorScale
    ) -> str:
        """Build track trace colored by time delta"""
        if len(coords) < 2:
            return ""

        svg = '<g class="delta-trace">\n'

        # Draw colored segments
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            delta = delta_data[i]

            # Color: green for positive (ref faster), red for negative (ref slower)
            color = scale.get_color(delta)
            svg += f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{self.config.line_width + 1}" stroke-linecap="round"/>\n'

        svg += '</g>\n'
        return svg

    def _build_delta_legend(
        self,
        scale: ColorScale,
        ref_label: str,
        comp_label: str
    ) -> str:
        """Build legend for delta track map"""
        legend_width = 180
        legend_height = 20

        x = self.config.width - legend_width - self.config.padding
        y = self.config.height - 60

        svg = f'<g class="delta-legend" transform="translate({x}, {y})">\n'

        # Gradient bar
        gradient_id = f"delta-gradient-{id(scale)}"
        svg += f'<defs><linearGradient id="{gradient_id}">\n'
        for i, color in enumerate(scale.colors):
            offset = i / (len(scale.colors) - 1) * 100
            svg += f'<stop offset="{offset}%" stop-color="{color}"/>\n'
        svg += '</linearGradient></defs>\n'

        svg += f'<rect x="0" y="0" width="{legend_width}" height="{legend_height}" fill="url(#{gradient_id})" rx="3"/>\n'

        # Labels
        svg += f'<text x="0" y="-5" fill="{self.config.title_color}" font-size="10" font-family="sans-serif">{ref_label} faster</text>\n'
        svg += f'<text x="{legend_width}" y="-5" fill="{self.config.title_color}" font-size="10" text-anchor="end" font-family="sans-serif">{comp_label} faster</text>\n'

        # Min/max labels
        svg += f'<text x="0" y="{legend_height + 12}" fill="{self.config.title_color}" font-size="9" font-family="sans-serif">{scale.min_val:.1f}s</text>\n'
        svg += f'<text x="{legend_width/2}" y="{legend_height + 12}" fill="{self.config.title_color}" font-size="9" text-anchor="middle" font-family="sans-serif">0s</text>\n'
        svg += f'<text x="{legend_width}" y="{legend_height + 12}" fill="{self.config.title_color}" font-size="9" text-anchor="end" font-family="sans-serif">+{scale.max_val:.1f}s</text>\n'

        svg += '</g>\n'
        return svg
