"""
G-G Diagram Visualization
Scatter plot visualization for friction circle analysis.

Generates both static (SVG) and interactive (Chart.js compatible) outputs.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json


@dataclass
class GGDiagramConfig:
    """Configuration for G-G diagram rendering"""
    width: int = 600
    height: int = 600
    padding: int = 60
    point_size: int = 3
    background_color: str = "#1a1a2e"
    grid_color: str = "#333344"
    axis_color: str = "#666677"
    title_color: str = "#ffffff"
    reference_circle_color: str = "#e74c3c"
    data_circle_color: str = "#f1c40f"
    show_reference_circle: bool = True
    show_data_circle: bool = True
    show_grid: bool = True
    show_axes: bool = True


class GGDiagram:
    """
    Generates G-G diagram visualizations.

    Supports SVG output and Chart.js compatible data format.
    """

    # Color schemes for different data attributes
    COLOR_SCHEMES = {
        'speed': {
            'name': 'Speed (mph)',
            'colors': ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c'],
            'default_range': (0, 120)
        },
        'throttle': {
            'name': 'Throttle %',
            'colors': ['#e74c3c', '#f1c40f', '#2ecc71'],
            'default_range': (0, 100)
        },
        'gear': {
            'name': 'Gear',
            'colors': ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#1abc9c'],
            'default_range': (1, 5)
        },
        'lap': {
            'name': 'Lap',
            'colors': ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f1c40f'],
            'default_range': (1, 10)
        },
        'default': {
            'name': 'G-Force',
            'colors': ['#3498db'],
            'default_range': (0, 1.5)
        }
    }

    def __init__(self, config: GGDiagramConfig = None):
        """
        Initialize G-G diagram generator.

        Args:
            config: GGDiagramConfig for rendering options
        """
        self.config = config or GGDiagramConfig()

    def render_svg(
        self,
        lat_acc: np.ndarray,
        lon_acc: np.ndarray,
        color_data: np.ndarray = None,
        color_scheme: str = 'default',
        reference_max_g: float = 1.3,
        data_max_g: float = None,
        title: str = "G-G Diagram"
    ) -> str:
        """
        Render G-G diagram as SVG.

        Args:
            lat_acc: Lateral acceleration (g)
            lon_acc: Longitudinal acceleration (g)
            color_data: Values for color coding (speed, throttle, etc.)
            color_scheme: Color scheme name
            reference_max_g: Reference circle radius (from vehicle config)
            data_max_g: Data-derived max g (95th percentile)
            title: Diagram title

        Returns:
            SVG string
        """
        # Calculate plot bounds (symmetric around origin)
        max_val = max(
            np.abs(lat_acc).max() if len(lat_acc) > 0 else 1.0,
            np.abs(lon_acc).max() if len(lon_acc) > 0 else 1.0,
            reference_max_g
        ) * 1.1

        # Transform to SVG coordinates
        def to_svg(lat_g, lon_g):
            cx = self.config.width / 2
            cy = self.config.height / 2
            scale = (min(self.config.width, self.config.height) - 2 * self.config.padding) / (2 * max_val)
            x = cx + lat_g * scale
            y = cy - lon_g * scale  # Flip Y axis
            return x, y

        # Build SVG
        svg = self._build_svg_header()
        svg += f'<rect width="{self.config.width}" height="{self.config.height}" fill="{self.config.background_color}"/>\n'

        # Grid
        if self.config.show_grid:
            svg += self._build_grid(max_val, to_svg)

        # Axes
        if self.config.show_axes:
            svg += self._build_axes(max_val, to_svg)

        # Reference circles
        if self.config.show_reference_circle:
            svg += self._build_reference_circle(reference_max_g, to_svg, self.config.reference_circle_color, "Max G")

        if self.config.show_data_circle and data_max_g:
            svg += self._build_reference_circle(data_max_g, to_svg, self.config.data_circle_color, "95th %ile")

        # Data points
        svg += self._build_data_points(lat_acc, lon_acc, color_data, color_scheme, to_svg)

        # Title
        svg += self._build_title(title)

        # Legend
        svg += self._build_legend(reference_max_g, data_max_g)

        svg += '</svg>'
        return svg

    def to_chartjs_data(
        self,
        lat_acc: np.ndarray,
        lon_acc: np.ndarray,
        color_data: np.ndarray = None,
        color_scheme: str = 'default',
        reference_max_g: float = 1.3,
        data_max_g: float = None,
        downsample: int = 5
    ) -> Dict:
        """
        Convert G-G data to Chart.js compatible format.

        Args:
            lat_acc: Lateral acceleration (g)
            lon_acc: Longitudinal acceleration (g)
            color_data: Values for color coding
            color_scheme: Color scheme name
            reference_max_g: Reference circle radius
            data_max_g: Data-derived max g
            downsample: Keep every Nth point to reduce data size

        Returns:
            Dictionary with Chart.js datasets
        """
        # Downsample data
        indices = range(0, len(lat_acc), downsample)
        lat = lat_acc[list(indices)]
        lon = lon_acc[list(indices)]

        # Build scatter data points
        points = [{"x": float(x), "y": float(y)} for x, y in zip(lat, lon)]

        # Color points if color_data provided
        colors = None
        if color_data is not None:
            colors_arr = color_data[list(indices)]
            scheme = self.COLOR_SCHEMES.get(color_scheme, self.COLOR_SCHEMES['default'])
            colors = [self._get_color(v, scheme) for v in colors_arr]

        return {
            "datasets": [
                {
                    "label": "G-G Data",
                    "data": points,
                    "backgroundColor": colors if colors else "rgba(52, 152, 219, 0.6)",
                    "pointRadius": 2
                }
            ],
            "reference_circles": [
                {"radius": reference_max_g, "label": "Max G", "color": "#e74c3c"},
                {"radius": data_max_g, "label": "95th %ile", "color": "#f1c40f"} if data_max_g else None
            ],
            "bounds": {
                "max_g": max(
                    float(np.abs(lat_acc).max()),
                    float(np.abs(lon_acc).max()),
                    reference_max_g
                ) * 1.1
            }
        }

    def _build_svg_header(self) -> str:
        """Build SVG header"""
        return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.config.width} {self.config.height}" width="{self.config.width}" height="{self.config.height}">
'''

    def _build_grid(self, max_val: float, to_svg) -> str:
        """Build grid lines"""
        svg = '<g class="grid">\n'
        cx, cy = to_svg(0, 0)

        # Grid circles at 0.5g intervals
        for g in np.arange(0.5, max_val + 0.5, 0.5):
            x0, y0 = to_svg(-g, 0)
            x1, _ = to_svg(g, 0)
            radius = (x1 - x0) / 2
            svg += f'<circle cx="{cx}" cy="{cy}" r="{radius:.1f}" fill="none" stroke="{self.config.grid_color}" stroke-width="1" stroke-dasharray="4,4"/>\n'

        svg += '</g>\n'
        return svg

    def _build_axes(self, max_val: float, to_svg) -> str:
        """Build axis lines"""
        svg = '<g class="axes">\n'
        cx, cy = to_svg(0, 0)

        # Horizontal axis (lateral)
        x0, _ = to_svg(-max_val, 0)
        x1, _ = to_svg(max_val, 0)
        svg += f'<line x1="{x0:.1f}" y1="{cy:.1f}" x2="{x1:.1f}" y2="{cy:.1f}" stroke="{self.config.axis_color}" stroke-width="1"/>\n'

        # Vertical axis (longitudinal)
        _, y0 = to_svg(0, -max_val)
        _, y1 = to_svg(0, max_val)
        svg += f'<line x1="{cx:.1f}" y1="{y0:.1f}" x2="{cx:.1f}" y2="{y1:.1f}" stroke="{self.config.axis_color}" stroke-width="1"/>\n'

        # Axis labels
        svg += f'<text x="{x1 - 10:.1f}" y="{cy - 10:.1f}" fill="{self.config.title_color}" font-size="12" font-family="sans-serif">Right</text>\n'
        svg += f'<text x="{x0 + 5:.1f}" y="{cy - 10:.1f}" fill="{self.config.title_color}" font-size="12" font-family="sans-serif">Left</text>\n'
        svg += f'<text x="{cx + 5:.1f}" y="{y1 + 15:.1f}" fill="{self.config.title_color}" font-size="12" font-family="sans-serif">Accel</text>\n'
        svg += f'<text x="{cx + 5:.1f}" y="{y0 - 5:.1f}" fill="{self.config.title_color}" font-size="12" font-family="sans-serif">Brake</text>\n'

        svg += '</g>\n'
        return svg

    def _build_reference_circle(self, radius: float, to_svg, color: str, label: str) -> str:
        """Build a reference circle"""
        cx, cy = to_svg(0, 0)
        x0, _ = to_svg(-radius, 0)
        x1, _ = to_svg(radius, 0)
        r = (x1 - x0) / 2

        svg = f'<circle cx="{cx}" cy="{cy}" r="{r:.1f}" fill="none" stroke="{color}" stroke-width="2"/>\n'
        svg += f'<text x="{cx + r + 5:.1f}" y="{cy:.1f}" fill="{color}" font-size="10" font-family="sans-serif">{label}: {radius:.2f}g</text>\n'
        return svg

    def _build_data_points(
        self,
        lat_acc: np.ndarray,
        lon_acc: np.ndarray,
        color_data: np.ndarray,
        color_scheme: str,
        to_svg
    ) -> str:
        """Build data points"""
        svg = '<g class="data-points">\n'

        scheme = self.COLOR_SCHEMES.get(color_scheme, self.COLOR_SCHEMES['default'])

        # Downsample for SVG (every 3rd point)
        for i in range(0, len(lat_acc), 3):
            x, y = to_svg(lat_acc[i], lon_acc[i])

            if color_data is not None:
                color = self._get_color(color_data[i], scheme)
            else:
                color = scheme['colors'][0]

            svg += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{self.config.point_size}" fill="{color}" opacity="0.7"/>\n'

        svg += '</g>\n'
        return svg

    def _get_color(self, value: float, scheme: Dict) -> str:
        """Get color for a value based on scheme"""
        colors = scheme['colors']
        min_val, max_val = scheme['default_range']

        if max_val == min_val:
            return colors[0]

        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0, min(1, normalized))

        idx = int(normalized * (len(colors) - 1))
        idx = min(idx, len(colors) - 1)

        return colors[idx]

    def _build_title(self, title: str) -> str:
        """Build title"""
        return f'<text x="{self.config.padding}" y="25" fill="{self.config.title_color}" font-size="16" font-weight="bold" font-family="sans-serif">{title}</text>\n'

    def _build_legend(self, reference_max_g: float, data_max_g: float) -> str:
        """Build legend"""
        svg = '<g class="legend">\n'
        y = self.config.height - 30

        # Reference circle
        svg += f'<circle cx="20" cy="{y}" r="6" fill="none" stroke="{self.config.reference_circle_color}" stroke-width="2"/>\n'
        svg += f'<text x="35" y="{y + 4}" fill="{self.config.title_color}" font-size="11" font-family="sans-serif">Max G: {reference_max_g:.2f}g</text>\n'

        # Data circle
        if data_max_g:
            svg += f'<circle cx="150" cy="{y}" r="6" fill="none" stroke="{self.config.data_circle_color}" stroke-width="2"/>\n'
            svg += f'<text x="165" y="{y + 4}" fill="{self.config.title_color}" font-size="11" font-family="sans-serif">95th %ile: {data_max_g:.2f}g</text>\n'

        svg += '</g>\n'
        return svg

    def save_svg(
        self,
        filepath: str,
        lat_acc: np.ndarray,
        lon_acc: np.ndarray,
        color_data: np.ndarray = None,
        color_scheme: str = 'default',
        reference_max_g: float = 1.3,
        data_max_g: float = None,
        title: str = "G-G Diagram"
    ):
        """Save G-G diagram as SVG file"""
        svg = self.render_svg(
            lat_acc, lon_acc, color_data, color_scheme,
            reference_max_g, data_max_g, title
        )
        Path(filepath).write_text(svg)
