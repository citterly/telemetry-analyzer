"""
Analysis package for telemetry data processing
Contains all existing analysis modules
"""

# Make analysis modules easily importable
from .data_loader import load_session_data, get_data_summary
from .lap_analyzer import analyze_session_laps
from .gear_calculator import analyze_lap_gearing
from .acceleration_analyzer import analyze_power_simple
from .track_plotter import plot_fastest_lap_analysis

__all__ = [
    'load_session_data',
    'get_data_summary', 
    'analyze_session_laps',
    'analyze_lap_gearing',
    'analyze_power_simple',
    'plot_fastest_lap_analysis'
]