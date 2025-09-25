"""
Configuration file for transmission analysis
Contains all constants, settings, and gear specifications
"""

from pathlib import Path
from typing import Dict, List, Tuple

# File paths
PROJECT_ROOT = Path(__file__).parent.parent
THIRD_PARTY_ROOT = PROJECT_ROOT / "third-party" / "AIM"
DLL_PATH = THIRD_PARTY_ROOT / "DLL-2022" / "MatLabXRK-2022-64-ReleaseU.dll"
DEPENDENCY_PATH = THIRD_PARTY_ROOT / "64"
SAMPLE_FILES_PATH = THIRD_PARTY_ROOT / "sample_files"

# Default session file
DEFAULT_SESSION = "20250713_094018_Road America_a_0411.xrk"

# Vehicle specifications
TIRE_SIZE = "275/35/18"  # Andy's current tire size
TIRE_CIRCUMFERENCE_METERS = 2.026  # Calculated from 275/35/18

# Current transmission setup
CURRENT_SETUP = {
    'name': 'Current Setup',
    'transmission_ratios': [2.20, 1.64, 1.28, 1.00],
    'final_drive': 3.55,
    'weight_lbs': 3450
}

# Alternative transmission options
TRANSMISSION_SCENARIOS = [
    CURRENT_SETUP,
    {
        'name': 'New Trans + Current Final',
        'transmission_ratios': [2.88, 1.91, 1.33, 1.00],
        'final_drive': 3.55,
        'weight_lbs': 3400  # 50 lbs lighter
    },
    {
        'name': 'New Trans + Shorter Final',
        'transmission_ratios': [2.88, 1.91, 1.33, 1.00],
        'final_drive': 3.73,
        'weight_lbs': 3400
    },
    {
        'name': 'New Trans + Taller Final',
        'transmission_ratios': [2.88, 1.91, 1.33, 1.00],
        'final_drive': 3.31,
        'weight_lbs': 3400
    }
]

# Engine specifications
ENGINE_SPECS = {
    'max_rpm': 8000,  # Absolute rev limiter
    'safe_rpm_limit': 7000,  # Safe operating limit (pro teams recommend staying under this)
    'shift_rpm': 6800,  # Suggested shift point for safety
    'power_band_min': 5500,  # Start of good power
    'power_band_max': 7000,  # End of safe power band
    'idle_rpm': 800
}

# RPM color zone definitions
RPM_ZONES = {
    'green': {'min': 0, 'max': 6000, 'label': 'Safe RPM', 'color': 'green'},
    'yellow': {'min': 6000, 'max': 7000, 'label': 'Power Band', 'color': 'gold'},
    'red': {'min': 7000, 'max': 10000, 'label': 'DANGER ZONE', 'color': 'red'}
}

# Track-specific settings
TRACK_CONFIG = {
    'name': 'Road America',
    'length_meters': 4048,  # 2.52 miles
    'main_straight_length': 800,  # Approximate
    'corner_count': 14,
    'start_finish_gps': (43.797875, -87.989638),  # Actual Road America start/finish line
    'turn_coordinates': {
        'T1': (43.792069, -87.989800),
        'T3': (43.791595, -87.995327),
        'T5': (43.801770, -87.992596),
        'T6': (43.801592, -87.996089),
        'T7': (43.799578, -87.996243),
        'T8': (43.797101, -87.999953),
        'Carousel': (43.794216, -87.999916),
        'Kink': (43.798512, -88.002654),
        'T12': (43.804928, -87.997402),
        'T13': (43.803806, -87.994009),
        'T14': (43.804001, -87.990058)
    }
}

# Data processing settings
PROCESSING_CONFIG = {
    'min_lap_time_seconds': 90,  # Minimum reasonable lap time
    'max_lap_time_seconds': 300,  # Maximum reasonable lap time
    'start_finish_threshold': 0.0001,  # GPS coordinate threshold for lap detection
    'rpm_interpolation_method': 'linear',
    'smoothing_window': 5  # Points for data smoothing
}

# Plotting settings
PLOT_CONFIG = {
    'figure_size': (14, 10),
    'point_size': 12,
    'alpha': 0.8,
    'grid_alpha': 0.3,
    'title_fontsize': 16,
    'label_fontsize': 12,
    'legend_fontsize': 10
}

def calculate_tire_circumference(tire_size: str) -> float:
    """
    Calculate tire circumference from tire size string
    Format: "275/35/18" = width/aspect_ratio/wheel_diameter
    """
    try:
        parts = tire_size.split('/')
        width_mm = int(parts[0])
        aspect_ratio = int(parts[1])
        wheel_diameter_inches = int(parts[2])
        
        # Calculate sidewall height
        sidewall_mm = width_mm * (aspect_ratio / 100)
        
        # Calculate total diameter in mm
        wheel_diameter_mm = wheel_diameter_inches * 25.4
        tire_diameter_mm = wheel_diameter_mm + (2 * sidewall_mm)
        
        # Calculate circumference in meters
        circumference_m = (tire_diameter_mm / 1000) * 3.14159
        
        return circumference_m
        
    except (ValueError, IndexError):
        # Fallback to default if parsing fails
        return 2.026

def theoretical_speed_at_rpm(rpm: float, gear_ratio: float, final_drive: float, 
                           tire_circumference: float = TIRE_CIRCUMFERENCE_METERS) -> float:
    """
    Calculate theoretical speed in m/s at given RPM and gear ratios
    """
    if rpm <= 0 or gear_ratio <= 0 or final_drive <= 0:
        return 0.0
    
    # Speed = (RPM / 60) * tire_circumference / (gear_ratio * final_drive)
    speed_ms = (rpm / 60) * tire_circumference / (gear_ratio * final_drive)
    return speed_ms

def theoretical_rpm_at_speed(speed_ms: float, gear_ratio: float, final_drive: float,
                           tire_circumference: float = TIRE_CIRCUMFERENCE_METERS) -> float:
    """
    Calculate theoretical RPM at given speed and gear ratios
    """
    if speed_ms <= 0 or gear_ratio <= 0 or final_drive <= 0:
        return 0.0
    
    # RPM = (speed_ms * 60 * gear_ratio * final_drive) / tire_circumference
    rpm = (speed_ms * 60 * gear_ratio * final_drive) / tire_circumference
    return rpm

def get_scenario_by_name(name: str) -> Dict:
    """Get transmission scenario by name"""
    for scenario in TRANSMISSION_SCENARIOS:
        if scenario['name'] == name:
            return scenario
    return CURRENT_SETUP

def print_config_summary():
    """Print configuration summary for verification"""
    print("Transmission Analysis Configuration")
    print("=" * 50)
    print(f"Tire Size: {TIRE_SIZE}")
    print(f"Tire Circumference: {TIRE_CIRCUMFERENCE_METERS:.3f} m")
    print(f"Default Session: {DEFAULT_SESSION}")
    print(f"\nTransmission Scenarios:")
    for scenario in TRANSMISSION_SCENARIOS:
        print(f"  {scenario['name']}: {scenario['transmission_ratios']} @ {scenario['final_drive']} final")
    print(f"\nEngine Specs:")
    print(f"  Rev Limiter: {ENGINE_SPECS['max_rpm']} RPM")
    print(f"  Power Band: {ENGINE_SPECS['power_band_min']}-{ENGINE_SPECS['power_band_max']} RPM")

if __name__ == "__main__":
    # Test the configuration
    print_config_summary()
    
    # Test tire calculation
    calculated = calculate_tire_circumference(TIRE_SIZE)
    print(f"\nTire circumference calculation test:")
    print(f"  Input: {TIRE_SIZE}")
    print(f"  Calculated: {calculated:.3f} m")
    print(f"  Config value: {TIRE_CIRCUMFERENCE_METERS:.3f} m")
    
    # Test speed/RPM calculations
    print(f"\nSpeed/RPM calculation tests:")
    test_rpm = 7000
    test_gear = 1.28  # 3rd gear
    test_final = 3.55
    
    speed = theoretical_speed_at_rpm(test_rpm, test_gear, test_final)
    rpm_back = theoretical_rpm_at_speed(speed, test_gear, test_final)
    
    print(f"  {test_rpm} RPM in gear {test_gear} = {speed:.1f} m/s ({speed * 2.237:.1f} mph)")
    print(f"  Round trip check: {rpm_back:.0f} RPM")
    
    print(f"\nFile paths:")
    print(f"  DLL: {DLL_PATH.exists()} - {DLL_PATH}")
    print(f"  Sample files: {SAMPLE_FILES_PATH.exists()} - {SAMPLE_FILES_PATH}")