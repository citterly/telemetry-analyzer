"""
Gear calculator module for determining current gear based on RPM and speed
Uses transmission ratios, final drive, and tire size for accurate calculations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .config import (
    TIRE_CIRCUMFERENCE_METERS, 
    CURRENT_SETUP, 
    TRANSMISSION_SCENARIOS,
    theoretical_rpm_at_speed,
    theoretical_speed_at_rpm
)

@dataclass
class GearInfo:
    """Information about gear usage at a specific point"""
    gear: int  # 0 = neutral/unknown, 1-6 = gear number
    confidence: float  # 0-1, how confident we are in this gear
    theoretical_rpm: float  # What RPM should be at this speed/gear
    rpm_difference: float  # Actual RPM - theoretical RPM
    speed_mph: float
    actual_rpm: float

class GearCalculator:
    """Calculator for determining gear from RPM and speed data"""
    
    def __init__(self, transmission_ratios: List[float], final_drive: float):
        self.transmission_ratios = transmission_ratios
        self.final_drive = final_drive
        self.tire_circumference = TIRE_CIRCUMFERENCE_METERS
        
    def calculate_gear_at_point(self, rpm: float, speed_mph: float) -> GearInfo:
        """Calculate most likely gear at a single RPM/speed point"""
        
        # Convert speed to m/s
        speed_ms = speed_mph / 2.237
        
        # Skip calculation for very low speeds or RPM
        if speed_ms < 2.0 or rpm < 1000:
            return GearInfo(
                gear=0, confidence=0.0, theoretical_rpm=0,
                rpm_difference=999999, speed_mph=speed_mph, actual_rpm=rpm
            )
        
        # Test each gear to find best match
        gear_candidates = []
        
        for gear_num, ratio in enumerate(self.transmission_ratios, 1):
            theoretical_rpm = theoretical_rpm_at_speed(
                speed_ms, ratio, self.final_drive, self.tire_circumference
            )
            
            rpm_diff = abs(rpm - theoretical_rpm)
            
            # Calculate confidence based on how close the RPM match is
            if rpm_diff < 200:
                confidence = 1.0 - (rpm_diff / 200)
            elif rpm_diff < 500:
                confidence = 0.5 - (rpm_diff / 1000)
            else:
                confidence = 0.0
            
            gear_candidates.append({
                'gear': gear_num,
                'theoretical_rpm': theoretical_rpm,
                'rpm_difference': rpm_diff,
                'confidence': max(0.0, confidence)
            })
        
        # Find best match
        best_match = max(gear_candidates, key=lambda x: x['confidence'])
        
        return GearInfo(
            gear=best_match['gear'] if best_match['confidence'] > 0.1 else 0,
            confidence=best_match['confidence'],
            theoretical_rpm=best_match['theoretical_rpm'],
            rpm_difference=best_match['rpm_difference'],
            speed_mph=speed_mph,
            actual_rpm=rpm
        )
    
    def calculate_gear_trace(self, rpm_array: np.ndarray, speed_mph_array: np.ndarray) -> List[GearInfo]:
        """Calculate gear for entire speed/RPM trace"""
        
        if len(rpm_array) != len(speed_mph_array):
            raise ValueError("RPM and speed arrays must be same length")
        
        gear_trace = []
        
        for i, (rpm, speed) in enumerate(zip(rpm_array, speed_mph_array)):
            gear_info = self.calculate_gear_at_point(rpm, speed)
            gear_trace.append(gear_info)
        
        # Apply smoothing to reduce gear jitter
        smoothed_trace = self._smooth_gear_trace(gear_trace)
        
        return smoothed_trace
    
    def _smooth_gear_trace(self, gear_trace: List[GearInfo]) -> List[GearInfo]:
        """Apply smoothing to reduce unrealistic gear changes"""
        
        if len(gear_trace) < 5:
            return gear_trace
        
        smoothed = gear_trace.copy()
        
        # Remove single-point gear anomalies
        for i in range(2, len(gear_trace) - 2):
            current_gear = gear_trace[i].gear
            prev_gear = gear_trace[i-1].gear
            next_gear = gear_trace[i+1].gear
            
            # If this gear is different from both neighbors and has low confidence
            if (current_gear != prev_gear and current_gear != next_gear and 
                gear_trace[i].confidence < 0.5):
                
                # Use the more common neighboring gear
                if prev_gear == next_gear:
                    smoothed[i] = GearInfo(
                        gear=prev_gear,
                        confidence=gear_trace[i].confidence * 0.7,  # Reduce confidence
                        theoretical_rpm=gear_trace[i].theoretical_rpm,
                        rpm_difference=gear_trace[i].rpm_difference,
                        speed_mph=gear_trace[i].speed_mph,
                        actual_rpm=gear_trace[i].actual_rpm
                    )
        
        return smoothed
    
    def get_gear_usage_summary(self, gear_trace: List[GearInfo]) -> Dict:
        """Generate summary statistics for gear usage"""
        
        total_points = len(gear_trace)
        gear_counts = {}
        gear_speeds = {}
        
        for gear_info in gear_trace:
            gear = gear_info.gear
            if gear > 0:  # Skip neutral/unknown
                if gear not in gear_counts:
                    gear_counts[gear] = 0
                    gear_speeds[gear] = []
                
                gear_counts[gear] += 1
                gear_speeds[gear].append(gear_info.speed_mph)
        
        # Calculate percentages and speed ranges
        summary = {}
        
        for gear, count in gear_counts.items():
            speeds = gear_speeds[gear]
            summary[gear] = {
                'usage_percent': (count / total_points) * 100,
                'sample_count': count,
                'speed_range_mph': {
                    'min': min(speeds),
                    'max': max(speeds),
                    'avg': sum(speeds) / len(speeds)
                }
            }
        
        return summary
    
    def find_shift_points(self, gear_trace: List[GearInfo], time_array: np.ndarray) -> List[Dict]:
        """Find gear shift points in the trace"""
        
        shifts = []
        
        for i in range(1, len(gear_trace)):
            prev_gear = gear_trace[i-1].gear
            current_gear = gear_trace[i].gear
            
            # Skip shifts involving neutral/unknown gears
            if prev_gear > 0 and current_gear > 0 and prev_gear != current_gear:
                shift = {
                    'time': time_array[i],
                    'from_gear': prev_gear,
                    'to_gear': current_gear,
                    'type': 'upshift' if current_gear > prev_gear else 'downshift',
                    'rpm': gear_trace[i-1].actual_rpm,
                    'speed_mph': gear_trace[i-1].speed_mph
                }
                shifts.append(shift)
        
        return shifts
    
    def calculate_theoretical_performance(self, scenario_name: str, max_rpm: float = 7500) -> Dict:
        """Calculate theoretical top speeds for a transmission scenario"""
        
        scenario = None
        for s in TRANSMISSION_SCENARIOS:
            if s['name'] == scenario_name:
                scenario = s
                break
        
        if not scenario:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        performance = {
            'scenario_name': scenario_name,
            'transmission_ratios': scenario['transmission_ratios'],
            'final_drive': scenario['final_drive'],
            'weight_lbs': scenario['weight_lbs'],
            'gear_top_speeds_mph': [],
            'gear_top_speeds_ms': []
        }
        
        for gear_num, ratio in enumerate(scenario['transmission_ratios'], 1):
            top_speed_ms = theoretical_speed_at_rpm(
                max_rpm, ratio, scenario['final_drive'], self.tire_circumference
            )
            top_speed_mph = top_speed_ms * 2.237
            
            performance['gear_top_speeds_mph'].append(top_speed_mph)
            performance['gear_top_speeds_ms'].append(top_speed_ms)
        
        return performance

def analyze_lap_gearing(lap_data: Dict, scenario_name: str = 'Current Setup') -> Tuple[List[GearInfo], Dict]:
    """
    Analyze gear usage for a complete lap
    Returns: (gear trace, summary statistics)
    """
    
    # Get scenario configuration
    scenario = None
    for s in TRANSMISSION_SCENARIOS:
        if s['name'] == scenario_name:
            scenario = s
            break
    
    if not scenario:
        raise ValueError(f"Scenario '{scenario_name}' not found")
    
    # Create gear calculator
    calculator = GearCalculator(
        scenario['transmission_ratios'],
        scenario['final_drive']
    )
    
    # Calculate gear trace
    rpm_array = lap_data['rpm']
    speed_array = lap_data.get('speed_mph', lap_data.get('speed_ms', []) * 2.237)
    
    if len(speed_array) == 0:
        raise ValueError("No speed data available in lap data")
    
    print(f"Calculating gear usage for {scenario_name}")
    print(f"  Transmission: {scenario['transmission_ratios']}")
    print(f"  Final drive: {scenario['final_drive']}")
    
    gear_trace = calculator.calculate_gear_trace(rpm_array, speed_array)
    
    # Generate summary
    gear_summary = calculator.get_gear_usage_summary(gear_trace)
    
    # Find shift points
    time_array = lap_data['time']
    shift_points = calculator.find_shift_points(gear_trace, time_array)
    
    # Calculate theoretical performance
    theoretical_perf = calculator.calculate_theoretical_performance(scenario_name)
    
    analysis_summary = {
        'scenario': scenario_name,
        'gear_usage': gear_summary,
        'shift_points': shift_points,
        'theoretical_performance': theoretical_perf,
        'lap_info': lap_data.get('lap_info', {}),
        'total_shifts': len(shift_points)
    }
    
    return gear_trace, analysis_summary

def print_gear_analysis(gear_trace: List[GearInfo], summary: Dict):
    """Print detailed gear analysis results"""
    
    print(f"\nGear Analysis Results - {summary['scenario']}")
    print("=" * 60)
    
    # Gear usage summary
    print("Gear Usage:")
    gear_usage = summary['gear_usage']
    
    if gear_usage:
        for gear in sorted(gear_usage.keys()):
            usage = gear_usage[gear]
            speed_range = usage['speed_range_mph']
            print(f"  Gear {gear}: {usage['usage_percent']:.1f}% "
                  f"({speed_range['min']:.1f} - {speed_range['max']:.1f} mph, "
                  f"avg: {speed_range['avg']:.1f} mph)")
    else:
        print("  No gear usage data available")
    
    # Shift points
    shifts = summary['shift_points']
    print(f"\nShift Points ({len(shifts)} total):")
    
    upshifts = [s for s in shifts if s['type'] == 'upshift']
    downshifts = [s for s in shifts if s['type'] == 'downshift']
    
    if upshifts:
        print("  Upshifts:")
        for shift in upshifts:
            print(f"    {shift['from_gear']}->{shift['to_gear']}: "
                  f"{shift['rpm']:.0f} RPM at {shift['speed_mph']:.1f} mph "
                  f"(t={shift['time']:.1f}s)")
    
    if downshifts:
        print("  Downshifts:")
        for shift in downshifts:
            print(f"    {shift['from_gear']}->{shift['to_gear']}: "
                  f"{shift['rpm']:.0f} RPM at {shift['speed_mph']:.1f} mph "
                  f"(t={shift['time']:.1f}s)")
    
    # Theoretical performance
    perf = summary['theoretical_performance']
    print(f"\nTheoretical Top Speeds at 7000 RPM (Safe Limit):")
    for i, speed in enumerate(perf['gear_top_speeds_mph'], 1):
        print(f"  Gear {i}: {speed:.1f} mph")
    
    # Safety analysis
    gear_trace_rpms = [g.actual_rpm for g in gear_trace if g.gear > 0]
    if gear_trace_rpms:
        max_rpm = max(gear_trace_rpms)
        danger_points = len([rpm for rpm in gear_trace_rpms if rpm > 7000])
        total_points = len(gear_trace_rpms)
        
        print(f"\nSafety Analysis:")
        print(f"  Max RPM: {max_rpm:.0f}")
        if max_rpm > 7000:
            print(f"  ⚠️  EXCEEDED SAFE LIMIT by {max_rpm - 7000:.0f} RPM")
        else:
            print(f"  ✅ Stayed within safe limit (7000 RPM)")
        
        danger_percent = (danger_points / total_points) * 100
        print(f"  Time over 7000 RPM: {danger_percent:.1f}% ({danger_points}/{total_points} points)")
        
        if danger_percent > 5:
            print(f"  🔴 RECOMMENDATION: Consider shorter gearing to reduce RPM")
        elif danger_percent > 1:
            print(f"  🟡 CAUTION: Minimize time over 7000 RPM")
        else:
            print(f"  🟢 RPM usage looks safe")

def debug_gear_calculations(transmission_ratios: List[float], final_drive: float):
    """Print theoretical speed ranges for each gear to validate calculations"""
    
    print(f"Theoretical Gear Speed Ranges (Transmission: {transmission_ratios}, Final: {final_drive})")
    print("=" * 70)
    
    tire_circ = TIRE_CIRCUMFERENCE_METERS
    
    for gear_num, ratio in enumerate(transmission_ratios, 1):
        # Calculate speed range from 1200 to 7200 RPM
        rpm_range = [1200, 2000, 3000, 4000, 5000, 6000, 7000, 7200]
        speeds_mph = []
        
        for rpm in rpm_range:
            speed_ms = theoretical_speed_at_rpm(rpm, ratio, final_drive, tire_circ)
            speed_mph = speed_ms * 2.237
            speeds_mph.append(speed_mph)
        
        print(f"Gear {gear_num} (ratio {ratio:.2f}):")
        print(f"  1200 RPM: {speeds_mph[0]:5.1f} mph")
        print(f"  2000 RPM: {speeds_mph[1]:5.1f} mph") 
        print(f"  3000 RPM: {speeds_mph[2]:5.1f} mph")
        print(f"  4000 RPM: {speeds_mph[3]:5.1f} mph")
        print(f"  5000 RPM: {speeds_mph[4]:5.1f} mph")
        print(f"  6000 RPM: {speeds_mph[5]:5.1f} mph")
        print(f"  7000 RPM: {speeds_mph[6]:5.1f} mph (SAFE LIMIT)")
        print(f"  7200 RPM: {speeds_mph[7]:5.1f} mph (DANGER)")
        print(f"  Realistic range: {speeds_mph[1]:.1f} - {speeds_mph[6]:.1f} mph")
        print()

if __name__ == "__main__":
    # Test the gear calculator
    print("Testing Simplified Physics-Based Gear Calculator")
    print("=" * 60)
    
    # First, show theoretical calculations
    debug_gear_calculations(CURRENT_SETUP['transmission_ratios'], CURRENT_SETUP['final_drive'])
    
    from .data_loader import load_session_data
    from .lap_analyzer import analyze_session_laps
    
    # Load session and get fastest lap
    session_data = load_session_data()
    if not session_data:
        print("Failed to load session data")
        exit(1)
    
    laps, fastest_lap_data = analyze_session_laps(session_data)
    if not fastest_lap_data:
        print("No fastest lap data available")
        exit(1)
    
    # Analyze gearing for fastest lap
    gear_trace, analysis_summary = analyze_lap_gearing(fastest_lap_data, 'Current Setup')
    
    # Print results
    print_gear_analysis(gear_trace, analysis_summary)
    
    # Show sample gear calculations
    print(f"\nSample Gear Calculations:")
    for i in range(0, len(gear_trace), len(gear_trace)//10):
        gear_info = gear_trace[i]
        if gear_info.gear > 0:
            print(f"  t={fastest_lap_data['time'][i]:5.1f}s: "
                  f"Gear {gear_info.gear} "
                  f"({gear_info.actual_rpm:.0f} RPM, {gear_info.speed_mph:.1f} mph, "
                  f"conf: {gear_info.confidence:.2f})")
