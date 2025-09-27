"""
Lap analyzer module for splitting session data into individual laps
and identifying the fastest lap
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..config.config_analysis import PROCESSING_CONFIG, TRACK_CONFIG

@dataclass
class LapInfo:
    """Information about a single lap"""
    lap_number: int
    start_index: int
    end_index: int
    start_time: float
    end_time: float
    lap_time: float
    max_speed_mph: float
    max_rpm: float
    avg_rpm: float
    sample_count: int

class LapAnalyzer:
    """Analyzer for detecting and splitting laps from session data"""
    
    def __init__(self, session_data: Dict):
        self.session_data = session_data
        self.laps = []
        self.fastest_lap = None
        
    def detect_laps(self) -> List[LapInfo]:
        """
        Detect individual laps by finding start/finish line crossings
        Uses specified GPS coordinates for Road America start/finish line
        """
        lat = self.session_data['latitude']
        lon = self.session_data['longitude']
        time = self.session_data['time']
        
        # Use actual Road America start/finish line coordinates
        from ..config.config_analysis import TRACK_CONFIG
        start_lat, start_lon = TRACK_CONFIG['start_finish_gps']
        
        print(f"Detecting laps using Road America start/finish at: {start_lat:.6f}, {start_lon:.6f}")
        
        # Debug GPS detection
        self.debug_lap_detection_fixed(start_lat, start_lon)
        
        # Calculate distance from start/finish line
        distances = np.sqrt((lat - start_lat)**2 + (lon - start_lon)**2)
        
        # Find crossings (close approaches to start/finish)
        threshold = PROCESSING_CONFIG['start_finish_threshold']
        crossings = self._find_crossings(distances, threshold, time)
        
        if len(crossings) < 2:
            print(f"Only {len(crossings)} crossings found, using time-based splitting")
            return self._split_by_time()
        
        # Build laps from crossings
        laps = []
        for i in range(len(crossings) - 1):
            lap_info = self._create_lap_info(i + 1, crossings[i], crossings[i + 1])
            if lap_info and self._is_valid_lap(lap_info):
                laps.append(lap_info)
        
        print(f"Detected {len(laps)} valid laps")
        self.laps = laps
        return laps
    
    def debug_lap_detection_fixed(self, start_lat: float, start_lon: float) -> None:
        """Print detailed debugging info for lap detection with fixed coordinates"""
        lat = self.session_data['latitude']
        lon = self.session_data['longitude']
        time = self.session_data['time']
        
        # Calculate distances to fixed start/finish
        distances = np.sqrt((lat - start_lat)**2 + (lon - start_lon)**2)
        
        print(f"GPS Lap Detection Debug (Fixed Coordinates):")
        print(f"  Start/Finish: {start_lat:.6f}, {start_lon:.6f}")
        print(f"  Distance range: {distances.min():.6f} - {distances.max():.6f}")
        print(f"  Current threshold: {PROCESSING_CONFIG['start_finish_threshold']:.6f}")
        
        # Find and show the closest approaches
        min_distance = distances.min()
        close_threshold = min_distance + 0.002  # Within ~200m of closest approach
        close_indices = np.where(distances <= close_threshold)[0]
        
        print(f"  Closest approaches to start/finish (within {close_threshold:.6f}):")
        
        # Group close approaches into separate passages
        lap_crossings = []
        if len(close_indices) > 0:
            current_group = [close_indices[0]]
            
            for i in range(1, len(close_indices)):
                time_gap = time[close_indices[i]] - time[close_indices[i-1]]
                if time_gap > 30:  # More than 30 seconds gap = new lap
                    # Show the closest point from current group
                    group_distances = [distances[idx] for idx in current_group]
                    closest_idx = current_group[np.argmin(group_distances)]
                    lap_crossings.append(closest_idx)
                    
                    current_group = [close_indices[i]]
                else:
                    current_group.append(close_indices[i])
            
            # Add the last group
            if current_group:
                group_distances = [distances[idx] for idx in current_group]
                closest_idx = current_group[np.argmin(group_distances)]
                lap_crossings.append(closest_idx)
        
        print(f"  Found {len(lap_crossings)} potential lap crossings:")
        for i, idx in enumerate(lap_crossings):
            print(f"    Lap {i+1}: Time {time[idx]:6.1f}s, Distance: {distances[idx]:.6f}, GPS: {lat[idx]:.6f}, {lon[idx]:.6f}")
        
        # Show time gaps between crossings
        if len(lap_crossings) > 1:
            time_gaps = [time[lap_crossings[i+1]] - time[lap_crossings[i]] for i in range(len(lap_crossings)-1)]
            print(f"  Lap times: {[f'{gap:.1f}s' for gap in time_gaps]}")
    
    def _find_crossings(self, distances: np.ndarray, threshold: float, time: np.ndarray) -> List[int]:
        """Find indices where car crosses start/finish line"""
        crossings = [0]  # Always start with first point
        
        # Look for close approaches to start/finish
        min_separation = 30  # Minimum seconds between crossings
        
        for i in range(100, len(distances) - 100):  # Skip beginning and end
            if distances[i] < threshold:
                # Check if this is a genuine crossing (was far away before)
                if (distances[i-50] > threshold * 3 and 
                    len(crossings) == 0 or 
                    time[i] - time[crossings[-1]] > min_separation):
                    crossings.append(i)
        
        # Always end with last point if we have crossings
        if len(crossings) > 1 and crossings[-1] != len(distances) - 1:
            crossings.append(len(distances) - 1)
        
        print(f"Found {len(crossings)} start/finish crossings")
        return crossings
    
    def _split_by_time(self) -> List[LapInfo]:
        """Fallback: split session into equal time segments if lap detection fails"""
        time = self.session_data['time']
        duration = time[-1] - time[0]
        
        # Estimate lap count based on session duration (assume ~2.5 min laps for Road America)
        estimated_lap_time = 150  # seconds
        estimated_laps = max(1, int(duration / estimated_lap_time))
        
        print(f"Splitting {duration:.1f}s session into {estimated_laps} estimated laps")
        
        laps = []
        segment_size = len(time) // estimated_laps
        
        for i in range(estimated_laps):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < estimated_laps - 1 else len(time) - 1
            
            lap_info = self._create_lap_info(i + 1, start_idx, end_idx)
            if lap_info:
                laps.append(lap_info)
        
        self.laps = laps
        return laps
    
    def _create_lap_info(self, lap_number: int, start_idx: int, end_idx: int) -> Optional[LapInfo]:
        """Create LapInfo object from indices"""
        try:
            time = self.session_data['time']
            rpm = self.session_data['rpm']
            
            lap_time = time[end_idx] - time[start_idx]
            
            # Get lap data slice
            lap_rpm = rpm[start_idx:end_idx+1]
            lap_speed = self.session_data.get('speed_mph', np.zeros_like(rpm))[start_idx:end_idx+1]
            
            return LapInfo(
                lap_number=lap_number,
                start_index=start_idx,
                end_index=end_idx,
                start_time=time[start_idx],
                end_time=time[end_idx],
                lap_time=lap_time,
                max_speed_mph=float(lap_speed.max()) if len(lap_speed) > 0 else 0,
                max_rpm=float(lap_rpm.max()) if len(lap_rpm) > 0 else 0,
                avg_rpm=float(lap_rpm.mean()) if len(lap_rpm) > 0 else 0,
                sample_count=end_idx - start_idx + 1
            )
            
        except Exception as e:
            print(f"Error creating lap info for lap {lap_number}: {e}")
            return None
    
    def _is_valid_lap(self, lap_info: LapInfo) -> bool:
        """Check if lap has reasonable characteristics"""
        min_time = PROCESSING_CONFIG['min_lap_time_seconds']
        max_time = PROCESSING_CONFIG['max_lap_time_seconds']
        
        if lap_info.lap_time < min_time or lap_info.lap_time > max_time:
            print(f"Lap {lap_info.lap_number}: {lap_info.lap_time:.1f}s - outside valid range ({min_time}-{max_time}s)")
            return False
        
        if lap_info.sample_count < 50:  # Minimum data points
            print(f"Lap {lap_info.lap_number}: Only {lap_info.sample_count} samples")
            return False
        
        return True
    
    def find_fastest_lap(self) -> Optional[LapInfo]:
        """Find the lap with the shortest lap time"""
        if not self.laps:
            return None
        
        fastest = min(self.laps, key=lambda lap: lap.lap_time)
        self.fastest_lap = fastest
        
        print(f"Fastest lap: #{fastest.lap_number} - {fastest.lap_time:.2f}s")
        return fastest
    
    def get_lap_data(self, lap_info: LapInfo) -> Dict:
        """Extract data for a specific lap"""
        start_idx = lap_info.start_index
        end_idx = lap_info.end_index
        
        # Extract all data for this lap
        lap_data = {
            'lap_info': lap_info,
            'time': self.session_data['time'][start_idx:end_idx+1] - self.session_data['time'][start_idx],  # Start from 0
            'latitude': self.session_data['latitude'][start_idx:end_idx+1],
            'longitude': self.session_data['longitude'][start_idx:end_idx+1],
            'rpm': self.session_data['rpm'][start_idx:end_idx+1],
        }
        
        # Add speed data if available
        if 'speed_mph' in self.session_data:
            lap_data['speed_mph'] = self.session_data['speed_mph'][start_idx:end_idx+1]
            lap_data['speed_ms'] = self.session_data['speed_ms'][start_idx:end_idx+1]
        
        return lap_data
    
    def get_fastest_lap_data(self) -> Optional[Dict]:
        """Get data for the fastest lap"""
        if not self.fastest_lap:
            self.find_fastest_lap()
        
        if not self.fastest_lap:
            return None
        
        return self.get_lap_data(self.fastest_lap)
    
    def print_lap_summary(self):
        """Print summary of all detected laps"""
        if not self.laps:
            print("No laps detected")
            return
        
        print(f"\nLap Analysis Summary:")
        print(f"{'Lap':<4} {'Time':<8} {'Max Speed':<10} {'Max RPM':<8} {'Avg RPM':<8} {'Samples':<8}")
        print("-" * 55)
        
        for lap in self.laps:
            print(f"{lap.lap_number:<4} {lap.lap_time:<8.2f} {lap.max_speed_mph:<10.1f} "
                  f"{lap.max_rpm:<8.0f} {lap.avg_rpm:<8.0f} {lap.sample_count:<8}")
        
        if self.fastest_lap:
            print(f"\nFastest Lap: #{self.fastest_lap.lap_number} ({self.fastest_lap.lap_time:.2f}s)")

def analyze_session_laps(session_data: Dict) -> Tuple[List[LapInfo], Optional[Dict]]:
    """
    Analyze session data to extract laps and find fastest lap
    Returns: (list of all laps, fastest lap data)
    """
    analyzer = LapAnalyzer(session_data)
    
    # Detect laps
    laps = analyzer.detect_laps()
    
    if not laps:
        print("No valid laps found in session")
        return [], None
    
    # Print summary
    analyzer.print_lap_summary()
    
    # Get fastest lap data
    fastest_lap_data = analyzer.get_fastest_lap_data()
    
    return laps, fastest_lap_data

if __name__ == "__main__":
    # Test the lap analyzer
    print("Testing Lap Analyzer")
    print("=" * 50)
    
    from .data_loader import load_session_data
    
    # Load session data
    session_data = load_session_data()
    
    if not session_data:
        print("Failed to load session data")
        exit(1)
    
    # Analyze laps
    laps, fastest_lap_data = analyze_session_laps(session_data)
    
    if fastest_lap_data:
        lap_info = fastest_lap_data['lap_info']
        print(f"\nFastest Lap Details:")
        print(f"  Lap #{lap_info.lap_number}")
        print(f"  Time: {lap_info.lap_time:.2f} seconds")
        print(f"  Max Speed: {lap_info.max_speed_mph:.1f} mph")
        print(f"  Max RPM: {lap_info.max_rpm:.0f}")
        print(f"  Data Points: {lap_info.sample_count}")
        print(f"  Time Range: {lap_info.start_time:.1f} - {lap_info.end_time:.1f} seconds")
        
        # Show RPM distribution for fastest lap
        lap_rpm = fastest_lap_data['rpm']
        print(f"  RPM Distribution:")
        print(f"    Min: {lap_rpm.min():.0f}")
        print(f"    Max: {lap_rpm.max():.0f}")
        print(f"    Avg: {lap_rpm.mean():.0f}")
    else:
        print("No fastest lap data available")
