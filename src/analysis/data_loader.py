"""
Data loader module for XRK file extraction
Handles all XRK file reading and data synchronization
"""

import ctypes
from ctypes import c_char_p, c_int, c_double, POINTER
import numpy as np
import os
from typing import Dict, Optional, Tuple
from pathlib import Path

from ..config.config_analysis import DLL_PATH, DEPENDENCY_PATH, SAMPLE_FILES_PATH, DEFAULT_SESSION

class XRKDataLoader:
    """Handler for loading and synchronizing XRK telemetry data"""
    
    def __init__(self):
        self.dll = None
        self.file_index = None
        
    def _setup_dll(self):
        """Load and configure the AIM DLL"""
        try:
            # Add dependency path for DLL loading
            if str(DEPENDENCY_PATH) not in os.environ['PATH']:
                os.environ['PATH'] = str(DEPENDENCY_PATH) + os.pathsep + os.environ['PATH']
            
            # Load DLL
            self.dll = ctypes.WinDLL(str(DLL_PATH))
            
            # Configure function signatures
            self._configure_dll_functions()
            
            return True
            
        except Exception as e:
            print(f"Failed to load DLL: {e}")
            return False
    
    def _configure_dll_functions(self):
        """Set up all DLL function signatures"""
        # File operations
        self.dll.open_file.argtypes = [c_char_p]
        self.dll.open_file.restype = c_int
        
        self.dll.close_file_i.argtypes = [c_int]
        self.dll.close_file_i.restype = c_int
        
        # Regular channel functions
        self.dll.get_channels_count.argtypes = [c_int]
        self.dll.get_channels_count.restype = c_int
        
        self.dll.get_channel_name.argtypes = [c_int, c_int]
        self.dll.get_channel_name.restype = c_char_p
        
        self.dll.get_channel_samples_count.argtypes = [c_int, c_int]
        self.dll.get_channel_samples_count.restype = c_int
        
        self.dll.get_channel_samples.argtypes = [c_int, c_int, POINTER(c_double), POINTER(c_double), c_int]
        self.dll.get_channel_samples.restype = c_int
        
        # GPS channel functions
        self.dll.get_GPS_channels_count.argtypes = [c_int]
        self.dll.get_GPS_channels_count.restype = c_int
        
        self.dll.get_GPS_channel_name.argtypes = [c_int, c_int]
        self.dll.get_GPS_channel_name.restype = c_char_p
        
        self.dll.get_GPS_channel_samples_count.argtypes = [c_int, c_int]
        self.dll.get_GPS_channel_samples_count.restype = c_int
        
        self.dll.get_GPS_channel_samples.argtypes = [c_int, c_int, POINTER(c_double), POINTER(c_double), c_int]
        self.dll.get_GPS_channel_samples.restype = c_int
    
    def open_file(self, filename: str) -> bool:
        """Open XRK file for reading"""
        if not self._setup_dll():
            return False
        
        # Build full path
        if not Path(filename).is_absolute():
            xrk_path = SAMPLE_FILES_PATH / filename
        else:
            xrk_path = Path(filename)
        
        if not xrk_path.exists():
            print(f"File not found: {xrk_path}")
            return False
        
        # Open file
        self.file_index = self.dll.open_file(str(xrk_path).encode('utf-8'))
        
        if self.file_index <= 0:
            print(f"Failed to open file: {xrk_path}")
            return False
        
        print(f"Successfully opened: {xrk_path.name}")
        return True
    
    def close_file(self):
        """Close the currently open file"""
        if self.dll and self.file_index:
            self.dll.close_file_i(self.file_index)
            self.file_index = None
    
    def extract_rpm_data(self) -> Optional[Dict]:
        """Extract RPM data from the file"""
        if not self.dll or not self.file_index:
            return None
        
        try:
            regular_count = self.dll.get_channels_count(self.file_index)
            
            for i in range(regular_count):
                name_ptr = self.dll.get_channel_name(self.file_index, i)
                if name_ptr:
                    name = name_ptr.decode('utf-8')
                    if name == 'RPM dup 3':  # The working RPM channel
                        return self._extract_channel_data(i, is_gps=False)
            
            print("RPM dup 3 channel not found")
            return None
            
        except Exception as e:
            print(f"Error extracting RPM data: {e}")
            return None
    
    def extract_gps_data(self) -> Dict[str, Dict]:
        """Extract GPS coordinate and speed data"""
        if not self.dll or not self.file_index:
            return {}
        
        gps_data = {}
        channels_to_extract = ['GPS Latitude', 'GPS Longitude', 'GPS Speed']
        
        try:
            gps_count = self.dll.get_GPS_channels_count(self.file_index)
            
            for i in range(gps_count):
                name_ptr = self.dll.get_GPS_channel_name(self.file_index, i)
                if name_ptr:
                    name = name_ptr.decode('utf-8')
                    if name in channels_to_extract:
                        channel_data = self._extract_channel_data(i, is_gps=True)
                        if channel_data:
                            gps_data[name] = channel_data
            
            return gps_data
            
        except Exception as e:
            print(f"Error extracting GPS data: {e}")
            return {}
    
    def _extract_channel_data(self, channel_index: int, is_gps: bool = False) -> Optional[Dict]:
        """Extract data from a specific channel"""
        try:
            # Get sample count and allocate arrays
            if is_gps:
                sample_count = self.dll.get_GPS_channel_samples_count(self.file_index, channel_index)
                get_samples_func = self.dll.get_GPS_channel_samples
            else:
                sample_count = self.dll.get_channel_samples_count(self.file_index, channel_index)
                get_samples_func = self.dll.get_channel_samples
            
            if sample_count <= 0:
                return None
            
            # Allocate arrays
            times_array = (c_double * sample_count)()
            values_array = (c_double * sample_count)()
            
            # Extract data
            result = get_samples_func(self.file_index, channel_index, times_array, values_array, sample_count)
            
            if result <= 0:
                return None
            
            # Convert to numpy arrays
            times = np.array([times_array[j] for j in range(result)])
            values = np.array([values_array[j] for j in range(result)])
            
            # Convert GPS time from milliseconds to seconds
            if is_gps:
                times = times / 1000.0
            
            return {
                'time': times,
                'values': values,
                'sample_count': result
            }
            
        except Exception as e:
            print(f"Error extracting channel {channel_index}: {e}")
            return None

def load_session_data(filename: str = DEFAULT_SESSION) -> Optional[Dict]:
    """
    Load complete session data from XRK file
    Returns synchronized RPM and GPS data
    """
    loader = XRKDataLoader()
    
    try:
        # Open file
        if not loader.open_file(filename):
            return None
        
        print(f"Loading session data from: {filename}")
        
        # Extract data
        rpm_data = loader.extract_rpm_data()
        gps_data = loader.extract_gps_data()
        
        if not rpm_data:
            print("Failed to extract RPM data")
            return None
        
        if 'GPS Latitude' not in gps_data or 'GPS Longitude' not in gps_data:
            print("Failed to extract GPS coordinate data")
            return None
        
        # Print data summary
        print(f"Data extracted successfully:")
        print(f"  RPM: {rpm_data['sample_count']} samples, {rpm_data['time'][0]:.1f} to {rpm_data['time'][-1]:.1f} seconds")
        
        for name, data in gps_data.items():
            print(f"  {name}: {data['sample_count']} samples, {data['time'][0]:.1f} to {data['time'][-1]:.1f} seconds")
        
        # Synchronize data by interpolating RPM to GPS time base
        gps_lat = gps_data['GPS Latitude']
        gps_lon = gps_data['GPS Longitude']
        
        # Interpolate RPM to GPS time points
        rpm_interp = np.interp(gps_lat['time'], rpm_data['time'], rpm_data['values'])
        
        # Build synchronized dataset
        session_data = {
            'time': gps_lat['time'],
            'latitude': gps_lat['values'],
            'longitude': gps_lon['values'],
            'rpm': rpm_interp,
            'session_duration': gps_lat['time'][-1] - gps_lat['time'][0],
            'sample_count': len(gps_lat['time']),
            'filename': filename
        }
        
        # Add GPS speed if available
        if 'GPS Speed' in gps_data:
            session_data['speed_ms'] = gps_data['GPS Speed']['values']
            session_data['speed_mph'] = gps_data['GPS Speed']['values'] * 2.237
        
        print(f"Synchronized dataset: {session_data['sample_count']} points, {session_data['session_duration']:.1f} second session")
        print(f"RPM range: {session_data['rpm'].min():.0f} - {session_data['rpm'].max():.0f}")
        
        return session_data
        
    except Exception as e:
        print(f"Error loading session data: {e}")
        return None
        
    finally:
        loader.close_file()

def get_data_summary(session_data: Dict) -> Dict:
    """Generate summary statistics for loaded session data"""
    if not session_data:
        return {}
    
    summary = {
        'filename': session_data.get('filename', 'Unknown'),
        'duration_minutes': session_data['session_duration'] / 60,
        'sample_count': session_data['sample_count'],
        'rpm_stats': {
            'min': float(session_data['rpm'].min()),
            'max': float(session_data['rpm'].max()),
            'mean': float(session_data['rpm'].mean()),
            'std': float(session_data['rpm'].std())
        }
    }
    
    if 'speed_mph' in session_data:
        summary['speed_stats'] = {
            'max_mph': float(session_data['speed_mph'].max()),
            'mean_mph': float(session_data['speed_mph'].mean())
        }
    
    # GPS coordinate ranges
    summary['track_bounds'] = {
        'lat_min': float(session_data['latitude'].min()),
        'lat_max': float(session_data['latitude'].max()),
        'lon_min': float(session_data['longitude'].min()),
        'lon_max': float(session_data['longitude'].max())
    }
    
    return summary

if __name__ == "__main__":
    # Test the data loader
    print("Testing XRK Data Loader")
    print("=" * 50)
    
    # Load default session
    session_data = load_session_data()
    
    if session_data:
        print("\nData loading successful!")
        
        # Print summary
        summary = get_data_summary(session_data)
        print(f"\nSession Summary:")
        print(f"  File: {summary['filename']}")
        print(f"  Duration: {summary['duration_minutes']:.1f} minutes")
        print(f"  Samples: {summary['sample_count']:,}")
        print(f"  RPM Range: {summary['rpm_stats']['min']:.0f} - {summary['rpm_stats']['max']:.0f}")
        
        if 'speed_stats' in summary:
            print(f"  Max Speed: {summary['speed_stats']['max_mph']:.1f} mph")
        
        print(f"  Track Bounds:")
        print(f"    Lat: {summary['track_bounds']['lat_min']:.6f} to {summary['track_bounds']['lat_max']:.6f}")
        print(f"    Lon: {summary['track_bounds']['lon_min']:.6f} to {summary['track_bounds']['lon_max']:.6f}")
        
    else:
        print("Data loading failed!")