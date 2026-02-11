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

from src.config.config import DLL_PATH, DEPENDENCY_PATH, SAMPLE_FILES_PATH
from src.config.vehicles import DEFAULT_SESSION
from src.utils.dataframe_helpers import SPEED_MS_TO_MPH


class XRKDataLoader:
    """
    Loader for AIM XRK telemetry files via AIM DLL.
    Provides methods to open files, extract channels, and close cleanly.
    """

    def __init__(self):
        self.dll = None
        self.file_index: Optional[int] = None

    # ---------------- DLL Setup ---------------- #

    def _setup_dll(self) -> bool:
        """Load AIM DLL and configure function signatures"""
        try:
            if not DLL_PATH.exists():
                print(f"⚠️ DLL not found at: {DLL_PATH}")
                return False

            # Ensure dependency path is on system PATH
            dep_path = str(DEPENDENCY_PATH)
            if dep_path not in os.environ["PATH"]:
                os.environ["PATH"] = dep_path + os.pathsep + os.environ["PATH"]

            self.dll = ctypes.WinDLL(str(DLL_PATH))
            self._configure_dll_functions()
            return True

        except Exception as e:
            print(f"❌ Failed to load DLL: {e}")
            return False

    def _configure_dll_functions(self):
        """Define DLL function signatures"""

        # File operations
        self.dll.open_file.argtypes = [c_char_p]
        self.dll.open_file.restype = c_int

        self.dll.close_file_i.argtypes = [c_int]
        self.dll.close_file_i.restype = c_int

        # Regular channel operations
        self.dll.get_channels_count.argtypes = [c_int]
        self.dll.get_channels_count.restype = c_int

        self.dll.get_channel_name.argtypes = [c_int, c_int]
        self.dll.get_channel_name.restype = c_char_p

        self.dll.get_channel_samples_count.argtypes = [c_int, c_int]
        self.dll.get_channel_samples_count.restype = c_int

        self.dll.get_channel_samples.argtypes = [
            c_int,
            c_int,
            POINTER(c_double),
            POINTER(c_double),
            c_int,
        ]
        self.dll.get_channel_samples.restype = c_int

        # GPS channel operations
        self.dll.get_GPS_channels_count.argtypes = [c_int]
        self.dll.get_GPS_channels_count.restype = c_int

        self.dll.get_GPS_channel_name.argtypes = [c_int, c_int]
        self.dll.get_GPS_channel_name.restype = c_char_p

        self.dll.get_GPS_channel_samples_count.argtypes = [c_int, c_int]
        self.dll.get_GPS_channel_samples_count.restype = c_int

        self.dll.get_GPS_channel_samples.argtypes = [
            c_int,
            c_int,
            POINTER(c_double),
            POINTER(c_double),
            c_int,
        ]
        self.dll.get_GPS_channel_samples.restype = c_int

    # ---------------- File Handling ---------------- #

    def open_file(self, filename: str) -> bool:
        """Open XRK file for reading.
        Accepts either a plain filename (resolved in SAMPLE_FILES_PATH)
        or a full/relative path to the XRK file.
        """
        if not self._setup_dll():
            return False

        fn = Path(filename)

        # Case 1: explicit path (absolute or relative) → use directly
        if fn.exists():
            xrk_path = fn.resolve()
        else:
            # Case 2: assume it's just a bare filename, prepend SAMPLE_FILES_PATH
            xrk_path = (SAMPLE_FILES_PATH / fn).resolve()

        if not xrk_path.exists():
            print(f"❌ File not found: {xrk_path}")
            return False

        # Open with DLL
        self.file_index = self.dll.open_file(str(xrk_path).encode("utf-8"))
        if self.file_index <= 0:
            print(f"❌ Failed to open file: {xrk_path}")
            return False

        # Store path for later (e.g. metadata)
        self.filepath = xrk_path
        print(f"✅ Opened XRK file: {xrk_path.name}")
        return True



    def close_file(self):
        """Close the currently open file"""
        if self.dll and self.file_index is not None:
            self.dll.close_file_i(self.file_index)
            self.file_index = None
            print("📁 File closed")

    # ---------------- Data Extraction ---------------- #

    def _extract_channel_data(self, channel_index: int, is_gps: bool = False) -> Optional[Dict]:
        """Extract data from a channel into numpy arrays"""
        try:
            if is_gps:
                sample_count = self.dll.get_GPS_channel_samples_count(self.file_index, channel_index)
                get_samples = self.dll.get_GPS_channel_samples
            else:
                sample_count = self.dll.get_channel_samples_count(self.file_index, channel_index)
                get_samples = self.dll.get_channel_samples

            if sample_count <= 0:
                return None

            times_array = (c_double * sample_count)()
            values_array = (c_double * sample_count)()

            result = get_samples(self.file_index, channel_index, times_array, values_array, sample_count)
            if result <= 0:
                return None

            times = np.array([times_array[i] for i in range(result)])
            values = np.array([values_array[i] for i in range(result)])

            # Normalize time to seconds
            # GPS channels are documented as milliseconds
            # Regular channels: auto-detect based on magnitude
            if is_gps:
                times = times / 1000.0  # ms → seconds
            elif len(times) > 0 and times[-1] > 1000:
                # If max time > 1000, assume milliseconds (a 16-min session would be ~1000s)
                times = times / 1000.0

            return {"time": times, "values": values, "sample_count": result}

        except Exception as e:
            print(f"❌ Error extracting channel {channel_index}: {e}")
            return None

    def extract_rpm_data(self) -> Optional[Dict]:
        """Extract the RPM channel data"""
        if not self.dll or self.file_index is None:
            return None

        try:
            channel_count = self.dll.get_channels_count(self.file_index)
            for i in range(channel_count):
                name_ptr = self.dll.get_channel_name(self.file_index, i)
                if not name_ptr:
                    continue
                name = name_ptr.decode("utf-8")
                if name == "RPM dup 3":  # The working RPM channel
                    return self._extract_channel_data(i, is_gps=False)

            print("⚠️ RPM channel 'RPM dup 3' not found")
            return None

        except Exception as e:
            print(f"❌ Error extracting RPM data: {e}")
            return None

    def extract_gps_data(self) -> Dict[str, Dict]:
        """Extract GPS channels (lat, lon, speed)"""
        if not self.dll or self.file_index is None:
            return {}

        gps_data: Dict[str, Dict] = {}
        channels = ["GPS Latitude", "GPS Longitude", "GPS Speed"]

        try:
            gps_count = self.dll.get_GPS_channels_count(self.file_index)
            for i in range(gps_count):
                name_ptr = self.dll.get_GPS_channel_name(self.file_index, i)
                if not name_ptr:
                    continue
                name = name_ptr.decode("utf-8")
                if name in channels:
                    channel_data = self._extract_channel_data(i, is_gps=True)
                    if channel_data:
                        gps_data[name] = channel_data

            return gps_data

        except Exception as e:
            print(f"❌ Error extracting GPS data: {e}")
            return {}




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
            session_data['speed_mph'] = gps_data['GPS Speed']['values'] * SPEED_MS_TO_MPH
        
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
    print("🔍 Smoke test: XRKDataLoader")
    session = load_session_data()
    if session:
        print("✅ Loaded session:", session['filename'])
        print(f"  Duration: {session['session_duration']:.1f} seconds")
        print(f"  Samples: {session['sample_count']}")
        print(f"  RPM range: {session['rpm'].min():.0f} – {session['rpm'].max():.0f}")
        if 'speed_mph' in session:
            print(f"  Max speed: {session['speed_mph'].max():.1f} mph")
    else:
        print("❌ Failed to load default session")
