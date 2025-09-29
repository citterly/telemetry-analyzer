"""
XRK Metadata Analyzer - Enhanced Version
Comprehensive analysis of XRK file format structure and metadata
Uses both DLL API and direct binary file analysis for complete metadata extraction
"""

import ctypes
from ctypes import c_char_p, c_int, c_double, POINTER
import numpy as np
import os
import json
import struct
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
import hashlib
from src.io.dll_interface import AIMDLL

from analysis.config import DLL_PATH, DEPENDENCY_PATH, SAMPLE_FILES_PATH, UNITS_XML_PATH

dll = AIMDLL()
if dll.setup():
    file_idx = dll.open("data/uploads/temp_20250713_093209_Road America_a_0410_1.xrk")
    channels = dll.get_channels(file_idx)
    print("Channels:", channels[:5])  # quick test
    dll.close(file_idx)

@dataclass
class ChannelMetadata:
    """Metadata for a single telemetry channel"""
    index: int
    name: str
    channel_type: str  # 'regular', 'gps', 'math', 'lap', etc.
    sample_count: int
    sample_rate: Optional[float] = None
    time_range: Optional[tuple] = None
    value_range: Optional[tuple] = None
    data_type: Optional[str] = None
    units: Optional[str] = None
    encoding: Optional[str] = None
    bit_depth: Optional[int] = None
    compression: Optional[str] = None
    
    # Additional metadata
    raw_offset: Optional[int] = None  # Byte offset in file
    raw_size: Optional[int] = None    # Size in bytes
    checksum: Optional[str] = None


@dataclass
class LoggerMetadata:
    """Data logger hardware/firmware metadata"""
    manufacturer: str = "AIM"
    model: Optional[str] = None
    serial_number: Optional[str] = None
    firmware_version: Optional[str] = None
    hardware_version: Optional[str] = None
    calibration_date: Optional[str] = None
    
    # Logger capabilities
    max_channels: Optional[int] = None
    max_sample_rate: Optional[int] = None
    memory_size_mb: Optional[int] = None
    gps_enabled: bool = False
    can_bus_enabled: bool = False
    analog_inputs: Optional[int] = None
    digital_inputs: Optional[int] = None


@dataclass
class SessionMetadata:
    """Session-specific metadata"""
    session_id: Optional[str] = None
    session_name: Optional[str] = None
    session_number: Optional[int] = None
    session_type: Optional[str] = None  # Practice, Qualifying, Race, Test
    
    date: Optional[str] = None
    time: Optional[str] = None
    timezone: Optional[str] = None
    
    track_name: Optional[str] = None
    track_length_meters: Optional[float] = None
    track_configuration: Optional[str] = None
    
    weather_conditions: Optional[str] = None
    ambient_temp: Optional[float] = None
    track_temp: Optional[float] = None
    
    duration_seconds: Optional[float] = None
    laps_completed: Optional[int] = None
    distance_covered_km: Optional[float] = None


@dataclass 
class VehicleMetadata:
    """Vehicle and setup metadata"""
    vehicle_id: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    vin: Optional[str] = None
    
    driver_name: Optional[str] = None
    driver_id: Optional[str] = None
    team_name: Optional[str] = None
    
    # Setup information
    setup_name: Optional[str] = None
    setup_version: Optional[int] = None
    fuel_capacity_liters: Optional[float] = None
    weight_kg: Optional[float] = None
    
    # Configuration notes
    notes: Optional[str] = None


@dataclass
class FileFormatMetadata:
    """File format and structure metadata"""
    format_signature: Optional[str] = None
    format_version: Optional[str] = None
    file_structure_type: Optional[str] = None  # Binary, XML hybrid, etc.
    
    header_offset: int = 0
    header_size: int = 0
    data_offset: int = 0
    data_size: int = 0
    footer_offset: Optional[int] = None
    footer_size: Optional[int] = None
    
    endianness: Optional[str] = None  # 'little' or 'big'
    encoding: Optional[str] = None
    compression: Optional[str] = None
    encryption: Optional[str] = None
    
    # File integrity
    checksum_type: Optional[str] = None
    checksum_value: Optional[str] = None
    
    # Raw binary sections found
    binary_sections: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class XRKFileMetadata:
    """Complete metadata extracted from XRK file"""
    # File information
    filename: str
    file_path: str
    file_size_bytes: int
    file_hash_md5: Optional[str] = None
    file_hash_sha256: Optional[str] = None
    
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    
    # Component metadata
    file_format: Optional[FileFormatMetadata] = None
    logger: Optional[LoggerMetadata] = None
    session: Optional[SessionMetadata] = None
    vehicle: Optional[VehicleMetadata] = None
    
    # Channel information
    regular_channels: List[ChannelMetadata] = field(default_factory=list)
    gps_channels: List[ChannelMetadata] = field(default_factory=list)
    calculated_channels: List[ChannelMetadata] = field(default_factory=list)
    
    # Channel summary
    total_channels: int = 0
    total_sample_count: int = 0
    unique_sample_rates: List[float] = field(default_factory=list)
    
    # Data quality metrics
    data_quality: Dict[str, Any] = field(default_factory=dict)
    
    # Raw analysis results
    raw_file_analysis: Dict[str, Any] = field(default_factory=dict)


class XRKBinaryAnalyzer:
    """Direct binary analysis of XRK files"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.file_size = self.file_path.stat().st_size
        self.binary_data = None
        
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive binary analysis"""
        with open(self.file_path, 'rb') as f:
            self.binary_data = f.read()
        
        analysis = {
            'file_structure': self._analyze_structure(),
            'header_analysis': self._analyze_header(),
            'data_blocks': self._find_data_blocks(),
            'strings': self._extract_strings(),
            'patterns': self._find_patterns(),
            'encoding_hints': self._detect_encoding()
        }
        
        return analysis
    
    def _analyze_structure(self) -> Dict[str, Any]:
        """Analyze overall file structure"""
        # Look for common file signatures and patterns
        signatures = {
            'aim_xrk': b'XRK',
            'aim_drk': b'DRK',
            'xml_header': b'<?xml',
            'binary_marker': b'\x00\x00\x00\x00',
        }
        
        structure = {
            'signatures_found': [],
            'likely_format': None,
            'sections': []
        }
        
        for name, sig in signatures.items():
            positions = self._find_all_occurrences(sig)
            if positions:
                structure['signatures_found'].append({
                    'name': name,
                    'signature': sig.hex(),
                    'positions': positions[:10]  # First 10 occurrences
                })
        
        # Analyze sections
        structure['sections'] = self._identify_sections()
        
        return structure
    
    def _analyze_header(self) -> Dict[str, Any]:
        """Analyze file header (typically first 1-4KB)"""
        header_size = min(4096, self.file_size)
        header = self.binary_data[:header_size]
        
        analysis = {
            'size': header_size,
            'magic_bytes': header[:16].hex(),
            'possible_version': None,
            'timestamp_candidates': [],
            'metadata_fields': []
        }
        
        # Look for version strings
        version_patterns = [rb'V\d+\.\d+', rb'v\d+\.\d+', rb'\d+\.\d+\.\d+']
        for pattern in version_patterns:
            import re
            matches = re.findall(pattern, header)
            if matches:
                analysis['possible_version'] = matches[0].decode('utf-8', errors='ignore')
                break
        
        # Look for timestamps (common Unix timestamp range)
        for i in range(0, len(header) - 4, 4):
            value = struct.unpack('<I', header[i:i+4])[0]
            if 1000000000 < value < 2000000000:  # Reasonable Unix timestamp range
                try:
                    dt = datetime.fromtimestamp(value)
                    if 2000 < dt.year < 2030:
                        analysis['timestamp_candidates'].append({
                            'offset': i,
                            'value': value,
                            'datetime': dt.isoformat()
                        })
                except:
                    pass
        
        # Look for structured data
        analysis['metadata_fields'] = self._parse_header_fields(header)
        
        return analysis
    
    def _find_data_blocks(self) -> List[Dict[str, Any]]:
        """Identify major data blocks in the file"""
        blocks = []
        
        # Look for repeating patterns that might indicate data blocks
        # Common block sizes in telemetry data
        common_sizes = [512, 1024, 2048, 4096, 8192]
        
        for block_size in common_sizes:
            if self.file_size > block_size * 10:  # Need at least 10 blocks
                # Sample first few potential blocks
                consistent = True
                block_headers = []
                
                for i in range(5):
                    offset = i * block_size
                    if offset + 16 < self.file_size:
                        header = self.binary_data[offset:offset+16]
                        block_headers.append(header)
                
                # Check if blocks have consistent structure
                if len(set(h[:4] for h in block_headers)) == 1:  # Same first 4 bytes
                    blocks.append({
                        'block_size': block_size,
                        'estimated_count': self.file_size // block_size,
                        'header_pattern': block_headers[0][:4].hex(),
                        'confidence': 'high' if len(set(block_headers)) == 1 else 'medium'
                    })
        
        return blocks
    
    def _extract_strings(self) -> Dict[str, List[str]]:
        """Extract readable strings from binary data"""
        import re
        
        # ASCII strings (minimum length 4)
        ascii_strings = re.findall(b'[\x20-\x7E]{4,}', self.binary_data)
        
        # UTF-16 strings (common in Windows applications)
        utf16_strings = []
        try:
            text = self.binary_data.decode('utf-16-le', errors='ignore')
            utf16_strings = re.findall(r'[\x20-\x7E]{4,}', text)
        except:
            pass
        
        # Categorize strings
        categorized = {
            'channel_names': [],
            'track_names': [],
            'timestamps': [],
            'versions': [],
            'configuration': [],
            'other': []
        }
        
        for s in ascii_strings[:500]:  # Analyze first 500 strings
            s_decoded = s.decode('ascii', errors='ignore')
            
            # Categorize based on content
            if any(kw in s_decoded.lower() for kw in ['rpm', 'speed', 'temp', 'pressure', 'throttle', 'brake']):
                categorized['channel_names'].append(s_decoded)
            elif any(kw in s_decoded.lower() for kw in ['road america', 'laguna', 'watkins', 'circuit']):
                categorized['track_names'].append(s_decoded)
            elif re.match(r'\d{4}[-/]\d{2}[-/]\d{2}', s_decoded):
                categorized['timestamps'].append(s_decoded)
            elif re.match(r'[vV]?\d+\.\d+', s_decoded):
                categorized['versions'].append(s_decoded)
            elif '=' in s_decoded or ':' in s_decoded:
                categorized['configuration'].append(s_decoded)
            else:
                if len(categorized['other']) < 50:  # Limit 'other' category
                    categorized['other'].append(s_decoded)
        
        return categorized
    
    def _find_patterns(self) -> Dict[str, Any]:
        """Find repeating patterns that might indicate data structure"""
        patterns = {
            'float_arrays': [],
            'int_arrays': [],
            'recurring_sequences': []
        }
        
        # Look for arrays of floats (common in telemetry)
        for offset in range(0, min(10000, self.file_size - 1000), 100):
            chunk = self.binary_data[offset:offset+1000]
            
            # Try interpreting as float array
            try:
                floats = struct.unpack('<' + 'f' * (len(chunk) // 4), chunk[:len(chunk) & ~3])
                # Check if values are reasonable (not NaN, not huge)
                if all(-10000 < f < 10000 and not np.isnan(f) for f in floats[:10]):
                    patterns['float_arrays'].append({
                        'offset': offset,
                        'sample_values': floats[:5],
                        'confidence': 'high'
                    })
            except:
                pass
        
        return patterns
    
    def _detect_encoding(self) -> Dict[str, Any]:
        """Detect data encoding and compression"""
        encoding_hints = {
            'likely_compressed': False,
            'likely_encrypted': False,
            'endianness': None,
            'entropy': 0
        }
        
        # Calculate entropy (high entropy suggests compression or encryption)
        byte_counts = np.bincount(np.frombuffer(self.binary_data[:10000], dtype=np.uint8))
        probabilities = byte_counts / byte_counts.sum()
        entropy = -np.sum(p * np.log2(p + 1e-10) for p in probabilities if p > 0)
        encoding_hints['entropy'] = float(entropy)
        
        if entropy > 7.5:
            encoding_hints['likely_compressed'] = True
        if entropy > 7.9:
            encoding_hints['likely_encrypted'] = True
        
        # Check endianness by looking for known values
        # Test with common telemetry values (e.g., RPM around 6000)
        test_value = 6000
        little_endian_bytes = struct.pack('<H', test_value)
        big_endian_bytes = struct.pack('>H', test_value)
        
        if little_endian_bytes in self.binary_data[:10000]:
            encoding_hints['endianness'] = 'little'
        elif big_endian_bytes in self.binary_data[:10000]:
            encoding_hints['endianness'] = 'big'
        
        return encoding_hints
    
    def _find_all_occurrences(self, pattern: bytes) -> List[int]:
        """Find all occurrences of a byte pattern"""
        positions = []
        start = 0
        while True:
            pos = self.binary_data.find(pattern, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
            if len(positions) >= 100:  # Limit to 100 occurrences
                break
        return positions
    
    def _identify_sections(self) -> List[Dict[str, Any]]:
        """Identify major sections in the file"""
        sections = []
        
        # Look for section markers (common patterns)
        markers = [
            b'\xFF\xFF\xFF\xFF',
            b'\x00\x00\x00\x00\x00\x00\x00\x00',
            b'DATA',
            b'HEAD',
            b'FOOT'
        ]
        
        for marker in markers:
            positions = self._find_all_occurrences(marker)
            if positions:
                for i, pos in enumerate(positions[:5]):  # First 5 occurrences
                    # Try to determine section size
                    next_pos = positions[i + 1] if i + 1 < len(positions) else self.file_size
                    sections.append({
                        'marker': marker.hex(),
                        'offset': pos,
                        'estimated_size': next_pos - pos,
                        'type': 'unknown'
                    })
        
        return sorted(sections, key=lambda x: x['offset'])
    
    def _parse_header_fields(self, header: bytes) -> List[Dict[str, Any]]:
        """Parse structured fields from header"""
        fields = []
        
        # Look for key-value pairs (common in metadata)
        # Format: null-terminated strings
        strings = header.split(b'\x00')
        for s in strings:
            if b'=' in s or b':' in s:
                try:
                    decoded = s.decode('utf-8', errors='ignore')
                    if len(decoded) > 3:
                        fields.append({
                            'type': 'key_value',
                            'content': decoded
                        })
                except:
                    pass
        
        return fields


class XRKMetadataExtractor:
    """Enhanced XRK file metadata extractor using both DLL and direct analysis"""
    
    def __init__(self):
        self.dll = None
        self.file_index = None
        
    def _setup_dll(self):
        """Load and configure the AIM DLL"""
        try:
            # Ensure dependency folder is in DLL search path
            if os.name == "nt":
                os.add_dll_directory(str(DEPENDENCY_PATH))
            else:
                if str(DEPENDENCY_PATH) not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = str(DEPENDENCY_PATH) + os.pathsep + os.environ["PATH"]

            # Point AIM SDK to units.xml (from config)
            os.environ["AIM_UNITS_FILE"] = str(UNITS_XML_PATH)

            # Load the DLL
            self.dll = ctypes.WinDLL(str(DLL_PATH))
            self._configure_dll_functions()
            return True

        except Exception as e:
            print(f"Warning: DLL not available ({e}), using binary analysis only")
            return False

    def _setup_dll_old(self):
        """Load and configure the AIM DLL"""
        try:
            if str(DEPENDENCY_PATH) not in os.environ['PATH']:
                os.environ['PATH'] = str(DEPENDENCY_PATH) + os.pathsep + os.environ['PATH']
            
            self.dll = ctypes.WinDLL(str(DLL_PATH))
            self._configure_dll_functions()
            return True
        except Exception as e:
            print(f"Warning: DLL not available ({e}), using binary analysis only")
            return False
    
    def _configure_dll_functions(self):
        """Configure DLL function signatures"""
        # Basic functions
        self.dll.open_file.argtypes = [c_char_p]
        self.dll.open_file.restype = c_int
        
        self.dll.close_file_i.argtypes = [c_int]
        self.dll.close_file_i.restype = c_int
        
        # Channel functions
        self.dll.get_channels_count.argtypes = [c_int]
        self.dll.get_channels_count.restype = c_int
        
        self.dll.get_channel_name.argtypes = [c_int, c_int]
        self.dll.get_channel_name.restype = c_char_p
        
        self.dll.get_channel_samples_count.argtypes = [c_int, c_int]
        self.dll.get_channel_samples_count.restype = c_int
        
        self.dll.get_GPS_channels_count.argtypes = [c_int]
        self.dll.get_GPS_channels_count.restype = c_int
        
        self.dll.get_GPS_channel_name.argtypes = [c_int, c_int]
        self.dll.get_GPS_channel_name.restype = c_char_p
        
        self.dll.get_GPS_channel_samples_count.argtypes = [c_int, c_int]
        self.dll.get_GPS_channel_samples_count.restype = c_int
    
    def analyze_file_comprehensive(self, file_path: str) -> XRKFileMetadata:
        """Perform comprehensive analysis using all available methods"""
        file_path = Path(file_path)
        
        # Initialize metadata
        metadata = XRKFileMetadata(
            filename=file_path.name,
            file_path=str(file_path),
            file_size_bytes=file_path.stat().st_size
        )
        
        # Calculate file hashes
        metadata.file_hash_md5 = self._calculate_hash(file_path, 'md5')
        metadata.file_hash_sha256 = self._calculate_hash(file_path, 'sha256')
        
        # File timestamps
        stat = file_path.stat()
        metadata.creation_date = datetime.fromtimestamp(stat.st_ctime).isoformat()
        metadata.modification_date = datetime.fromtimestamp(stat.st_mtime).isoformat()
        
        # Binary analysis (always available)
        print("Performing binary analysis...")
        binary_analyzer = XRKBinaryAnalyzer(str(file_path))
        metadata.raw_file_analysis = binary_analyzer.analyze()
        
        # Extract metadata from binary analysis
        self._extract_from_binary_analysis(metadata)
        
        # DLL-based analysis (if available)
        if self._setup_dll():
            print("Performing DLL-based analysis...")
            self.file_index = self.dll.open_file(str(file_path).encode('utf-8'))
            if self.file_index > 0:
                try:
                    self._extract_dll_metadata(metadata)
                finally:
                    self.dll.close_file_i(self.file_index)
        
        # Combine and reconcile data from both sources
        self._reconcile_metadata(metadata)
        
        # Calculate summary statistics
        metadata.total_channels = len(metadata.regular_channels) + len(metadata.gps_channels) + len(metadata.calculated_channels)
        metadata.total_sample_count = sum(ch.sample_count for ch in metadata.regular_channels + metadata.gps_channels)
        
        # Extract unique sample rates
        sample_rates = [ch.sample_rate for ch in metadata.regular_channels + metadata.gps_channels if ch.sample_rate]
        metadata.unique_sample_rates = sorted(list(set(sample_rates)))
        
        return metadata
    
    def _calculate_hash(self, file_path: Path, algorithm: str) -> str:
        """Calculate file hash"""
        hash_obj = hashlib.md5() if algorithm == 'md5' else hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    def _extract_from_binary_analysis(self, metadata: XRKFileMetadata):
        """Extract metadata from binary analysis results"""
        raw = metadata.raw_file_analysis
        
        # File format information
        metadata.file_format = FileFormatMetadata()
        
        if 'file_structure' in raw:
            structure = raw['file_structure']
            for sig in structure.get('signatures_found', []):
                if sig['name'] == 'aim_xrk':
                    metadata.file_format.format_signature = 'AIM XRK'
                elif sig['name'] == 'xml_header':
                    metadata.file_format.file_structure_type = 'XML/Binary Hybrid'
        
        if 'encoding_hints' in raw:
            hints = raw['encoding_hints']
            metadata.file_format.endianness = hints.get('endianness')
            if hints.get('likely_compressed'):
                metadata.file_format.compression = 'Detected'
            if hints.get('likely_encrypted'):
                metadata.file_format.encryption = 'Possible'
        
        # Logger information from strings
        metadata.logger = LoggerMetadata()
        
        if 'strings' in raw:
            strings = raw['strings']
            
            # Parse version strings
            for v in strings.get('versions', []):
                if 'firmware' in v.lower():
                    metadata.logger.firmware_version = v
                elif not metadata.file_format.format_version:
                    metadata.file_format.format_version = v
            
            # Parse configuration strings
            for config in strings.get('configuration', []):
                if 'serial' in config.lower():
                    parts = config.split('=')
                    if len(parts) == 2:
                        metadata.logger.serial_number = parts[1].strip()
        
        # Session information from filename and strings
        metadata.session = SessionMetadata()
        self._parse_session_from_filename(metadata)
        
        # Track names from strings
        if 'strings' in raw:
            track_names = raw['strings'].get('track_names', [])
            if track_names and not metadata.session.track_name:
                metadata.session.track_name = track_names[0]
        
        # Data quality assessment
        metadata.data_quality = self._assess_data_quality(raw)
    
    def _extract_dll_metadata(self, metadata: XRKFileMetadata):
        """Extract metadata using DLL functions"""
        # Regular channels
        regular_count = self.dll.get_channels_count(self.file_index)
        for i in range(regular_count):
            name_ptr = self.dll.get_channel_name(self.file_index, i)
            if name_ptr:
                name = name_ptr.decode('utf-8')
                sample_count = self.dll.get_channel_samples_count(self.file_index, i)
                
                channel = ChannelMetadata(
                    index=i,
                    name=name,
                    channel_type='regular',
                    sample_count=sample_count,
                    units=self._guess_units_from_name(name)
                )
                metadata.regular_channels.append(channel)
        
        # GPS channels
        gps_count = self.dll.get_GPS_channels_count(self.file_index)
        for i in range(gps_count):
            name_ptr = self.dll.get_GPS_channel_name(self.file_index, i)
            if name_ptr:
                name = name_ptr.decode('utf-8')
                sample_count = self.dll.get_GPS_channel_samples_count(self.file_index, i)
                
                channel = ChannelMetadata(
                    index=i,
                    name=name,
                    channel_type='gps',
                    sample_count=sample_count,
                    units=self._guess_units_from_name(name)
                )
                metadata.gps_channels.append(channel)
    
    def _parse_session_from_filename(self, metadata: XRKFileMetadata):
        """Parse session info from filename"""
        filename = metadata.filename
        
        # AIM format: YYYYMMDD_HHMMSS_Track_Session_ID.xrk
        parts = filename.replace('.xrk', '').split('_')
        
        if len(parts) >= 2:
            # Date
            if len(parts[0]) == 8 and parts[0].isdigit():
                date_str = parts[0]
                metadata.session.date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            
            # Time
            if len(parts[1]) == 6 and parts[1].isdigit():
                time_str = parts[1]
                metadata.session.time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
            
            # Track name (combine middle parts)
            if len(parts) > 2:
                track_parts = []
                for part in parts[2:]:
                    if not (part.startswith('a_') or part.isdigit()):
                        track_parts.append(part)
                    else:
                        break
                
                if track_parts:
                    metadata.session.track_name = ' '.join(track_parts)
    
    def _guess_units_from_name(self, name: str) -> Optional[str]:
        """Guess measurement units from channel name"""
        name_lower = name.lower()
        
        unit_map = {
            'rpm': 'RPM',
            'speed': 'mph',
            'temp': '°F',
            'pressure': 'psi',
            'voltage': 'V',
            'throttle': '%',
            'brake': '%',
            'latitude': 'degrees',
            'longitude': 'degrees'
        }
        
        for keyword, unit in unit_map.items():
            if keyword in name_lower:
                return unit
        
        return None
    
    def _assess_data_quality(self, raw_analysis: Dict) -> Dict[str, Any]:
        """Assess data quality from raw analysis"""
        quality = {
            'file_integrity': 'good',
            'compression_detected': False,
            'encryption_detected': False,
            'data_density': 0,
            'estimated_completeness': 100
        }
        
        if 'encoding_hints' in raw_analysis:
            hints = raw_analysis['encoding_hints']
            quality['compression_detected'] = hints.get('likely_compressed', False)
            quality['encryption_detected'] = hints.get('likely_encrypted', False)
            
            # Use entropy as a quality indicator
            entropy = hints.get('entropy', 0)
            if entropy < 3:
                quality['data_density'] = 'sparse'
            elif entropy < 6:
                quality['data_density'] = 'normal'
            else:
                quality['data_density'] = 'dense'
        
        return quality
    
    def _reconcile_metadata(self, metadata: XRKFileMetadata):
        """Reconcile metadata from different sources"""
        # Prefer DLL data for channels (more accurate)
        # Prefer binary analysis for file structure
        # Combine both for comprehensive view
        
        # Add any channels found in strings but not in DLL
        if 'strings' in metadata.raw_file_analysis:
            channel_names_from_strings = metadata.raw_file_analysis['strings'].get('channel_names', [])
            existing_names = [ch.name for ch in metadata.regular_channels + metadata.gps_channels]
            
            for name in channel_names_from_strings:
                if name not in existing_names:
                    # Found a channel in binary that wasn't exposed by DLL
                    metadata.calculated_channels.append(ChannelMetadata(
                        index=-1,
                        name=name,
                        channel_type='calculated',
                        sample_count=0,
                        units=self._guess_units_from_name(name)
                    ))


def print_comprehensive_metadata(metadata: XRKFileMetadata):
    """Print a comprehensive, formatted metadata report"""
    print(f"\n{'='*100}")
    print(f"XRK FILE COMPREHENSIVE METADATA ANALYSIS")
    print(f"{'='*100}")
    
    # File Information
    print(f"\n📁 FILE INFORMATION")
    print(f"{'─'*50}")
    print(f"   Filename:        {metadata.filename}")
    print(f"   Path:            {metadata.file_path}")
    print(f"   Size:            {metadata.file_size_bytes:,} bytes ({metadata.file_size_bytes/1024/1024:.2f} MB)")
    print(f"   MD5 Hash:        {metadata.file_hash_md5}")
    print(f"   SHA256 Hash:     {metadata.file_hash_sha256}")
    print(f"   Created:         {metadata.creation_date}")
    print(f"   Modified:        {metadata.modification_date}")
    
    # File Format
    if metadata.file_format:
        print(f"\n📊 FILE FORMAT")
        print(f"{'─'*50}")
        ff = metadata.file_format
        print(f"   Format:          {ff.format_signature or 'Unknown'}")
        print(f"   Version:         {ff.format_version or 'Unknown'}")
        print(f"   Structure:       {ff.file_structure_type or 'Binary'}")
        print(f"   Endianness:      {ff.endianness or 'Unknown'}")
        print(f"   Compression:     {ff.compression or 'None detected'}")
        print(f"   Encryption:      {ff.encryption or 'None detected'}")
        
        if ff.binary_sections:
            print(f"   Binary Sections: {len(ff.binary_sections)} identified")
    
    # Logger Information
    if metadata.logger:
        print(f"\n🔧 DATA LOGGER")
        print(f"{'─'*50}")
        logger = metadata.logger
        print(f"   Manufacturer:    {logger.manufacturer}")
        print(f"   Model:           {logger.model or 'Unknown'}")
        print(f"   Serial Number:   {logger.serial_number or 'Unknown'}")
        print(f"   Firmware:        {logger.firmware_version or 'Unknown'}")
        
        if logger.gps_enabled is not None:
            print(f"   GPS Enabled:     {'Yes' if logger.gps_enabled else 'No'}")
        if logger.can_bus_enabled is not None:
            print(f"   CAN Bus:         {'Yes' if logger.can_bus_enabled else 'No'}")
    
    # Session Information
    if metadata.session:
        print(f"\n🏁 SESSION INFORMATION")
        print(f"{'─'*50}")
        session = metadata.session
        
        if session.date:
            print(f"   Date:            {session.date}")
        if session.time:
            print(f"   Time:            {session.time}")
        if session.track_name:
            print(f"   Track:           {session.track_name}")
        if session.session_type:
            print(f"   Type:            {session.session_type}")
        if session.duration_seconds:
            minutes = int(session.duration_seconds // 60)
            seconds = session.duration_seconds % 60
            print(f"   Duration:        {minutes}:{seconds:05.2f}")
        if session.laps_completed:
            print(f"   Laps:            {session.laps_completed}")
        
        # Weather if available
        if session.ambient_temp or session.track_temp:
            print(f"\n   Weather Conditions:")
            if session.ambient_temp:
                print(f"     Ambient:       {session.ambient_temp}°F")
            if session.track_temp:
                print(f"     Track:         {session.track_temp}°F")
    
    # Vehicle Information
    if metadata.vehicle:
        print(f"\n🏎️  VEHICLE INFORMATION")
        print(f"{'─'*50}")
        vehicle = metadata.vehicle
        
        if vehicle.driver_name:
            print(f"   Driver:          {vehicle.driver_name}")
        if vehicle.team_name:
            print(f"   Team:            {vehicle.team_name}")
        if vehicle.make and vehicle.model:
            print(f"   Vehicle:         {vehicle.make} {vehicle.model}")
        if vehicle.year:
            print(f"   Year:            {vehicle.year}")
        if vehicle.weight_kg:
            print(f"   Weight:          {vehicle.weight_kg} kg ({vehicle.weight_kg * 2.205:.0f} lbs)")
        if vehicle.setup_name:
            print(f"   Setup:           {vehicle.setup_name}")
    
    # Channel Information
    print(f"\n📈 TELEMETRY CHANNELS")
    print(f"{'─'*50}")
    print(f"   Total Channels:  {metadata.total_channels}")
    print(f"   Total Samples:   {metadata.total_sample_count:,}")
    
    if metadata.unique_sample_rates:
        print(f"   Sample Rates:    {', '.join(f'{r:.1f} Hz' for r in metadata.unique_sample_rates)}")
    
    # Regular Channels
    if metadata.regular_channels:
        print(f"\n   Standard Channels ({len(metadata.regular_channels)}):")
        # Group channels by category
        categories = {}
        for ch in metadata.regular_channels:
            category = _categorize_channel(ch.name)
            if category not in categories:
                categories[category] = []
            categories[category].append(ch)
        
        for category, channels in sorted(categories.items()):
            print(f"\n     {category}:")
            for ch in channels[:5]:  # Show first 5 in each category
                units = f" [{ch.units}]" if ch.units else ""
                rate = f" @ {ch.sample_rate:.1f}Hz" if ch.sample_rate else ""
                print(f"       • {ch.name}{units}: {ch.sample_count:,} samples{rate}")
            
            if len(channels) > 5:
                print(f"       ... and {len(channels) - 5} more")
    
    # GPS Channels
    if metadata.gps_channels:
        print(f"\n   GPS Channels ({len(metadata.gps_channels)}):")
        for ch in metadata.gps_channels:
            units = f" [{ch.units}]" if ch.units else ""
            rate = f" @ {ch.sample_rate:.1f}Hz" if ch.sample_rate else ""
            print(f"     • {ch.name}{units}: {ch.sample_count:,} samples{rate}")
    
    # Calculated/Math Channels
    if metadata.calculated_channels:
        print(f"\n   Calculated/Math Channels ({len(metadata.calculated_channels)}):")
        for ch in metadata.calculated_channels[:10]:
            units = f" [{ch.units}]" if ch.units else ""
            print(f"     • {ch.name}{units}")
        
        if len(metadata.calculated_channels) > 10:
            print(f"     ... and {len(metadata.calculated_channels) - 10} more")
    
    # Data Quality
    if metadata.data_quality:
        print(f"\n✅ DATA QUALITY ASSESSMENT")
        print(f"{'─'*50}")
        dq = metadata.data_quality
        
        print(f"   File Integrity:  {dq.get('file_integrity', 'Unknown')}")
        print(f"   Data Density:    {dq.get('data_density', 'Unknown')}")
        print(f"   Completeness:    {dq.get('estimated_completeness', 0)}%")
        
        if dq.get('compression_detected'):
            print(f"   ⚠️  Compression detected - may affect direct parsing")
        if dq.get('encryption_detected'):
            print(f"   ⚠️  Possible encryption - limited metadata extraction")
    
    # Raw Analysis Summary
    if metadata.raw_file_analysis:
        print(f"\n🔍 BINARY ANALYSIS INSIGHTS")
        print(f"{'─'*50}")
        
        # Strings found
        if 'strings' in metadata.raw_file_analysis:
            strings = metadata.raw_file_analysis['strings']
            total_strings = sum(len(v) for v in strings.values() if isinstance(v, list))
            print(f"   Readable Strings Found: {total_strings}")
            
            # Show interesting findings
            if strings.get('track_names'):
                print(f"   Track References: {', '.join(strings['track_names'][:3])}")
            if strings.get('timestamps'):
                print(f"   Timestamps Found: {', '.join(strings['timestamps'][:3])}")
        
        # Data blocks
        if 'data_blocks' in metadata.raw_file_analysis:
            blocks = metadata.raw_file_analysis['data_blocks']
            if blocks:
                print(f"   Data Block Structure:")
                for block in blocks[:3]:
                    print(f"     • {block['block_size']} byte blocks ({block['estimated_count']} estimated)")
        
        # File structure
        if 'file_structure' in metadata.raw_file_analysis:
            structure = metadata.raw_file_analysis['file_structure']
            if structure.get('sections'):
                print(f"   File Sections Identified: {len(structure['sections'])}")


def _categorize_channel(channel_name: str) -> str:
    """Categorize channel by name for organized display"""
    name_lower = channel_name.lower()
    
    # Define categories and their keywords
    categories = {
        'Engine': ['rpm', 'throttle', 'map', 'lambda', 'afr', 'fuel', 'ignition', 'knock'],
        'Temperature': ['temp', 'egt', 'cht', 'oil', 'water', 'coolant', 'intake'],
        'Pressure': ['pressure', 'boost', 'oil_p', 'fuel_p', 'brake_p'],
        'Suspension': ['damper', 'ride_height', 'suspension', 'shock', 'spring'],
        'Brakes': ['brake', 'brake_temp', 'brake_pressure', 'pad'],
        'Steering': ['steering', 'steer'],
        'Acceleration': ['accel', 'g_lat', 'g_lon', 'g_vert'],
        'Wheels/Tires': ['wheel', 'tire', 'slip'],
        'Electrical': ['voltage', 'current', 'battery'],
        'Driver Inputs': ['clutch', 'gear', 'handbrake'],
        'GPS/Position': ['gps', 'latitude', 'longitude', 'altitude', 'heading'],
        'Timing': ['lap', 'sector', 'split'],
        'Aero': ['aero', 'downforce', 'drag', 'ride_height']
    }
    
    for category, keywords in categories.items():
        if any(kw in name_lower for kw in keywords):
            return category
    
    return 'Other'


def export_metadata_report(metadata: XRKFileMetadata, output_path: str = None):
    """Export comprehensive metadata report to multiple formats"""
    import json
    from datetime import datetime
    
    if not output_path:
        output_path = f"xrk_metadata_{metadata.filename.replace('.xrk', '')}_{datetime.now():%Y%m%d_%H%M%S}"
    
    # JSON export (complete data)
    json_file = f"{output_path}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(asdict(metadata), f, indent=2, default=str, ensure_ascii=False)
    print(f"\n💾 JSON metadata saved to: {json_file}")
    
    # HTML report (formatted for viewing)
    html_file = f"{output_path}.html"
    html_content = generate_html_report(metadata)
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"📄 HTML report saved to: {html_file}")
    
    # CSV channel list (for spreadsheet analysis)
    csv_file = f"{output_path}_channels.csv"
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        f.write("Channel Name,Type,Sample Count,Sample Rate,Units,Category\n")
        
        all_channels = metadata.regular_channels + metadata.gps_channels + metadata.calculated_channels
        for ch in all_channels:
            category = _categorize_channel(ch.name)
            f.write(f'"{ch.name}",{ch.channel_type},{ch.sample_count},{ch.sample_rate or ""},{ch.units or ""},{category}\n')
    
    print(f"📊 Channel list CSV saved to: {csv_file}")




def export_metadata_report_old(metadata: XRKFileMetadata, output_path: str = None):
    """Export comprehensive metadata report to multiple formats"""
    import json
    from datetime import datetime
    
    if not output_path:
        output_path = f"xrk_metadata_{metadata.filename.replace('.xrk', '')}_{datetime.now():%Y%m%d_%H%M%S}"
    
    # JSON export (complete data)
    json_file = f"{output_path}.json"
    with open(json_file, 'w') as f:
        json.dump(asdict(metadata), f, indent=2, default=str)
    print(f"\n💾 JSON metadata saved to: {json_file}")
    
    # HTML report (formatted for viewing)
    html_file = f"{output_path}.html"
    html_content = generate_html_report(metadata)
    with open(html_file, 'w') as f:
        f.write(html_content)
    print(f"📄 HTML report saved to: {html_file}")
    
    # CSV channel list (for spreadsheet analysis)
    csv_file = f"{output_path}_channels.csv"
    with open(csv_file, 'w') as f:
        f.write("Channel Name,Type,Sample Count,Sample Rate,Units,Category\n")
        
        all_channels = metadata.regular_channels + metadata.gps_channels + metadata.calculated_channels
        for ch in all_channels:
            category = _categorize_channel(ch.name)
            f.write(f'"{ch.name}",{ch.channel_type},{ch.sample_count},{ch.sample_rate or ""},{ch.units or ""},{category}\n')
    
    print(f"📊 Channel list CSV saved to: {csv_file}")


def generate_html_report(metadata: XRKFileMetadata) -> str:
    """Generate HTML report with styling"""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>XRK Metadata Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 30px; }
        .info-grid { display: grid; grid-template-columns: 200px 1fr; gap: 10px; margin: 10px 0; }
        .label { font-weight: bold; color: #7f8c8d; }
        .value { color: #2c3e50; }
        .channel-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        .channel-table th { background: #3498db; color: white; padding: 8px; text-align: left; }
        .channel-table td { padding: 8px; border-bottom: 1px solid #ecf0f1; }
        .channel-table tr:hover { background: #f8f9fa; }
        .warning { background: #f39c12; color: white; padding: 5px 10px; border-radius: 4px; }
        .success { background: #27ae60; color: white; padding: 5px 10px; border-radius: 4px; }
        .section { margin: 20px 0; padding: 15px; background: #ecf0f1; border-radius: 6px; }
    </style>
</head>
<body>
    <div class="container">
"""
    
    html += f"""
        <h1>XRK Metadata Analysis Report</h1>
        <div class="section">
            <h2>📁 File Information</h2>
            <div class="info-grid">
                <div class="label">Filename:</div><div class="value">{metadata.filename}</div>
                <div class="label">Size:</div><div class="value">{metadata.file_size_bytes:,} bytes ({metadata.file_size_bytes/1024/1024:.2f} MB)</div>
                <div class="label">MD5 Hash:</div><div class="value" style="font-family: monospace;">{metadata.file_hash_md5}</div>
                <div class="label">Created:</div><div class="value">{metadata.creation_date}</div>
                <div class="label">Modified:</div><div class="value">{metadata.modification_date}</div>
            </div>
        </div>
"""
    
    if metadata.session:
        session = metadata.session
        html += f"""
        <div class="section">
            <h2>🏁 Session Information</h2>
            <div class="info-grid">
                <div class="label">Date:</div><div class="value">{session.date or 'Unknown'}</div>
                <div class="label">Time:</div><div class="value">{session.time or 'Unknown'}</div>
                <div class="label">Track:</div><div class="value">{session.track_name or 'Unknown'}</div>
                <div class="label">Session Type:</div><div class="value">{session.session_type or 'Unknown'}</div>
            </div>
        </div>
"""
    
    # Channels table
    html += """
        <div class="section">
            <h2>📈 Telemetry Channels</h2>
            <table class="channel-table">
                <thead>
                    <tr>
                        <th>Channel Name</th>
                        <th>Type</th>
                        <th>Samples</th>
                        <th>Rate (Hz)</th>
                        <th>Units</th>
                        <th>Category</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    all_channels = metadata.regular_channels + metadata.gps_channels
    for ch in sorted(all_channels, key=lambda x: x.name):
        category = _categorize_channel(ch.name)
        rate = f"{ch.sample_rate:.1f}" if ch.sample_rate else "-"
        html += f"""
                    <tr>
                        <td>{ch.name}</td>
                        <td>{ch.channel_type}</td>
                        <td>{ch.sample_count:,}</td>
                        <td>{rate}</td>
                        <td>{ch.units or "-"}</td>
                        <td>{category}</td>
                    </tr>
"""
    
    html += """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
    
    return html


if __name__ == "__main__":
    import sys
    
    # Get file path from command line or use default
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Look for a sample file
        xrk_files = list(SAMPLE_FILES_PATH.glob("*.xrk"))
        if not xrk_files:
            print("No XRK files found!")
            print("Usage: python xrk_metadata_analyzer.py <path_to_xrk_file>")
            sys.exit(1)
        
        file_path = xrk_files[0]
        print(f"Using sample file: {file_path}")
    
    print(f"\nAnalyzing: {file_path}")
    print("This comprehensive analysis may take a moment...\n")
    
    # Perform comprehensive analysis
    extractor = XRKMetadataExtractor()
    metadata = extractor.analyze_file_comprehensive(str(file_path))
    
    # Print comprehensive report
    print_comprehensive_metadata(metadata)
    
    # Export reports
    export_metadata_report(metadata)
    
    print(f"\n✨ Analysis complete!")
    print(f"This analysis examined {metadata.file_size_bytes:,} bytes of data")
    print(f"Found {metadata.total_channels} channels with {metadata.total_sample_count:,} total samples")