"""
File Management Module for Telemetry Analysis
Handles XRK file import, metadata extraction, and file selection
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib

# Import your existing analysis modules
from analysis.data_loader import load_session_data, get_data_summary
from analysis.lap_analyzer import analyze_session_laps


@dataclass
class FileMetadata:
    """Minimal metadata structure that can be expanded over time"""
    filename: str
    file_path: str
    file_size_bytes: int
    file_hash: str  # For duplicate detection
    import_date: str
    
    # XRK-specific metadata (extracted from file)
    session_duration_seconds: Optional[float] = None
    sample_count: Optional[int] = None
    track_name: Optional[str] = None
    session_date: Optional[str] = None
    
    # Analysis metadata (populated after processing)
    total_laps: Optional[int] = None
    fastest_lap_time: Optional[float] = None
    max_speed_mph: Optional[float] = None
    max_rpm: Optional[float] = None
    processed: bool = False
    
    # Expandable custom attributes
    custom_attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_attributes is None:
            self.custom_attributes = {}


class FileManager:
    """Manages XRK files and their metadata"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.uploads_dir = self.data_dir / "uploads"
        self.metadata_dir = self.data_dir / "metadata"
        
        # Ensure directories exist
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Index of all files (filename -> metadata)
        self.file_index: Dict[str, FileMetadata] = {}
        self._load_existing_files()
    
    def import_file(self, source_path: str, custom_attributes: Optional[Dict] = None) -> FileMetadata:
        """
        Import an XRK file and extract basic metadata
        """
        source_path = Path(source_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        if not source_path.name.endswith('.xrk'):
            raise ValueError("Only XRK files are supported")
        
        # Calculate file hash for duplicate detection
        file_hash = self._calculate_file_hash(source_path)
        
        # Check for duplicates
        existing_file = self._find_by_hash(file_hash)
        if existing_file:
            print(f"File already imported: {existing_file.filename}")
            return existing_file
        
        # Copy file to uploads directory
        dest_path = self.uploads_dir / source_path.name
        
        # Handle filename conflicts
        counter = 1
        while dest_path.exists():
            stem = source_path.stem
            suffix = source_path.suffix
            dest_path = self.uploads_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        shutil.copy2(source_path, dest_path)
        
        # Create metadata
        metadata = FileMetadata(
            filename=dest_path.name,
            file_path=str(dest_path),
            file_size_bytes=dest_path.stat().st_size,
            file_hash=file_hash,
            import_date=datetime.now().isoformat(),
            custom_attributes=custom_attributes or {}
        )
        
        # Extract XRK metadata
        self._extract_xrk_metadata(metadata)
        
        # Save metadata and update index
        self._save_metadata(metadata)
        self.file_index[metadata.filename] = metadata
        
        print(f"Imported: {metadata.filename} ({metadata.file_size_bytes:,} bytes)")
        return metadata
    
    def get_file_list(self, filter_processed: Optional[bool] = None) -> List[FileMetadata]:
        """Get list of all imported files, optionally filtered by processing status"""
        files = list(self.file_index.values())
        
        if filter_processed is not None:
            files = [f for f in files if f.processed == filter_processed]
        
        # Sort by import date (newest first)
        files.sort(key=lambda x: x.import_date, reverse=True)
        return files
    
    def get_file_metadata(self, filename: str) -> Optional[FileMetadata]:
        """Get metadata for a specific file"""
        return self.file_index.get(filename)
    
    def update_metadata(self, filename: str, updates: Dict[str, Any]) -> bool:
        """Update metadata for a file"""
        if filename not in self.file_index:
            return False
        
        metadata = self.file_index[filename]
        
        # Update standard fields
        for key, value in updates.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
            else:
                # Add to custom attributes
                metadata.custom_attributes[key] = value
        
        # Save updated metadata
        self._save_metadata(metadata)
        return True
    
    def process_file(self, filename: str) -> Dict[str, Any]:
        """Process a file and update its metadata with analysis results"""
        if filename not in self.file_index:
            raise ValueError(f"File not found: {filename}")
        
        metadata = self.file_index[filename]
        file_path = Path(metadata.file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found on disk: {file_path}")
        
        print(f"Processing: {filename}")
        
        try:
            # Load session data using your existing code
            session_data = load_session_data(str(file_path))
            if not session_data:
                raise Exception("Failed to load session data")
            
            # Analyze laps
            laps, fastest_lap_data = analyze_session_laps(session_data)
            
            # Update metadata with analysis results
            updates = {
                'total_laps': len(laps) if laps else 0,
                'processed': True
            }
            
            if fastest_lap_data:
                lap_info = fastest_lap_data['lap_info']
                updates.update({
                    'fastest_lap_time': lap_info.lap_time,
                    'max_speed_mph': lap_info.max_speed_mph,
                    'max_rpm': lap_info.max_rpm
                })
            
            self.update_metadata(filename, updates)
            
            # Return processing results
            results = {
                'session_data': session_data,
                'laps': laps,
                'fastest_lap_data': fastest_lap_data,
                'metadata': self.file_index[filename]
            }
            
            print(f"Processed: {filename} - {len(laps) if laps else 0} laps")
            return results
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            self.update_metadata(filename, {'processed': False, 'error': str(e)})
            raise
    
    def delete_file(self, filename: str) -> bool:
        """Delete a file and its metadata"""
        if filename not in self.file_index:
            return False
        
        metadata = self.file_index[filename]
        
        # Delete physical file
        file_path = Path(metadata.file_path)
        if file_path.exists():
            file_path.unlink()
        
        # Delete metadata file
        metadata_path = self.metadata_dir / f"{filename}.json"
        if metadata_path.exists():
            metadata_path.unlink()
        
        # Remove from index
        del self.file_index[filename]
        
        print(f"Deleted: {filename}")
        return True
    
    def search_files(self, **criteria) -> List[FileMetadata]:
        """Search files by various criteria"""
        results = []
        
        for metadata in self.file_index.values():
            match = True
            
            for key, value in criteria.items():
                if key == 'track_name' and metadata.track_name:
                    if value.lower() not in metadata.track_name.lower():
                        match = False
                        break
                elif key == 'min_lap_time' and metadata.fastest_lap_time:
                    if metadata.fastest_lap_time > value:
                        match = False
                        break
                elif key == 'max_lap_time' and metadata.fastest_lap_time:
                    if metadata.fastest_lap_time < value:
                        match = False
                        break
                elif key == 'processed':
                    if metadata.processed != value:
                        match = False
                        break
            
            if match:
                results.append(metadata)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics about imported files"""
        files = list(self.file_index.values())
        processed_files = [f for f in files if f.processed]
        
        stats = {
            'total_files': len(files),
            'processed_files': len(processed_files),
            'unprocessed_files': len(files) - len(processed_files),
            'total_size_mb': sum(f.file_size_bytes for f in files) / (1024 * 1024),
        }
        
        if processed_files:
            lap_times = [f.fastest_lap_time for f in processed_files if f.fastest_lap_time]
            if lap_times:
                stats.update({
                    'fastest_overall_lap': min(lap_times),
                    'slowest_overall_lap': max(lap_times),
                    'average_lap_time': sum(lap_times) / len(lap_times)
                })
        
        return stats
    
    # Private methods
    
    def _load_existing_files(self):
        """Load metadata for existing files"""
        self.file_index = {}
        
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                
                metadata = FileMetadata(**data)
                self.file_index[metadata.filename] = metadata
                
            except Exception as e:
                print(f"Error loading metadata from {metadata_file}: {e}")
    
    def _save_metadata(self, metadata: FileMetadata):
        """Save metadata to disk"""
        metadata_path = self.metadata_dir / f"{metadata.filename}.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file for duplicate detection"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _find_by_hash(self, file_hash: str) -> Optional[FileMetadata]:
        """Find existing file by hash"""
        for metadata in self.file_index.values():
            if metadata.file_hash == file_hash:
                return metadata
        return None
    
    def _extract_xrk_metadata(self, metadata: FileMetadata):
        """Extract basic metadata from XRK file"""
        try:
            # Use your existing data loader to get basic info
            session_data = load_session_data(metadata.file_path)
            
            if session_data:
                summary = get_data_summary(session_data)
                
                metadata.session_duration_seconds = session_data.get('session_duration')
                metadata.sample_count = session_data.get('sample_count')
                metadata.track_name = "Road America"  # Could be extracted from filename or config
                
                # Try to extract date from filename (format: YYYYMMDD_HHMMSS_...)
                filename_parts = metadata.filename.split('_')
                if len(filename_parts) >= 2 and len(filename_parts[0]) == 8:
                    try:
                        date_str = filename_parts[0]
                        metadata.session_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    except:
                        pass
                        
        except Exception as e:
            print(f"Could not extract XRK metadata: {e}")


# Command-line interface for testing
if __name__ == "__main__":
    import sys
    
    fm = FileManager()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python file_manager.py import <xrk_file>")
        print("  python file_manager.py list")
        print("  python file_manager.py process <filename>")
        print("  python file_manager.py stats")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "import" and len(sys.argv) > 2:
        source_file = sys.argv[2]
        try:
            metadata = fm.import_file(source_file)
            print(f"Success: {metadata.filename}")
        except Exception as e:
            print(f"Error: {e}")
    
    elif command == "list":
        files = fm.get_file_list()
        print(f"Found {len(files)} files:")
        for f in files:
            status = "✓" if f.processed else "○"
            print(f"  {status} {f.filename} ({f.file_size_bytes:,} bytes)")
    
    elif command == "process" and len(sys.argv) > 2:
        filename = sys.argv[2]
        try:
            results = fm.process_file(filename)
            print(f"Processed: {filename}")
            print(f"  Laps: {len(results['laps'])}")
            if results['fastest_lap_data']:
                lap_time = results['fastest_lap_data']['lap_info'].lap_time
                print(f"  Fastest lap: {lap_time:.2f}s")
        except Exception as e:
            print(f"Error: {e}")
    
    elif command == "stats":
        stats = fm.get_stats()
        print("Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    else:
        print("Invalid command")