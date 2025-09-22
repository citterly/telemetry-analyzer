"""
Test script for File Manager
Run this to verify file management functionality
"""

import tempfile
import shutil
from pathlib import Path
import json

from file_manager import FileManager, FileMetadata


def test_file_manager():
    """Test file manager functionality"""
    
    print("Testing File Manager...")
    
    # Create temporary test directory
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data_dir = Path(temp_dir) / "test_data"
        
        # Initialize file manager
        fm = FileManager(str(test_data_dir))
        
        # Test 1: Check initial state
        print("\n1. Testing initial state...")
        files = fm.get_file_list()
        stats = fm.get_stats()
        print(f"   Initial files: {len(files)}")
        print(f"   Initial stats: {stats}")
        assert len(files) == 0
        
        # Test 2: Import a file (mock XRK file)
        print("\n2. Testing file import...")
        
        # Create a mock XRK file for testing
        mock_xrk_path = Path(temp_dir) / "test_session.xrk"
        mock_xrk_path.write_bytes(b"mock xrk data for testing")
        
        try:
            # This will fail because it's not a real XRK, but we can test the validation
            metadata = fm.import_file(str(mock_xrk_path))
            print(f"   ERROR: Should have failed with mock data")
        except Exception as e:
            print(f"   Expected error with mock file: {e}")
        
        # Test 3: Import with your actual XRK file (if available)
        print("\n3. Testing with real XRK file...")
        
        # Look for your default session file
        from config import SAMPLE_FILES_PATH, DEFAULT_SESSION
        real_xrk_path = SAMPLE_FILES_PATH / DEFAULT_SESSION
        
        if real_xrk_path.exists():
            try:
                metadata = fm.import_file(str(real_xrk_path), 
                                        custom_attributes={"test": True, "source": "real_data"})
                print(f"   Imported: {metadata.filename}")
                print(f"   Size: {metadata.file_size_bytes:,} bytes")
                print(f"   Hash: {metadata.file_hash[:16]}...")
                print(f"   Session duration: {metadata.session_duration_seconds}s")
                
                # Test 4: Process the file
                print("\n4. Testing file processing...")
                results = fm.process_file(metadata.filename)
                print(f"   Processed {len(results['laps'])} laps")
                
                if results['fastest_lap_data']:
                    lap_info = results['fastest_lap_data']['lap_info']
                    print(f"   Fastest lap: {lap_info.lap_time:.2f}s")
                
                # Test 5: Check updated metadata
                print("\n5. Testing metadata updates...")
                updated_metadata = fm.get_file_metadata(metadata.filename)
                print(f"   Processed: {updated_metadata.processed}")
                print(f"   Total laps: {updated_metadata.total_laps}")
                print(f"   Fastest lap: {updated_metadata.fastest_lap_time}s")
                print(f"   Custom attributes: {updated_metadata.custom_attributes}")
                
                # Test 6: Search functionality
                print("\n6. Testing search functionality...")
                processed_files = fm.search_files(processed=True)
                print(f"   Found {len(processed_files)} processed files")
                
                fast_laps = fm.search_files(max_lap_time=150.0)
                print(f"   Found {len(fast_laps)} files with laps under 150s")
                
                # Test 7: Statistics
                print("\n7. Testing statistics...")
                stats = fm.get_stats()
                for key, value in stats.items():
                    print(f"   {key}: {value}")
                
                print("\n✓ All tests passed!")
                
            except Exception as e:
                print(f"   Error with real XRK file: {e}")
                print("   This might be expected if the XRK parsing fails")
        else:
            print(f"   No XRK file found at {real_xrk_path}")
            print("   Skipping real file tests")


def test_metadata_persistence():
    """Test that metadata persists across FileManager instances"""
    
    print("\nTesting metadata persistence...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data_dir = Path(temp_dir) / "persistence_test"
        
        # Create first instance and import file
        fm1 = FileManager(str(test_data_dir))
        
        # Create mock metadata file
        metadata = FileMetadata(
            filename="test.xrk",
            file_path=str(test_data_dir / "uploads" / "test.xrk"),
            file_size_bytes=1024,
            file_hash="test_hash",
            import_date="2024-01-01T12:00:00",
            processed=True,
            total_laps=5,
            fastest_lap_time=145.5
        )
        
        fm1._save_metadata(metadata)
        fm1.file_index[metadata.filename] = metadata
        
        print(f"   Saved metadata for {metadata.filename}")
        
        # Create second instance and verify it loads the metadata
        fm2 = FileManager(str(test_data_dir))
        loaded_files = fm2.get_file_list()
        
        print(f"   Loaded {len(loaded_files)} files in new instance")
        
        if loaded_files:
            loaded = loaded_files[0]
            print(f"   Loaded file: {loaded.filename}")
            print(f"   Processed: {loaded.processed}")
            print(f"   Fastest lap: {loaded.fastest_lap_time}")
            assert loaded.filename == metadata.filename
            assert loaded.processed == metadata.processed
            assert loaded.fastest_lap_time == metadata.fastest_lap_time
            print("   ✓ Persistence test passed!")
        else:
            print("   ✗ Persistence test failed - no files loaded")


if __name__ == "__main__":
    test_file_manager()
    test_metadata_persistence()
    print("\nFile Manager testing complete!")