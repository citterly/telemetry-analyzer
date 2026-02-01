"""
Phase 9 Performance Validation Test

Tests that all views load within specified time limits:
- Session overview loads in < 2 seconds
- Quadrant deep dive loads in < 1 second
- Zone detail loads in < 1 second
- Comparison overlay loads in < 2 seconds
- Lap switching completes in < 0.5 seconds
"""

import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from src.features.gg_analysis import GGAnalyzer
from src.config.vehicles import get_active_vehicle


def find_test_parquet():
    """Find a test Parquet file"""
    data_dir = Path("data")

    # Look for any parquet file in exports
    exports_dir = data_dir / "exports"
    if exports_dir.exists():
        for pq in exports_dir.rglob("*.parquet"):
            return pq

    # Look in uploads
    uploads_dir = data_dir / "uploads"
    if uploads_dir.exists():
        for pq in uploads_dir.glob("*.parquet"):
            return pq

    return None


def measure_time(func, *args, **kwargs):
    """Measure execution time of a function"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def test_session_overview(parquet_path):
    """Test: Session overview loads in < 2 seconds"""
    print("\n1. Testing Session Overview Load Time...")

    # Get vehicle config
    vehicle = get_active_vehicle()
    max_lateral_g = getattr(vehicle, 'max_lateral_g', 1.3)
    max_braking_g = getattr(vehicle, 'max_braking_g', max_lateral_g * 1.1)
    power_limited_accel_g = getattr(vehicle, 'power_limited_accel_g', 0.4)

    # Create analyzer
    analyzer = GGAnalyzer(
        max_g_reference=max_lateral_g,
        max_braking_g=max_braking_g,
        power_limited_accel_g=power_limited_accel_g
    )

    # Measure full analysis (no lap filter)
    result, elapsed = measure_time(
        analyzer.analyze_from_parquet,
        str(parquet_path),
        session_id=parquet_path.stem
    )

    # Convert to JSON (what the frontend receives)
    json_result, json_elapsed = measure_time(result.to_dict)

    total_time = elapsed + json_elapsed

    print(f"   Analysis time: {elapsed:.3f}s")
    print(f"   JSON conversion: {json_elapsed:.3f}s")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Requirement: < 2.0s")

    if total_time < 2.0:
        print("   ✓ PASS")
        return True
    else:
        print("   ✗ FAIL")
        return False


def test_lap_switching(parquet_path):
    """Test: Lap switching completes in < 0.5 seconds"""
    print("\n2. Testing Lap Switching Performance...")

    # Get vehicle config
    vehicle = get_active_vehicle()
    max_lateral_g = getattr(vehicle, 'max_lateral_g', 1.3)
    max_braking_g = getattr(vehicle, 'max_braking_g', max_lateral_g * 1.1)
    power_limited_accel_g = getattr(vehicle, 'power_limited_accel_g', 0.4)

    # Create analyzer
    analyzer = GGAnalyzer(
        max_g_reference=max_lateral_g,
        max_braking_g=max_braking_g,
        power_limited_accel_g=power_limited_accel_g
    )

    # First, analyze to get lap numbers
    full_result = analyzer.analyze_from_parquet(
        str(parquet_path),
        session_id=parquet_path.stem
    )

    if not full_result.lap_numbers or len(full_result.lap_numbers) < 2:
        print("   ⚠ SKIP - Not enough laps for testing")
        return True

    # Test switching to a specific lap
    lap_to_test = full_result.lap_numbers[0]

    result, elapsed = measure_time(
        analyzer.analyze_from_parquet,
        str(parquet_path),
        session_id=parquet_path.stem,
        lap_filter=lap_to_test
    )

    # Convert to JSON
    json_result, json_elapsed = measure_time(result.to_dict)

    total_time = elapsed + json_elapsed

    print(f"   Filtered analysis (Lap {lap_to_test}): {elapsed:.3f}s")
    print(f"   JSON conversion: {json_elapsed:.3f}s")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Requirement: < 0.5s")

    if total_time < 0.5:
        print("   ✓ PASS")
        return True
    else:
        print("   ✗ FAIL")
        return False


def test_quadrant_deep_dive(parquet_path):
    """Test: Quadrant deep dive loads in < 1 second"""
    print("\n3. Testing Quadrant Deep Dive Performance...")

    # Get vehicle config
    vehicle = get_active_vehicle()
    max_lateral_g = getattr(vehicle, 'max_lateral_g', 1.3)
    max_braking_g = getattr(vehicle, 'max_braking_g', max_lateral_g * 1.1)
    power_limited_accel_g = getattr(vehicle, 'power_limited_accel_g', 0.4)

    # Create analyzer
    analyzer = GGAnalyzer(
        max_g_reference=max_lateral_g,
        max_braking_g=max_braking_g,
        power_limited_accel_g=power_limited_accel_g
    )

    # Analyze session
    result, elapsed = measure_time(
        analyzer.analyze_from_parquet,
        str(parquet_path),
        session_id=parquet_path.stem
    )

    # Extract quadrant data (what the deep dive view needs)
    quadrants = result.quadrants

    # Simulate JSON serialization for quadrant view
    def serialize_quadrant_view():
        return {
            "quadrants": [
                {
                    "name": q.name,
                    "display_name": q.display_name,
                    "max_g": q.max_g,
                    "avg_g": q.avg_g,
                    "utilization_pct": q.utilization_pct,
                    "time_spent_pct": q.time_spent_pct,
                    "points_count": q.points_count,
                    "color": q.color
                }
                for q in quadrants
            ]
        }

    json_result, json_elapsed = measure_time(serialize_quadrant_view)

    total_time = elapsed + json_elapsed

    print(f"   Analysis time: {elapsed:.3f}s")
    print(f"   Quadrant serialization: {json_elapsed:.3f}s")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Requirement: < 1.0s")

    if total_time < 1.0:
        print("   ✓ PASS")
        return True
    else:
        print("   ✗ FAIL")
        return False


def test_zone_detail(parquet_path):
    """Test: Zone detail loads in < 1 second"""
    print("\n4. Testing Zone Detail Performance...")

    # Get vehicle config
    vehicle = get_active_vehicle()
    max_lateral_g = getattr(vehicle, 'max_lateral_g', 1.3)
    max_braking_g = getattr(vehicle, 'max_braking_g', max_lateral_g * 1.1)
    power_limited_accel_g = getattr(vehicle, 'power_limited_accel_g', 0.4)

    # Create analyzer
    analyzer = GGAnalyzer(
        max_g_reference=max_lateral_g,
        max_braking_g=max_braking_g,
        power_limited_accel_g=power_limited_accel_g
    )

    # Analyze session
    result, elapsed = measure_time(
        analyzer.analyze_from_parquet,
        str(parquet_path),
        session_id=parquet_path.stem
    )

    # Extract zone data (what the zone detail view needs)
    zones = result.low_utilization_zones + result.power_limited_zones

    # Simulate JSON serialization for zone view
    def serialize_zone_view():
        return {
            "low_utilization_zones": [z.to_dict() for z in result.low_utilization_zones],
            "power_limited_zones": [z.to_dict() for z in result.power_limited_zones]
        }

    json_result, json_elapsed = measure_time(serialize_zone_view)

    total_time = elapsed + json_elapsed

    print(f"   Analysis time: {elapsed:.3f}s")
    print(f"   Zone serialization: {json_elapsed:.3f}s")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Requirement: < 1.0s")

    if total_time < 1.0:
        print("   ✓ PASS")
        return True
    else:
        print("   ✗ FAIL")
        return False


def test_comparison_overlay(parquet_path):
    """Test: Comparison overlay loads in < 2 seconds"""
    print("\n5. Testing Comparison Overlay Performance...")

    # Get vehicle config
    vehicle = get_active_vehicle()
    max_lateral_g = getattr(vehicle, 'max_lateral_g', 1.3)
    max_braking_g = getattr(vehicle, 'max_braking_g', max_lateral_g * 1.1)
    power_limited_accel_g = getattr(vehicle, 'power_limited_accel_g', 0.4)

    # Create analyzer
    analyzer = GGAnalyzer(
        max_g_reference=max_lateral_g,
        max_braking_g=max_braking_g,
        power_limited_accel_g=power_limited_accel_g
    )

    # First, analyze to get lap numbers
    full_result = analyzer.analyze_from_parquet(
        str(parquet_path),
        session_id=parquet_path.stem
    )

    if not full_result.lap_numbers or len(full_result.lap_numbers) < 2:
        print("   ⚠ SKIP - Not enough laps for comparison testing")
        return True

    # Simulate comparison mode: analyze two different laps
    lap_a = full_result.lap_numbers[0]
    lap_b = full_result.lap_numbers[1] if len(full_result.lap_numbers) > 1 else lap_a

    # Measure time to analyze both laps
    start = time.perf_counter()

    result_a = analyzer.analyze_from_parquet(
        str(parquet_path),
        session_id=parquet_path.stem,
        lap_filter=lap_a
    )

    result_b = analyzer.analyze_from_parquet(
        str(parquet_path),
        session_id=parquet_path.stem,
        lap_filter=lap_b
    )

    # Simulate comparison JSON generation
    comparison_data = {
        "lap_a": result_a.to_dict(),
        "lap_b": result_b.to_dict()
    }

    elapsed = time.perf_counter() - start

    print(f"   Analyze lap {lap_a} + lap {lap_b}: {elapsed:.3f}s")
    print(f"   Requirement: < 2.0s")

    if elapsed < 2.0:
        print("   ✓ PASS")
        return True
    else:
        print("   ✗ FAIL")
        return False


def main():
    """Run all performance tests"""
    print("=" * 60)
    print("Phase 9: Performance Validation")
    print("=" * 60)

    # Find test data
    parquet_path = find_test_parquet()

    if parquet_path is None:
        print("\n✗ ERROR: No test Parquet file found")
        print("Please run the app and upload/process a session first")
        return False

    print(f"\nUsing test file: {parquet_path}")

    # Run all tests
    results = []

    results.append(test_session_overview(parquet_path))
    results.append(test_quadrant_deep_dive(parquet_path))
    results.append(test_zone_detail(parquet_path))
    results.append(test_comparison_overlay(parquet_path))
    results.append(test_lap_switching(parquet_path))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if all(results):
        print("\n✓ All performance requirements met!")
        return True
    else:
        print("\n✗ Some performance requirements not met")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
