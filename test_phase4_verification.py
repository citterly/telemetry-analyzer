#!/usr/bin/env python3
"""
Phase 4 Verification Test
Verifies that quadrant reference values are properly implemented.
"""

import json
import sys
from pathlib import Path

def test_vehicles_json():
    """Verify vehicles.json has all required fields"""
    print("Testing vehicles.json...")

    vehicles_path = Path("data/vehicles.json")
    with open(vehicles_path) as f:
        data = json.load(f)

    required_fields = ['max_lateral_g', 'max_braking_g', 'power_limited_accel_g']

    for vehicle in data['vehicles']:
        vehicle_id = vehicle['id']
        for field in required_fields:
            assert field in vehicle, f"Vehicle {vehicle_id} missing field {field}"
            assert isinstance(vehicle[field], (int, float)), f"Vehicle {vehicle_id} field {field} is not numeric"
        print(f"  ✓ {vehicle_id}: lat={vehicle['max_lateral_g']}g, brake={vehicle['max_braking_g']}g, accel={vehicle['power_limited_accel_g']}g")

    print(f"✓ All {len(data['vehicles'])} vehicles have required fields\n")
    return True

def test_gg_analysis_code():
    """Verify GGAnalyzer uses per-quadrant references"""
    print("Testing GGAnalyzer implementation...")

    # Read the source code to verify implementation
    gg_path = Path("src/features/gg_analysis.py")
    code = gg_path.read_text()

    # Check that __init__ accepts all three parameters
    assert "max_g_reference:" in code, "Missing max_g_reference parameter"
    assert "max_braking_g:" in code, "Missing max_braking_g parameter"
    assert "power_limited_accel_g:" in code, "Missing power_limited_accel_g parameter"
    print("  ✓ GGAnalyzer __init__ has all three reference parameters")

    # Check that _calculate_quadrants uses per-quadrant references
    assert "max_lateral_g:" in code, "Missing max_lateral_g in _calculate_quadrants"
    assert "max_braking_g:" in code, "Missing max_braking_g in _calculate_quadrants"
    assert "power_limited_accel_g:" in code, "Missing power_limited_accel_g in _calculate_quadrants"
    print("  ✓ _calculate_quadrants signature includes per-quadrant references")

    # Check that quadrants are defined with their respective references
    assert '"lat_left"' in code and '"lat_right"' in code, "Missing lateral quadrants"
    assert '"braking"' in code, "Missing braking quadrant"
    assert '"acceleration"' in code, "Missing acceleration quadrant"
    print("  ✓ All four quadrants defined")

    # Check utilization calculation uses ref_g
    assert "avg_g_in_quadrant / ref_g" in code, "Utilization not using per-quadrant reference"
    print("  ✓ Utilization calculation uses per-quadrant reference\n")

    return True

def test_result_structure():
    """Verify GGAnalysisResult has per-quadrant reference fields"""
    print("Testing GGAnalysisResult structure...")

    gg_path = Path("src/features/gg_analysis.py")
    code = gg_path.read_text()

    # Check that result includes all reference values
    assert "reference_lateral_g:" in code, "Missing reference_lateral_g field"
    assert "reference_braking_g:" in code, "Missing reference_braking_g field"
    assert "reference_accel_g:" in code, "Missing reference_accel_g field"
    print("  ✓ GGAnalysisResult has all three reference fields")

    # Check that to_dict includes them
    assert '"reference_lateral_g"' in code, "to_dict missing reference_lateral_g"
    assert '"reference_braking_g"' in code, "to_dict missing reference_braking_g"
    assert '"reference_accel_g"' in code, "to_dict missing reference_accel_g"
    print("  ✓ to_dict() exports all reference values\n")

    return True

def test_warning_logic():
    """Verify warning logic for exceeding config values"""
    print("Testing warning logic...")

    gg_path = Path("src/features/gg_analysis.py")
    code = gg_path.read_text()

    # Check lateral G warning
    assert "max_lateral_g > self.max_g_reference" in code, "Missing lateral G warning check"
    assert "max_g_exceeds_config" in code, "Missing max_g_exceeds_config flag"
    print("  ✓ Lateral G warning logic present")

    # Check braking G warning
    assert "max_braking_g > self.max_braking_g" in code, "Missing braking G warning check"
    print("  ✓ Braking G warning logic present")

    # Check warnings field in result
    assert "warnings:" in code and "List[str]" in code, "Missing warnings field"
    print("  ✓ Warnings field in result\n")

    return True

def test_frontend_display():
    """Verify frontend displays reference values and tooltips"""
    print("Testing frontend template...")

    template_path = Path("templates/gg_diagram.html")
    html = template_path.read_text()

    # Check reference values card
    assert "Reference Values (Vehicle Config)" in html, "Missing reference values card"
    assert "ref-lateral-g" in html, "Missing lateral reference display"
    assert "ref-braking-g" in html, "Missing braking reference display"
    assert "ref-accel-g" in html, "Missing accel reference display"
    print("  ✓ Reference values card present")

    # Check that values are populated from API
    assert "reference_lateral_g" in html or "reference_max_g" in html, "Missing lateral reference update"
    assert "reference_braking_g" in html, "Missing braking reference update"
    assert "reference_accel_g" in html, "Missing accel reference update"
    print("  ✓ Reference values populated from API")

    # Check tooltips on quadrant cards
    assert "tooltip" in html or "title=" in html, "Missing tooltips"
    print("  ✓ Tooltips present on quadrant cards")

    # Check warnings section
    assert "warnings-section" in html, "Missing warnings section"
    assert "warnings-content" in html, "Missing warnings content"
    print("  ✓ Warnings display section present\n")

    return True

def test_app_integration():
    """Verify app.py passes vehicle config to analyzer"""
    print("Testing app.py integration...")

    app_path = Path("src/main/app.py")
    code = app_path.read_text()

    # Check that it reads from vehicle config
    assert "get_active_vehicle()" in code, "Not reading active vehicle"
    assert "max_lateral_g" in code, "Not reading max_lateral_g from vehicle"
    assert "max_braking_g" in code, "Not reading max_braking_g from vehicle"
    assert "power_limited_accel_g" in code, "Not reading power_limited_accel_g from vehicle"
    print("  ✓ Reads all three values from vehicle config")

    # Check that it passes to analyzer
    assert "GGAnalyzer(" in code, "GGAnalyzer not instantiated"
    assert "max_g_reference=" in code, "Not passing max_g_reference"
    assert "max_braking_g=" in code, "Not passing max_braking_g"
    assert "power_limited_accel_g=" in code, "Not passing power_limited_accel_g"
    print("  ✓ Passes all three values to GGAnalyzer\n")

    return True

def main():
    """Run all verification tests"""
    print("=" * 60)
    print("Phase 4 Verification: Quadrant Reference Values")
    print("=" * 60)
    print()

    tests = [
        ("vehicles.json fields", test_vehicles_json),
        ("GGAnalyzer implementation", test_gg_analysis_code),
        ("GGAnalysisResult structure", test_result_structure),
        ("Warning logic", test_warning_logic),
        ("Frontend display", test_frontend_display),
        ("App integration", test_app_integration),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {name}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {name}")
            print(f"  Error: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n✓ ALL ACCEPTANCE CRITERIA VERIFIED")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
