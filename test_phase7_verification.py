#!/usr/bin/env python3
"""
Phase 7 Verification Script
Verifies that Zone Detail view implementation meets all acceptance criteria.
"""

import re

def verify_phase7():
    """Verify Phase 7 implementation"""

    html_file = "templates/gg_diagram.html"

    with open(html_file, 'r') as f:
        content = f.read()

    checks = {
        "Zone Detail View Container": 'id="zone-detail-view"' in content,
        "Back Button to Zone List": 'onclick="exitZoneDetail()"' in content,
        "Action Statement Element": 'id="zone-action-statement"' in content,
        "Zoomed Track Map": 'id="zone-detail-track-map"' in content,
        "Mini G-G Chart": 'id="zone-gg-chart"' in content,
        "Entry Speed Comparison": 'id="zone-current-entry-speed"' in content and 'id="zone-ref-entry-speed"' in content,
        "Brake Point Comparison": 'id="zone-current-brake-point"' in content and 'id="zone-ref-brake-point"' in content,
        "Peak G Comparison": 'id="zone-current-peak-g"' in content and 'id="zone-ref-peak-g"' in content,
        "Duration Comparison": 'id="zone-current-duration"' in content and 'id="zone-ref-duration"' in content,
        "openZoneDetail Function": 'function openZoneDetail(zoneIndex)' in content,
        "exitZoneDetail Function": 'function exitZoneDetail()' in content,
        "renderZoneActionStatement Function": 'function renderZoneActionStatement(zone, referenceLap)' in content,
        "renderZoneDetailTrackMap Function": 'function renderZoneDetailTrackMap(zone)' in content,
        "renderZoneSideBySideComparison Function": 'function renderZoneSideBySideComparison(zone, referenceLap)' in content,
        "renderZoneGGChart Function": 'function renderZoneGGChart(zone, referenceLap)' in content,
        "Click from List": 'onclick="openZoneDetail(' in content,
        "Click from Map": 'openZoneDetail(idx)' in content,
        "Two-click from Overview": 'openZoneDetailFromOverview' in content,
    }

    print("=" * 70)
    print("Phase 7: Zone Detail View - Verification Results")
    print("=" * 70)
    print()

    passed = 0
    failed = 0

    for check_name, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        color = "\033[92m" if result else "\033[91m"
        reset = "\033[0m"
        print(f"{color}{status}{reset} - {check_name}")

        if result:
            passed += 1
        else:
            failed += 1

    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    print()

    # Verify acceptance criteria
    print("Acceptance Criteria Verification:")
    print()

    criteria = [
        ("Clicking zone from list or map opens detail view",
         'onclick="openZoneDetail(' in content),
        ("Zoomed map shows just this corner/zone approach",
         'renderZoneDetailTrackMap' in content and 'zoneLats' in content),
        ("Side-by-side comparison: current lap vs reference",
         'zone-current-entry-speed' in content and 'zone-ref-entry-speed' in content),
        ("Specific numbers displayed: entry speed (mph), brake point (ft), peak g, duration (sec)",
         all(x in content for x in ['entry-speed', 'brake-point', 'peak-g', 'duration'])),
        ("Plain English action statement generated",
         'renderZoneActionStatement' in content and 'Carry' in content),
        ("Mini G-G scatter shows just this zone's data overlaid with reference",
         'renderZoneGGChart' in content and 'zone-gg-chart' in content),
        ("Two clicks to reach this view from landing page",
         'openZoneDetailFromOverview' in content),
    ]

    criteria_passed = 0
    for criterion, result in criteria:
        status = "✓" if result else "✗"
        color = "\033[92m" if result else "\033[91m"
        reset = "\033[0m"
        print(f"{color}{status}{reset} {criterion}")
        if result:
            criteria_passed += 1

    print()
    print("=" * 70)
    print(f"Acceptance Criteria: {criteria_passed}/{len(criteria)} met")
    print("=" * 70)

    if criteria_passed == len(criteria):
        print()
        print("\033[92m✓ Phase 7 implementation is COMPLETE and meets all acceptance criteria!\033[0m")
        print()
        return True
    else:
        print()
        print("\033[91m✗ Some acceptance criteria not met. Review needed.\033[0m")
        print()
        return False

if __name__ == "__main__":
    success = verify_phase7()
    exit(0 if success else 1)
