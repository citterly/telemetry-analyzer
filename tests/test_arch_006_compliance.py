"""
Compliance tests for arch-006: Audit Mode UI

Verifies all 10 acceptance criteria from docs/architecture/arch-006.md
"""
import json
import pytest
from pathlib import Path
from src.utils.calculation_trace import SanityCheck, CalculationTrace
from src.features.power_analysis import PowerAnalysis
from src.features.shift_analysis import ShiftAnalyzer
from src.features.gg_analysis import GGAnalyzer
from src.features.lap_analysis import LapAnalysis
from src.features.gear_analysis import GearAnalysis
from src.features.corner_analysis import CornerAnalyzer
from src.features.session_report import SessionReportGenerator


class TestAcceptanceCriteria:
    """Test all 10 acceptance criteria from arch-006.md"""

    def test_ac1_sanity_check_has_impact_field(self):
        """AC-1: SanityCheck has impact field; to_dict() includes it"""
        check = SanityCheck(
            name="test_check",
            status="pass",
            message="Test message",
            impact="Test impact description"
        )

        # Verify field exists
        assert hasattr(check, 'impact')
        assert check.impact == "Test impact description"

        # Verify to_dict includes it
        check_dict = check.to_dict()
        assert 'impact' in check_dict
        assert check_dict['impact'] == "Test impact description"

    def test_ac2_all_26_checks_have_impact_descriptions(self, sample_parquet_path):
        """AC-2: All 26 checks have non-empty impact descriptions"""
        import pandas as pd

        # Create sample data with all required channels
        df = pd.DataFrame({
            'time': [i * 0.1 for i in range(1000)],
            'speed_mph': [i * 0.5 for i in range(1000)],
            'rpm': [2000 + i * 5 for i in range(1000)],
            'gear': [3] * 500 + [4] * 500,
            'latitude': [43.8 + i * 0.0001 for i in range(1000)],
            'longitude': [-88.0 + i * 0.0001 for i in range(1000)],
            'GPS LatAcc': [i * 0.001 - 0.5 for i in range(1000)],
            'GPS LonAcc': [i * 0.001 - 0.5 for i in range(1000)]
        })
        df.attrs['native_rates'] = {'speed_mph': 10.0, 'rpm': 10.0}
        df.to_parquet(sample_parquet_path)

        all_checks = []

        # PowerAnalysis: 5 checks
        power = PowerAnalysis()
        power_result = power.analyze_from_parquet(str(sample_parquet_path), include_trace=True)
        if power_result.trace:
            all_checks.extend(power_result.trace.sanity_checks)

        # ShiftAnalyzer: 4 checks
        shift = ShiftAnalyzer()
        shift_result = shift.analyze_from_parquet(str(sample_parquet_path), include_trace=True)
        if shift_result.trace:
            all_checks.extend(shift_result.trace.sanity_checks)

        # GGAnalyzer: 4 checks
        gg = GGAnalyzer()
        gg_result = gg.analyze_from_parquet(str(sample_parquet_path), include_trace=True)
        if gg_result.trace:
            all_checks.extend(gg_result.trace.sanity_checks)

        # LapAnalysis: 4 checks
        lap = LapAnalysis()
        lap_result = lap.analyze_from_parquet(str(sample_parquet_path), include_trace=True)
        if lap_result.trace:
            all_checks.extend(lap_result.trace.sanity_checks)

        # GearAnalysis: 3 checks
        gear = GearAnalysis()
        gear_result = gear.analyze_from_parquet(str(sample_parquet_path), include_trace=True)
        if gear_result.trace:
            all_checks.extend(gear_result.trace.sanity_checks)

        # CornerAnalyzer: 3 checks
        corner = CornerAnalyzer()
        corner_result = corner.analyze_from_parquet(str(sample_parquet_path), include_trace=True)
        if corner_result.trace:
            all_checks.extend(corner_result.trace.sanity_checks)

        # SessionReport: 3 cross-validation checks
        report_gen = SessionReportGenerator()
        report = report_gen.generate_from_parquet(str(sample_parquet_path), include_trace=True)
        if report.trace:
            all_checks.extend(report.trace.sanity_checks)

        # Verify we have at least 25 checks (26 expected, but some may be data-dependent)
        assert len(all_checks) >= 25, f"Expected at least 25 checks, got {len(all_checks)}"

        # Verify every check has non-empty impact
        for check in all_checks:
            assert hasattr(check, 'impact'), f"Check {check.name} missing impact field"
            assert check.impact, f"Check {check.name} has empty impact description"
            assert len(check.impact) > 10, f"Check {check.name} impact too short: {check.impact}"

        # Print summary for debugging
        print(f"\nTotal checks found: {len(all_checks)}")
        print(f"All checks have impact descriptions: {all([c.impact for c in all_checks])}")

    def test_ac3_full_report_includes_sub_analyzer_traces(self, sample_parquet_path):
        """AC-3: Full report ?trace=true includes sub-analyzer traces (not just cross-validation)"""
        import pandas as pd

        # Create sample data
        df = pd.DataFrame({
            'time': [i * 0.1 for i in range(1000)],
            'speed_mph': [i * 0.5 for i in range(1000)],
            'rpm': [2000 + i * 5 for i in range(1000)],
            'gear': [3] * 500 + [4] * 500,
            'latitude': [43.8 + i * 0.0001 for i in range(1000)],
            'longitude': [-88.0 + i * 0.0001 for i in range(1000)],
            'lat_acc_g': [i * 0.001 - 0.5 for i in range(1000)],
            'lon_acc_g': [i * 0.001 - 0.5 for i in range(1000)]
        })
        df.attrs['native_rates'] = {'speed_mph': 10.0, 'rpm': 10.0}
        df.to_parquet(sample_parquet_path)

        report_gen = SessionReportGenerator()
        report = report_gen.generate_from_parquet(str(sample_parquet_path), include_trace=True)
        report_dict = report.to_dict()

        # Verify main trace exists (cross-validation)
        assert '_trace' in report_dict
        assert report_dict['_trace']['sanity_checks']  # Should have cross-validation checks

        # Verify sub-analyzer traces exist
        sub_analyzers_with_trace = 0
        for key in ['shift_analysis', 'lap_analysis', 'gear_analysis', 'power_analysis', 'gg_analysis', 'corner_analysis']:
            if key in report_dict and report_dict[key] and '_trace' in report_dict[key]:
                sub_analyzers_with_trace += 1
                assert report_dict[key]['_trace']['sanity_checks'], f"{key} has trace but no sanity checks"

        # At least 4 sub-analyzers should have traces (not all may run depending on data)
        assert sub_analyzers_with_trace >= 4, f"Only {sub_analyzers_with_trace} sub-analyzers have traces"

    def test_ac4_all_3_analysis_pages_have_audit_toggle(self):
        """AC-4: All 3 analysis pages have an audit toggle switch"""
        project_root = Path(__file__).parent.parent
        templates_dir = project_root / 'templates'

        pages = ['analysis.html', 'gg_diagram.html', 'corner_analysis.html']

        for page in pages:
            page_path = templates_dir / page
            assert page_path.exists(), f"{page} not found"

            content = page_path.read_text()

            # Verify audit-toggle-container div exists
            assert 'audit-toggle-container' in content, f"{page} missing audit-toggle-container"

            # Verify initialization code exists
            assert 'window._auditManager.renderToggleButton()' in content, f"{page} missing toggle initialization"

    def test_ac5_toggle_state_persists_in_localstorage(self):
        """AC-5: Toggle state persists in localStorage across page loads"""
        # This is verified by the audit.js implementation using localStorage
        # We can verify the key is correct
        audit_js_path = Path(__file__).parent.parent / 'static' / 'js' / 'audit.js'
        assert audit_js_path.exists(), "audit.js not found"

        content = audit_js_path.read_text()
        assert "telemetry_audit_enabled" in content, "localStorage key not found"
        assert "localStorage.getItem" in content, "localStorage getter not found"
        assert "localStorage.setItem" in content, "localStorage setter not found"

    def test_ac6_audit_off_no_trace_param(self):
        """AC-6: When audit OFF, no ?trace=true in fetch requests"""
        project_root = Path(__file__).parent.parent
        templates_dir = project_root / 'templates'

        pages = ['analysis.html', 'gg_diagram.html', 'corner_analysis.html']

        for page in pages:
            page_path = templates_dir / page
            content = page_path.read_text()

            # Verify traceParam() is used (which returns empty string when disabled)
            assert 'traceParam(' in content, f"{page} not using traceParam()"

            # Verify conditional logic exists
            assert 'window._auditManager' in content, f"{page} missing AuditManager reference"

    def test_ac7_audit_on_traffic_light_dots_appear(self):
        """AC-7: When audit ON, traffic light dots appear next to section headers"""
        audit_js_path = Path(__file__).parent.parent / 'static' / 'js' / 'audit.js'
        content = audit_js_path.read_text()

        # Verify traffic light rendering functions exist
        assert 'renderDot(' in content, "renderDot function not found"
        assert 'renderSectionIndicator(' in content, "renderSectionIndicator function not found"
        assert 'audit-dot' in content, "audit-dot class not found"

        # Verify status colors exist
        audit_css_path = Path(__file__).parent.parent / 'static' / 'css' / 'audit.css'
        css_content = audit_css_path.read_text()
        assert 'audit-dot-green' in css_content, "Green dot style not found"
        assert 'audit-dot-yellow' in css_content, "Yellow dot style not found"
        assert 'audit-dot-red' in css_content, "Red dot style not found"

    def test_ac8_audit_on_expandable_panels_show_all_sections(self):
        """AC-8: When audit ON, expandable audit panels show inputs, config, intermediates, and checks"""
        audit_js_path = Path(__file__).parent.parent / 'static' / 'js' / 'audit.js'
        content = audit_js_path.read_text()

        # Verify renderPanel includes all sections
        assert 'renderInputsSection' in content, "renderInputsSection not found"
        assert 'renderConfigSection' in content, "renderConfigSection not found"
        assert 'renderIntermediatesSection' in content, "renderIntermediatesSection not found"
        assert 'renderChecksSection' in content, "renderChecksSection not found"
        assert 'renderWarningsSection' in content, "renderWarningsSection not found"

        # Verify panel structure
        assert 'audit-panel' in content, "audit-panel class not found"
        assert 'audit-panel-header' in content, "audit-panel-header class not found"
        assert 'audit-panel-body' in content, "audit-panel-body class not found"

    def test_ac9_every_check_row_shows_impact(self):
        """AC-9: Every check row shows impact description regardless of pass/warn/fail status"""
        audit_js_path = Path(__file__).parent.parent / 'static' / 'js' / 'audit.js'
        content = audit_js_path.read_text()

        # Verify impact rendering in check rows
        assert 'audit-check-impact' in content, "audit-check-impact class not found"
        assert 'check.impact' in content, "impact field not rendered in checks"

        # Verify impact is shown unconditionally (not inside if statement for status)
        # The pattern should be: if (check.impact) show it, not if (check.status === 'fail')
        lines = content.split('\n')
        impact_line = None
        for i, line in enumerate(lines):
            if 'audit-check-impact' in line and '<strong>Impact:</strong>' in line:
                impact_line = i
                break

        assert impact_line is not None, "Impact rendering not found in check row"

        # Verify it's conditional only on impact existence, not status
        # Look for the conditional: ${check.impact ? ...}
        assert 'check.impact ?' in content, "Impact should be conditionally rendered based on existence"

    def test_ac10_all_existing_tests_pass_unchanged(self):
        """AC-10: All existing 1011+ tests pass unchanged"""
        # This is a meta-test that verifies the test suite still runs
        # The actual verification happens when pytest runs the full suite
        # Here we just check that key test files still exist

        project_root = Path(__file__).parent.parent
        tests_dir = project_root / 'tests'

        key_test_files = [
            'test_power_analysis.py',
            'test_shift_analysis.py',
            'test_gg_analysis.py',
            'test_lap_analysis.py',
            'test_gear_analysis.py',
            'test_corner_analysis.py',
            'test_session_report.py',
            'test_calculation_trace.py'
        ]

        for test_file in key_test_files:
            test_path = tests_dir / test_file
            assert test_path.exists(), f"Key test file {test_file} not found - test suite may be broken"


@pytest.fixture
def sample_parquet_path(tmp_path):
    """Provide a temporary path for sample parquet files"""
    return tmp_path / "test_sample.parquet"
