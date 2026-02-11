"""
Compliance tests for arch-003: Analyzer registry and plugin pattern.

Verifies all 10 acceptance criteria from docs/architecture/arch-003.md.
"""

import importlib
import inspect
import numpy as np
import pandas as pd
import pytest
import tempfile
import os
from dataclasses import fields


# ── AC-1: AnalyzerRegistry exists at src/features/registry.py ──


class TestAcceptanceCriteria:
    """Tests verifying each acceptance criterion from the design doc."""

    def test_ac1_registry_exists(self):
        """AC-1: AnalyzerRegistry exists at src/features/registry.py"""
        from src.features.registry import AnalyzerRegistry, AnalyzerRegistration, analyzer_registry
        assert AnalyzerRegistry is not None
        assert AnalyzerRegistration is not None
        assert analyzer_registry is not None
        assert isinstance(analyzer_registry, AnalyzerRegistry)

    def test_ac2_all_six_analyzers_registered(self):
        """AC-2: All 6 analyzers registered — registry.list_registered() returns 6 keys"""
        from src.features.registry import analyzer_registry
        # Ensure all modules are imported (triggers registration)
        import src.features  # noqa: F401

        registered = analyzer_registry.list_registered()
        assert len(registered) == 6, f"Expected 6, got {len(registered)}: {registered}"
        expected_keys = {"shifts", "laps", "gears", "power", "gg", "corners"}
        assert set(registered) == expected_keys

    def test_ac3_metadata_on_each_analyzer(self):
        """AC-3: Metadata on each analyzer — registry_key, required_channels, config_params"""
        from src.features.shift_analysis import ShiftAnalyzer
        from src.features.lap_analysis import LapAnalysis
        from src.features.gear_analysis import GearAnalysis
        from src.features.power_analysis import PowerAnalysis
        from src.features.gg_analysis import GGAnalyzer
        from src.features.corner_analysis import CornerAnalyzer

        analyzers = [ShiftAnalyzer, LapAnalysis, GearAnalysis,
                     PowerAnalysis, GGAnalyzer, CornerAnalyzer]

        for cls in analyzers:
            assert hasattr(cls, 'registry_key'), f"{cls.__name__} missing registry_key"
            assert hasattr(cls, 'required_channels'), f"{cls.__name__} missing required_channels"
            assert hasattr(cls, 'optional_channels'), f"{cls.__name__} missing optional_channels"
            assert hasattr(cls, 'config_params'), f"{cls.__name__} missing config_params"
            assert cls.registry_key is not None, f"{cls.__name__}.registry_key is None"
            assert isinstance(cls.required_channels, list), f"{cls.__name__}.required_channels not a list"

    def test_ac4_analyze_from_channels_method(self):
        """AC-4: analyze_from_channels() method on each analyzer accepts SessionChannels"""
        from src.features.shift_analysis import ShiftAnalyzer
        from src.features.lap_analysis import LapAnalysis
        from src.features.gear_analysis import GearAnalysis
        from src.features.power_analysis import PowerAnalysis
        from src.features.gg_analysis import GGAnalyzer
        from src.features.corner_analysis import CornerAnalyzer

        analyzers = [ShiftAnalyzer, LapAnalysis, GearAnalysis,
                     PowerAnalysis, GGAnalyzer, CornerAnalyzer]

        for cls in analyzers:
            assert hasattr(cls, 'analyze_from_channels'), (
                f"{cls.__name__} missing analyze_from_channels"
            )
            # Should be callable (a method)
            instance = cls.__new__(cls)
            assert callable(getattr(instance, 'analyze_from_channels'))

    def test_ac5_session_report_uses_registry(self):
        """AC-5: SessionReportGenerator uses registry — no hardcoded analyzer imports in generate method"""
        import src.features.session_report as mod
        source = inspect.getsource(mod.SessionReportGenerator)

        # Should NOT have hardcoded _run_lap_analysis, _run_shift_analysis, etc.
        assert '_run_lap_analysis' not in source
        assert '_run_shift_analysis' not in source
        assert '_run_gear_analysis' not in source
        assert '_run_power_analysis' not in source

        # Should reference analyzer_registry
        assert 'analyzer_registry' in source

    def test_ac6_dynamic_report_assembly(self):
        """AC-6: Dynamic report assembly — SessionReport.sub_reports dict keyed by registry_key"""
        from src.features.session_report import SessionReport, SessionMetadata, SessionSummary

        # SessionReport should have sub_reports field
        field_names = [f.name for f in fields(SessionReport)]
        assert 'sub_reports' in field_names

        # sub_reports should be a dict
        meta = SessionMetadata(session_id="test", track_name="test", vehicle_setup="test",
                               analysis_timestamp="", data_source="test",
                               total_duration_seconds=0.0, sample_count=0)
        summary = SessionSummary(total_laps=0, fastest_lap_time=0.0,
                                 fastest_lap_number=0, average_lap_time=0.0,
                                 total_shifts=0, max_speed_mph=0.0,
                                 max_rpm=0.0, max_power_hp=0.0,
                                 max_braking_g=0.0, improvement_trend="")
        report = SessionReport(metadata=meta, summary=summary)
        assert isinstance(report.sub_reports, dict)

    def test_ac7_backward_compatible(self):
        """AC-7: Backward compatible — report.lap_analysis, report.shift_analysis still work"""
        from src.features.session_report import SessionReport, SessionMetadata, SessionSummary

        meta = SessionMetadata(session_id="test", track_name="test", vehicle_setup="test",
                               analysis_timestamp="", data_source="test",
                               total_duration_seconds=0.0, sample_count=0)
        summary = SessionSummary(total_laps=0, fastest_lap_time=0.0,
                                 fastest_lap_number=0, average_lap_time=0.0,
                                 total_shifts=0, max_speed_mph=0.0,
                                 max_rpm=0.0, max_power_hp=0.0,
                                 max_braking_g=0.0, improvement_trend="")

        # Create a report with sub_reports
        report = SessionReport(
            metadata=meta, summary=summary,
            sub_reports={"laps": "mock_lap_report", "shifts": "mock_shift_report"},
        )

        # Backward-compat typed fields should be synced via __post_init__
        assert report.lap_analysis == "mock_lap_report"
        assert report.shift_analysis == "mock_shift_report"

    def test_ac8_new_analyzer_registration_trivial(self):
        """AC-8: New analyzer registration is trivial — only create class + register"""
        from src.features.base_analyzer import BaseAnalyzer
        from src.features.registry import AnalyzerRegistry

        # Create a fresh registry for isolation
        test_registry = AnalyzerRegistry()

        class DummyAnalyzer(BaseAnalyzer):
            registry_key = "dummy"
            required_channels = ["speed"]
            optional_channels = []
            config_params = []

            def analyze_from_parquet(self, parquet_path, **kwargs):
                pass

        test_registry.register(DummyAnalyzer)

        assert "dummy" in test_registry
        assert test_registry.get("dummy").analyzer_class is DummyAnalyzer
        assert len(test_registry) == 1

    def test_ac9_all_existing_tests_pass(self):
        """AC-9: All existing tests pass — verified by test suite (meta-check)"""
        # This is verified by the full test suite running without regression.
        # Here we just confirm the critical modules import cleanly.
        from src.features import (
            ShiftAnalyzer, LapAnalysis, GearAnalysis,
            PowerAnalysis, SessionReportGenerator, SessionReport,
            analyzer_registry,
        )
        assert len(analyzer_registry) == 6

    def test_ac10_trace_cross_validation_dynamic(self):
        """AC-10: Trace cross-validation works dynamically — iterates over registered results"""
        import src.features.session_report as mod
        source = inspect.getsource(mod.SessionReportGenerator._run_cross_validation_checks)

        # Should iterate over sub_reports, not hardcode analyzer names
        assert 'sub_reports' in source or 'report.sub_reports' in source


# ── Integration tests ──


class TestRegistryIntegration:
    """Integration tests for the registry pattern working end-to-end."""

    @pytest.fixture
    def sample_parquet(self):
        """Create sample Parquet file with channels for multiple analyzers."""
        n = 1000
        time = np.linspace(0, 100, n)
        lat = 43.797875 + 0.005 * np.sin(time / 25)
        lon = -87.989638 + 0.005 * np.cos(time / 25)
        rpm = 4000 + 2000 * np.sin(time / 20)
        speed = 60 + 30 * np.sin(time / 15)
        lat_acc = 0.5 * np.sin(time / 10)
        lon_acc = 0.3 * np.cos(time / 12)

        df = pd.DataFrame({
            "GPS Latitude": lat,
            "GPS Longitude": lon,
            "RPM": rpm,
            "GPS Speed": speed,
            "GPS_LatAcc": lat_acc,
            "GPS_LonAcc": lon_acc,
        }, index=time)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name)
            yield f.name

        try:
            os.unlink(f.name)
        except Exception:
            pass

    def test_generate_report_runs_all_available_analyzers(self, sample_parquet):
        """Session report should run all analyzers whose channels are available."""
        from src.features.session_report import SessionReportGenerator

        gen = SessionReportGenerator()
        report = gen.generate_from_parquet(sample_parquet)

        # With lat, lon, speed, rpm, lat_acc, lon_acc — all 6 should run
        assert len(report.sub_reports) == 6
        expected = {"shifts", "laps", "gears", "power", "gg", "corners"}
        assert set(report.sub_reports.keys()) == expected

    def test_generate_report_skips_missing_channels(self):
        """Analyzers whose required channels are missing should be skipped."""
        from src.features.session_report import SessionReportGenerator

        # Only speed — no GPS, no RPM, no acc
        n = 500
        time = np.linspace(0, 50, n)
        speed = 60 + 30 * np.sin(time / 15)

        df = pd.DataFrame({"GPS Speed": speed}, index=time)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name)

        try:
            gen = SessionReportGenerator()
            report = gen.generate_from_parquet(f.name)

            # Only power should run (requires just speed)
            assert "power" in report.sub_reports
            # These need rpm or gps, should be skipped
            assert "shifts" not in report.sub_reports
            assert "laps" not in report.sub_reports
            assert "gears" not in report.sub_reports
            assert "gg" not in report.sub_reports
            assert "corners" not in report.sub_reports
        finally:
            os.unlink(f.name)

    def test_registry_create_instance(self):
        """Registry can create analyzer instances with config params."""
        from src.features.registry import analyzer_registry
        import src.features  # noqa: F401

        instance = analyzer_registry.create_instance("power", vehicle_mass_kg=1500)
        from src.features.power_analysis import PowerAnalysis
        assert isinstance(instance, PowerAnalysis)

    def test_registry_reset_for_test_isolation(self):
        """Registry.reset() clears all registrations for test isolation."""
        from src.features.registry import AnalyzerRegistry

        reg = AnalyzerRegistry()
        from src.features.base_analyzer import BaseAnalyzer

        class Temp(BaseAnalyzer):
            registry_key = "temp"
            required_channels = []
            optional_channels = []
            config_params = []
            def analyze_from_parquet(self, p, **kw): pass

        reg.register(Temp)
        assert len(reg) == 1
        reg.reset()
        assert len(reg) == 0

    def test_backward_compat_typed_fields_synced(self, sample_parquet):
        """Backward-compat typed fields on SessionReport are synced from sub_reports."""
        from src.features.session_report import SessionReportGenerator

        gen = SessionReportGenerator()
        report = gen.generate_from_parquet(sample_parquet)

        # Typed fields should match sub_reports entries
        if "laps" in report.sub_reports:
            assert report.lap_analysis is report.sub_reports["laps"]
        if "shifts" in report.sub_reports:
            assert report.shift_analysis is report.sub_reports["shifts"]
        if "gears" in report.sub_reports:
            assert report.gear_analysis is report.sub_reports["gears"]
        if "power" in report.sub_reports:
            assert report.power_analysis is report.sub_reports["power"]

    def test_to_dict_includes_all_sub_reports(self, sample_parquet):
        """SessionReport.to_dict() includes all sub-reports dynamically."""
        from src.features.session_report import SessionReportGenerator

        gen = SessionReportGenerator()
        report = gen.generate_from_parquet(sample_parquet)
        d = report.to_dict()

        # All sub_reports should appear in the dict
        for key in report.sub_reports:
            assert key in d or any(key in str(v) for v in d.values()), (
                f"sub_report '{key}' not in to_dict() output"
            )

    def test_trace_records_dynamic_intermediates(self, sample_parquet):
        """Trace intermediates reflect dynamically discovered analyzers."""
        from src.features.session_report import SessionReportGenerator

        gen = SessionReportGenerator()
        report = gen.generate_from_parquet(sample_parquet, include_trace=True)

        intermediates = report.trace.intermediates
        assert "sub_analyzers_run" in intermediates
        assert intermediates["sub_analyzers_run"] == len(report.sub_reports)

        # Each registered analyzer should have a *_analysis_ok intermediate
        from src.features.registry import analyzer_registry
        for key in analyzer_registry.list_registered():
            assert f"{key}_analysis_ok" in intermediates
