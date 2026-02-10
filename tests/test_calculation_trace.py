"""Tests for safeguard-001: Trace infrastructure.

Tests the CalculationTrace and SanityCheck dataclasses, BaseAnalysisReport
trace integration, and BaseAnalyzer._create_trace() factory.
"""

import json
from dataclasses import dataclass, field
from typing import Optional

import pytest

from src.utils.calculation_trace import CalculationTrace, SanityCheck
from src.features.base_analyzer import BaseAnalyzer, BaseAnalysisReport


class TestSanityCheck:
    """Tests for SanityCheck dataclass."""

    def test_creation_minimal(self):
        check = SanityCheck(name="test_check", status="pass", message="OK")
        assert check.name == "test_check"
        assert check.status == "pass"
        assert check.message == "OK"
        assert check.expected is None
        assert check.actual is None
        assert check.severity == "warning"

    def test_creation_full(self):
        check = SanityCheck(
            name="power_plausible",
            status="fail",
            message="800+ HP from a Miata",
            expected=150,
            actual=850,
            severity="error",
        )
        assert check.expected == 150
        assert check.actual == 850
        assert check.severity == "error"

    def test_to_dict_minimal(self):
        check = SanityCheck(name="x", status="pass", message="ok")
        d = check.to_dict()
        assert d == {
            "name": "x",
            "status": "pass",
            "message": "ok",
            "severity": "warning",
        }
        assert "expected" not in d
        assert "actual" not in d

    def test_to_dict_full(self):
        check = SanityCheck(
            name="mass_check", status="warn", message="10% off",
            expected=1565.0, actual=1720.0, severity="warning",
        )
        d = check.to_dict()
        assert d["expected"] == 1565.0
        assert d["actual"] == 1720.0

    def test_to_dict_json_serializable(self):
        check = SanityCheck(
            name="test", status="pass", message="ok",
            expected=42, actual=42.0,
        )
        serialized = json.dumps(check.to_dict())
        assert isinstance(serialized, str)


class TestCalculationTrace:
    """Tests for CalculationTrace dataclass."""

    def test_creation(self):
        trace = CalculationTrace(
            analyzer_name="PowerAnalysis",
            timestamp="2026-02-10T12:00:00+00:00",
        )
        assert trace.analyzer_name == "PowerAnalysis"
        assert trace.inputs == {}
        assert trace.config == {}
        assert trace.intermediates == {}
        assert trace.sanity_checks == []
        assert trace.warnings == []

    def test_record_input(self):
        trace = CalculationTrace(analyzer_name="Test", timestamp="t")
        trace.record_input("speed_column", "GPS Speed")
        trace.record_input("sample_count", 9780)
        assert trace.inputs == {"speed_column": "GPS Speed", "sample_count": 9780}

    def test_record_config(self):
        trace = CalculationTrace(analyzer_name="Test", timestamp="t")
        trace.record_config("vehicle_mass_kg", 1565.0)
        assert trace.config["vehicle_mass_kg"] == 1565.0

    def test_record_intermediate(self):
        trace = CalculationTrace(analyzer_name="Test", timestamp="t")
        trace.record_intermediate("max_raw_power_hp", 342.5)
        assert trace.intermediates["max_raw_power_hp"] == 342.5

    def test_add_check(self):
        trace = CalculationTrace(analyzer_name="Test", timestamp="t")
        trace.add_check("check_1", "pass", "OK")
        trace.add_check("check_2", "fail", "Bad", expected=1.2, actual=3.0, severity="error")
        assert len(trace.sanity_checks) == 2
        assert trace.sanity_checks[0].name == "check_1"
        assert trace.sanity_checks[1].severity == "error"

    def test_has_failures_false(self):
        trace = CalculationTrace(analyzer_name="Test", timestamp="t")
        trace.add_check("a", "pass", "OK")
        trace.add_check("b", "warn", "Hmm")
        assert trace.has_failures is False

    def test_has_failures_true(self):
        trace = CalculationTrace(analyzer_name="Test", timestamp="t")
        trace.add_check("a", "pass", "OK")
        trace.add_check("b", "fail", "Bad")
        assert trace.has_failures is True

    def test_has_warnings_false(self):
        trace = CalculationTrace(analyzer_name="Test", timestamp="t")
        trace.add_check("a", "pass", "OK")
        assert trace.has_warnings is False

    def test_has_warnings_true(self):
        trace = CalculationTrace(analyzer_name="Test", timestamp="t")
        trace.add_check("a", "warn", "Hmm")
        assert trace.has_warnings is True

    def test_has_failures_empty(self):
        trace = CalculationTrace(analyzer_name="Test", timestamp="t")
        assert trace.has_failures is False
        assert trace.has_warnings is False

    def test_to_dict(self):
        trace = CalculationTrace(
            analyzer_name="PowerAnalysis",
            timestamp="2026-02-10T12:00:00+00:00",
        )
        trace.record_input("speed_column", "GPS Speed")
        trace.record_config("mass_kg", 1565.0)
        trace.record_intermediate("max_hp", 342.5)
        trace.add_check("plausible", "pass", "OK")
        trace.warnings.append("RPM column missing")

        d = trace.to_dict()
        assert d["analyzer_name"] == "PowerAnalysis"
        assert d["timestamp"] == "2026-02-10T12:00:00+00:00"
        assert d["inputs"] == {"speed_column": "GPS Speed"}
        assert d["config"] == {"mass_kg": 1565.0}
        assert d["intermediates"] == {"max_hp": 342.5}
        assert len(d["sanity_checks"]) == 1
        assert d["sanity_checks"][0]["name"] == "plausible"
        assert d["warnings"] == ["RPM column missing"]
        assert d["has_failures"] is False
        assert d["has_warnings"] is False

    def test_to_dict_json_serializable(self):
        trace = CalculationTrace(analyzer_name="Test", timestamp="t")
        trace.record_input("count", 100)
        trace.record_config("mass", 1565.0)
        trace.record_intermediate("power", 342.5)
        trace.add_check("ok", "pass", "fine", expected=1.0, actual=1.0)
        serialized = json.dumps(trace.to_dict())
        roundtrip = json.loads(serialized)
        assert roundtrip["analyzer_name"] == "Test"
        assert len(roundtrip["sanity_checks"]) == 1


class TestBaseAnalysisReportTrace:
    """Tests for trace integration in BaseAnalysisReport."""

    def test_trace_dict_without_trace(self):
        """_trace_dict returns empty dict when no trace is set."""
        @dataclass
        class TestReport(BaseAnalysisReport):
            value: int = 0
            def to_dict(self):
                result = {"value": self.value}
                result.update(self._trace_dict())
                return result

        report = TestReport(value=42)
        assert report._trace_dict() == {}
        assert "_trace" not in report.to_dict()

    def test_trace_dict_with_trace(self):
        """_trace_dict returns trace when set on the report."""
        @dataclass
        class TestReport(BaseAnalysisReport):
            value: int = 0
            def to_dict(self):
                result = {"value": self.value}
                result.update(self._trace_dict())
                return result

        report = TestReport(value=42)
        trace = CalculationTrace(analyzer_name="Test", timestamp="t")
        trace.add_check("ok", "pass", "fine")
        report.trace = trace

        d = report.to_dict()
        assert "_trace" in d
        assert d["_trace"]["analyzer_name"] == "Test"
        assert len(d["_trace"]["sanity_checks"]) == 1

    def test_trace_dict_json_serializable_with_trace(self):
        """Full to_dict with trace can be JSON serialized."""
        @dataclass
        class TestReport(BaseAnalysisReport):
            value: float = 0.0
            def to_dict(self):
                result = {"value": self.value}
                result.update(self._trace_dict())
                return result

        report = TestReport(value=3.14)
        trace = CalculationTrace(analyzer_name="Test", timestamp="t")
        trace.record_input("x", 1)
        trace.add_check("c", "warn", "hmm", expected=1, actual=2)
        report.trace = trace

        serialized = json.dumps(report.to_dict())
        roundtrip = json.loads(serialized)
        assert roundtrip["value"] == 3.14
        assert roundtrip["_trace"]["analyzer_name"] == "Test"


class TestBaseAnalyzerCreateTrace:
    """Tests for BaseAnalyzer._create_trace() factory."""

    def test_create_trace(self):
        """_create_trace returns a properly initialized CalculationTrace."""
        @dataclass
        class DummyReport(BaseAnalysisReport):
            def to_dict(self):
                return {}

        class DummyAnalyzer(BaseAnalyzer):
            def analyze_from_parquet(self, parquet_path, session_id=None,
                                     include_trace=False, **kwargs):
                return DummyReport()

        analyzer = DummyAnalyzer()
        trace = analyzer._create_trace("DummyAnalyzer")

        assert isinstance(trace, CalculationTrace)
        assert trace.analyzer_name == "DummyAnalyzer"
        assert trace.timestamp  # non-empty ISO timestamp
        assert trace.inputs == {}
        assert trace.config == {}
        assert trace.intermediates == {}
        assert trace.sanity_checks == []
        assert trace.warnings == []

    def test_create_trace_has_valid_timestamp(self):
        """Timestamp is a valid ISO 8601 string."""
        class DummyAnalyzer(BaseAnalyzer):
            def analyze_from_parquet(self, parquet_path, session_id=None,
                                     include_trace=False, **kwargs):
                pass

        analyzer = DummyAnalyzer()
        trace = analyzer._create_trace("Test")
        # Should parse without error
        from datetime import datetime
        dt = datetime.fromisoformat(trace.timestamp)
        assert dt.year == 2026

    def test_include_trace_parameter_in_signature(self):
        """BaseAnalyzer.analyze_from_parquet accepts include_trace parameter."""
        import inspect
        sig = inspect.signature(BaseAnalyzer.analyze_from_parquet)
        assert "include_trace" in sig.parameters
        param = sig.parameters["include_trace"]
        assert param.default is False


class TestBackwardCompatibility:
    """Verify existing analyzers still work with updated base classes."""

    def test_all_analyzers_still_inherit_base(self):
        """All 7 analyzers still inherit BaseAnalyzer."""
        from src.features.shift_analysis import ShiftAnalyzer
        from src.features.lap_analysis import LapAnalysis
        from src.features.gear_analysis import GearAnalysis
        from src.features.power_analysis import PowerAnalysis
        from src.features.gg_analysis import GGAnalyzer
        from src.features.corner_analysis import CornerAnalyzer
        from src.features.session_report import SessionReportGenerator

        for cls in [ShiftAnalyzer, LapAnalysis, GearAnalysis,
                    PowerAnalysis, GGAnalyzer, CornerAnalyzer,
                    SessionReportGenerator]:
            assert issubclass(cls, BaseAnalyzer), f"{cls.__name__} should inherit BaseAnalyzer"

    def test_all_reports_still_inherit_base(self):
        """All 7 report classes still inherit BaseAnalysisReport."""
        from src.features.shift_analysis import ShiftReport
        from src.features.lap_analysis import LapAnalysisReport
        from src.features.gear_analysis import GearAnalysisReport
        from src.features.power_analysis import PowerAnalysisReport
        from src.features.gg_analysis import GGAnalysisResult
        from src.features.corner_analysis import CornerAnalysisResult
        from src.features.session_report import SessionReport

        for cls in [ShiftReport, LapAnalysisReport, GearAnalysisReport,
                    PowerAnalysisReport, GGAnalysisResult, CornerAnalysisResult,
                    SessionReport]:
            assert issubclass(cls, BaseAnalysisReport), \
                f"{cls.__name__} should inherit BaseAnalysisReport"

    def test_imports_from_features_init(self):
        """Public API imports still work."""
        from src.features import BaseAnalyzer, BaseAnalysisReport
        assert BaseAnalyzer is not None
        assert BaseAnalysisReport is not None

    def test_calculation_trace_importable(self):
        """CalculationTrace is importable from utils."""
        from src.utils.calculation_trace import CalculationTrace, SanityCheck
        assert CalculationTrace is not None
        assert SanityCheck is not None
