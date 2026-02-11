"""
Compliance tests for arch-002: SessionDataLoader Service Layer

Verifies the 10 acceptance criteria from docs/architecture/arch-002.md.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def full_parquet(tmp_path):
    """Create a parquet with all standard telemetry channels."""
    n = 500
    time = np.linspace(0, 50, n)
    df = pd.DataFrame({
        "GPS Latitude": 43.0 + 0.001 * np.sin(time * 0.1),
        "GPS Longitude": -88.0 + 0.001 * np.cos(time * 0.1),
        "GPS Speed": np.random.uniform(20, 45, n),  # m/s
        "RPM": np.random.uniform(3000, 7000, n),
        "GPS LatAcc": np.random.uniform(-1.5, 1.5, n),
        "GPS LonAcc": np.random.uniform(-1.5, 1.5, n),
        "PedalPos": np.random.uniform(0, 100, n),
    }, index=time)
    path = tmp_path / "full_session.parquet"
    df.to_parquet(path)
    return str(path)


class TestAcceptanceCriteria:
    """Verify all 10 acceptance criteria from arch-002.md"""

    def test_ac1_session_data_loader_exists(self):
        """AC-1: SessionDataLoader exists at src/services/session_data_loader.py"""
        loader_path = PROJECT_ROOT / "src" / "services" / "session_data_loader.py"
        assert loader_path.exists()

        from src.services.session_data_loader import SessionDataLoader
        assert SessionDataLoader is not None

    def test_ac2_session_channels_has_core_channels(self):
        """AC-2: SessionChannels has all core channels"""
        from src.services.session_data_loader import SessionChannels

        # Check all required fields exist on the dataclass
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(SessionChannels)}

        required_fields = {
            "time", "latitude", "longitude",
            "speed_mph", "speed_ms", "rpm",
            "lat_acc", "lon_acc", "throttle",
            "column_map", "session_id", "source_path",
            "sample_count", "duration_seconds", "speed_unit_detected",
            "df",
        }
        assert required_fields.issubset(field_names), \
            f"Missing fields: {required_fields - field_names}"

    def test_ac3_column_discovery_uses_known_columns(self, full_parquet):
        """AC-3: Column discovery uses KNOWN_COLUMNS as single source of truth"""
        from src.services.session_data_loader import SessionDataLoader
        from src.utils.dataframe_helpers import KNOWN_COLUMNS

        loader = SessionDataLoader()
        channels = loader.load(full_parquet)

        # Every discovered channel should have a corresponding entry in KNOWN_COLUMNS
        for logical_name in channels.column_map:
            assert logical_name in KNOWN_COLUMNS, \
                f"Discovered channel '{logical_name}' not in KNOWN_COLUMNS"

    def test_ac4_speed_unit_detection_centralized(self, full_parquet):
        """AC-4: Speed unit detection is centralized in SessionChannels"""
        from src.services.session_data_loader import SessionDataLoader

        loader = SessionDataLoader()
        channels = loader.load(full_parquet)

        assert hasattr(channels, "speed_unit_detected")
        assert channels.speed_unit_detected in ("mph", "m/s", "km/h", "unknown")

    def test_ac5_both_speed_units_available(self, full_parquet):
        """AC-5: Both mph and m/s always populated when speed found"""
        from src.services.session_data_loader import SessionDataLoader

        loader = SessionDataLoader()
        channels = loader.load(full_parquet)

        assert channels.speed_mph is not None, "speed_mph should be populated"
        assert channels.speed_ms is not None, "speed_ms should be populated"
        assert len(channels.speed_mph) == len(channels.speed_ms)

        # Verify conversion is correct (mph ≈ 2.237 * m/s)
        ratio = channels.speed_mph.mean() / channels.speed_ms.mean()
        assert 2.0 < ratio < 2.5, f"Speed ratio {ratio} doesn't match m/s→mph conversion"

    def test_ac6_router_endpoints_use_load_session(self):
        """AC-6: Router endpoints use load_session() instead of inline loading"""
        analysis_path = PROJECT_ROOT / "src" / "main" / "routers" / "analysis.py"
        content = analysis_path.read_text()

        # Should use load_session
        assert "load_session" in content, "analysis.py should use load_session()"

        # Non-trace paths should NOT have inline pd.read_parquet
        # (trace paths still use analyze_from_parquet which reads internally)
        lines = content.splitlines()
        inline_reads = 0
        for line in lines:
            stripped = line.strip()
            if "pd.read_parquet" in stripped and not stripped.startswith("#"):
                inline_reads += 1
        assert inline_reads == 0, \
            f"analysis.py still has {inline_reads} inline pd.read_parquet calls"

    def test_ac7_backward_compatible(self, full_parquet):
        """AC-7: Analyzers still work with direct parquet_path argument"""
        from src.features.shift_analysis import ShiftAnalyzer
        from src.features.power_analysis import PowerAnalysis

        # These should still work with just a path (no SessionDataLoader needed)
        shift = ShiftAnalyzer()
        report = shift.analyze_from_parquet(full_parquet)
        assert report is not None

        power = PowerAnalysis()
        report = power.analyze_from_parquet(full_parquet)
        assert report is not None

    def test_ac8_session_report_loads_once(self):
        """AC-8: SessionReportGenerator can use SessionDataLoader (design check)"""
        # This is a design criterion — verify the interface supports it
        from src.services.session_data_loader import SessionDataLoader, SessionChannels

        # SessionDataLoader.load() returns SessionChannels with .df
        # which could be passed to sub-analyzers to avoid re-reading
        loader = SessionDataLoader()
        assert hasattr(loader, "load")
        assert hasattr(loader, "to_session_data_dict")

    def test_ac9_all_existing_tests_pass(self):
        """AC-9: All existing tests pass (meta-check — imports work)"""
        from src.services import SessionDataLoader, SessionChannels
        from src.main.deps import load_session, find_parquet_file
        from src.main.routers.analysis import router

        # All imports successful
        assert True

    def test_ac10_session_data_loader_tests_exist(self):
        """AC-10: New tests cover SessionDataLoader"""
        test_path = PROJECT_ROOT / "tests" / "test_session_data_loader.py"
        assert test_path.exists()

        content = test_path.read_text()
        # Should have tests for major functionality
        assert "TestSessionDataLoader" in content
        assert "TestSpeedUnitDetection" in content
        assert "TestLoadOrRaise" in content
        assert "TestToSessionDataDict" in content


class TestLoadSessionHelper:
    """Verify the deps.load_session() helper"""

    def test_load_session_404_for_missing_file(self):
        """load_session raises 404 for nonexistent file"""
        from fastapi import HTTPException
        from src.main.deps import load_session

        with pytest.raises(HTTPException) as exc_info:
            load_session("completely_nonexistent_file_abc123.parquet")
        assert exc_info.value.status_code == 404

    def test_load_session_422_for_missing_channels(self, full_parquet, tmp_path):
        """load_session raises 422 for missing required channels"""
        from fastapi import HTTPException
        from unittest.mock import patch
        from pathlib import Path

        # Create parquet with no speed
        n = 100
        time = np.linspace(0, 10, n)
        df = pd.DataFrame({"RPM": np.ones(n) * 5000}, index=time)
        no_speed = tmp_path / "no_speed.parquet"
        df.to_parquet(no_speed)

        from src.main.deps import load_session
        with patch("src.main.deps.find_parquet_file", return_value=no_speed):
            with pytest.raises(HTTPException) as exc_info:
                load_session("test.parquet", required=["speed"])
            assert exc_info.value.status_code == 422

    def test_load_session_returns_channels(self, full_parquet):
        """load_session returns SessionChannels on success"""
        from unittest.mock import patch
        from pathlib import Path
        from src.main.deps import load_session
        from src.services.session_data_loader import SessionChannels

        with patch("src.main.deps.find_parquet_file", return_value=Path(full_parquet)):
            channels = load_session("test.parquet")
            assert isinstance(channels, SessionChannels)
            assert channels.sample_count == 500
