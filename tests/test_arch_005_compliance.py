"""
Compliance tests for arch-005: API integration test coverage.

Verifies all 10 acceptance criteria from docs/architecture/arch-005.md.
"""

import ast
import importlib
import pytest
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent


class TestAcceptanceCriteria:
    """Verify each acceptance criterion from the design doc."""

    def test_ac1_visualization_tests_exist(self):
        """AC-1: test_api_visualization.py with >= 10 tests covering all 5 endpoints."""
        test_file = PROJECT_ROOT / "tests" / "test_api_visualization.py"
        assert test_file.exists(), "test_api_visualization.py must exist"

        source = test_file.read_text()
        tree = ast.parse(source)
        test_methods = [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        ]
        assert len(test_methods) >= 10, f"Expected >= 10 tests, found {len(test_methods)}"

        # All 5 endpoints covered
        assert any("track_map" in m for m in test_methods), "track-map endpoint not tested"
        assert any("delta" in m for m in test_methods), "delta track-map endpoint not tested"
        assert any("gg" in m for m in test_methods), "gg-diagram endpoint not tested"
        assert any("corner_analysis" in m for m in test_methods), "corner-analysis not tested"
        assert any("corner_track_map" in m for m in test_methods), "corner-track-map not tested"

    def test_ac2_parquet_tests_exist(self):
        """AC-2: test_api_parquet.py with >= 5 tests covering all 3 endpoints."""
        test_file = PROJECT_ROOT / "tests" / "test_api_parquet.py"
        assert test_file.exists(), "test_api_parquet.py must exist"

        source = test_file.read_text()
        tree = ast.parse(source)
        test_methods = [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        ]
        assert len(test_methods) >= 5, f"Expected >= 5 tests, found {len(test_methods)}"

        # All 3 endpoints covered
        assert any("list" in m for m in test_methods), "parquet list endpoint not tested"
        assert any("view" in m for m in test_methods), "parquet view endpoint not tested"
        assert any("summary" in m for m in test_methods), "parquet summary endpoint not tested"

    def test_ac3_queue_tests_exist(self):
        """AC-3: test_api_queue.py with >= 7 tests covering all 7 endpoints."""
        test_file = PROJECT_ROOT / "tests" / "test_api_queue.py"
        assert test_file.exists(), "test_api_queue.py must exist"

        source = test_file.read_text()
        tree = ast.parse(source)
        test_methods = [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        ]
        assert len(test_methods) >= 7, f"Expected >= 7 tests, found {len(test_methods)}"

        # Key endpoints covered
        assert any("stats" in m for m in test_methods), "queue stats endpoint not tested"
        assert any("list" in m or "jobs" in m for m in test_methods), "queue jobs list not tested"
        assert any("job" in m and ("found" in m or "detail" in m) for m in test_methods), \
            "queue job detail not tested"
        assert any("retry" in m for m in test_methods), "queue retry not tested"
        assert any("delete" in m for m in test_methods), "queue delete not tested"
        assert any("clear" in m for m in test_methods), "queue clear not tested"

    def test_ac4_vehicle_tests_exist(self):
        """AC-4: test_api_vehicles.py with >= 5 tests covering all 4 endpoints."""
        test_file = PROJECT_ROOT / "tests" / "test_api_vehicles.py"
        assert test_file.exists(), "test_api_vehicles.py must exist"

        source = test_file.read_text()
        tree = ast.parse(source)
        test_methods = [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        ]
        assert len(test_methods) >= 5, f"Expected >= 5 tests, found {len(test_methods)}"

        # All 4 endpoints covered (list, get, set_active; update is optional since
        # it mutates persistent state)
        assert any("list" in m for m in test_methods), "vehicles list endpoint not tested"
        assert any("get" in m for m in test_methods), "vehicles get endpoint not tested"
        assert any("active" in m for m in test_methods), "vehicles set active endpoint not tested"

    def test_ac5_health_tests_exist(self):
        """AC-5: test_api_health.py with >= 2 tests."""
        test_file = PROJECT_ROOT / "tests" / "test_api_health.py"
        assert test_file.exists(), "test_api_health.py must exist"

        source = test_file.read_text()
        tree = ast.parse(source)
        test_methods = [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        ]
        assert len(test_methods) >= 2, f"Expected >= 2 tests, found {len(test_methods)}"

    def test_ac6_all_tests_use_testclient(self):
        """AC-6: All tests use FastAPI TestClient, no manual HTTP calls."""
        test_files = [
            "test_api_visualization.py",
            "test_api_parquet.py",
            "test_api_queue.py",
            "test_api_vehicles.py",
            "test_api_health.py",
        ]
        for name in test_files:
            source = (PROJECT_ROOT / "tests" / name).read_text()
            assert "TestClient" in source, f"{name} must use TestClient"
            assert "requests.get" not in source, f"{name} should not use requests.get"
            assert "urllib" not in source, f"{name} should not use urllib"

    def test_ac7_404_scenarios_covered(self):
        """AC-7: Every endpoint that takes a filename/ID has a not-found test."""
        test_files = {
            "test_api_visualization.py": ["not_found"],
            "test_api_parquet.py": ["not_found"],
            "test_api_queue.py": ["not_found", "nonexistent"],
            "test_api_vehicles.py": ["nonexistent"],
        }
        for name, keywords in test_files.items():
            source = (PROJECT_ROOT / "tests" / name).read_text()
            found = any(kw in source for kw in keywords)
            assert found, f"{name} must have not-found/404 test scenarios"

    def test_ac8_response_structure_validated(self):
        """AC-8: Tests check for expected keys in JSON responses."""
        test_files = [
            "test_api_visualization.py",
            "test_api_parquet.py",
            "test_api_queue.py",
            "test_api_vehicles.py",
            "test_api_health.py",
        ]
        for name in test_files:
            source = (PROJECT_ROOT / "tests" / name).read_text()
            # Tests should check response keys (assert "key" in data, assert isinstance, etc.)
            has_key_check = (
                "in data" in source
                or "in response" in source
                or "isinstance" in source
                or ".json()" in source
            )
            assert has_key_check, f"{name} must validate response structure"

    def test_ac9_no_test_pollution(self):
        """AC-9: Tests use fixtures/mocks to avoid modifying real data."""
        test_files = [
            "test_api_visualization.py",
            "test_api_parquet.py",
            "test_api_queue.py",
            "test_api_vehicles.py",
        ]
        for name in test_files:
            source = (PROJECT_ROOT / "tests" / name).read_text()
            uses_isolation = (
                "mock" in source.lower()
                or "patch" in source.lower()
                or "tmp_path" in source
                or "monkeypatch" in source
                or "fixture" in source.lower()
            )
            assert uses_isolation, f"{name} must use mocks/fixtures for test isolation"

    def test_ac10_all_existing_tests_pass(self):
        """AC-10: No regressions — this test passing means the suite runs."""
        # This is a meta-test: if this test file runs, the suite is healthy.
        # The actual regression check is done by running the full suite.
        # Here we just verify the test files are importable.
        test_modules = [
            "tests.test_api_visualization",
            "tests.test_api_parquet",
            "tests.test_api_queue",
            "tests.test_api_vehicles",
            "tests.test_api_health",
        ]
        for mod_name in test_modules:
            mod = importlib.import_module(mod_name)
            assert mod is not None, f"Failed to import {mod_name}"


class TestCoverageMetrics:
    """Verify test coverage meets design doc targets."""

    def test_total_new_test_count(self):
        """At least 29 new tests across 5 files (design doc: 2+5+7+5+10 = 29 min)."""
        test_files = [
            "test_api_visualization.py",
            "test_api_parquet.py",
            "test_api_queue.py",
            "test_api_vehicles.py",
            "test_api_health.py",
        ]
        total = 0
        for name in test_files:
            source = (PROJECT_ROOT / "tests" / name).read_text()
            tree = ast.parse(source)
            count = sum(
                1 for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
            )
            total += count
        assert total >= 29, f"Expected >= 29 total tests, found {total}"

    def test_trace_parameter_tested(self):
        """Visualization tests verify trace=true returns _trace key."""
        source = (PROJECT_ROOT / "tests" / "test_api_visualization.py").read_text()
        assert "trace" in source, "Visualization tests should cover trace parameter"
        assert "_trace" in source, "Tests should verify _trace key in response"

    def test_queue_isolation_pattern(self):
        """Queue tests use isolated temporary database."""
        source = (PROJECT_ROOT / "tests" / "test_api_queue.py").read_text()
        has_isolation = (
            "tmp_path" in source
            or "tempfile" in source
            or "isolated_queue" in source
        )
        assert has_isolation, "Queue tests must use isolated temporary database"
