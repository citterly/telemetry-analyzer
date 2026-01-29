"""
Tests for transmission comparison feature
"""

import os
import sys
import numpy as np
import pytest
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.transmission_comparison import (
    TransmissionComparison,
    TransmissionComparisonReport,
    GearComparison,
    ScenarioPerformance
)
from src.config.vehicle_config import TRANSMISSION_SCENARIOS


class TestGearComparison:
    """Tests for GearComparison dataclass"""

    def test_gear_comparison_creation(self):
        """Test creating a gear comparison"""
        gc = GearComparison(
            gear_number=2,
            current_ratio=1.64,
            proposed_ratio=1.91,
            ratio_difference_pct=16.5,
            current_top_speed_mph=85.0,
            proposed_top_speed_mph=73.0,
            speed_difference_mph=-12.0,
            rpm_difference_at_60mph=800
        )
        assert gc.gear_number == 2
        assert gc.current_ratio == 1.64
        assert gc.speed_difference_mph == -12.0


class TestScenarioPerformance:
    """Tests for ScenarioPerformance dataclass"""

    def test_scenario_performance_creation(self):
        """Test creating scenario performance"""
        sp = ScenarioPerformance(
            name="Test Setup",
            transmission_ratios=[2.88, 1.91, 1.33, 1.00],
            final_drive=3.55,
            weight_lbs=3400,
            gear_top_speeds_mph=[50, 75, 110, 145],
            gear_top_speeds_at_redline=[55, 82, 120, 160],
            overlap_analysis={},
            power_band_coverage={}
        )
        assert sp.name == "Test Setup"
        assert len(sp.transmission_ratios) == 4


class TestTransmissionComparison:
    """Tests for TransmissionComparison"""

    @pytest.fixture
    def comparison(self):
        """Create comparison analyzer"""
        return TransmissionComparison()

    def test_init_defaults(self, comparison):
        """Test initialization with defaults"""
        assert comparison.tire_circumference > 0
        assert comparison.safe_rpm > 0
        assert comparison.redline_rpm > comparison.safe_rpm

    def test_init_custom(self):
        """Test initialization with custom values"""
        comp = TransmissionComparison(
            tire_circumference=2.0,
            safe_rpm=6500,
            redline_rpm=7500,
            power_band=(5000, 6500)
        )
        assert comp.tire_circumference == 2.0
        assert comp.safe_rpm == 6500
        assert comp.power_band == (5000, 6500)

    def test_compare_returns_report(self, comparison):
        """Test compare returns TransmissionComparisonReport"""
        report = comparison.compare()
        assert isinstance(report, TransmissionComparisonReport)

    def test_compare_with_scenario_names(self, comparison):
        """Test compare with specific scenario names"""
        report = comparison.compare(
            current_name="Current Setup",
            proposed_name="New Trans + Current Final"
        )
        assert report.current_setup.name == "Current Setup"
        assert report.proposed_setup.name == "New Trans + Current Final"

    def test_compare_has_gear_comparisons(self, comparison):
        """Test that compare generates gear comparisons"""
        report = comparison.compare()
        assert len(report.gear_comparisons) > 0
        assert all(isinstance(gc, GearComparison) for gc in report.gear_comparisons)

    def test_compare_has_recommendations(self, comparison):
        """Test that compare generates recommendations"""
        report = comparison.compare()
        assert isinstance(report.recommendations, list)
        assert len(report.recommendations) >= 1

    def test_compare_has_summary(self, comparison):
        """Test that compare generates summary"""
        report = comparison.compare()
        assert isinstance(report.summary, dict)
        assert "top_speed_change_mph" in report.summary
        assert "weight_change_lbs" in report.summary

    def test_compare_timestamp(self, comparison):
        """Test that report has timestamp"""
        report = comparison.compare()
        assert report.analysis_timestamp is not None
        # Should be valid ISO format
        from datetime import datetime
        datetime.fromisoformat(report.analysis_timestamp)

    def test_list_available_scenarios(self, comparison):
        """Test listing available scenarios"""
        scenarios = comparison.list_available_scenarios()
        assert isinstance(scenarios, list)
        assert len(scenarios) > 0
        assert "Current Setup" in scenarios

    def test_get_scenario_details(self, comparison):
        """Test getting scenario details"""
        details = comparison.get_scenario_details("Current Setup")
        assert details is not None
        assert "transmission_ratios" in details
        assert "final_drive" in details

    def test_get_nonexistent_scenario(self, comparison):
        """Test getting nonexistent scenario returns default"""
        details = comparison.get_scenario_details("Nonexistent")
        # Should return default setup
        assert details is not None

    def test_report_to_dict(self, comparison):
        """Test report serialization to dict"""
        report = comparison.compare()
        data = report.to_dict()

        assert "analysis_timestamp" in data
        assert "current_setup" in data
        assert "proposed_setup" in data
        assert "gear_comparisons" in data
        assert "recommendations" in data

    def test_report_to_json(self, comparison):
        """Test report serialization to JSON"""
        report = comparison.compare()
        json_str = report.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "current_setup" in parsed

    def test_performance_calculation(self, comparison):
        """Test performance metrics are calculated"""
        report = comparison.compare()

        # Current setup should have performance data
        assert len(report.current_setup.gear_top_speeds_mph) > 0
        assert len(report.current_setup.power_band_coverage) > 0

        # Proposed setup should have performance data
        assert len(report.proposed_setup.gear_top_speeds_mph) > 0

    def test_overlap_analysis(self, comparison):
        """Test gear overlap analysis"""
        report = comparison.compare()

        # Should have overlap analysis
        assert len(report.current_setup.overlap_analysis) > 0

        # Check overlap structure
        for gear, overlap in report.current_setup.overlap_analysis.items():
            assert "to_gear" in overlap
            assert "overlap_mph" in overlap
            assert "has_gap" in overlap


class TestTransmissionComparisonWithSessionData:
    """Tests for session data analysis"""

    @pytest.fixture
    def comparison(self):
        return TransmissionComparison()

    @pytest.fixture
    def sample_session(self):
        """Generate sample session data"""
        time = np.linspace(0, 60, 600)  # 60 seconds at 10 Hz

        # Simulate acceleration run through gears
        rpm = np.zeros(600)
        speed = np.zeros(600)

        # Gear 1: 0-10s
        rpm[0:100] = np.linspace(2000, 6500, 100)
        speed[0:100] = np.linspace(0, 35, 100)

        # Gear 2: 10-20s
        rpm[100:200] = np.linspace(4500, 6500, 100)
        speed[100:200] = np.linspace(35, 60, 100)

        # Gear 3: 20-40s
        rpm[200:400] = np.linspace(4800, 6800, 200)
        speed[200:400] = np.linspace(60, 100, 200)

        # Gear 4: 40-60s
        rpm[400:600] = np.linspace(5000, 6500, 200)
        speed[400:600] = np.linspace(100, 130, 200)

        return {"rpm": rpm, "speed": speed, "time": time}

    def test_compare_with_session_data(self, comparison, sample_session):
        """Test comparison with session data"""
        report = comparison.compare_with_session_data(
            rpm_data=sample_session["rpm"],
            speed_data=sample_session["speed"]
        )

        assert isinstance(report, TransmissionComparisonReport)
        assert report.session_based_analysis is not None

    def test_session_analysis_fields(self, comparison, sample_session):
        """Test session analysis contains expected fields"""
        report = comparison.compare_with_session_data(
            rpm_data=sample_session["rpm"],
            speed_data=sample_session["speed"]
        )

        analysis = report.session_based_analysis
        assert "avg_rpm_difference" in analysis
        assert "max_rpm_difference" in analysis
        assert "pct_time_higher_rpm" in analysis
        assert "sample_points" in analysis

    def test_session_analysis_generates_recommendations(self, comparison, sample_session):
        """Test that session analysis adds recommendations"""
        report = comparison.compare_with_session_data(
            rpm_data=sample_session["rpm"],
            speed_data=sample_session["speed"]
        )

        # Should have at least the base recommendations
        assert len(report.recommendations) >= 1


class TestGearComparisonCalculations:
    """Tests for specific gear comparison calculations"""

    def test_ratio_difference_calculation(self):
        """Test ratio difference percentage calculation"""
        comparison = TransmissionComparison()
        report = comparison.compare(
            current_name="Current Setup",
            proposed_name="New Trans + Current Final"
        )

        # First gear comparison
        gc1 = report.gear_comparisons[0]

        # Calculate expected difference
        current_ratio = 2.20
        proposed_ratio = 2.88
        expected_diff = ((proposed_ratio - current_ratio) / current_ratio) * 100

        assert abs(gc1.ratio_difference_pct - expected_diff) < 0.5

    def test_top_speed_calculations(self):
        """Test top speed calculations are reasonable"""
        comparison = TransmissionComparison()
        report = comparison.compare()

        # Top speeds should increase with each gear
        speeds = report.current_setup.gear_top_speeds_mph
        for i in range(len(speeds) - 1):
            assert speeds[i + 1] > speeds[i], f"Gear {i + 2} should be faster than gear {i + 1}"


class TestAllScenarios:
    """Test all defined scenarios"""

    def test_all_scenarios_valid(self):
        """Test that all scenarios can be compared"""
        comparison = TransmissionComparison()
        scenarios = comparison.list_available_scenarios()

        for scenario in scenarios:
            report = comparison.compare(
                current_name="Current Setup",
                proposed_name=scenario
            )
            assert report is not None

    def test_scenario_count(self):
        """Test expected number of scenarios"""
        assert len(TRANSMISSION_SCENARIOS) >= 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
