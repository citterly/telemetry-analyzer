"""
Tests for gear_calculator module (feat-050)

Unit tests for GearCalculator, GearInfo, and gear analysis functions.
"""

import pytest
import numpy as np
from typing import List


class TestGearInfo:
    """Tests for GearInfo dataclass"""

    def test_gear_info_creation(self):
        """Test creating a GearInfo instance"""
        from src.analysis.gear_calculator import GearInfo

        info = GearInfo(
            gear=3,
            confidence=0.95,
            theoretical_rpm=5500,
            rpm_difference=50,
            speed_mph=100,
            actual_rpm=5550
        )

        assert info.gear == 3
        assert info.confidence == 0.95
        assert info.theoretical_rpm == 5500
        assert info.rpm_difference == 50
        assert info.speed_mph == 100
        assert info.actual_rpm == 5550

    def test_gear_info_neutral(self):
        """Test GearInfo for neutral/unknown state"""
        from src.analysis.gear_calculator import GearInfo

        info = GearInfo(
            gear=0,
            confidence=0.0,
            theoretical_rpm=0,
            rpm_difference=999999,
            speed_mph=0,
            actual_rpm=800
        )

        assert info.gear == 0
        assert info.confidence == 0.0


class TestGearCalculator:
    """Tests for GearCalculator class"""

    @pytest.fixture
    def calculator(self):
        """Create a GearCalculator with known transmission ratios"""
        from src.analysis.gear_calculator import GearCalculator

        # Use BMW M3 ratios: [2.20, 1.64, 1.28, 1.00]
        return GearCalculator(
            transmission_ratios=[2.20, 1.64, 1.28, 1.00],
            final_drive=3.55
        )

    def test_calculator_init(self, calculator):
        """Test GearCalculator initialization"""
        assert len(calculator.transmission_ratios) == 4
        assert calculator.final_drive == 3.55
        assert calculator.tire_circumference > 0

    def test_calculate_gear_at_point_low_speed(self, calculator):
        """Test gear calculation returns neutral at low speed"""
        result = calculator.calculate_gear_at_point(rpm=1000, speed_mph=1)

        assert result.gear == 0
        assert result.confidence == 0.0

    def test_calculate_gear_at_point_low_rpm(self, calculator):
        """Test gear calculation returns neutral at low RPM"""
        result = calculator.calculate_gear_at_point(rpm=500, speed_mph=50)

        assert result.gear == 0
        assert result.confidence == 0.0

    def test_calculate_gear_at_point_first_gear(self, calculator):
        """Test gear calculation identifies first gear"""
        # First gear (ratio 2.20) with final drive 3.55 at 6000 RPM
        # Theoretical speed = RPM * tire_circ / (gear_ratio * final_drive * 60) * 2.237 mph
        # ~6000 * 2.026 / (2.20 * 3.55 * 60) * 2.237 = ~58 mph
        result = calculator.calculate_gear_at_point(rpm=6000, speed_mph=58)

        assert result.gear == 1
        assert result.confidence > 0

    def test_calculate_gear_at_point_high_gear(self, calculator):
        """Test gear calculation identifies fourth gear"""
        # Fourth gear (ratio 1.00) with final drive 3.55 at 6000 RPM
        # Theoretical speed = ~6000 * 2.026 / (1.00 * 3.55 * 60) * 2.237 = ~128 mph
        result = calculator.calculate_gear_at_point(rpm=6000, speed_mph=128)

        assert result.gear == 4
        assert result.confidence > 0.5

    def test_calculate_gear_at_point_returns_confidence(self, calculator):
        """Test that confidence reflects RPM match quality"""
        # Perfect match should have high confidence
        result = calculator.calculate_gear_at_point(rpm=6000, speed_mph=95)
        assert 0.0 <= result.confidence <= 1.0

    def test_calculate_gear_trace_empty(self, calculator):
        """Test gear trace with empty arrays"""
        rpm_array = np.array([])
        speed_array = np.array([])

        result = calculator.calculate_gear_trace(rpm_array, speed_array)
        assert result == []

    def test_calculate_gear_trace_mismatched_lengths(self, calculator):
        """Test gear trace raises error with mismatched array lengths"""
        rpm_array = np.array([5000, 5500, 6000])
        speed_array = np.array([50, 60])

        with pytest.raises(ValueError):
            calculator.calculate_gear_trace(rpm_array, speed_array)

    def test_calculate_gear_trace_single_point(self, calculator):
        """Test gear trace with single point"""
        rpm_array = np.array([5000])
        speed_array = np.array([80])

        result = calculator.calculate_gear_trace(rpm_array, speed_array)
        assert len(result) == 1

    def test_calculate_gear_trace_multiple_points(self, calculator):
        """Test gear trace with multiple points"""
        rpm_array = np.array([5000, 5500, 6000, 6500, 7000])
        speed_array = np.array([70, 80, 90, 95, 100])

        result = calculator.calculate_gear_trace(rpm_array, speed_array)
        assert len(result) == 5
        for gear_info in result:
            assert hasattr(gear_info, 'gear')
            assert hasattr(gear_info, 'confidence')

    def test_smooth_gear_trace_removes_jitter(self, calculator):
        """Test that smoothing removes single-point gear anomalies"""
        from src.analysis.gear_calculator import GearInfo

        # Create a trace with a single-point anomaly
        trace = [
            GearInfo(gear=3, confidence=0.9, theoretical_rpm=5000, rpm_difference=50, speed_mph=80, actual_rpm=5050),
            GearInfo(gear=3, confidence=0.9, theoretical_rpm=5100, rpm_difference=50, speed_mph=82, actual_rpm=5150),
            GearInfo(gear=2, confidence=0.3, theoretical_rpm=5200, rpm_difference=200, speed_mph=84, actual_rpm=5400),  # anomaly
            GearInfo(gear=3, confidence=0.9, theoretical_rpm=5300, rpm_difference=50, speed_mph=86, actual_rpm=5350),
            GearInfo(gear=3, confidence=0.9, theoretical_rpm=5400, rpm_difference=50, speed_mph=88, actual_rpm=5450),
        ]

        smoothed = calculator._smooth_gear_trace(trace)

        # The anomaly point should be smoothed to match neighbors
        assert smoothed[2].gear == 3

    def test_get_gear_usage_summary_empty(self, calculator):
        """Test gear usage summary with empty trace"""
        summary = calculator.get_gear_usage_summary([])
        assert summary == {}

    def test_get_gear_usage_summary_single_gear(self, calculator):
        """Test gear usage summary with single gear"""
        from src.analysis.gear_calculator import GearInfo

        trace = [
            GearInfo(gear=3, confidence=0.9, theoretical_rpm=5000, rpm_difference=50, speed_mph=80, actual_rpm=5050),
            GearInfo(gear=3, confidence=0.9, theoretical_rpm=5100, rpm_difference=50, speed_mph=82, actual_rpm=5150),
            GearInfo(gear=3, confidence=0.9, theoretical_rpm=5200, rpm_difference=50, speed_mph=84, actual_rpm=5250),
        ]

        summary = calculator.get_gear_usage_summary(trace)

        assert 3 in summary
        assert summary[3]['usage_percent'] == 100.0
        assert summary[3]['sample_count'] == 3
        assert summary[3]['speed_range_mph']['min'] == 80
        assert summary[3]['speed_range_mph']['max'] == 84

    def test_get_gear_usage_summary_multiple_gears(self, calculator):
        """Test gear usage summary with multiple gears"""
        from src.analysis.gear_calculator import GearInfo

        trace = [
            GearInfo(gear=2, confidence=0.9, theoretical_rpm=5000, rpm_difference=50, speed_mph=60, actual_rpm=5050),
            GearInfo(gear=3, confidence=0.9, theoretical_rpm=5100, rpm_difference=50, speed_mph=80, actual_rpm=5150),
            GearInfo(gear=3, confidence=0.9, theoretical_rpm=5200, rpm_difference=50, speed_mph=82, actual_rpm=5250),
            GearInfo(gear=4, confidence=0.9, theoretical_rpm=5300, rpm_difference=50, speed_mph=100, actual_rpm=5350),
        ]

        summary = calculator.get_gear_usage_summary(trace)

        assert 2 in summary
        assert 3 in summary
        assert 4 in summary
        assert summary[2]['usage_percent'] == 25.0
        assert summary[3]['usage_percent'] == 50.0
        assert summary[4]['usage_percent'] == 25.0

    def test_find_shift_points_no_shifts(self, calculator):
        """Test shift detection with no gear changes"""
        from src.analysis.gear_calculator import GearInfo

        trace = [
            GearInfo(gear=3, confidence=0.9, theoretical_rpm=5000, rpm_difference=50, speed_mph=80, actual_rpm=5050),
            GearInfo(gear=3, confidence=0.9, theoretical_rpm=5100, rpm_difference=50, speed_mph=82, actual_rpm=5150),
        ]
        time_array = np.array([0.0, 1.0])

        shifts = calculator.find_shift_points(trace, time_array)
        assert shifts == []

    def test_find_shift_points_upshift(self, calculator):
        """Test upshift detection"""
        from src.analysis.gear_calculator import GearInfo

        trace = [
            GearInfo(gear=2, confidence=0.9, theoretical_rpm=6500, rpm_difference=50, speed_mph=60, actual_rpm=6550),
            GearInfo(gear=3, confidence=0.9, theoretical_rpm=5000, rpm_difference=50, speed_mph=60, actual_rpm=5050),
        ]
        time_array = np.array([0.0, 1.0])

        shifts = calculator.find_shift_points(trace, time_array)

        assert len(shifts) == 1
        assert shifts[0]['type'] == 'upshift'
        assert shifts[0]['from_gear'] == 2
        assert shifts[0]['to_gear'] == 3

    def test_find_shift_points_downshift(self, calculator):
        """Test downshift detection"""
        from src.analysis.gear_calculator import GearInfo

        trace = [
            GearInfo(gear=4, confidence=0.9, theoretical_rpm=4000, rpm_difference=50, speed_mph=80, actual_rpm=4050),
            GearInfo(gear=3, confidence=0.9, theoretical_rpm=5500, rpm_difference=50, speed_mph=80, actual_rpm=5550),
        ]
        time_array = np.array([0.0, 1.0])

        shifts = calculator.find_shift_points(trace, time_array)

        assert len(shifts) == 1
        assert shifts[0]['type'] == 'downshift'
        assert shifts[0]['from_gear'] == 4
        assert shifts[0]['to_gear'] == 3

    def test_find_shift_points_ignores_neutral(self, calculator):
        """Test that shifts involving neutral are ignored"""
        from src.analysis.gear_calculator import GearInfo

        trace = [
            GearInfo(gear=0, confidence=0.0, theoretical_rpm=0, rpm_difference=999, speed_mph=5, actual_rpm=800),
            GearInfo(gear=1, confidence=0.9, theoretical_rpm=5000, rpm_difference=50, speed_mph=30, actual_rpm=5050),
        ]
        time_array = np.array([0.0, 1.0])

        shifts = calculator.find_shift_points(trace, time_array)
        assert shifts == []

    def test_calculate_theoretical_performance(self, calculator):
        """Test theoretical performance calculation"""
        perf = calculator.calculate_theoretical_performance('Current Setup', max_rpm=7000)

        assert perf['scenario_name'] == 'Current Setup'
        assert len(perf['gear_top_speeds_mph']) == 4
        assert len(perf['gear_top_speeds_ms']) == 4

        # Top speed in higher gears should be higher
        assert perf['gear_top_speeds_mph'][3] > perf['gear_top_speeds_mph'][0]

    def test_calculate_theoretical_performance_invalid_scenario(self, calculator):
        """Test theoretical performance with invalid scenario"""
        with pytest.raises(ValueError):
            calculator.calculate_theoretical_performance('Nonexistent Scenario')


class TestModuleFunctions:
    """Tests for module-level functions"""

    def test_analyze_lap_gearing_invalid_scenario(self):
        """Test analyze_lap_gearing with invalid scenario"""
        from src.analysis.gear_calculator import analyze_lap_gearing

        lap_data = {
            'rpm': np.array([5000, 5500, 6000]),
            'speed_mph': np.array([70, 80, 90]),
            'time': np.array([0.0, 1.0, 2.0])
        }

        with pytest.raises(ValueError):
            analyze_lap_gearing(lap_data, 'Nonexistent Scenario')

    def test_analyze_lap_gearing_valid(self, capsys):
        """Test analyze_lap_gearing with valid data"""
        from src.analysis.gear_calculator import analyze_lap_gearing

        # Use realistic RPM/speed combinations for 4th gear
        lap_data = {
            'rpm': np.array([5000, 5500, 6000, 6500, 7000]),
            'speed_mph': np.array([100, 110, 120, 130, 140]),
            'time': np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        }

        gear_trace, summary = analyze_lap_gearing(lap_data, 'Current Setup')

        assert len(gear_trace) == 5
        assert 'scenario' in summary
        assert 'gear_usage' in summary
        assert 'shift_points' in summary
        assert 'theoretical_performance' in summary

    def test_debug_gear_calculations(self, capsys):
        """Test debug_gear_calculations output"""
        from src.analysis.gear_calculator import debug_gear_calculations

        debug_gear_calculations([2.20, 1.64, 1.28, 1.00], 3.55)

        captured = capsys.readouterr()
        assert 'Gear 1' in captured.out
        assert 'Gear 4' in captured.out
        assert 'RPM' in captured.out
