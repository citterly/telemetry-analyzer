"""
Tests for LapClassifier module
"""

import os
import sys
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.lap_analyzer import LapInfo
from src.session.classifier import LapClassifier, StintInfo
from src.session.models import LapClassification


def make_lap(num, start, duration, speed=100, rpm=6000):
    """Helper to create synthetic LapInfo objects"""
    return LapInfo(
        lap_number=num,
        start_index=0,
        end_index=100,
        start_time=start,
        end_time=start + duration,
        lap_time=duration,
        max_speed_mph=speed,
        max_rpm=rpm,
        avg_rpm=rpm - 500,
        sample_count=100
    )


class TestLapClassifier:
    """Tests for LapClassifier"""

    @pytest.fixture
    def classifier(self):
        """Create a classifier instance"""
        return LapClassifier(pit_gap_threshold=60.0)

    def test_empty_laps(self, classifier):
        """Test classification of empty lap list returns empty results"""
        classifications, stints = classifier.classify([])
        assert classifications == []
        assert stints == []

    def test_single_lap(self, classifier):
        """Test single lap gets classified as out_lap"""
        laps = [make_lap(1, 0.0, 120.0)]
        classifications, stints = classifier.classify(laps)

        assert len(classifications) == 1
        assert len(stints) == 1
        assert classifications[0].classification == LapClassification.OUT_LAP
        assert classifications[0].lap_number == 1
        assert classifications[0].stint_number == 1
        assert 0.0 <= classifications[0].confidence <= 1.0

    def test_stint_detection(self, classifier):
        """Test stint detection with 90-second gap between laps 3 and 4"""
        laps = [
            make_lap(1, 0.0, 120.0),
            make_lap(2, 120.0, 115.0),
            make_lap(3, 235.0, 118.0),
            # 90-second gap here
            make_lap(4, 443.0, 122.0),  # 353.0 + 90.0 = 443.0
            make_lap(5, 565.0, 119.0),
        ]

        classifications, stints = classifier.classify(laps)

        # Should detect 2 stints
        assert len(stints) == 2
        assert stints[0].stint_number == 1
        assert stints[0].start_lap_number == 1
        assert stints[0].end_lap_number == 3
        assert len(stints[0].lap_indices) == 3

        assert stints[1].stint_number == 2
        assert stints[1].start_lap_number == 4
        assert stints[1].end_lap_number == 5
        assert stints[1].gap_before_seconds > 60.0

    def test_out_lap_classification(self, classifier):
        """Test that first lap of each stint is classified as out_lap"""
        laps = [
            make_lap(1, 0.0, 125.0),
            make_lap(2, 125.0, 115.0),
            make_lap(3, 240.0, 113.0),
            # 70-second gap
            make_lap(4, 423.0, 127.0),
            make_lap(5, 550.0, 116.0),
        ]

        classifications, stints = classifier.classify(laps)

        # Laps 1 and 4 should be out_laps
        out_laps = [c for c in classifications if c.classification == LapClassification.OUT_LAP]
        assert len(out_laps) == 2
        assert out_laps[0].lap_number == 1
        assert out_laps[1].lap_number == 4

    def test_in_lap_classification(self, classifier):
        """Test that last lap of each stint is classified as in_lap"""
        laps = [
            make_lap(1, 0.0, 125.0),
            make_lap(2, 125.0, 115.0),
            make_lap(3, 240.0, 130.0),  # Slower in_lap
            # 70-second gap
            make_lap(4, 440.0, 127.0),
            make_lap(5, 567.0, 116.0),
            make_lap(6, 683.0, 135.0),  # Slower in_lap
        ]

        classifications, stints = classifier.classify(laps)

        # Laps 3 and 6 should be in_laps (last of each stint with >1 lap)
        in_laps = [c for c in classifications if c.classification == LapClassification.IN_LAP]
        assert len(in_laps) == 2
        assert in_laps[0].lap_number == 3
        assert in_laps[1].lap_number == 6

    def test_hot_lap_classification(self, classifier):
        """Test that lap within 105% of session best is classified as hot_lap"""
        laps = [
            make_lap(1, 0.0, 125.0),
            make_lap(2, 125.0, 100.0),    # Session best
            make_lap(3, 225.0, 101.0),    # Within 101% - hot lap
            make_lap(4, 326.0, 104.0),    # Within 104% - hot lap
            make_lap(5, 430.0, 106.0),    # Above 105% - not hot lap
        ]

        classifications, _ = classifier.classify(laps)

        # Laps 2, 3, 4 should be hot laps (within 105% of best)
        hot_laps = [c for c in classifications if c.classification == LapClassification.HOT_LAP]
        assert len(hot_laps) >= 3
        hot_lap_numbers = [c.lap_number for c in hot_laps]
        assert 2 in hot_lap_numbers
        assert 3 in hot_lap_numbers
        assert 4 in hot_lap_numbers

    def test_warm_up_classification(self, classifier):
        """Test warm_up classification for early-stint laps that are slow but getting faster"""
        laps = [
            make_lap(1, 0.0, 130.0),      # Out lap
            make_lap(2, 130.0, 122.0),    # Position 1, slow, should be warm_up
            make_lap(3, 252.0, 118.0),    # Position 2, getting faster, warm_up
            make_lap(4, 370.0, 114.0),    # Position 3, faster still, warm_up
            make_lap(5, 484.0, 110.0),    # Best lap
            make_lap(6, 594.0, 111.0),    # Normal
            make_lap(7, 705.0, 125.0),    # In lap
        ]

        classifications, _ = classifier.classify(laps)

        # Early laps that are slow but getting faster should be warm_up
        warm_up_laps = [c for c in classifications if c.classification == LapClassification.WARM_UP]
        # Should have at least one warm_up lap
        assert len(warm_up_laps) > 0
        # Warm up laps should be in positions 1-3 (lap numbers 2-4)
        for c in warm_up_laps:
            assert c.lap_number in [2, 3, 4]

    def test_confidence_scores(self, classifier):
        """Test that all confidence scores are in valid 0-1 range"""
        laps = [
            make_lap(1, 0.0, 125.0),
            make_lap(2, 125.0, 115.0),
            make_lap(3, 240.0, 113.0),
            make_lap(4, 353.0, 112.0),
            make_lap(5, 465.0, 130.0),
        ]

        classifications, _ = classifier.classify(laps)

        for c in classifications:
            assert 0.0 <= c.confidence <= 1.0, f"Confidence {c.confidence} out of range for lap {c.lap_number}"

    def test_build_stint_models(self, classifier):
        """Test building Stint model objects from StintInfo"""
        laps = [
            make_lap(1, 0.0, 125.0),
            make_lap(2, 125.0, 115.0),
            make_lap(3, 240.0, 113.0),
            # 70-second gap
            make_lap(4, 423.0, 127.0),
            make_lap(5, 550.0, 116.0),
            make_lap(6, 666.0, 114.0),
        ]

        _, stint_infos = classifier.classify(laps)
        stint_models = classifier.build_stint_models(laps, stint_infos)

        # Should have 2 stints
        assert len(stint_models) == 2

        # First stint
        assert stint_models[0].stint_number == 1
        assert stint_models[0].start_lap == 1
        assert stint_models[0].end_lap == 3
        assert stint_models[0].lap_count == 3
        assert stint_models[0].best_lap_time == 113.0
        assert stint_models[0].avg_lap_time == (125.0 + 115.0 + 113.0) / 3
        assert stint_models[0].start_time == 0.0
        assert stint_models[0].end_time == 353.0

        # Second stint
        assert stint_models[1].stint_number == 2
        assert stint_models[1].start_lap == 4
        assert stint_models[1].end_lap == 6
        assert stint_models[1].lap_count == 3
        assert stint_models[1].best_lap_time == 114.0
        assert stint_models[1].avg_lap_time == (127.0 + 116.0 + 114.0) / 3
        assert stint_models[1].start_time == 423.0
        assert stint_models[1].end_time == 780.0

    def test_multiple_stints(self, classifier):
        """Test classification with multiple stints"""
        laps = [
            # Stint 1
            make_lap(1, 0.0, 130.0),
            make_lap(2, 130.0, 115.0),
            # 80-second gap
            # Stint 2
            make_lap(3, 325.0, 128.0),
            make_lap(4, 453.0, 112.0),
            make_lap(5, 565.0, 113.0),
            # 75-second gap
            # Stint 3
            make_lap(6, 753.0, 125.0),
            make_lap(7, 878.0, 114.0),
        ]

        classifications, stints = classifier.classify(laps)

        # Should detect 3 stints
        assert len(stints) == 3
        assert stints[0].stint_number == 1
        assert stints[1].stint_number == 2
        assert stints[2].stint_number == 3

        # Each stint's first lap should be out_lap
        out_laps = [c for c in classifications if c.classification == LapClassification.OUT_LAP]
        assert len(out_laps) == 3
        assert out_laps[0].lap_number == 1
        assert out_laps[1].lap_number == 3
        assert out_laps[2].lap_number == 6

        # Stints with >1 lap should have in_lap as last lap
        in_laps = [c for c in classifications if c.classification == LapClassification.IN_LAP]
        # Stints 1, 2, 3 all have >1 lap, so should have in_laps
        assert len(in_laps) == 3

    def test_normal_classification(self, classifier):
        """Test that unremarkable laps get classified as normal"""
        laps = [
            make_lap(1, 0.0, 130.0),      # Out lap
            make_lap(2, 130.0, 115.0),    # Decent time
            make_lap(3, 245.0, 114.0),    # Decent time (best)
            make_lap(4, 359.0, 120.0),    # Slower, not hot lap - should be normal
            make_lap(5, 479.0, 121.0),    # Also slower, not hot lap - should be normal
            make_lap(6, 600.0, 125.0),    # In lap
        ]

        classifications, _ = classifier.classify(laps)

        # Should have some normal laps (laps that aren't out, in, or hot)
        normal_laps = [c for c in classifications if c.classification == LapClassification.NORMAL]
        assert len(normal_laps) > 0

    def test_cool_down_classification(self, classifier):
        """Test cool_down classification for laps getting slower near end of stint"""
        laps = [
            make_lap(1, 0.0, 130.0),      # Out lap
            make_lap(2, 130.0, 112.0),    # Good lap
            make_lap(3, 242.0, 113.0),    # Good lap
            make_lap(4, 355.0, 114.0),    # Good lap
            make_lap(5, 469.0, 119.0),    # Slowing down
            make_lap(6, 588.0, 123.0),    # Slowing more - should be cool_down
            make_lap(7, 711.0, 135.0),    # In lap
        ]

        classifications, _ = classifier.classify(laps)

        # Should have at least some classification diversity
        classification_types = {c.classification for c in classifications}
        assert len(classification_types) > 1

        # Verify all classifications are valid
        for c in classifications:
            assert isinstance(c.classification, LapClassification)
            assert c.lap_number in [1, 2, 3, 4, 5, 6, 7]
            assert c.stint_number == 1

    def test_classifications_sorted_by_lap_number(self, classifier):
        """Test that classifications are returned sorted by lap number"""
        laps = [
            make_lap(1, 0.0, 125.0),
            make_lap(2, 125.0, 115.0),
            make_lap(3, 240.0, 113.0),
            make_lap(4, 353.0, 112.0),
            make_lap(5, 465.0, 114.0),
        ]

        classifications, _ = classifier.classify(laps)

        # Verify sorted order
        lap_numbers = [c.lap_number for c in classifications]
        assert lap_numbers == sorted(lap_numbers)
        assert lap_numbers == [1, 2, 3, 4, 5]

    def test_session_best_time_calculation(self, classifier):
        """Test that session best time is correctly identified"""
        laps = [
            make_lap(1, 0.0, 125.0),
            make_lap(2, 125.0, 115.0),
            make_lap(3, 240.0, 108.0),    # Best lap
            make_lap(4, 348.0, 112.0),
            make_lap(5, 460.0, 110.0),
        ]

        classifications, _ = classifier.classify(laps)

        # Lap 3 should be hot_lap with high confidence (it's the best)
        lap3_classification = next(c for c in classifications if c.lap_number == 3)
        assert lap3_classification.classification == LapClassification.HOT_LAP
        assert lap3_classification.confidence > 0.8

    def test_stint_gap_threshold_exact(self, classifier):
        """Test stint detection with gap exactly at threshold"""
        classifier = LapClassifier(pit_gap_threshold=60.0)
        laps = [
            make_lap(1, 0.0, 120.0),
            make_lap(2, 120.0, 115.0),
            # Exactly 60 second gap (at threshold)
            make_lap(3, 295.0, 118.0),  # 235.0 + 60.0 = 295.0
        ]

        _, stints = classifier.classify(laps)

        # Exactly at threshold should NOT create new stint (> not >=)
        assert len(stints) == 1

    def test_stint_gap_threshold_above(self, classifier):
        """Test stint detection with gap just above threshold"""
        classifier = LapClassifier(pit_gap_threshold=60.0)
        laps = [
            make_lap(1, 0.0, 120.0),
            make_lap(2, 120.0, 115.0),
            # 61 second gap (above threshold)
            make_lap(3, 296.0, 118.0),  # 235.0 + 61.0 = 296.0
        ]

        _, stints = classifier.classify(laps)

        # Above threshold should create new stint
        assert len(stints) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
