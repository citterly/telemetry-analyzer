"""
Lap auto-classification engine.

Classifies laps as out_lap, in_lap, warm_up, cool_down, hot_lap, or normal
based on timing patterns and stint position. Each classification includes
a confidence score (0.0-1.0).
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.analysis.lap_analyzer import LapInfo
from src.session.models import LapClassification, LapClassificationResult, Stint

logger = logging.getLogger(__name__)


@dataclass
class StintInfo:
    """Detected stint from gap analysis."""
    stint_number: int
    lap_indices: List[int]  # indices into the laps list
    start_lap_number: int
    end_lap_number: int
    gap_before_seconds: float = 0.0  # gap from previous stint


class LapClassifier:
    """
    Auto-classifies laps based on timing patterns and stint position.

    Algorithm:
    1. Detect stints by finding gaps >60s between consecutive laps
    2. Within each stint, classify based on position and timing:
       - out_lap: first lap of stint
       - in_lap: last lap of stint (if stint has >1 lap)
       - warm_up: laps 2-4 of stint, >108% of stint best, building speed
       - cool_down: last 2 laps before in_lap, >108% of preceding avg, slowing
       - hot_lap: within 105% of session best time
       - normal: everything else
    3. Assign confidence scores
    """

    def __init__(self, pit_gap_threshold: float = 60.0):
        self.pit_gap_threshold = pit_gap_threshold

    def classify(self, laps: List[LapInfo]) -> Tuple[List[LapClassificationResult], List[StintInfo]]:
        """
        Classify all laps and detect stints.

        Args:
            laps: List of LapInfo from lap detection (must have at least 1 lap)

        Returns:
            Tuple of (classifications, stints)
        """
        if not laps:
            return [], []

        # Step 1: Detect stints
        stints = self._detect_stints(laps)

        # Step 2: Find session best time
        session_best = min(lap.lap_time for lap in laps)

        # Step 3: Classify each lap
        classifications = []
        for stint in stints:
            stint_laps = [laps[i] for i in stint.lap_indices]
            stint_best = min(l.lap_time for l in stint_laps)
            stint_classifications = self._classify_stint(
                stint_laps, stint, session_best, stint_best
            )
            classifications.extend(stint_classifications)

        # Sort by lap number
        classifications.sort(key=lambda c: c.lap_number)

        return classifications, stints

    def _detect_stints(self, laps: List[LapInfo]) -> List[StintInfo]:
        """Find gaps >threshold between consecutive laps to split into stints."""
        if not laps:
            return []

        stints = []
        current_indices = [0]

        for i in range(1, len(laps)):
            gap = laps[i].start_time - laps[i-1].end_time
            if gap > self.pit_gap_threshold:
                # New stint
                stints.append(StintInfo(
                    stint_number=len(stints) + 1,
                    lap_indices=current_indices,
                    start_lap_number=laps[current_indices[0]].lap_number,
                    end_lap_number=laps[current_indices[-1]].lap_number,
                    gap_before_seconds=0.0 if not stints else gap,
                ))
                current_indices = [i]
            else:
                current_indices.append(i)

        # Final stint
        gap = 0.0
        if stints and len(laps) > 1:
            gap = laps[current_indices[0]].start_time - laps[current_indices[0]-1].end_time if current_indices[0] > 0 else 0.0
        stints.append(StintInfo(
            stint_number=len(stints) + 1,
            lap_indices=current_indices,
            start_lap_number=laps[current_indices[0]].lap_number,
            end_lap_number=laps[current_indices[-1]].lap_number,
            gap_before_seconds=gap,
        ))

        return stints

    def _classify_stint(
        self,
        stint_laps: List[LapInfo],
        stint: StintInfo,
        session_best: float,
        stint_best: float,
    ) -> List[LapClassificationResult]:
        """Classify laps within a single stint."""
        results = []
        n = len(stint_laps)

        for i, lap in enumerate(stint_laps):
            classification, confidence, flags = self._classify_single_lap(
                lap, i, n, stint_laps, session_best, stint_best
            )
            results.append(LapClassificationResult(
                lap_number=lap.lap_number,
                classification=classification,
                confidence=confidence,
                stint_number=stint.stint_number,
                flags=flags,
                lap_time=lap.lap_time,
            ))

        return results

    def _classify_single_lap(
        self,
        lap: LapInfo,
        position: int,
        stint_length: int,
        stint_laps: List[LapInfo],
        session_best: float,
        stint_best: float,
    ) -> Tuple[LapClassification, float, List[str]]:
        """
        Classify a single lap based on its position and timing.

        Returns (classification, confidence, flags)
        """
        flags = []

        # Out lap: first lap of stint
        if position == 0:
            confidence = 0.95 if stint_length > 1 else 0.70
            return LapClassification.OUT_LAP, confidence, flags

        # In lap: last lap of stint (only if stint has >1 lap)
        if position == stint_length - 1 and stint_length > 1:
            # Higher confidence if lap is significantly slower
            if lap.lap_time > stint_best * 1.08:
                confidence = 0.90
            else:
                confidence = 0.75
            return LapClassification.IN_LAP, confidence, flags

        # Warm up: laps 2-4 (positions 1-3), slower than best, getting faster
        if 1 <= position <= 3 and stint_length > 4:
            is_slow = lap.lap_time > stint_best * 1.08
            getting_faster = position == 1 or lap.lap_time < stint_laps[position - 1].lap_time
            if is_slow and getting_faster:
                confidence = 0.80 if is_slow else 0.55
                return LapClassification.WARM_UP, confidence, flags

        # Cool down: second-to-last and third-to-last laps, getting slower
        if stint_length > 3 and position >= stint_length - 3 and position < stint_length - 1:
            # Check if slowing relative to preceding laps
            preceding = stint_laps[max(0, position - 3):position]
            if preceding:
                avg_preceding = sum(l.lap_time for l in preceding) / len(preceding)
                is_slow = lap.lap_time > avg_preceding * 1.08
                getting_slower = lap.lap_time > stint_laps[position - 1].lap_time
                if is_slow and getting_slower:
                    confidence = 0.75
                    return LapClassification.COOL_DOWN, confidence, flags

        # Hot lap: within 105% of session best
        if lap.lap_time <= session_best * 1.05:
            ratio = lap.lap_time / session_best
            if ratio <= 1.02:
                confidence = 0.95
            elif ratio <= 1.03:
                confidence = 0.85
            else:
                confidence = 0.70
            return LapClassification.HOT_LAP, confidence, flags

        # Normal: fallback
        confidence = 0.60
        return LapClassification.NORMAL, confidence, flags

    def build_stint_models(
        self, laps: List[LapInfo], stints: List[StintInfo]
    ) -> List[Stint]:
        """Convert StintInfo + LapInfo into Stint model objects for database storage."""
        result = []
        for stint_info in stints:
            stint_laps = [laps[i] for i in stint_info.lap_indices]
            lap_times = [l.lap_time for l in stint_laps]
            result.append(Stint(
                stint_number=stint_info.stint_number,
                start_lap=stint_info.start_lap_number,
                end_lap=stint_info.end_lap_number,
                lap_count=len(stint_laps),
                best_lap_time=min(lap_times) if lap_times else None,
                avg_lap_time=sum(lap_times) / len(lap_times) if lap_times else None,
                start_time=stint_laps[0].start_time if stint_laps else 0.0,
                end_time=stint_laps[-1].end_time if stint_laps else 0.0,
            ))
        return result
