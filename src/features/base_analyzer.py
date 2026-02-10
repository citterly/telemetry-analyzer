"""
Base classes for telemetry analyzers.

Provides BaseAnalyzer ABC and BaseAnalysisReport for consistent
interfaces across all analysis modules.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseAnalysisReport:
    """Base class for all analysis result/report objects.

    All analyzer results must implement to_dict() for JSON serialization.
    Subclasses are typically @dataclass decorated, so this base avoids
    __init__ to not conflict with dataclass generation.
    """

    def to_dict(self) -> dict:
        """Serialize report to a JSON-safe dictionary."""
        raise NotImplementedError


class BaseAnalyzer(ABC):
    """Base class for all telemetry analyzers.

    All analyzers must support loading from Parquet files.
    Array-based analysis methods vary per analyzer due to different
    required channels, so only analyze_from_parquet is standardized.
    """

    @abstractmethod
    def analyze_from_parquet(
        self,
        parquet_path: str,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> BaseAnalysisReport:
        """Analyze a session from a Parquet file.

        Args:
            parquet_path: Path to the Parquet file.
            session_id: Optional identifier for the session.
            **kwargs: Analyzer-specific options (e.g., lap_filter, track_name).

        Returns:
            An analysis report object (subclass of BaseAnalysisReport).
        """
        raise NotImplementedError
