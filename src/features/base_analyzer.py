"""
Base classes for telemetry analyzers.

Provides BaseAnalyzer ABC and BaseAnalysisReport for consistent
interfaces across all analysis modules.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

from src.utils.calculation_trace import CalculationTrace


class BaseAnalysisReport:
    """Base class for all analysis result/report objects.

    All analyzer results must implement to_dict() for JSON serialization.
    Subclasses are typically @dataclass decorated, so this base avoids
    __init__ to not conflict with dataclass generation.

    Trace support: After constructing a report, set report.trace = trace
    to attach a CalculationTrace. Use _trace_dict() in to_dict() to
    include it in serialization when present.
    """

    def to_dict(self) -> dict:
        """Serialize report to a JSON-safe dictionary."""
        raise NotImplementedError

    def _trace_dict(self) -> dict:
        """Return trace dict if present, empty dict if not.

        Subclasses should merge this into their to_dict() output:
            result = { ... }
            result.update(self._trace_dict())
            return result
        """
        trace = getattr(self, 'trace', None)
        if trace is not None:
            return {"_trace": trace.to_dict()}
        return {}


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
        include_trace: bool = False,
        **kwargs,
    ) -> BaseAnalysisReport:
        """Analyze a session from a Parquet file.

        Args:
            parquet_path: Path to the Parquet file.
            session_id: Optional identifier for the session.
            include_trace: If True, attach a CalculationTrace to the report
                with inputs, config, intermediates, and sanity checks.
            **kwargs: Analyzer-specific options (e.g., lap_filter, track_name).

        Returns:
            An analysis report object (subclass of BaseAnalysisReport).
        """
        raise NotImplementedError

    def _create_trace(self, analyzer_name: str) -> CalculationTrace:
        """Create a new trace object.

        Called by subclasses when include_trace=True.
        """
        return CalculationTrace(
            analyzer_name=analyzer_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
