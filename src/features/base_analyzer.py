"""
Base classes for telemetry analyzers.

Provides BaseAnalyzer ABC and BaseAnalysisReport for consistent
interfaces across all analysis modules.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import List, Optional

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

    Registry metadata (override in subclasses to enable auto-registration):
        registry_key: Unique identifier (e.g., "shifts", "laps")
        required_channels: Channels that must be present (e.g., ["rpm", "speed"])
        optional_channels: Channels used if available (e.g., ["latitude"])
        config_params: Constructor kwargs from session config (e.g., ["track_name"])
    """

    # Registry metadata — override in subclasses
    registry_key: Optional[str] = None
    required_channels: List[str] = []
    optional_channels: List[str] = []
    config_params: List[str] = []

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

    def analyze_from_channels(self, channels, session_id="unknown",
                              include_trace=False, **kwargs):
        """Analyze from pre-loaded SessionChannels.

        Override in subclass to map SessionChannels fields to
        analyze_from_arrays() parameters.

        Args:
            channels: SessionChannels with discovered channel data.
            session_id: Session identifier.
            include_trace: If True, attach trace to report.
            **kwargs: Extra config (e.g., track_name).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement analyze_from_channels"
        )

    def _create_trace(self, analyzer_name: str) -> CalculationTrace:
        """Create a new trace object.

        Called by subclasses when include_trace=True.
        """
        return CalculationTrace(
            analyzer_name=analyzer_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
