"""
Analyzer registry for auto-discovery and plugin pattern.

Analyzers register themselves at import time via class attributes.
SessionReportGenerator iterates the registry instead of hardcoding sub-analyzers.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from .base_analyzer import BaseAnalyzer


@dataclass
class AnalyzerRegistration:
    """Metadata about a registered analyzer."""
    key: str
    analyzer_class: Type[BaseAnalyzer]
    required_channels: List[str]
    optional_channels: List[str]
    config_params: List[str]


class AnalyzerRegistry:
    """Registry for analyzer classes with auto-discovery support."""

    def __init__(self):
        self._analyzers: Dict[str, AnalyzerRegistration] = {}

    def register(self, analyzer_class: Type[BaseAnalyzer]) -> None:
        """Register an analyzer class. Reads metadata from class attributes."""
        key = getattr(analyzer_class, 'registry_key', None)
        if key is None:
            raise ValueError(
                f"{analyzer_class.__name__} has no registry_key class attribute"
            )
        self._analyzers[key] = AnalyzerRegistration(
            key=key,
            analyzer_class=analyzer_class,
            required_channels=list(getattr(analyzer_class, 'required_channels', [])),
            optional_channels=list(getattr(analyzer_class, 'optional_channels', [])),
            config_params=list(getattr(analyzer_class, 'config_params', [])),
        )

    def get(self, key: str) -> Optional[AnalyzerRegistration]:
        """Get registration by key."""
        return self._analyzers.get(key)

    def list_registered(self) -> List[str]:
        """List all registered analyzer keys."""
        return list(self._analyzers.keys())

    def create_instance(self, key: str, **config) -> BaseAnalyzer:
        """Create an analyzer instance, passing only recognized config params."""
        reg = self._analyzers[key]
        filtered = {k: v for k, v in config.items() if k in reg.config_params}
        return reg.analyzer_class(**filtered)

    def reset(self):
        """Clear all registrations. For testing only."""
        self._analyzers.clear()

    def __iter__(self):
        return iter(self._analyzers.items())

    def __len__(self):
        return len(self._analyzers)

    def __contains__(self, key: str) -> bool:
        return key in self._analyzers


# Singleton instance
analyzer_registry = AnalyzerRegistry()
