"""
Analysis context management.

Provides persistent "what am I analyzing?" state across analysis pages.
"""

from .models import AnalysisContext, AnalysisScope, ScopeMode
from .storage import ContextStorage, get_context_storage

__all__ = [
    "AnalysisContext",
    "AnalysisScope",
    "ScopeMode",
    "ContextStorage",
    "get_context_storage",
]
