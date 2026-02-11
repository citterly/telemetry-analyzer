"""
Context storage using in-memory dictionary.

For multi-user scenarios, this could be upgraded to Redis.
Current implementation: Simple in-memory dict (single-user desktop app).
"""

import json
from datetime import datetime, timezone
from typing import Dict, Optional

from .models import AnalysisContext, AnalysisScope, ScopeMode


class ContextStorage:
    """
    Manages analysis context persistence.

    Current implementation: In-memory dictionary (single-user desktop app)
    Future: Redis for multi-user/multi-instance deployments, or browser localStorage
    """

    def __init__(self):
        self._contexts: Dict[str, dict] = {}
        self._default_user = "default"  # Single-user mode

    def set_context(self, context: AnalysisContext, user_id: str = None) -> None:
        """Store analysis context."""
        user_id = user_id or self._default_user
        context.last_accessed = datetime.now(timezone.utc)
        self._contexts[user_id] = context.to_dict()

    def get_context(self, user_id: str = None) -> Optional[AnalysisContext]:
        """Retrieve analysis context."""
        user_id = user_id or self._default_user
        data = self._contexts.get(user_id)
        if data:
            context = AnalysisContext.from_dict(data)
            # Update last accessed
            context.last_accessed = datetime.now(timezone.utc)
            self.set_context(context, user_id)
            return context
        return None

    def clear_context(self, user_id: str = None) -> None:
        """Remove analysis context."""
        user_id = user_id or self._default_user
        if user_id in self._contexts:
            del self._contexts[user_id]

    def set_scope(
        self,
        session_ids: list,
        mode: str = "single",
        baseline_session_id: Optional[str] = None,
        filters: Optional[dict] = None,
        user_id: str = None
    ) -> AnalysisContext:
        """
        Convenience method to set a new scope.

        Creates or updates the analysis context with a new scope.
        """
        scope = AnalysisScope(
            mode=ScopeMode(mode),
            session_ids=session_ids,
            baseline_session_id=baseline_session_id,
            filters=filters,
        )

        active_session_id = session_ids[0] if session_ids else ""

        context = AnalysisContext(
            scope=scope,
            active_session_id=active_session_id,
        )

        self.set_context(context, user_id)
        return context

    def add_comparison(
        self,
        session_id: str,
        role: str = "comparison",
        user_id: str = None
    ) -> Optional[AnalysisContext]:
        """
        Add a session to the current context for comparison.

        Args:
            session_id: Session to add
            role: "baseline" or "comparison"
            user_id: User identifier (default: "default")

        Returns:
            Updated context, or None if no context exists
        """
        context = self.get_context(user_id)
        if not context:
            return None

        if role == "baseline":
            context.scope.baseline_session_id = session_id
        else:
            # Add to session_ids if not already present
            if session_id not in context.scope.session_ids:
                context.scope.session_ids.append(session_id)

        # Update mode to multi if we now have baseline or multiple sessions
        if context.scope.baseline_session_id or len(context.scope.session_ids) > 1:
            context.scope.mode = ScopeMode.MULTI

        self.set_context(context, user_id)
        return context


# Singleton instance
_context_storage: Optional[ContextStorage] = None


def get_context_storage() -> ContextStorage:
    """Get the singleton ContextStorage instance."""
    global _context_storage
    if _context_storage is None:
        _context_storage = ContextStorage()
    return _context_storage
