"""
Analysis context API endpoints.

Manages user's current "what am I analyzing?" state.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from src.context import get_context_storage, AnalysisContext, ScopeMode


router = APIRouter(prefix="/api/context", tags=["context"])


# Request/Response models
class SetScopeRequest(BaseModel):
    mode: str = "single"
    session_ids: List[str]
    baseline_session_id: Optional[str] = None
    filters: Optional[dict] = None


class AddComparisonRequest(BaseModel):
    session_id: str
    role: str = "comparison"  # "baseline" or "comparison"


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.post("/set")
def set_context(request: SetScopeRequest):
    """
    Set the analysis scope.

    Creates or replaces the current analysis context.
    """
    storage = get_context_storage()

    context = storage.set_scope(
        session_ids=request.session_ids,
        mode=request.mode,
        baseline_session_id=request.baseline_session_id,
        filters=request.filters,
    )

    return context.to_dict()


@router.get("/current")
def get_current_context():
    """
    Get the current analysis context.

    Returns null if no context is set.
    """
    storage = get_context_storage()
    context = storage.get_context()

    if context:
        return context.to_dict()
    return None


@router.post("/add-comparison")
def add_comparison(request: AddComparisonRequest):
    """
    Add a session to the current context for comparison.

    Returns 404 if no context exists.
    """
    storage = get_context_storage()
    context = storage.add_comparison(
        session_id=request.session_id,
        role=request.role,
    )

    if not context:
        raise HTTPException(404, "No analysis context set")

    return context.to_dict()


@router.delete("/clear")
def clear_context():
    """Remove the current analysis context."""
    storage = get_context_storage()
    storage.clear_context()
    return {"status": "cleared"}
