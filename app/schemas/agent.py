"""Pydantic models for the agent API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Body for POST /agent/ask."""

    question: str = Field(min_length=3, max_length=2000, examples=[
        "What changed on Tuesday, and is it safe to restore?"
    ])  # fmt: skip


class ToolCallOut(BaseModel):
    name: str
    input: Any
    is_error: bool


class AskResponse(BaseModel):
    """Agent answer plus machine-checkable citations."""

    answer: str
    cited_snapshot_ids: list[str]
    cited_incident_ids: list[int]
    unverified_ids: list[str] = Field(
        description="Ids the model mentioned that do not exist (possible hallucinations)."
    )
    recommended_snapshot_id: str | None = Field(
        description="Validated clean snapshot. Restoring still requires POST /restore."
    )
    recommendation_warning: str | None
    time_range: str | None
    retrieved_keys: list[str]
    tool_calls: list[ToolCallOut]
    model: str
    stop_reason: str | None


class SearchHitOut(BaseModel):
    key: str
    kind: str
    timestamp: datetime
    score: float
    text: str


class SearchResponse(BaseModel):
    """Retrieval-only result (no LLM call) — handy for debugging RAG."""

    time_range: str | None
    widened: bool
    hits: list[SearchHitOut]
