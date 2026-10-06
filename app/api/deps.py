"""FastAPI dependencies.

Shared resources (session factory, storage client) are created once at startup
and stored on ``app.state``; dependencies read them from the request. Tests
swap them by setting different objects on ``app.state`` — no globals to patch.
"""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import BackgroundTasks, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.agent.claude_agent import RecoveryAgent
from app.agent.retriever import Retriever
from app.config import Settings
from app.detection.model import Detector
from app.storage.s3_client import S3Storage


def get_db(request: Request) -> Iterator[Session]:
    """Yield a DB session for the duration of a request."""
    session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


def get_storage(request: Request) -> S3Storage:
    """Return the shared S3 storage wrapper."""
    return request.app.state.storage


def get_app_settings(request: Request) -> Settings:
    """Return the settings the app was built with."""
    return request.app.state.settings


def get_detector_optional(request: Request) -> Detector | None:
    """Return the loaded detector, or None if no model has been trained."""
    return request.app.state.detector


def get_detector(request: Request) -> Detector:
    """Return the detector or fail with 503 (the API stays up without a model)."""
    detector = request.app.state.detector
    if detector is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "No detection model loaded. Run `make train` (python -m ml.train) and restart.",
        )
    return detector


def get_retriever(request: Request) -> Retriever:
    """Shared retriever (vector index + date parsing)."""
    return request.app.state.retriever


def get_agent(request: Request) -> RecoveryAgent:
    """Build the Claude agent on first use (lazy: the API runs without credentials)."""
    state = request.app.state
    if state.agent is None:
        settings: Settings = state.settings
        try:
            import anthropic

            client = state.agent_client or anthropic.Anthropic()
        except Exception as exc:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                f"Claude is not configured (set ANTHROPIC_API_KEY): {exc}",
            ) from exc
        state.agent = RecoveryAgent(
            client,
            state.retriever,
            model=settings.anthropic_model,
            effort=settings.anthropic_effort,
            use_fallbacks=settings.anthropic_fallbacks,
            max_turns=settings.agent_max_turns,
            top_k=settings.agent_top_k,
        )
    return state.agent


def schedule_reindex(request: Request, background_tasks: BackgroundTasks) -> None:
    """Refresh the RAG index after the response is sent (snapshot/incident/restore)."""
    background_tasks.add_task(
        request.app.state.retriever.index.sync_with_factory, request.app.state.session_factory
    )
