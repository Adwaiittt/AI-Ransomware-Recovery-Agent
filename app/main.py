"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from app.agent.embeddings import Embedder, make_embedder
from app.agent.indexer import VectorIndex
from app.agent.retriever import Retriever
from app.api import agent, backup, detection, health, restore
from app.config import Settings, get_settings
from app.db.session import init_db, make_engine, make_session_factory
from app.detection.model import Detector, ModelNotAvailableError
from app.logging_config import RequestIdMiddleware, configure_logging
from app.storage.s3_client import S3Storage, StorageError

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    storage: S3Storage | None = None,
    detector: Detector | None = None,
    agent_client: Any | None = None,
    embedder: Embedder | None = None,
) -> FastAPI:
    """Build the app. Dependencies can be injected (tests use this).

    If no detector is injected, the model is loaded from ``settings.model_path``;
    a missing model is not fatal — backups still work and detection endpoints
    return 503 until ``make train`` has been run.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = make_engine(settings.database_url)
        init_db(engine)
        app.state.settings = settings
        app.state.session_factory = make_session_factory(engine)
        app.state.storage = storage or S3Storage.from_settings(settings)
        settings.watch_dir.mkdir(parents=True, exist_ok=True)
        try:
            app.state.storage.ensure_bucket()
        except StorageError as exc:
            # Start anyway; /health reports degraded and backup calls return 503.
            logger.warning("Object store not ready at startup: %s", exc)
        app.state.detector = detector
        if app.state.detector is None:
            try:
                app.state.detector = Detector.load(settings.model_path)
            except ModelNotAvailableError as exc:
                logger.warning("Detection disabled: %s", exc)
        # RAG index + agent. The embedder loads lazily (first search), and the
        # Claude client is created on the first /agent/ask, so neither a model
        # download nor an API key is needed just to start the API.
        index = VectorIndex(embedder or make_embedder(settings.embedding_model))
        app.state.retriever = Retriever(index, settings.agent_timezone)
        app.state.agent_client = agent_client
        app.state.agent = None
        yield
        engine.dispose()

    app = FastAPI(
        title="AI Ransomware Recovery Agent",
        version="0.4.0",
        description="Backup, ransomware detection, and AI-assisted recovery.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    app.include_router(health.router)
    app.include_router(backup.router)
    app.include_router(detection.router)
    app.include_router(restore.router)
    app.include_router(agent.router)
    return app


app = create_app()
