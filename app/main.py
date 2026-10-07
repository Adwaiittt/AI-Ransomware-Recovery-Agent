"""FastAPI application factory."""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.agent.embeddings import Embedder, make_embedder
from app.agent.indexer import VectorIndex
from app.agent.retriever import Retriever
from app.api import agent, backup, detection, health, lab, restore
from app.config import Settings, get_settings
from app.db.session import init_db, make_engine, make_session_factory
from app.detection.model import Detector, ModelNotAvailableError
from app.logging_config import RequestIdMiddleware, configure_logging
from app.storage.s3_client import S3Storage, StorageError

logger = logging.getLogger(__name__)

UI_DIR = Path(__file__).parent / "ui"


def _warm_embedder(index: VectorIndex) -> None:
    try:
        index.embedder.embed(["warm-up"])
    except Exception:  # never fatal: the first real search just loads it instead
        logger.warning("embedding model warm-up failed", exc_info=True)


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
        if (
            storage is None
            and settings.storage_mode == "local"
            and not (settings.aws_access_key_id and settings.aws_secret_access_key)
        ):
            logger.error(
                "STORAGE_MODE=local needs AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
                "(the object store's credentials). Run `python scripts/init_env.py` or "
                "copy .env.example to .env and fill them in. Backups will fail until then."
            )
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
        # Warm the embedder in the background (MiniLM + torch import takes ~25 s in
        # the container) so the first search/re-index doesn't pay for it.
        threading.Thread(target=_warm_embedder, args=(index,), daemon=True).start()
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
    app.include_router(lab.router)

    # Dashboard: static files only, calls the same JSON API (same origin).
    app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    @app.middleware("http")
    async def ui_security_headers(request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        if request.url.path.startswith("/ui"):
            # Strict CSP: no inline script/style, no third-party origins. Data shown
            # in the UI (file names!) comes from a disk ransomware controls.
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
            )
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "no-referrer"
        return response

    return app


app = create_app()
