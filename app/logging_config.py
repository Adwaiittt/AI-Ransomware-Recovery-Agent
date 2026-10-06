"""Structured (JSON) logging and per-request IDs.

Every log line emitted while handling a request carries that request's ID, so
one alert or restore can be traced across log lines. The ID comes from an
incoming ``X-Request-ID`` header (if it looks safe) or is generated, and is
echoed back in the response header.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import time
import uuid
from datetime import UTC, datetime

from starlette.types import ASGIApp, Message, Receive, Scope, Send

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")  # never reflect arbitrary header content
_STD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}

access_logger = logging.getLogger("app.access")


class JsonFormatter(logging.Formatter):
    """One JSON object per line; ``extra={...}`` fields become top-level keys."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = request_id_var.get()
        if rid:
            payload["request_id"] = rid
        for key, value in vars(record).items():
            if key not in _STD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        return True


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Install a single root handler (idempotent; safe to call from app and worker)."""
    handler = logging.StreamHandler()
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s")
        )
        handler.addFilter(_RequestIdFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    for noisy in ("botocore", "boto3", "s3transfer", "urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class RequestIdMiddleware:
    """Pure-ASGI middleware: request ID context + one access log line per request.

    Pure ASGI (not BaseHTTPMiddleware) so the contextvar is visible in sync
    endpoints run in the threadpool and in background tasks.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = dict(scope.get("headers") or []).get(b"x-request-id", b"").decode("latin-1")
        rid = incoming if _SAFE_ID.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(rid)
        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                message.setdefault("headers", [])
                message["headers"] = [*message["headers"], (b"x-request-id", rid.encode())]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            access_logger.info(
                "request",
                extra={
                    "method": scope["method"],
                    "path": scope["path"],
                    "status": status_code,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                },
            )
            request_id_var.reset(token)
