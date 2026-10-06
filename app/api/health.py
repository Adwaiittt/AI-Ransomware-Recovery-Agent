"""Liveness/readiness endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_storage
from app.storage.s3_client import S3Storage

router = APIRouter(tags=["health"])


@router.get("/health")
def health(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    storage: S3Storage = Depends(get_storage),
) -> dict[str, object]:
    """Report DB and object-store reachability; 503 if either is down."""
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    storage_ok = storage.ping()
    ok = db_ok and storage_ok
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ok" if ok else "degraded",
        "database": db_ok,
        "storage": storage_ok,
        # informational: the API is useful without these
        "detection_model": request.app.state.detector is not None,
    }
