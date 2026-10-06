"""Backup (snapshot) endpoints.

Handlers are plain ``def`` (not ``async def``): boto3 and file hashing are
blocking, and FastAPI runs sync handlers in a threadpool so the event loop
stays responsive.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from app.api.deps import get_app_settings, get_db, get_storage, schedule_reindex
from app.backup.snapshotter import (
    SnapshotError,
    SnapshotNotFoundError,
    create_snapshot,
    diff_snapshots,
    get_snapshot,
    list_snapshots,
)
from app.config import Settings
from app.db.models import Snapshot
from app.schemas.backup import (
    CreateBackupRequest,
    SnapshotDetail,
    SnapshotDiffOut,
    SnapshotList,
    SnapshotSummary,
)
from app.storage.s3_client import S3Storage, StorageError

router = APIRouter(prefix="/backups", tags=["backups"])


def _get_or_404(db: Session, snapshot_id: str) -> Snapshot:
    try:
        return get_snapshot(db, snapshot_id)
    except SnapshotNotFoundError:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Snapshot {snapshot_id!r} not found"
        ) from None


@router.post("", response_model=SnapshotSummary, status_code=status.HTTP_201_CREATED)
def trigger_backup(
    request: Request,
    background_tasks: BackgroundTasks,
    body: CreateBackupRequest | None = None,
    db: Session = Depends(get_db),
    storage: S3Storage = Depends(get_storage),
    settings: Settings = Depends(get_app_settings),
) -> Snapshot:
    """Snapshot the configured watch directory now."""
    try:
        snapshot = create_snapshot(
            db,
            storage,
            settings.watch_dir,
            settings.exclude_patterns,
            label=body.label if body else None,
        )
    except SnapshotError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except StorageError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    schedule_reindex(request, background_tasks)
    return snapshot


@router.get("", response_model=SnapshotList)
def get_backups(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> SnapshotList:
    """List snapshots, newest first."""
    rows, total = list_snapshots(db, limit=limit, offset=offset)
    return SnapshotList(
        total=total,
        limit=limit,
        offset=offset,
        items=[SnapshotSummary.model_validate(r) for r in rows],
    )


@router.get("/{snapshot_id}", response_model=SnapshotDetail)
def get_backup(snapshot_id: str, db: Session = Depends(get_db)) -> Snapshot:
    """Return one snapshot including its file records."""
    return _get_or_404(db, snapshot_id)


@router.get("/{snapshot_id}/diff/{other_id}", response_model=SnapshotDiffOut)
def get_backup_diff(
    snapshot_id: str, other_id: str, db: Session = Depends(get_db)
) -> SnapshotDiffOut:
    """Diff ``snapshot_id`` (base) against ``other_id`` (target)."""
    base = _get_or_404(db, snapshot_id)
    target = _get_or_404(db, other_id)
    diff = diff_snapshots(base, target)
    return SnapshotDiffOut(
        base_id=diff.base_id,
        target_id=diff.target_id,
        summary=diff.summary(),
        added=diff.added,
        removed=diff.removed,
        modified=diff.modified,
        renamed=diff.renamed,
        extension_changed=diff.extension_changed,
    )
