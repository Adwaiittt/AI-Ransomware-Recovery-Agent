"""Restore endpoints."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from app.api.deps import get_app_settings, get_db, get_storage, schedule_reindex
from app.backup.snapshotter import SnapshotNotFoundError
from app.config import Settings
from app.db.models import RestoreJob
from app.restore.restorer import (
    FileAction,
    IncidentNotFoundForRestoreError,
    RestoreError,
    list_restore_jobs,
    restore_candidates,
    run_restore,
)
from app.schemas.restore import (
    CandidateOut,
    CandidatesResponse,
    FileActionOut,
    RestoreJobOut,
    RestoreRequest,
    RestoreResponse,
)
from app.storage.s3_client import S3Storage

router = APIRouter(prefix="/restore", tags=["restore"])

MAX_FILES_IN_RESPONSE = 1000


@router.get("/candidates", response_model=CandidatesResponse)
def candidates(
    incident_id: int | None = Query(None, description="Default: earliest open incident."),
    db: Session = Depends(get_db),
) -> CandidatesResponse:
    """Snapshots newest first; the recommended one is the last clean before the incident."""
    try:
        c = restore_candidates(db, incident_id)
    except IncidentNotFoundForRestoreError:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Incident {incident_id} not found"
        ) from None
    started = c.incident.started_at if c.incident else None
    return CandidatesResponse(
        reference_incident_id=c.incident.id if c.incident else None,
        incident_started_at=started,
        last_clean_snapshot_id=c.last_clean.id if c.last_clean else None,
        candidates=[
            CandidateOut(
                id=s.id,
                created_at=s.created_at,
                state=s.state,
                file_count=s.file_count,
                before_incident=(s.created_at < started) if started else None,
                recommended=c.last_clean is not None and s.id == c.last_clean.id,
            )
            for s in c.snapshots
        ],
    )


@router.post("", response_model=RestoreResponse)
def restore(
    body: RestoreRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    storage: S3Storage = Depends(get_storage),
    settings: Settings = Depends(get_app_settings),
) -> RestoreResponse:
    """Plan (dry run, default) or perform a restore, verifying SHA-256 of every file."""
    try:
        job, plan, outcome = run_restore(
            db,
            storage,
            snapshot_id=body.snapshot_id,
            target_path=body.target_path,
            watch_dir=settings.watch_dir,
            restore_dir=settings.restore_dir,
            dry_run=body.dry_run,
            force=body.force,
            quarantine_extras=body.quarantine_extras,
        )
    except SnapshotNotFoundError:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Snapshot {body.snapshot_id!r} not found"
        ) from None
    except RestoreError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    schedule_reindex(request, background_tasks)
    return RestoreResponse(
        job=RestoreJobOut.model_validate(job),
        summary={
            "create": plan.count(FileAction.create),
            "overwrite": plan.count(FileAction.overwrite),
            "unchanged": plan.count(FileAction.unchanged),
            "extra": len(plan.extra_files),
        },
        files=[
            FileActionOut(path=f.path, action=f.action.value, size=f.size)
            for f in plan.files[:MAX_FILES_IN_RESPONSE]
        ],
        files_truncated=len(plan.files) > MAX_FILES_IN_RESPONSE,
        extra_files=plan.extra_files[:MAX_FILES_IN_RESPONSE],
        quarantined=outcome.quarantined if outcome else [],
        quarantine_dir=str(outcome.quarantine_dir) if outcome and outcome.quarantine_dir else None,
        failures=outcome.failed if outcome else [],
    )


@router.get("/jobs", response_model=list[RestoreJobOut])
def jobs(limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)) -> list[RestoreJob]:
    """Restore audit log, newest first."""
    return list_restore_jobs(db, limit)
