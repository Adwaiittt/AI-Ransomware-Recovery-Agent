"""Detection endpoints: on-demand scan, incidents, status."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import (
    get_app_settings,
    get_db,
    get_detector,
    get_detector_optional,
    schedule_reindex,
)
from app.backup.snapshotter import SnapshotNotFoundError
from app.config import Settings
from app.db.models import Incident, IncidentStatus, MonitorHeartbeat
from app.detection.incidents import IncidentNotFoundError, list_incidents, resolve_incident
from app.detection.model import Detector
from app.detection.scan import NoBaselineError, run_scan
from app.schemas.detection import (
    DetectionStatus,
    IncidentOut,
    MonitorStatus,
    ScanRequest,
    ScanResponse,
    TopFeature,
    WindowOut,
)

router = APIRouter(prefix="/detection", tags=["detection"])


@router.post("/scan", response_model=ScanResponse)
def scan(
    request: Request,
    background_tasks: BackgroundTasks,
    body: ScanRequest | None = None,
    db: Session = Depends(get_db),
    detector: Detector = Depends(get_detector),
    settings: Settings = Depends(get_app_settings),
) -> ScanResponse:
    """Score changes between a snapshot and the live directory (or another snapshot)."""
    body = body or ScanRequest()
    try:
        result = run_scan(
            db,
            detector,
            watch_dir=settings.watch_dir,
            exclude_patterns=settings.exclude_patterns,
            window_seconds=settings.detection_window_seconds,
            cooldown_seconds=settings.incident_cooldown_seconds,
            base_snapshot_id=body.base_snapshot_id,
            target_snapshot_id=body.target_snapshot_id,
            record=body.record_incident,
        )
    except NoBaselineError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except SnapshotNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Snapshot {exc} not found") from exc

    if result.incident is not None:
        schedule_reindex(request, background_tasks)
    return ScanResponse(
        base_snapshot_id=result.base_snapshot_id,
        target=result.target,
        verdict="alert" if result.is_alert else "clean",
        max_score=result.max_score,
        threshold=detector.threshold,
        diff_summary=result.diff_summary,
        windows=[
            WindowOut(
                start=w.start,
                end=w.end,
                event_count=w.event_count,
                score=w.detection.score,
                is_alert=w.detection.is_alert,
                top_features=[TopFeature(**f) for f in w.detection.top_features],
            )
            for w in result.windows
        ],
        incident=IncidentOut.model_validate(result.incident) if result.incident else None,
    )


@router.get("/incidents", response_model=list[IncidentOut])
def incidents(
    status_filter: IncidentStatus | None = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[Incident]:
    """List incidents, newest first."""
    return list_incidents(db, status_filter, limit, offset)


@router.post("/incidents/{incident_id}/resolve", response_model=IncidentOut)
def resolve(
    incident_id: int,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> Incident:
    """Close an incident (after recovery) so new snapshots are trusted as clean again."""
    try:
        incident = resolve_incident(db, incident_id)
    except IncidentNotFoundError:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Incident {incident_id} not found"
        ) from None
    schedule_reindex(request, background_tasks)
    return incident


@router.get("/status", response_model=DetectionStatus)
def detection_status(
    db: Session = Depends(get_db),
    detector: Detector | None = Depends(get_detector_optional),
    settings: Settings = Depends(get_app_settings),
) -> DetectionStatus:
    """Model info, watcher liveness (heartbeat), and open incident count."""
    hb = db.scalars(select(MonitorHeartbeat).limit(1)).first()
    if hb is None:
        monitor = MonitorStatus(alive=False)
    else:
        # Alive if it checked in within ~3 windows.
        stale_after = timedelta(seconds=3 * settings.detection_window_seconds)
        monitor = MonitorStatus(
            alive=datetime.now(UTC) - hb.last_seen <= stale_after,
            last_seen=hb.last_seen,
            watch_dir=hb.watch_dir,
            windows_scored=hb.windows_scored,
            last_score=hb.last_score,
            alerts=hb.alerts,
        )
    open_count = db.scalar(
        select(func.count()).select_from(Incident).where(Incident.status == IncidentStatus.open)
    )
    return DetectionStatus(
        model_loaded=detector is not None,
        model=detector.info() if detector else None,
        monitor=monitor,
        open_incidents=open_count or 0,
    )
