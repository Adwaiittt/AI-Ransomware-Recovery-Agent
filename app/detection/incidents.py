"""Incident persistence: create/merge incidents and taint snapshots."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db.models import Incident, IncidentStatus, Snapshot, SnapshotState
from app.detection.model import Detection

AFFECTED_FILES_CAP = 500  # keep rows small; the count is stored separately


class IncidentNotFoundError(LookupError):
    """Raised when an incident id does not exist."""


def record_detection(
    session: Session,
    *,
    detection: Detection,
    features: dict[str, float],
    started_at: datetime,
    ended_at: datetime,
    affected_files: Sequence[str],
    source: str,
    model_name: str,
    threshold: float,
    cooldown_seconds: float,
) -> Incident:
    """Persist an alerting window as a new incident or merge into an open one.

    Merging: one attack spans many 10 s windows. If an open incident ended
    within ``cooldown_seconds`` of this window, extend it rather than opening
    dozens of incidents for the same event.

    Side effect: every *clean* snapshot taken at/after ``started_at`` becomes
    ``suspect`` — it may already contain encrypted files.
    """
    cutoff = started_at - timedelta(seconds=cooldown_seconds)
    incident = session.scalars(
        select(Incident)
        .where(Incident.status == IncidentStatus.open, Incident.ended_at >= cutoff)
        .order_by(Incident.ended_at.desc())
        .limit(1)
    ).first()

    if incident is None:
        incident = Incident(
            created_at=ended_at,
            started_at=started_at,
            ended_at=ended_at,
            source=source,
            status=IncidentStatus.open,
            score=detection.score,
            threshold=threshold,
            model_name=model_name,
            windows=1,
            top_features=detection.top_features,
            features=features,
            affected_files=sorted(set(affected_files))[:AFFECTED_FILES_CAP],
            affected_file_count=len(set(affected_files)),
        )
        session.add(incident)
    else:
        incident.started_at = min(incident.started_at, started_at)
        incident.ended_at = max(incident.ended_at, ended_at)
        incident.windows += 1
        if detection.score > incident.score:
            incident.score = detection.score
            incident.top_features = detection.top_features
            incident.features = features
        merged = set(incident.affected_files) | set(affected_files)
        incident.affected_files = sorted(merged)[:AFFECTED_FILES_CAP]
        incident.affected_file_count = max(incident.affected_file_count, len(merged))

    session.execute(
        update(Snapshot)
        .where(Snapshot.created_at >= incident.started_at, Snapshot.state == SnapshotState.clean)
        .values(state=SnapshotState.suspect)
    )
    session.commit()
    return incident


def list_incidents(
    session: Session,
    status: IncidentStatus | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Incident]:
    """Incidents newest first, optionally filtered by status."""
    stmt = select(Incident).order_by(Incident.started_at.desc(), Incident.id.desc())
    if status is not None:
        stmt = stmt.where(Incident.status == status)
    return list(session.scalars(stmt.limit(limit).offset(offset)))


def resolve_incident(session: Session, incident_id: int) -> Incident:
    """Mark an incident resolved so new snapshots are trusted again."""
    incident = session.get(Incident, incident_id)
    if incident is None:
        raise IncidentNotFoundError(incident_id)
    incident.status = IncidentStatus.resolved
    session.commit()
    return incident
