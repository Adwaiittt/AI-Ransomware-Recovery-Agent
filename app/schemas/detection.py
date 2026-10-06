"""Pydantic models for the detection API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import IncidentStatus


class ScanRequest(BaseModel):
    """Body for POST /detection/scan. Defaults: latest snapshot vs. live directory."""

    base_snapshot_id: str | None = None
    target_snapshot_id: str | None = Field(
        default=None, description="Compare against another snapshot instead of the live dir."
    )
    record_incident: bool = True


class TopFeature(BaseModel):
    """One feature that pushed the score up."""

    feature: str
    value: float
    benign_mean: float
    z_score: float


class WindowOut(BaseModel):
    """Score for one reconstructed time window."""

    start: datetime
    end: datetime
    event_count: int
    score: float
    is_alert: bool
    top_features: list[TopFeature]


class IncidentOut(BaseModel):
    """A recorded incident."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    started_at: datetime
    ended_at: datetime
    source: str
    status: IncidentStatus
    score: float
    threshold: float
    model_name: str
    windows: int
    top_features: list[TopFeature]
    features: dict[str, float]
    affected_file_count: int
    affected_files: list[str]


class ScanResponse(BaseModel):
    """Result of a scan."""

    base_snapshot_id: str
    target: str
    verdict: str  # "alert" | "clean"
    max_score: float
    threshold: float
    diff_summary: dict[str, Any]
    windows: list[WindowOut]
    incident: IncidentOut | None


class MonitorStatus(BaseModel):
    """Liveness of the watcher process, from its heartbeat row."""

    alive: bool
    last_seen: datetime | None = None
    watch_dir: str | None = None
    windows_scored: int = 0
    last_score: float = 0.0
    alerts: int = 0


class DetectionStatus(BaseModel):
    """Overall detection subsystem status."""

    model_loaded: bool
    model: dict[str, Any] | None
    monitor: MonitorStatus
    open_incidents: int
