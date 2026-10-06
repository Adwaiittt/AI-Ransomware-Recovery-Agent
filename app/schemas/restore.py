"""Pydantic models for the restore API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import SnapshotState


class CandidateOut(BaseModel):
    """One snapshot as a restore candidate."""

    id: str
    created_at: datetime
    state: SnapshotState
    file_count: int
    before_incident: bool | None  # None when there is no reference incident
    recommended: bool


class CandidatesResponse(BaseModel):
    """Restore candidates relative to the reference incident."""

    reference_incident_id: int | None
    incident_started_at: datetime | None
    last_clean_snapshot_id: str | None
    candidates: list[CandidateOut]


class RestoreRequest(BaseModel):
    """Body for POST /restore.

    ``dry_run`` defaults to **true**: the safe call only reports what would change.
    """

    snapshot_id: str
    target_path: str | None = Field(
        default=None,
        description="Default RESTORE_DIR/<snapshot_id>. Use the watch dir for in-place.",
    )
    dry_run: bool = True
    force: bool = Field(default=False, description="Allow restoring a suspect/infected snapshot.")
    quarantine_extras: bool = Field(
        default=False,
        description="In-place only: move files not in the snapshot (e.g. *.locked) aside.",
    )


class FileActionOut(BaseModel):
    """Planned/performed action for one file."""

    path: str
    action: str
    size: int


class RestoreJobOut(BaseModel):
    """Audit row for a restore."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    finished_at: datetime | None
    snapshot_id: str
    target_path: str
    in_place: bool
    dry_run: bool
    status: str
    files_total: int
    files_restored: int
    files_unchanged: int
    files_verified: int
    files_failed: int
    extra_files: int
    error: str | None


class RestoreResponse(BaseModel):
    """Result of POST /restore."""

    job: RestoreJobOut
    summary: dict[str, int]
    files: list[FileActionOut]
    files_truncated: bool
    extra_files: list[str]
    quarantined: list[str]
    quarantine_dir: str | None
    failures: list[dict[str, str]]
