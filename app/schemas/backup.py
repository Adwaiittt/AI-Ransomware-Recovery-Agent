"""Pydantic request/response models for the backup API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import SnapshotState


class CreateBackupRequest(BaseModel):
    """Body for POST /backups. The source dir is server-configured (WATCH_DIR),
    not client-supplied, so the API cannot be used to read arbitrary paths."""

    label: str | None = Field(default=None, max_length=255, examples=["before-upgrade"])


class FileRecordOut(BaseModel):
    """One file inside a snapshot."""

    model_config = ConfigDict(from_attributes=True)

    path: str
    size: int
    sha256: str
    entropy: float
    extension: str
    mtime: datetime


class SnapshotSummary(BaseModel):
    """Snapshot metadata without the file list."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime
    label: str | None
    state: SnapshotState
    source_dir: str
    manifest_key: str
    file_count: int
    total_bytes: int
    uploaded_files: int
    uploaded_bytes: int
    mean_entropy: float


class SnapshotDetail(SnapshotSummary):
    """Snapshot metadata including every file record."""

    files: list[FileRecordOut]


class SnapshotList(BaseModel):
    """Paginated list of snapshots."""

    total: int
    limit: int
    offset: int
    items: list[SnapshotSummary]


class DiffSummary(BaseModel):
    """Aggregate change counts."""

    added: int
    removed: int
    modified: int
    renamed: int
    extension_changed: int
    unchanged: int
    mean_entropy_delta: float


class SnapshotDiffOut(BaseModel):
    """Full diff between two snapshots."""

    base_id: str
    target_id: str
    summary: DiffSummary
    added: list[dict]
    removed: list[dict]
    modified: list[dict]
    renamed: list[dict]
    extension_changed: list[dict]
