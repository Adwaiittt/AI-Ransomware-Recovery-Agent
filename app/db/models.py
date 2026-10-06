"""SQLAlchemy ORM models for backup metadata."""

from __future__ import annotations

import enum
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Dialect,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    TypeDecorator,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class UTCDateTime(TypeDecorator[datetime]):
    """Store datetimes as UTC and always return them timezone-aware.

    SQLite has no timezone type and hands back naive datetimes; without this,
    API responses would silently lose the "Z" and time comparisons in later
    phases (incident vs snapshot timestamps) could mix naive and aware values.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class SnapshotState(enum.StrEnum):
    """Trust state of a snapshot.

    Every snapshot starts ``clean``. The detector (Phase 2) downgrades snapshots
    taken after an incident to ``suspect``; an analyst/restore flow (Phase 3) can
    confirm ``infected``. "Last clean snapshot" logic only trusts ``clean``.
    """

    clean = "clean"
    suspect = "suspect"
    infected = "infected"


class Snapshot(Base):
    """One point-in-time backup of the watched directory."""

    __tablename__ = "snapshots"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    source_dir: Mapped[str] = mapped_column(String(1024))
    label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    state: Mapped[SnapshotState] = mapped_column(
        Enum(SnapshotState, native_enum=False, length=16),
        default=SnapshotState.clean,
        index=True,
    )
    manifest_key: Mapped[str] = mapped_column(String(512))

    file_count: Mapped[int] = mapped_column(Integer, default=0)
    total_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    uploaded_files: Mapped[int] = mapped_column(Integer, default=0)
    uploaded_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    mean_entropy: Mapped[float] = mapped_column(Float, default=0.0)

    files: Mapped[list[FileRecord]] = relationship(
        back_populates="snapshot", cascade="all, delete-orphan", lazy="selectin"
    )


class FileRecord(Base):
    """A single file as it existed in a given snapshot."""

    __tablename__ = "file_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("snapshots.id", ondelete="CASCADE"), index=True
    )
    path: Mapped[str] = mapped_column(String(1024))  # POSIX path relative to source_dir
    size: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    entropy: Mapped[float] = mapped_column(Float)  # Shannon entropy, bits/byte (0..8)
    extension: Mapped[str] = mapped_column(String(64))
    mtime: Mapped[datetime] = mapped_column(UTCDateTime())
    object_key: Mapped[str] = mapped_column(String(512))

    snapshot: Mapped[Snapshot] = relationship(back_populates="files")

    __table_args__ = (Index("ix_file_records_snapshot_path", "snapshot_id", "path", unique=True),)


class IncidentStatus(enum.StrEnum):
    """Lifecycle of an incident. ``open`` incidents taint new snapshots as suspect."""

    open = "open"
    resolved = "resolved"


class Incident(Base):
    """A detected burst of ransomware-like activity.

    ``started_at`` is the first event in the first alerting window; Phase 3's
    "last clean snapshot" is the newest clean snapshot strictly before it.
    """

    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    ended_at: Mapped[datetime] = mapped_column(UTCDateTime())
    source: Mapped[str] = mapped_column(String(32))  # "monitor" | "scan"
    status: Mapped[IncidentStatus] = mapped_column(
        Enum(IncidentStatus, native_enum=False, length=16),
        default=IncidentStatus.open,
        index=True,
    )
    score: Mapped[float] = mapped_column(Float)  # max window score seen
    threshold: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(64))
    windows: Mapped[int] = mapped_column(Integer, default=1)
    top_features: Mapped[list[dict]] = mapped_column(JSON, default=list)
    features: Mapped[dict] = mapped_column(JSON, default=dict)  # highest-scoring window
    affected_files: Mapped[list[str]] = mapped_column(JSON, default=list)
    affected_file_count: Mapped[int] = mapped_column(Integer, default=0)


class MonitorHeartbeat(Base):
    """Single-row table the watcher process updates every window.

    The API runs in a different process/container, so this is how
    ``GET /detection/status`` knows whether the monitor is alive.
    """

    __tablename__ = "monitor_heartbeat"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    last_seen: Mapped[datetime] = mapped_column(UTCDateTime())
    watch_dir: Mapped[str] = mapped_column(String(1024))
    windows_scored: Mapped[int] = mapped_column(Integer, default=0)
    last_score: Mapped[float] = mapped_column(Float, default=0.0)
    alerts: Mapped[int] = mapped_column(Integer, default=0)


class RestoreJob(Base):
    """Audit record of every restore (dry runs included).

    Restores are the most dangerous operation in the system, so each one is
    logged with who/what/where and the verification outcome. The agent also
    reads these to answer "was anything restored already?".
    """

    __tablename__ = "restore_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("snapshots.id"), index=True)
    target_path: Mapped[str] = mapped_column(String(1024))
    in_place: Mapped[bool] = mapped_column(default=False)
    dry_run: Mapped[bool] = mapped_column(default=True)
    status: Mapped[str] = mapped_column(String(16))  # planned | succeeded | failed
    files_total: Mapped[int] = mapped_column(Integer, default=0)
    files_restored: Mapped[int] = mapped_column(Integer, default=0)
    files_unchanged: Mapped[int] = mapped_column(Integer, default=0)
    files_verified: Mapped[int] = mapped_column(Integer, default=0)
    files_failed: Mapped[int] = mapped_column(Integer, default=0)
    extra_files: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(String(2048), nullable=True)


class AgentChunk(Base):
    """One retrievable text chunk derived from backup metadata (RAG corpus).

    Embeddings live here (bytes) so the FAISS index is a disposable in-memory
    cache rebuilt from this table — no separate index file to keep in sync.
    ``text_hash`` lets re-indexing skip chunks whose text did not change.
    """

    __tablename__ = "agent_chunks"

    row_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(128), unique=True)  # e.g. "incident:3"
    kind: Mapped[str] = mapped_column(String(32), index=True)
    timestamp: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    text: Mapped[str] = mapped_column(String(8000))
    text_hash: Mapped[str] = mapped_column(String(64))
    embedder: Mapped[str] = mapped_column(String(128))
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    snapshot_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    incident_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
