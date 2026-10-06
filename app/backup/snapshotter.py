"""Snapshot creation, lookup, and diffing.

Write order for a snapshot is: blobs -> manifest -> DB commit. If anything fails
midway the DB never references a missing blob; at worst some unreferenced
(content-addressed, harmless) blobs remain in the bucket.
"""

from __future__ import annotations

import logging
import secrets
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.backup.manifest import FileEntry, scan_directory
from app.db.models import FileRecord, Incident, IncidentStatus, Snapshot, SnapshotState
from app.storage.s3_client import S3Storage, blob_key, manifest_key

logger = logging.getLogger(__name__)

# SQLite caps bound parameters per statement; query known hashes in batches.
_IN_CLAUSE_BATCH = 500
_UPLOAD_WORKERS = 8

# One snapshot at a time per process: two concurrent scans of the same tree
# would waste I/O and race on "previous snapshot" bookkeeping.
_snapshot_lock = threading.Lock()


class SnapshotError(RuntimeError):
    """Raised for invalid snapshot requests (e.g. missing source directory)."""


class SnapshotNotFoundError(LookupError):
    """Raised when a snapshot id does not exist."""


def new_snapshot_id(now: datetime) -> str:
    """Sortable, collision-resistant id: ``snap-20261005T120000Z-1a2b3c``."""
    return f"snap-{now.strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(3)}"


def _known_hashes(session: Session, hashes: Iterable[str]) -> set[str]:
    """Return the subset of ``hashes`` already referenced by some snapshot."""
    pending = list(set(hashes))
    known: set[str] = set()
    for i in range(0, len(pending), _IN_CLAUSE_BATCH):
        batch = pending[i : i + _IN_CLAUSE_BATCH]
        rows = session.scalars(
            select(FileRecord.sha256).where(FileRecord.sha256.in_(batch)).distinct()
        )
        known.update(rows)
    return known


def create_snapshot(
    session: Session,
    storage: S3Storage,
    source_dir: Path,
    exclude_patterns: Iterable[str] = (),
    label: str | None = None,
) -> Snapshot:
    """Scan ``source_dir``, upload new content, write the manifest, record metadata.

    Dedup: a blob is uploaded only if its hash is unknown to the DB *and* absent
    from the bucket. The DB check avoids a HEAD request for every unchanged file;
    the bucket check keeps us correct if the DB was wiped but the bucket was not.
    """
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        raise SnapshotError(f"Source directory does not exist: {source_dir}")

    with _snapshot_lock:
        now = datetime.now(UTC)
        snapshot_id = new_snapshot_id(now)
        entries = scan_directory(source_dir, exclude_patterns)

        known = _known_hashes(session, (e.sha256 for e in entries))
        # One representative file per unseen hash (duplicates share a blob).
        pending: dict[str, FileEntry] = {}
        for entry in entries:
            if entry.sha256 not in known:
                pending.setdefault(entry.sha256, entry)

        def _upload_if_missing(entry: FileEntry) -> bool:
            key = blob_key(entry.sha256)
            if storage.object_exists(key):
                return False
            storage.upload_file(source_dir / entry.path, key, entry.sha256)
            return True

        # Uploads are network-bound, so a small thread pool hides per-request
        # latency (boto3 clients are thread-safe). Hashing above stays serial:
        # it is disk-bound and parallel reads on one disk rarely help.
        with ThreadPoolExecutor(max_workers=_UPLOAD_WORKERS) as pool:
            uploaded = [
                e
                for e, did in zip(
                    pending.values(), pool.map(_upload_if_missing, pending.values()), strict=True
                )
                if did
            ]
        uploaded_bytes = sum(e.size for e in uploaded)

        previous = latest_snapshot(session)
        total_bytes = sum(e.size for e in entries)
        mean_entropy = (sum(e.entropy for e in entries) / len(entries)) if entries else 0.0

        manifest = {
            "snapshot_id": snapshot_id,
            "timestamp": now.isoformat(),
            "source_dir": str(source_dir),
            "label": label,
            "previous_snapshot_id": previous.id if previous else None,
            "file_count": len(entries),
            "total_bytes": total_bytes,
            "files": [e.to_dict() | {"object_key": blob_key(e.sha256)} for e in entries],
        }
        mkey = manifest_key(snapshot_id)
        storage.put_json(mkey, manifest)

        # While an incident is open the tree may already be partly encrypted, so
        # the snapshot is captured (evidence!) but never trusted as a restore point.
        state = SnapshotState.suspect if has_open_incident(session) else SnapshotState.clean

        snapshot = Snapshot(
            id=snapshot_id,
            created_at=now,
            state=state,
            source_dir=str(source_dir),
            label=label,
            manifest_key=mkey,
            file_count=len(entries),
            total_bytes=total_bytes,
            uploaded_files=len(uploaded),
            uploaded_bytes=uploaded_bytes,
            mean_entropy=round(mean_entropy, 4),
            files=[_to_record(e) for e in entries],
        )
        session.add(snapshot)
        session.commit()
        logger.info(
            "snapshot created",
            extra={
                "snapshot_id": snapshot_id,
                "files": len(entries),
                "uploaded_files": len(uploaded),
                "uploaded_bytes": uploaded_bytes,
            },
        )
        return snapshot


def _to_record(entry: FileEntry) -> FileRecord:
    return FileRecord(
        path=entry.path,
        size=entry.size,
        sha256=entry.sha256,
        entropy=entry.entropy,
        extension=entry.extension,
        mtime=entry.mtime,
        object_key=blob_key(entry.sha256),
    )


# -- queries --------------------------------------------------------------------
def latest_snapshot(session: Session) -> Snapshot | None:
    """Most recent snapshot, or None."""
    return session.scalars(
        select(Snapshot).order_by(Snapshot.created_at.desc(), Snapshot.id.desc()).limit(1)
    ).first()


def has_open_incident(session: Session) -> bool:
    """True if any incident is still open."""
    stmt = select(Incident.id).where(Incident.status == IncidentStatus.open).limit(1)
    return session.scalar(stmt) is not None


def list_snapshots(
    session: Session, limit: int = 50, offset: int = 0
) -> tuple[list[Snapshot], int]:
    """Return a page of snapshots (newest first) and the total count."""
    total = session.scalar(select(func.count()).select_from(Snapshot)) or 0
    rows = session.scalars(
        select(Snapshot)
        .order_by(Snapshot.created_at.desc(), Snapshot.id.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return list(rows), total


def get_snapshot(session: Session, snapshot_id: str) -> Snapshot:
    """Fetch a snapshot or raise SnapshotNotFoundError."""
    snap = session.get(Snapshot, snapshot_id)
    if snap is None:
        raise SnapshotNotFoundError(snapshot_id)
    return snap


# -- diff -----------------------------------------------------------------------
@dataclass
class SnapshotDiff:
    """Structured difference between two snapshots (``base`` -> ``target``)."""

    base_id: str
    target_id: str
    added: list[dict[str, Any]] = field(default_factory=list)
    removed: list[dict[str, Any]] = field(default_factory=list)
    modified: list[dict[str, Any]] = field(default_factory=list)
    renamed: list[dict[str, Any]] = field(default_factory=list)
    extension_changed: list[dict[str, Any]] = field(default_factory=list)
    unchanged_count: int = 0

    def summary(self) -> dict[str, Any]:
        """Aggregate counts + entropy stats â€” compact evidence for the detector/agent."""
        deltas = [m["entropy_delta"] for m in self.modified] + [
            c["entropy_delta"] for c in self.extension_changed
        ]
        return {
            "added": len(self.added),
            "removed": len(self.removed),
            "modified": len(self.modified),
            "renamed": len(self.renamed),
            "extension_changed": len(self.extension_changed),
            "unchanged": self.unchanged_count,
            "mean_entropy_delta": round(sum(deltas) / len(deltas), 4) if deltas else 0.0,
        }


def _strip_last_suffix(path: str) -> str:
    p = PurePosixPath(path)
    return str(p.with_suffix("")) if p.suffix else path


class FileLike(Protocol):
    """Anything diffable: an ORM ``FileRecord`` or a live-scan ``FileEntry``."""

    path: str
    size: int
    sha256: str
    entropy: float
    extension: str
    mtime: datetime


def diff_snapshots(base: Snapshot, target: Snapshot) -> SnapshotDiff:
    """Diff two stored snapshots (see :func:`diff_file_sets`)."""
    return diff_file_sets(base.id, base.files, target.id, target.files)


def diff_file_sets(
    base_id: str,
    base_files: Iterable[FileLike],
    target_id: str,
    target_files: Iterable[FileLike],
) -> SnapshotDiff:
    """Compare two file sets by path and content hash.

    Works on stored snapshots *and* on a live directory scan, which is how
    ``POST /detection/scan`` compares "last snapshot vs. disk right now".

    Classification, in order:
      * same path, same hash          -> unchanged
      * same path, different hash     -> modified (with entropy delta)
      * removed path + added path with the *same hash*  -> renamed
      * removed ``a/report.docx`` + added ``a/report.docx.locked`` (or
        ``a/report.enc``) -> extension_changed. This is the classic ransomware
        footprint, so it is surfaced separately instead of add+remove noise.
      * everything left -> added / removed
    """
    old = {f.path: f for f in base_files}
    new = {f.path: f for f in target_files}
    diff = SnapshotDiff(base_id=base_id, target_id=target_id)

    for path in old.keys() & new.keys():
        o, n = old[path], new[path]
        if o.sha256 == n.sha256:
            diff.unchanged_count += 1
        else:
            diff.modified.append(
                {
                    "path": path,
                    "size_before": o.size,
                    "size_after": n.size,
                    "entropy_before": o.entropy,
                    "entropy_after": n.entropy,
                    "entropy_delta": round(n.entropy - o.entropy, 4),
                    "mtime": n.mtime.isoformat(),
                }
            )

    removed = {p: old[p] for p in old.keys() - new.keys()}
    added = {p: new[p] for p in new.keys() - old.keys()}

    # Renames: match on identical content hash.
    removed_by_hash: dict[str, list[str]] = {}
    for p, rec in removed.items():
        removed_by_hash.setdefault(rec.sha256, []).append(p)
    for p in sorted(added):
        candidates = removed_by_hash.get(added[p].sha256)
        if candidates:
            old_path = candidates.pop(0)
            diff.renamed.append(
                {
                    "old_path": old_path,
                    "new_path": p,
                    "sha256": added[p].sha256,
                    "mtime": added[p].mtime.isoformat(),
                }
            )
            del removed[old_path]
            del added[p]

    # Extension changes: the new path minus its last suffix equals the old path
    # (appended extension) or the old path minus its suffix (replaced extension).
    removed_index: dict[str, str] = {}
    for p in removed:
        removed_index.setdefault(p, p)
        removed_index.setdefault(_strip_last_suffix(p), p)
    for p in sorted(added):
        stem = _strip_last_suffix(p)
        old_path = removed_index.get(stem)
        if old_path is None or old_path not in removed:
            continue
        o, n = removed.pop(old_path), added.pop(p)
        diff.extension_changed.append(
            {
                "old_path": old_path,
                "new_path": p,
                "old_extension": o.extension,
                "new_extension": n.extension,
                "content_changed": o.sha256 != n.sha256,
                "entropy_before": o.entropy,
                "entropy_after": n.entropy,
                "entropy_delta": round(n.entropy - o.entropy, 4),
                "mtime": n.mtime.isoformat(),
            }
        )

    diff.added = [
        {
            "path": p,
            "size": r.size,
            "entropy": r.entropy,
            "extension": r.extension,
            "mtime": r.mtime.isoformat(),
        }
        for p, r in sorted(added.items())
    ]
    diff.removed = [
        {"path": p, "size": r.size, "entropy": r.entropy, "extension": r.extension}
        for p, r in sorted(removed.items())
    ]
    diff.modified.sort(key=lambda m: m["path"])
    return diff
