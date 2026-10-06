"""Find the last clean snapshot, plan a restore, execute it, verify every file.

Safety model:
  * Default target is a *separate* directory (``RESTORE_DIR/<snapshot_id>``).
    Restoring over the live tree (``in_place``) must be asked for explicitly.
  * Only ``clean`` snapshots restore without ``force``.
  * Every destination path is resolved and must stay inside the target root
    (a tampered DB row like ``../../etc/passwd`` cannot escape).
  * Each blob is downloaded to a temp file, its SHA-256 checked, and only then
    atomically moved into place (``os.replace``), so a failed or corrupted
    download never leaves a half-written file behind.
  * Files present in the target but absent from the snapshot (``*.locked``,
    ransom notes) are never deleted; optionally they are *quarantined* (moved
    aside) so evidence is preserved.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import shutil
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.backup.manifest import iter_files
from app.backup.snapshotter import get_snapshot
from app.db.models import Incident, IncidentStatus, RestoreJob, Snapshot, SnapshotState
from app.storage.s3_client import S3Storage

logger = logging.getLogger(__name__)

_restore_lock = threading.Lock()


class RestoreError(RuntimeError):
    """Invalid restore request (bad target, unsafe path, untrusted snapshot)."""


class IncidentNotFoundForRestoreError(LookupError):
    """Raised when a referenced incident id does not exist."""


# -- choosing the snapshot -------------------------------------------------------
def select_last_clean(snapshots: Iterable[Snapshot], before: datetime | None) -> Snapshot | None:
    """Newest ``clean`` snapshot strictly before ``before`` (or newest clean if None).

    Pure function (no DB) so the core rule is trivially unit-testable:
    suspect/infected snapshots are never chosen, and a snapshot taken at the
    exact incident start time is not considered "before" it.
    """
    eligible = [
        s
        for s in snapshots
        if s.state is SnapshotState.clean and (before is None or s.created_at < before)
    ]
    return max(eligible, key=lambda s: (s.created_at, s.id), default=None)


def reference_incident(session: Session, incident_id: int | None = None) -> Incident | None:
    """The incident restore is measured against.

    Explicit ``incident_id`` wins; otherwise the *earliest* open incident, because
    anything after the first sign of compromise is untrustworthy.
    """
    if incident_id is not None:
        incident = session.get(Incident, incident_id)
        if incident is None:
            raise IncidentNotFoundForRestoreError(incident_id)
        return incident
    return session.scalars(
        select(Incident)
        .where(Incident.status == IncidentStatus.open)
        .order_by(Incident.started_at.asc())
        .limit(1)
    ).first()


@dataclass
class Candidates:
    """Restore candidates and the recommended (last clean) snapshot."""

    incident: Incident | None
    last_clean: Snapshot | None
    snapshots: list[Snapshot]


def restore_candidates(session: Session, incident_id: int | None = None) -> Candidates:
    """All snapshots newest first, plus the last clean one before the reference incident."""
    incident = reference_incident(session, incident_id)
    snapshots = list(
        session.scalars(select(Snapshot).order_by(Snapshot.created_at.desc(), Snapshot.id.desc()))
    )
    before = incident.started_at if incident else None
    return Candidates(incident, select_last_clean(snapshots, before), snapshots)


# -- planning -------------------------------------------------------------------
class FileAction(StrEnum):
    """What a restore will do to one file."""

    create = "create"  # missing in target
    overwrite = "overwrite"  # present but content differs
    unchanged = "unchanged"  # already identical


@dataclass(frozen=True)
class FilePlan:
    """Planned action for one file."""

    path: str
    action: FileAction
    size: int
    sha256: str
    object_key: str
    mtime: datetime


@dataclass
class RestorePlan:
    """Everything a restore would do; a dry run returns exactly this."""

    snapshot_id: str
    target: Path
    in_place: bool
    files: list[FilePlan] = field(default_factory=list)
    extra_files: list[str] = field(default_factory=list)

    def count(self, action: FileAction) -> int:
        return sum(1 for f in self.files if f.action is action)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def safe_join(root: Path, rel: str) -> Path:
    """Join a stored relative path to ``root`` and refuse anything that escapes it."""
    dest = (root / rel).resolve()
    if not dest.is_relative_to(root) or dest == root:
        raise RestoreError(f"Unsafe path in snapshot: {rel!r}")
    return dest


def resolve_target(
    target_path: str | None, snapshot_id: str, watch_dir: Path, restore_dir: Path
) -> tuple[Path, bool]:
    """Validate the requested target; returns (resolved path, in_place).

    Allowed: anywhere under RESTORE_DIR (default ``RESTORE_DIR/<snapshot_id>``),
    or exactly the watch directory for an in-place restore. Sub-folders of the
    watch dir are refused: they would nest a second copy inside the live tree.
    """
    watch = watch_dir.resolve()
    restore_root = restore_dir.resolve()
    target = (Path(target_path) if target_path else restore_root / snapshot_id).resolve()
    if target == watch:
        return target, True
    if target.is_relative_to(restore_root) and target != restore_root:
        return target, False
    raise RestoreError(
        f"Target must be the watch directory ({watch}) or inside RESTORE_DIR ({restore_root})."
    )


def plan_restore(snapshot: Snapshot, target: Path, in_place: bool) -> RestorePlan:
    """Compare the snapshot with what is on disk at ``target`` (no writes)."""
    plan = RestorePlan(snapshot_id=snapshot.id, target=target, in_place=in_place)
    wanted = set()
    for rec in sorted(snapshot.files, key=lambda r: r.path):
        dest = safe_join(target, rec.path)
        wanted.add(dest)
        if dest.is_file() and not dest.is_symlink():
            action = FileAction.unchanged if _sha256(dest) == rec.sha256 else FileAction.overwrite
        else:
            action = FileAction.create
        plan.files.append(
            FilePlan(rec.path, action, rec.size, rec.sha256, rec.object_key, rec.mtime)
        )
    if target.is_dir():
        plan.extra_files = sorted(
            p.relative_to(target).as_posix()
            for p in iter_files(target)
            if p.resolve() not in wanted
        )
    return plan


# -- execution ------------------------------------------------------------------
@dataclass
class RestoreOutcome:
    """Result of executing a plan."""

    restored: int = 0
    verified: int = 0
    failed: list[dict[str, str]] = field(default_factory=list)
    quarantined: list[str] = field(default_factory=list)
    quarantine_dir: Path | None = None


def execute_plan(
    plan: RestorePlan,
    storage: S3Storage,
    *,
    quarantine_dir: Path | None = None,
) -> RestoreOutcome:
    """Download + verify + atomically place each file; optionally quarantine extras."""
    out = RestoreOutcome(quarantine_dir=quarantine_dir)
    plan.target.mkdir(parents=True, exist_ok=True)
    for fp in plan.files:
        dest = safe_join(plan.target, fp.path)
        if fp.action is FileAction.unchanged:
            out.verified += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.restore-{secrets.token_hex(4)}.tmp")
        try:
            storage.download_file(fp.object_key, tmp)
            if _sha256(tmp) != fp.sha256:
                raise RestoreError("downloaded blob hash mismatch (corrupted object?)")
            os.replace(tmp, dest)  # atomic on the same filesystem
            ts = fp.mtime.timestamp()
            os.utime(dest, (ts, ts))  # keep original mtimes for fidelity
            out.restored += 1
            if _sha256(dest) == fp.sha256:
                out.verified += 1
            else:
                out.failed.append({"path": fp.path, "error": "post-restore verification failed"})
        except Exception as exc:
            out.failed.append({"path": fp.path, "error": str(exc)})
            tmp.unlink(missing_ok=True)

    if quarantine_dir is not None:
        for rel in plan.extra_files:
            src = safe_join(plan.target, rel)
            dst = quarantine_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            out.quarantined.append(rel)
    return out


def run_restore(
    session: Session,
    storage: S3Storage,
    *,
    snapshot_id: str,
    target_path: str | None,
    watch_dir: Path,
    restore_dir: Path,
    dry_run: bool = True,
    force: bool = False,
    quarantine_extras: bool = False,
) -> tuple[RestoreJob, RestorePlan, RestoreOutcome | None]:
    """Validate, plan, (optionally) execute, and audit a restore."""
    snapshot = get_snapshot(session, snapshot_id)
    if snapshot.state is not SnapshotState.clean and not force:
        raise RestoreError(
            f"Snapshot {snapshot_id} is {snapshot.state.value}; "
            "pass force=true to restore it anyway."
        )
    target, in_place = resolve_target(target_path, snapshot_id, watch_dir, restore_dir)
    if quarantine_extras and not in_place:
        raise RestoreError("quarantine_extras only applies to in-place restores.")

    with _restore_lock:
        job = RestoreJob(
            created_at=datetime.now(UTC),
            snapshot_id=snapshot.id,
            target_path=str(target),
            in_place=in_place,
            dry_run=dry_run,
            status="planned",
        )
        session.add(job)
        session.flush()  # assigns job.id (used for the quarantine folder name)

        plan = plan_restore(snapshot, target, in_place)
        job.files_total = len(plan.files)
        job.files_unchanged = plan.count(FileAction.unchanged)
        job.extra_files = len(plan.extra_files)

        outcome: RestoreOutcome | None = None
        if not dry_run:
            qdir = (
                restore_dir.resolve() / "quarantine" / f"job-{job.id}"
                if quarantine_extras
                else None
            )
            try:
                outcome = execute_plan(plan, storage, quarantine_dir=qdir)
                job.files_restored = outcome.restored
                job.files_verified = outcome.verified
                job.files_failed = len(outcome.failed)
                job.status = "failed" if outcome.failed else "succeeded"
                if outcome.failed:
                    job.error = f"{len(outcome.failed)} file(s) failed; first: {outcome.failed[0]}"
            except Exception as exc:  # unexpected (e.g. storage down mid-restore)
                job.status = "failed"
                job.error = str(exc)[:2000]
                logger.exception("restore job %s failed", job.id)
        job.finished_at = datetime.now(UTC)
        session.commit()
        logger.info(
            "restore job finished",
            extra={"job_id": job.id, "snapshot_id": snapshot.id, "status": job.status},
        )
        return job, plan, outcome


def list_restore_jobs(session: Session, limit: int = 50) -> list[RestoreJob]:
    """Most recent restore jobs first."""
    return list(session.scalars(select(RestoreJob).order_by(RestoreJob.id.desc()).limit(limit)))
