"""Turn backup metadata into text chunks and keep a FAISS index over them.

Chunk kinds (one row each, keyed so re-indexing is idempotent):
  snapshot:<id>        what a snapshot contains + its trust state
  diff:<prev>-><id>    what changed between consecutive snapshots (with evidence)
  incident:<id>        detector alert: time range, score, top features, files
  restore:<id>         restore audit record

Indexing is *incremental*: chunks are rebuilt from the DB (cheap), but only
those whose text hash changed are re-embedded (the expensive part). The FAISS
index itself is in-memory and rebuilt from stored embeddings after each sync.

Freshness: ``ensure_fresh`` compares a cheap DB fingerprint (counts, latest
timestamps, state tallies) with the one seen at the last sync. That catches
changes written by *other processes* — e.g. incidents recorded by the watcher
container — without any cross-process signalling.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import faiss
import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.agent.embeddings import Embedder
from app.backup.snapshotter import diff_snapshots
from app.db.models import AgentChunk, Incident, RestoreJob, Snapshot

logger = logging.getLogger(__name__)

MAX_EXAMPLES = 8  # file examples per chunk; counts carry the rest


@dataclass(frozen=True)
class ChunkSpec:
    """A chunk before embedding."""

    key: str
    kind: str
    timestamp: datetime
    text: str
    snapshot_id: str | None = None
    incident_id: int | None = None


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC (%A)")


def _snapshot_chunk(s: Snapshot) -> ChunkSpec:
    exts: dict[str, int] = {}
    for f in s.files:
        exts[f.extension or "(none)"] = exts.get(f.extension or "(none)", 0) + 1
    top_exts = ", ".join(f"{e}: {n}" for e, n in sorted(exts.items(), key=lambda kv: -kv[1])[:8])
    high = sum(1 for f in s.files if f.entropy > 7.5)
    text = (
        f"Snapshot {s.id} taken {_ts(s.created_at)}. State: {s.state.value}. "
        f"Label: {s.label or 'none'}. {s.file_count} files, {s.total_bytes} bytes, "
        f"mean entropy {s.mean_entropy:.2f} bits/byte, {high} files above 7.5 bits/byte. "
        f"New content uploaded: {s.uploaded_files} files ({s.uploaded_bytes} bytes). "
        f"Extensions: {top_exts or 'none'}."
    )
    return ChunkSpec(f"snapshot:{s.id}", "snapshot", s.created_at, text, snapshot_id=s.id)


def _diff_chunk(prev: Snapshot, cur: Snapshot) -> ChunkSpec:
    d = diff_snapshots(prev, cur)
    sm = d.summary()
    lines = [
        f"Changes between snapshot {prev.id} ({_ts(prev.created_at)}, {prev.state.value}) and "
        f"snapshot {cur.id} ({_ts(cur.created_at)}, {cur.state.value}): "
        f"{sm['added']} added, {sm['removed']} removed, {sm['modified']} modified, "
        f"{sm['renamed']} renamed, {sm['extension_changed']} extension changes, "
        f"{sm['unchanged']} unchanged. Mean entropy delta of rewritten files: "
        f"{sm['mean_entropy_delta']:+.2f} bits/byte."
    ]
    if d.extension_changed:
        new_exts = sorted({c["new_extension"] for c in d.extension_changed})
        ex = "; ".join(
            f"{c['old_path']} -> {c['new_path']} (entropy {c['entropy_before']:.2f}->"
            f"{c['entropy_after']:.2f})"
            for c in d.extension_changed[:MAX_EXAMPLES]
        )
        lines.append(f"Extension changes to {', '.join(new_exts)}: {ex}.")
    if d.modified:
        top = sorted(d.modified, key=lambda m: -m["entropy_delta"])[:MAX_EXAMPLES]
        ex = "; ".join(
            f"{m['path']} ({m['entropy_before']:.2f}->{m['entropy_after']:.2f})" for m in top
        )
        lines.append(f"Largest entropy increases among modified files: {ex}.")
    if d.added:
        lines.append("Added: " + ", ".join(a["path"] for a in d.added[:MAX_EXAMPLES]) + ".")
    if d.removed:
        lines.append("Removed: " + ", ".join(r["path"] for r in d.removed[:MAX_EXAMPLES]) + ".")
    return ChunkSpec(
        f"diff:{prev.id}->{cur.id}", "diff", cur.created_at, " ".join(lines), snapshot_id=cur.id
    )


def _incident_chunk(i: Incident) -> ChunkSpec:
    feats = ", ".join(
        f"{f['feature']}={f['value']} (z={f['z_score']})" for f in (i.top_features or [])
    )
    files = ", ".join(i.affected_files[:MAX_EXAMPLES])
    text = (
        f"Incident {i.id} ({i.status.value}) detected by {i.source}: ransomware-like activity "
        f"from {_ts(i.started_at)} to {_ts(i.ended_at)}. Model {i.model_name} score "
        f"{i.score:.3f} (threshold {i.threshold:.3f}) over {i.windows} window(s). "
        f"Top contributing features: {feats or 'n/a'}. {i.affected_file_count} files affected, "
        f"e.g. {files or 'n/a'}. Snapshots taken after the incident start are suspect."
    )
    return ChunkSpec(f"incident:{i.id}", "incident", i.started_at, text, incident_id=i.id)


def _restore_chunk(j: RestoreJob) -> ChunkSpec:
    kind = "dry-run plan" if j.dry_run else "restore"
    where = "in place over the watched directory" if j.in_place else f"into {j.target_path}"
    text = (
        f"Restore job {j.id} ({kind}, status {j.status}) at {_ts(j.created_at)}: snapshot "
        f"{j.snapshot_id} {where}. {j.files_total} files in snapshot, {j.files_restored} "
        f"restored, {j.files_unchanged} already identical, {j.files_verified} SHA-256 verified, "
        f"{j.files_failed} failed, {j.extra_files} extra files not in the snapshot."
    )
    return ChunkSpec(f"restore:{j.id}", "restore", j.created_at, text, snapshot_id=j.snapshot_id)


def build_chunks(session: Session) -> list[ChunkSpec]:
    """Derive every chunk from the current DB state."""
    snaps = list(session.scalars(select(Snapshot).order_by(Snapshot.created_at, Snapshot.id)))
    chunks = [_snapshot_chunk(s) for s in snaps]
    chunks += [_diff_chunk(a, b) for a, b in zip(snaps, snaps[1:], strict=False)]
    chunks += [_incident_chunk(i) for i in session.scalars(select(Incident))]
    # Dry runs change nothing; indexing them let planning noise outrank the
    # incidents and diffs that actually answer "what happened?" (seen in the UI).
    # They remain in restore_jobs / GET /restore/jobs for auditing.
    real_jobs = select(RestoreJob).where(RestoreJob.dry_run.is_(False))
    chunks += [_restore_chunk(j) for j in session.scalars(real_jobs)]
    return chunks


def db_fingerprint(session: Session) -> tuple:
    """Cheap summary that changes whenever indexable metadata changes."""
    snap = session.execute(select(func.count(Snapshot.id), func.max(Snapshot.created_at))).one()
    states = tuple(
        session.execute(
            select(Snapshot.state, func.count()).group_by(Snapshot.state).order_by(Snapshot.state)
        ).all()
    )
    inc = session.execute(
        select(func.count(Incident.id), func.max(Incident.ended_at), func.sum(Incident.windows))
    ).one()
    inc_status = tuple(
        session.execute(
            select(Incident.status, func.count())
            .group_by(Incident.status)
            .order_by(Incident.status)
        ).all()
    )
    jobs = session.execute(select(func.count(RestoreJob.id), func.max(RestoreJob.id))).one()
    return (tuple(snap), states, tuple(inc), inc_status, tuple(jobs))


@dataclass(frozen=True)
class Hit:
    """A retrieved chunk with its similarity score."""

    key: str
    kind: str
    timestamp: datetime
    text: str
    score: float
    snapshot_id: str | None
    incident_id: int | None


class VectorIndex:
    """Incrementally synced chunk store + in-memory FAISS inner-product index."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._index: faiss.IndexIDMap2 | None = None
        self._fingerprint: tuple | None = None
        self._lock = threading.RLock()

    # -- sync -------------------------------------------------------------------
    def sync(self, session: Session) -> dict[str, int]:
        """Bring stored chunks + FAISS index in line with the DB. Returns stats."""
        with self._lock:
            # Fingerprint BEFORE reading the data. Embedding can take seconds (the
            # first call loads the model); if another process writes meanwhile,
            # an end-of-sync fingerprint would describe data this index never
            # saw and mark it "fresh" forever. A start-of-sync fingerprint can
            # only cause one extra sync, never a stale index.
            fingerprint = db_fingerprint(session)
            specs = {c.key: c for c in build_chunks(session)}
            existing = {row.key: row for row in session.scalars(select(AgentChunk))}

            stale = [k for k in existing if k not in specs]
            if stale:
                session.execute(delete(AgentChunk).where(AgentChunk.key.in_(stale)))

            to_embed: list[ChunkSpec] = []
            for key, spec in specs.items():
                h = hashlib.sha256(spec.text.encode()).hexdigest()
                row = existing.get(key)
                if row is None or row.text_hash != h or row.embedder != self.embedder.name:
                    to_embed.append(spec)
            vectors = self.embedder.embed([c.text for c in to_embed])
            for spec, vec in zip(to_embed, vectors, strict=True):
                row = existing.get(spec.key) or AgentChunk(key=spec.key)
                row.kind = spec.kind
                row.timestamp = spec.timestamp
                row.text = spec.text
                row.text_hash = hashlib.sha256(spec.text.encode()).hexdigest()
                row.embedder = self.embedder.name
                row.embedding = np.asarray(vec, dtype=np.float32).tobytes()
                row.snapshot_id = spec.snapshot_id
                row.incident_id = spec.incident_id
                session.add(row)
            session.commit()

            self._rebuild(session)
            self._fingerprint = fingerprint
            stats = {"chunks": len(specs), "embedded": len(to_embed), "removed": len(stale)}
            logger.info("agent index synced", extra=stats)
            return stats

    def _rebuild(self, session: Session) -> None:
        rows = session.execute(select(AgentChunk.row_id, AgentChunk.embedding)).all()
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.embedder.dim))
        if rows:
            ids = np.array([r[0] for r in rows], dtype=np.int64)
            mat = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
            index.add_with_ids(mat, ids)
        self._index = index

    def ensure_fresh(self, session: Session) -> bool:
        """Sync only if the DB changed since the last sync. Returns True if it synced."""
        with self._lock:
            if self._index is not None and self._fingerprint == db_fingerprint(session):
                return False
            self.sync(session)
            return True

    def sync_with_factory(self, factory: sessionmaker[Session]) -> None:
        """Background-task entry point (own session, never raises)."""
        try:
            with factory() as session:
                self.ensure_fresh(session)
        except Exception:  # an index refresh must never break the request that triggered it
            logger.exception("background agent re-index failed")

    # -- search -----------------------------------------------------------------
    def search(
        self,
        session: Session,
        query: str,
        k: int = 8,
        start: datetime | None = None,
        end: datetime | None = None,
        kinds: Sequence[str] | None = None,
    ) -> list[Hit]:
        """Semantic search, optionally restricted to a time window and chunk kinds.

        Filtering happens *inside* FAISS via an ID selector (candidate ids come
        from an indexed SQL range query), so a narrow date range still returns
        the top-k matches within that range rather than top-k overall minus
        whatever falls outside.
        """
        with self._lock:
            if self._index is None:
                self.sync(session)
            assert self._index is not None
            if self._index.ntotal == 0:
                return []

            params = None
            if start or end or kinds:
                stmt = select(AgentChunk.row_id)
                if start:
                    stmt = stmt.where(AgentChunk.timestamp >= start)
                if end:
                    stmt = stmt.where(AgentChunk.timestamp < end)
                if kinds:
                    stmt = stmt.where(AgentChunk.kind.in_(list(kinds)))
                allowed = np.array(list(session.scalars(stmt)), dtype=np.int64)
                if allowed.size == 0:
                    return []
                params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(allowed))

            qv = self.embedder.embed([query])
            n = min(k, self._index.ntotal)
            scores, ids = self._index.search(qv, n, params=params)

        found = [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i != -1]
        if not found:
            return []
        rows = {r.row_id: r for r in session.scalars(
            select(AgentChunk).where(AgentChunk.row_id.in_([i for i, _ in found]))
        )}  # fmt: skip
        return [
            Hit(
                key=rows[i].key,
                kind=rows[i].kind,
                timestamp=rows[i].timestamp,
                text=rows[i].text,
                score=round(s, 4),
                snapshot_id=rows[i].snapshot_id,
                incident_id=rows[i].incident_id,
            )
            for i, s in found
            if i in rows
        ]
