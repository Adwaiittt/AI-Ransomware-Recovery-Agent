"""WindowProcessor / EventCollector / incident bookkeeping (no real watchdog thread)."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileMovedEvent

from app.backup.snapshotter import create_snapshot
from app.config import Settings
from app.db.models import Incident, MonitorHeartbeat, SnapshotState
from app.db.session import make_engine, make_session_factory
from app.detection.features import EventType
from app.detection.incidents import record_detection
from app.detection.model import Detection, Detector
from app.detection.monitor import EventCollector, RawEvent, WindowProcessor
from app.storage.s3_client import S3Storage


@pytest.fixture
def processor(settings: Settings, db: Session, detector: Detector) -> WindowProcessor:
    factory = make_session_factory(make_engine(settings.database_url))
    return WindowProcessor(
        settings.watch_dir,
        factory,
        detector,
        window_seconds=10.0,
        cooldown_seconds=60.0,
        entropy_sample_bytes=1 << 20,
    )


def _simulate_attack(root: Path, files: list[Path]) -> list[RawEvent]:
    """Overwrite with random bytes + rename, returning the events watchdog would emit."""
    raw: list[RawEvent] = []
    now = time.time()
    for i, p in enumerate(files):
        rel = p.relative_to(root).as_posix()
        p.write_bytes(os.urandom(p.stat().st_size))
        p.rename(p.with_name(p.name + ".locked"))
        raw += [
            RawEvent(now + i * 0.01, EventType.modified, rel),
            RawEvent(now + i * 0.01 + 0.001, EventType.moved, rel, rel + ".locked"),
        ]
    return raw


def test_attack_window_creates_incident_and_taints_snapshots(
    processor: WindowProcessor, db: Session, storage: S3Storage, settings: Settings, make_tree
) -> None:  # type: ignore[no-untyped-def]
    root = settings.watch_dir
    files = make_tree(root, 40)
    clean_snap = create_snapshot(db, storage, root)
    assert processor.seed_cache_from_latest_snapshot() >= 40

    det = processor.process(_simulate_attack(root, files), time.time())
    assert det is not None and det.is_alert

    db.expire_all()
    incident = db.scalars(select(Incident)).one()
    assert incident.source == "monitor"
    assert incident.affected_file_count == 80  # 40 originals + 40 .locked names
    assert any(f["feature"] for f in incident.top_features)
    # entropy_before came from the snapshot-seeded cache -> large positive delta
    assert incident.features["mean_entropy_delta"] > 2.0

    assert db.get(type(clean_snap), clean_snap.id).state is SnapshotState.clean
    assert create_snapshot(db, storage, root).state is SnapshotState.suspect

    hb = db.scalars(select(MonitorHeartbeat)).one()
    assert hb.windows_scored == 1 and hb.alerts == 1


def test_benign_window_no_incident(
    processor: WindowProcessor, db: Session, settings: Settings
) -> None:
    p = settings.watch_dir / "docs" / "notes.txt"
    p.write_text(p.read_text() + "one more line\n")
    det = processor.process([RawEvent(time.time(), EventType.modified, "docs/notes.txt")], 0)
    assert det is not None and not det.is_alert
    assert db.scalars(select(Incident)).first() is None


def test_idle_window_still_heartbeats(processor: WindowProcessor, db: Session) -> None:
    assert processor.process([], time.time()) is None
    assert db.scalars(select(MonitorHeartbeat)).one().windows_scored == 0


def test_alerts_within_cooldown_merge(db: Session) -> None:
    t0 = datetime.now(UTC)
    kwargs = dict(
        features={"x": 1.0},
        source="monitor",
        model_name="m",
        threshold=0.5,
        cooldown_seconds=60,
    )
    record_detection(
        db, detection=Detection(0.8, True), started_at=t0, ended_at=t0 + timedelta(seconds=10),
        affected_files=["a"], **kwargs,
    )  # fmt: skip
    record_detection(
        db, detection=Detection(0.95, True), started_at=t0 + timedelta(seconds=30),
        ended_at=t0 + timedelta(seconds=40), affected_files=["b"], **kwargs,
    )  # fmt: skip
    record_detection(
        db, detection=Detection(0.7, True), started_at=t0 + timedelta(minutes=10),
        ended_at=t0 + timedelta(minutes=10, seconds=5), affected_files=["c"], **kwargs,
    )  # fmt: skip
    first, second = sorted(db.scalars(select(Incident)), key=lambda i: i.id)
    assert first.windows == 2 and first.score == 0.95
    assert first.affected_files == ["a", "b"]
    assert second.windows == 1


def test_snapshot_taken_during_attack_is_marked_suspect(
    db: Session, storage: S3Storage, watch_dir: Path
) -> None:
    snap = create_snapshot(db, storage, watch_dir)
    record_detection(
        db, detection=Detection(0.9, True),
        started_at=snap.created_at - timedelta(seconds=5),
        ended_at=snap.created_at + timedelta(seconds=5),
        affected_files=[], features={}, source="monitor", model_name="m",
        threshold=0.5, cooldown_seconds=60,
    )  # fmt: skip
    db.refresh(snap)
    assert snap.state is SnapshotState.suspect


def test_collector_filters_and_relativizes(tmp_path: Path) -> None:
    root = tmp_path / "w"
    root.mkdir()
    c = EventCollector(root, ["*.tmp"])
    c.on_any_event(FileCreatedEvent(str(root / "a" / "x.txt")))
    c.on_any_event(FileModifiedEvent(str(root / "skip.tmp")))  # excluded
    c.on_any_event(FileCreatedEvent(str(tmp_path / "outside.txt")))  # outside root
    c.on_any_event(FileMovedEvent(str(root / "b.txt"), str(root / "b.txt.locked")))
    c.on_any_event(FileMovedEvent(str(tmp_path / "in.txt"), str(root / "in.txt")))  # moved in
    events = c.drain()
    assert [(e.event_type, e.path, e.dest_path) for e in events] == [
        (EventType.created, "a/x.txt", None),
        (EventType.moved, "b.txt", "b.txt.locked"),
        (EventType.created, "in.txt", None),
    ]
    assert c.drain() == []
