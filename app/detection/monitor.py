"""Live filesystem monitor (the "watcher" service).

Architecture:
    watchdog Observer thread --(raw events)--> EventCollector (thread-safe buffer)
    main loop, every window:  drain buffer -> WindowProcessor.process()
        -> enrich with entropy -> features -> model score -> heartbeat
        -> on alert: record/merge incident + mark snapshots suspect

Entropy is measured once per path at window close (not on every raw event):
one save fires several ``modified`` events, and by window close a
write-then-rename has finished, so we read the *final* file.

Run:  python -m app.detection.monitor
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

from app.backup.manifest import is_excluded, sample_entropy
from app.backup.snapshotter import latest_snapshot
from app.config import Settings, get_settings
from app.db.models import MonitorHeartbeat
from app.db.session import init_db, make_engine, make_session_factory
from app.detection.features import EventType, FileEvent, compute_window_features
from app.detection.incidents import record_detection
from app.detection.model import Detection, Detector
from app.logging_config import configure_logging

logger = logging.getLogger(__name__)

_WATCHDOG_TYPES = {
    "created": EventType.created,
    "modified": EventType.modified,
    "deleted": EventType.deleted,
    "moved": EventType.moved,
}


@dataclass(frozen=True, slots=True)
class RawEvent:
    """Un-enriched event as received from watchdog (paths relative to root)."""

    timestamp: float
    event_type: EventType
    path: str
    dest_path: str | None = None


class EventCollector(FileSystemEventHandler):
    """watchdog handler that buffers relevant events; drained once per window."""

    def __init__(self, root: Path, exclude_patterns: list[str]) -> None:
        self.root = root.resolve()
        self.exclude_patterns = exclude_patterns
        self._buf: list[RawEvent] = []
        self._lock = threading.Lock()

    def _rel(self, path: str | bytes) -> str | None:
        p = Path(path.decode() if isinstance(path, bytes) else path)
        try:
            rel = p.resolve().relative_to(self.root).as_posix()
        except (ValueError, OSError):
            return None
        return None if is_excluded(rel, self.exclude_patterns) else rel

    def on_any_event(self, event: FileSystemEvent) -> None:
        etype = _WATCHDOG_TYPES.get(event.event_type)
        if etype is None or event.is_directory:
            return
        src = self._rel(event.src_path)
        dest = self._rel(event.dest_path) if etype is EventType.moved else None
        if src is None and dest is None:
            return
        if etype is EventType.moved and (src is None or dest is None):
            # Moved in from / out to an untracked location: treat as create / delete.
            etype, src, dest = (
                (EventType.created, dest, None) if src is None else (EventType.deleted, src, None)
            )
        with self._lock:
            self._buf.append(RawEvent(time.time(), etype, src, dest))  # type: ignore[arg-type]

    def drain(self) -> list[RawEvent]:
        """Return and clear buffered events."""
        with self._lock:
            out, self._buf = self._buf, []
        return out


class WindowProcessor:
    """Turns one window of raw events into a score and (maybe) an incident."""

    def __init__(
        self,
        root: Path,
        session_factory: sessionmaker[Session],
        detector: Detector,
        *,
        window_seconds: float,
        cooldown_seconds: float,
        entropy_sample_bytes: int,
    ) -> None:
        self.root = root.resolve()
        self.session_factory = session_factory
        self.detector = detector
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self.entropy_sample_bytes = entropy_sample_bytes
        # Last known entropy per path -> provides "entropy_before" for rewrites.
        self.entropy_cache: dict[str, float] = {}
        self.windows_scored = 0
        self.alerts = 0

    def seed_cache_from_latest_snapshot(self) -> int:
        """Prime the entropy cache from the newest snapshot; returns entries loaded."""
        with self.session_factory() as session:
            snap = latest_snapshot(session)
            if snap is not None:
                self.entropy_cache.update({f.path: f.entropy for f in snap.files})
        return len(self.entropy_cache)

    def _measure(self, rel: str) -> float | None:
        try:
            return sample_entropy(self.root / rel, self.entropy_sample_bytes)
        except (FileNotFoundError, PermissionError, IsADirectoryError, OSError):
            return None

    def enrich(self, raw: list[RawEvent]) -> list[FileEvent]:
        """Attach entropy_before/after and update the cache.

        Writes are measured at their *final* name (write -> rename sequences).
        """
        moved_to = {r.path: r.dest_path for r in raw if r.event_type is EventType.moved}

        def final(p: str) -> str:
            seen: set[str] = set()
            while p in moved_to and p not in seen and moved_to[p]:
                seen.add(p)
                p = moved_to[p]  # type: ignore[assignment]
            return p

        measured: dict[str, float | None] = {}
        events: list[FileEvent] = []
        for r in raw:
            if r.event_type in (EventType.created, EventType.modified):
                target = final(r.path)
                if target not in measured:
                    measured[target] = self._measure(target)
                before = self.entropy_cache.get(r.path)
                if r.event_type is EventType.created:
                    before = None
                events.append(
                    FileEvent(
                        r.timestamp,
                        r.event_type,
                        r.path,
                        entropy_before=before,
                        entropy_after=measured[target],
                    )
                )
            else:
                events.append(FileEvent(r.timestamp, r.event_type, r.path, dest_path=r.dest_path))

        # Update cache after computing deltas so "before" reflects the old state.
        for r in raw:
            if r.event_type is EventType.deleted:
                self.entropy_cache.pop(r.path, None)
            elif r.event_type is EventType.moved and r.dest_path and r.path in self.entropy_cache:
                self.entropy_cache[r.dest_path] = self.entropy_cache.pop(r.path)
        for path, ent in measured.items():
            if ent is not None:
                self.entropy_cache[path] = ent
        return events

    def process(self, raw: list[RawEvent], window_end: float) -> Detection | None:
        """Score one window; record heartbeat and incident. Returns None if idle."""
        detection: Detection | None = None
        if raw:
            events = self.enrich(raw)
            features = compute_window_features(events, self.window_seconds)
            detection = self.detector.score(features)
            self.windows_scored += 1
            if detection.is_alert:
                self.alerts += 1
                self._record(events, features, detection)
                logger.warning(
                    "ALERT score=%.3f events=%d top=%s",
                    detection.score,
                    len(events),
                    [f["feature"] for f in detection.top_features],
                )
            else:
                logger.debug("window score=%.3f events=%d", detection.score, len(events))
        self._heartbeat(window_end, detection.score if detection else 0.0)
        return detection

    def _record(self, events: list[FileEvent], features: dict[str, float], det: Detection) -> None:
        affected = sorted({e.dest_path or e.path for e in events})
        with self.session_factory() as session:
            record_detection(
                session,
                detection=det,
                features=features,
                started_at=datetime.fromtimestamp(min(e.timestamp for e in events), UTC),
                ended_at=datetime.fromtimestamp(max(e.timestamp for e in events), UTC),
                affected_files=affected,
                source="monitor",
                model_name=self.detector.model_name,
                threshold=self.detector.threshold,
                cooldown_seconds=self.cooldown_seconds,
            )

    def _heartbeat(self, now: float, last_score: float) -> None:
        with self.session_factory() as session:
            hb = session.scalars(select(MonitorHeartbeat).limit(1)).first()
            if hb is None:
                hb = MonitorHeartbeat(id=1, watch_dir=str(self.root))
                session.add(hb)
            hb.last_seen = datetime.fromtimestamp(now, UTC)
            hb.watch_dir = str(self.root)
            hb.windows_scored = self.windows_scored
            hb.last_score = last_score
            hb.alerts = self.alerts
            session.commit()


def run_monitor(settings: Settings, stop: threading.Event | None = None) -> None:
    """Blocking loop: watch ``settings.watch_dir`` until ``stop`` is set (or Ctrl+C)."""
    stop = stop or threading.Event()
    root = settings.watch_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)

    engine = make_engine(settings.database_url)
    init_db(engine)
    factory = make_session_factory(engine)
    detector = Detector.load(settings.model_path)

    processor = WindowProcessor(
        root,
        factory,
        detector,
        window_seconds=settings.detection_window_seconds,
        cooldown_seconds=settings.incident_cooldown_seconds,
        entropy_sample_bytes=settings.entropy_sample_bytes,
    )
    seeded = processor.seed_cache_from_latest_snapshot()
    collector = EventCollector(root, settings.exclude_patterns)
    observer = PollingObserver(timeout=1.0) if settings.monitor_polling else Observer()
    observer.schedule(collector, str(root), recursive=True)
    observer.start()
    logger.info(
        "monitor watching %s (window=%ss, model=%s, threshold=%.3f, cache=%d files)",
        root,
        settings.detection_window_seconds,
        detector.model_name,
        detector.threshold,
        seeded,
    )
    try:
        while not stop.wait(settings.detection_window_seconds):
            processor.process(collector.drain(), time.time())
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join(timeout=5)
        engine.dispose()


def check_heartbeat(settings: Settings) -> bool:
    """True if the monitor wrote a heartbeat within 3 windows (container healthcheck)."""
    engine = make_engine(settings.database_url)
    try:
        with make_session_factory(engine)() as session:
            hb = session.scalars(select(MonitorHeartbeat).limit(1)).first()
    finally:
        engine.dispose()
    if hb is None:
        return False
    age = datetime.now(UTC) - hb.last_seen
    return age.total_seconds() <= 3 * settings.detection_window_seconds


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run the monitor, or ``--check`` its heartbeat."""
    import argparse

    parser = argparse.ArgumentParser(description="Ransomware activity monitor")
    parser.add_argument("--check", action="store_true", help="exit 0 if heartbeat is fresh")
    args = parser.parse_args(argv)
    settings = get_settings()
    if args.check:
        return 0 if check_heartbeat(settings) else 1
    configure_logging(settings.log_level, settings.log_format)
    run_monitor(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
