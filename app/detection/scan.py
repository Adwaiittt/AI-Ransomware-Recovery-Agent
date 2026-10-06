"""On-demand scan: compare a snapshot with the live directory (or another snapshot).

The live monitor sees real-time events. A scan only sees two *states*, so it
rebuilds approximate events from the diff and uses file mtimes as timestamps:

  modified           -> modified event (entropy before/after from both sides)
  extension_changed  -> modified (if content changed) + moved event
  renamed            -> moved event
  added / removed    -> created / deleted event

Events are then cut into the same fixed windows as training, and every window
is scored; the scan verdict is the worst window. Because the windows come from
mtimes, an attack that touched 300 files in 5 seconds still looks like a burst,
while the same number of edits spread over a day does not.

Known approximation: deletions have no mtime, so they are placed at the median
timestamp of the other events (or "now" if there are none).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.backup.manifest import scan_directory
from app.backup.snapshotter import (
    SnapshotDiff,
    diff_file_sets,
    diff_snapshots,
    get_snapshot,
    latest_snapshot,
)
from app.db.models import Incident
from app.detection.features import (
    EventType,
    FileEvent,
    compute_window_features,
    split_into_windows,
)
from app.detection.incidents import record_detection
from app.detection.model import Detection, Detector


class NoBaselineError(RuntimeError):
    """Raised when there is no snapshot to compare against."""


@dataclass
class WindowResult:
    """Score for one reconstructed window."""

    start: datetime
    end: datetime
    event_count: int
    features: dict[str, float]
    detection: Detection
    affected_files: list[str]


@dataclass
class ScanResult:
    """Outcome of a scan."""

    base_snapshot_id: str
    target: str  # snapshot id or "live"
    diff_summary: dict[str, object]
    windows: list[WindowResult] = field(default_factory=list)
    incident: Incident | None = None

    @property
    def max_score(self) -> float:
        return max((w.detection.score for w in self.windows), default=0.0)

    @property
    def is_alert(self) -> bool:
        return any(w.detection.is_alert for w in self.windows)


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def events_from_diff(diff: SnapshotDiff, now: datetime | None = None) -> list[FileEvent]:
    """Reconstruct approximate FileEvents from a snapshot diff (see module docstring)."""
    events: list[FileEvent] = []
    for m in diff.modified:
        events.append(
            FileEvent(
                _ts(m["mtime"]),
                EventType.modified,
                m["path"],
                entropy_before=m["entropy_before"],
                entropy_after=m["entropy_after"],
            )
        )
    for c in diff.extension_changed:
        t = _ts(c["mtime"])
        if c["content_changed"]:
            events.append(
                FileEvent(
                    t,
                    EventType.modified,
                    c["old_path"],
                    entropy_before=c["entropy_before"],
                    entropy_after=c["entropy_after"],
                )
            )
        events.append(FileEvent(t, EventType.moved, c["old_path"], dest_path=c["new_path"]))
    for r in diff.renamed:
        events.append(
            FileEvent(_ts(r["mtime"]), EventType.moved, r["old_path"], dest_path=r["new_path"])
        )
    for a in diff.added:
        events.append(
            FileEvent(_ts(a["mtime"]), EventType.created, a["path"], entropy_after=a["entropy"])
        )
    if diff.removed:
        anchor = (
            statistics.median(e.timestamp for e in events)
            if events
            else (now or datetime.now(UTC)).timestamp()
        )
        events += [FileEvent(anchor, EventType.deleted, r["path"]) for r in diff.removed]
    return events


def _affected(events: list[FileEvent]) -> list[str]:
    paths = {e.dest_path or e.path for e in events}
    return sorted(paths)


def run_scan(
    session: Session,
    detector: Detector,
    *,
    watch_dir: Path,
    exclude_patterns: list[str],
    window_seconds: float,
    cooldown_seconds: float,
    base_snapshot_id: str | None = None,
    target_snapshot_id: str | None = None,
    record: bool = True,
) -> ScanResult:
    """Diff base (default: latest snapshot) vs target (default: live dir) and score it."""
    base = get_snapshot(session, base_snapshot_id) if base_snapshot_id else latest_snapshot(session)
    if base is None:
        raise NoBaselineError("No snapshot exists yet; create a baseline with POST /backups.")

    if target_snapshot_id:
        target_label = target_snapshot_id
        diff = diff_snapshots(base, get_snapshot(session, target_snapshot_id))
    else:
        target_label = "live"
        live = scan_directory(watch_dir.resolve(), exclude_patterns)
        diff = diff_file_sets(base.id, base.files, "live", live)

    result = ScanResult(base_snapshot_id=base.id, target=target_label, diff_summary=diff.summary())
    windows = split_into_windows(events_from_diff(diff), window_seconds)
    feature_rows = [compute_window_features(w, window_seconds) for w in windows]
    for evs, feats, det in zip(
        windows, feature_rows, detector.score_many(feature_rows), strict=True
    ):
        result.windows.append(
            WindowResult(
                start=datetime.fromtimestamp(evs[0].timestamp, UTC),
                end=datetime.fromtimestamp(evs[-1].timestamp, UTC),
                event_count=len(evs),
                features=feats,
                detection=det,
                affected_files=_affected(evs),
            )
        )

    alerts = [w for w in result.windows if w.detection.is_alert]
    if record and alerts:
        worst = max(alerts, key=lambda w: w.detection.score)
        result.incident = record_detection(
            session,
            detection=worst.detection,
            features=worst.features,
            started_at=min(w.start for w in alerts),
            ended_at=max(w.end for w in alerts),
            affected_files=sorted({f for w in alerts for f in w.affected_files}),
            source="scan",
            model_name=detector.model_name,
            threshold=detector.threshold,
            cooldown_seconds=cooldown_seconds,
        )
    return result
