"""Window-level features for ransomware detection.

This module is the single source of truth for features: the synthetic dataset
generator (``ml/generate_dataset.py``), the live watchdog monitor, and the
snapshot-diff scan all build ``FileEvent`` lists and call
:func:`compute_window_features`. Training and serving therefore cannot drift
apart (no train/serve skew).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

HIGH_ENTROPY_BITS = 7.5

# Formats that are *legitimately* high-entropy because they are compressed.
# Office OOXML/ODF files are zip containers, so they belong here too.
COMPRESSED_EXTS = frozenset(
    {
        ".zip", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".zst",
        ".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic",
        ".mp3", ".mp4", ".mkv", ".mov", ".avi", ".flac", ".ogg",
        ".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp",
        ".pdf", ".jar", ".apk", ".whl", ".gpg", ".pgp",
    }
)  # fmt: skip

# Extensions a normal workstation produces. Anything else counts as "unknown",
# which is what ransomware-appended suffixes (.locked, .crypt, .x7f2a) look like.
KNOWN_EXTS = COMPRESSED_EXTS | frozenset(
    {
        "", ".txt", ".md", ".rst", ".csv", ".tsv", ".json", ".yaml", ".yml", ".toml",
        ".xml", ".html", ".htm", ".css", ".log", ".ini", ".cfg", ".conf",
        ".doc", ".xls", ".ppt", ".rtf", ".svg", ".bmp", ".tif", ".tiff", ".wav",
        ".py", ".pyc", ".js", ".ts", ".tsx", ".jsx", ".java", ".class", ".c", ".h",
        ".cpp", ".hpp", ".o", ".obj", ".so", ".dll", ".exe", ".a", ".lib", ".go",
        ".rs", ".rb", ".php", ".sh", ".ps1", ".bat", ".sql", ".db", ".sqlite",
        ".ipynb", ".lock", ".map", ".d", ".tmp", ".bak", ".swp",
    }
)  # fmt: skip


class EventType(StrEnum):
    """Filesystem event kinds (mirrors watchdog's event types)."""

    created = "created"
    modified = "modified"
    deleted = "deleted"
    moved = "moved"


@dataclass(frozen=True, slots=True)
class FileEvent:
    """One filesystem event, already enriched with entropy where known.

    ``entropy_before`` is the last known entropy of the file prior to this event
    (from a snapshot or an earlier observation); ``entropy_after`` is measured
    after the write. Both are None when not applicable/unknown.
    """

    timestamp: float  # epoch seconds
    event_type: EventType
    path: str  # POSIX path relative to the watch root (source path for moves)
    dest_path: str | None = None  # moves only
    entropy_before: float | None = None
    entropy_after: float | None = None


FEATURE_NAMES: tuple[str, ...] = (
    "files_written_per_sec",
    "renames_per_sec",
    "pct_extension_changed",
    "mean_entropy_delta",
    "frac_high_entropy",
    "frac_high_entropy_unexpected",
    "unknown_ext_count",
    "delete_create_ratio",
    "unique_dirs",
    "events_per_sec",
)


def ext_of(path: str) -> str:
    """Lower-cased last suffix of a POSIX path ("" if none)."""
    return PurePosixPath(path).suffix.lower()


def _parent(path: str) -> str:
    return str(PurePosixPath(path).parent)


def compute_window_features(events: Sequence[FileEvent], window_seconds: float) -> dict[str, float]:
    """Aggregate a window of events into the fixed feature vector.

    Writes are de-duplicated per path: editors and OSes fire several
    ``modified`` events for one save, and counting each would make a single
    large save look like a write burst.

    Feature meanings:
      files_written_per_sec        distinct paths created/modified per second
      renames_per_sec              move events per second
      pct_extension_changed        moves that changed the suffix / distinct touched paths
      mean_entropy_delta           mean(after - before) where both known (in-place rewrites)
      frac_high_entropy            written files with entropy > 7.5
      frac_high_entropy_unexpected same, excluding formats that are compressed by design
                                   (zip/jpg/docx...) -- the main false-positive killer
      unknown_ext_count            distinct new names whose suffix isn't a known type
      delete_create_ratio          deletes / (creates + 1); "encrypt copy, delete original"
      unique_dirs                  distinct parent directories touched (spread)
      events_per_sec               raw event rate
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    written: dict[str, FileEvent] = {}  # final path -> last write event
    creates = deletes = moves = ext_changes = 0
    touched: set[str] = set()
    new_names: set[str] = set()
    moved_to: dict[str, str] = {}

    for ev in events:
        touched.add(ev.path)
        if ev.event_type is EventType.created:
            creates += 1
            written[ev.path] = ev
            new_names.add(ev.path)
        elif ev.event_type is EventType.modified:
            written[ev.path] = ev
        elif ev.event_type is EventType.deleted:
            deletes += 1
        elif ev.event_type is EventType.moved and ev.dest_path is not None:
            moves += 1
            touched.add(ev.dest_path)
            new_names.add(ev.dest_path)
            moved_to[ev.path] = ev.dest_path
            if ext_of(ev.path) != ext_of(ev.dest_path):
                ext_changes += 1

    # Resolve each written path to its final name (write-then-rename is the
    # typical ransomware sequence), so extension checks see ".locked".
    def final_name(p: str) -> str:
        seen = set()
        while p in moved_to and p not in seen:
            seen.add(p)
            p = moved_to[p]
        return p

    entropy_after = [
        (final_name(p), e.entropy_after) for p, e in written.items() if e.entropy_after is not None
    ]
    deltas = [
        e.entropy_after - e.entropy_before
        for e in written.values()
        if e.entropy_after is not None and e.entropy_before is not None
    ]
    high = [(p, h) for p, h in entropy_after if h > HIGH_ENTROPY_BITS]
    high_unexpected = [p for p, _ in high if ext_of(p) not in COMPRESSED_EXTS]
    n_entropy = len(entropy_after)

    return {
        "files_written_per_sec": len(written) / window_seconds,
        "renames_per_sec": moves / window_seconds,
        "pct_extension_changed": ext_changes / len(touched) if touched else 0.0,
        "mean_entropy_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "frac_high_entropy": len(high) / n_entropy if n_entropy else 0.0,
        "frac_high_entropy_unexpected": len(high_unexpected) / n_entropy if n_entropy else 0.0,
        "unknown_ext_count": float(sum(1 for p in new_names if ext_of(p) not in KNOWN_EXTS)),
        "delete_create_ratio": deletes / (creates + 1),
        "unique_dirs": float(len({_parent(p) for p in touched})),
        "events_per_sec": len(events) / window_seconds,
    }


def to_vector(features: dict[str, float]) -> list[float]:
    """Order a feature dict per FEATURE_NAMES (what the model expects)."""
    return [float(features[name]) for name in FEATURE_NAMES]


def split_into_windows(events: Iterable[FileEvent], window_seconds: float) -> list[list[FileEvent]]:
    """Bucket events into consecutive fixed windows by timestamp (empty buckets dropped)."""
    ordered = sorted(events, key=lambda e: e.timestamp)
    if not ordered:
        return []
    start = ordered[0].timestamp
    buckets: dict[int, list[FileEvent]] = {}
    for ev in ordered:
        buckets.setdefault(int((ev.timestamp - start) // window_seconds), []).append(ev)
    return [buckets[k] for k in sorted(buckets)]
