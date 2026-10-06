"""Per-file metadata: SHA-256, Shannon entropy, size, extension, mtime.

Hash and entropy are computed in a single streaming pass so each file is read
exactly once and memory stays flat regardless of file size.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

CHUNK_SIZE = 1024 * 1024  # 1 MiB


@dataclass(frozen=True, slots=True)
class FileEntry:
    """Metadata for one file inside a snapshot."""

    path: str  # POSIX path relative to the snapshot root
    size: int
    sha256: str
    entropy: float  # bits per byte, 0.0 (constant) .. 8.0 (uniform random)
    extension: str  # lower-case, includes the dot, "" if none
    mtime: datetime

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation used in the manifest."""
        data = asdict(self)
        data["mtime"] = self.mtime.isoformat()
        return data


def shannon_entropy_from_counts(counts: np.ndarray) -> float:
    """Shannon entropy H = -sum(p * log2 p) over a 256-bin byte histogram.

    Plain text sits around 4-5 bits/byte; compressed or encrypted data is ~7.9+.
    That gap is the main signal ransomware leaves behind.
    """
    total = int(counts.sum())
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-(probs * np.log2(probs)).sum())


def shannon_entropy(data: bytes) -> float:
    """Shannon entropy of an in-memory byte string (bits/byte)."""
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    return shannon_entropy_from_counts(counts)


def hash_and_entropy(path: Path) -> tuple[str, float, int]:
    """Stream ``path`` once and return (sha256 hex, entropy, size in bytes)."""
    hasher = hashlib.sha256()
    counts = np.zeros(256, dtype=np.int64)
    size = 0
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK_SIZE):
            hasher.update(chunk)
            counts += np.bincount(np.frombuffer(chunk, dtype=np.uint8), minlength=256)
            size += len(chunk)
    return hasher.hexdigest(), shannon_entropy_from_counts(counts), size


def sample_entropy(path: Path, max_bytes: int = CHUNK_SIZE) -> float:
    """Entropy of the first ``max_bytes`` of a file (cheap enough for live monitoring)."""
    with path.open("rb") as fh:
        return shannon_entropy(fh.read(max_bytes))


def file_extension(path: str) -> str:
    """Return the final lower-cased suffix (``report.docx.locked`` -> ``.locked``)."""
    return Path(path).suffix.lower()


def is_excluded(rel_path: str, patterns: Iterable[str]) -> bool:
    """True if the relative path or its basename matches any glob pattern."""
    name = rel_path.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(rel_path, p) for p in patterns)


def iter_files(root: Path, exclude_patterns: Iterable[str] = ()) -> Iterator[Path]:
    """Yield regular files under ``root`` in deterministic order.

    Symlinks are skipped deliberately: following them could back up (or later
    restore over) files outside the watched directory.
    """
    patterns = list(exclude_patterns)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in sorted(filenames):
            full = Path(dirpath) / name
            if full.is_symlink() or not full.is_file():
                continue
            rel = full.relative_to(root).as_posix()
            if is_excluded(rel, patterns):
                continue
            yield full


def build_entry(root: Path, full_path: Path) -> FileEntry:
    """Compute the FileEntry for one file under ``root``."""
    sha, entropy, size = hash_and_entropy(full_path)
    rel = full_path.relative_to(root).as_posix()
    mtime = datetime.fromtimestamp(full_path.stat().st_mtime, tz=UTC)
    return FileEntry(
        path=rel,
        size=size,
        sha256=sha,
        entropy=round(entropy, 4),
        extension=file_extension(rel),
        mtime=mtime,
    )


def scan_directory(root: Path, exclude_patterns: Iterable[str] = ()) -> list[FileEntry]:
    """Build FileEntry records for every eligible file under ``root``.

    Files that disappear or become unreadable mid-scan are skipped rather than
    failing the whole snapshot (common when apps write temp files).
    """
    entries: list[FileEntry] = []
    for full in iter_files(root, exclude_patterns):
        try:
            entries.append(build_entry(root, full))
        except (FileNotFoundError, PermissionError):
            continue
    return entries
