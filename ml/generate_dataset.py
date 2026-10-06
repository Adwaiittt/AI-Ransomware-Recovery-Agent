"""Generate a labeled dataset of 10-second activity windows.

Each sample is produced by *simulating a stream of FileEvents* for one scenario
and then running the exact same ``compute_window_features`` used in production.
We generate events (not feature values) so that the features keep their real
correlations — e.g. a rename burst automatically raises renames_per_sec,
pct_extension_changed and unknown_ext_count together.

Benign scenarios include deliberately hard negatives:
  * photo_import / legit_encrypted_backup  -> many high-entropy writes
  * browser_cache / ml_checkpoint          -> high entropy, no/unknown extension
  * cloud_sync                             -> '.partial' temp names renamed away
  * zip_and_cleanup                        -> high delete/create ratio
  * bulk_rename / log_rotation / office_save -> renames and extension changes
Attack scenarios include deliberately hard positives:
  * slow_attack        -> few files per window
  * inplace_stealth    -> no renames, keeps the original extension
  * partial_encryption -> only part of each file encrypted (entropy ~6.5-7.6)
Telemetry is then degraded (entropy noise, missing "before" values, attacks
starting mid-window) so the task is not trivially separable.

Usage:  python -m ml.generate_dataset --out ml/data/dataset.csv --seed 42
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from app.detection.features import EventType, FileEvent, compute_window_features

WINDOW = 10.0
Rng = np.random.Generator

# extension pool, (entropy low, entropy high) in bits/byte
FILE_TYPES: dict[str, tuple[list[str], tuple[float, float]]] = {
    "text": ([".txt", ".md", ".csv", ".log", ".json"], (3.8, 5.3)),
    "code": ([".py", ".js", ".ts", ".java", ".c", ".go"], (4.2, 5.6)),
    "office": ([".docx", ".xlsx", ".pptx"], (7.5, 7.97)),
    "legacy_office": ([".doc", ".xls", ".rtf"], (4.5, 6.8)),
    "image": ([".jpg", ".png", ".heic"], (7.6, 7.99)),
    "pdf": ([".pdf"], (6.8, 7.95)),
    "archive": ([".zip", ".7z", ".gz"], (7.95, 7.999)),
    "binary": ([".o", ".pyc", ".class", ".so"], (5.0, 6.6)),
}
VICTIM_TYPES = ["text", "code", "office", "legacy_office", "image", "pdf"]
RANSOM_EXTS = [".locked", ".crypt", ".encrypted", ".enc", ".crypted", ".lck", ".r4nd"]
DIRS = [
    "docs", "docs/reports", "docs/finance", "projects/app/src", "projects/app/build",
    "photos/2026", "photos/2025", "music", "downloads", "desktop", "notes",
    "projects/ml/data", "projects/web/node_modules/pkg",
]  # fmt: skip


# -- helpers --------------------------------------------------------------------
def _ts(rng: Rng, n: int) -> np.ndarray:
    return np.sort(rng.uniform(0, WINDOW, size=n))


def _entropy(rng: Rng, ftype: str) -> float:
    lo, hi = FILE_TYPES[ftype][1]
    return float(rng.uniform(lo, hi))


def _name(rng: Rng, ftype: str, dirs: list[str] | None = None) -> str:
    d = rng.choice(dirs or DIRS)
    ext = rng.choice(FILE_TYPES[ftype][0])
    return f"{d}/file_{rng.integers(0, 10**7)}{ext}"


def _encrypted_entropy(rng: Rng) -> float:
    # Ciphertext of files >= a few KB measures 7.97-8.0 bits/byte, but a file of
    # n bytes can never exceed log2(n): small encrypted files read much lower.
    if rng.random() < 0.2:
        return float(rng.uniform(6.8, 7.9))
    return float(rng.uniform(7.97, 8.0))


def _random_ext(rng: Rng) -> str:
    return "." + "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), size=5))


# Explicit constructors: positional FileEvent(...) is too easy to misorder.
def _c(t: float, path: str, entropy: float) -> FileEvent:
    return FileEvent(float(t), EventType.created, path, entropy_after=entropy)


def _m(t: float, path: str, before: float, after: float) -> FileEvent:
    return FileEvent(float(t), EventType.modified, path, entropy_before=before, entropy_after=after)


def _d(t: float, path: str) -> FileEvent:
    return FileEvent(float(t), EventType.deleted, path)


def _mv(t: float, src: str, dst: str) -> FileEvent:
    return FileEvent(float(t), EventType.moved, src, dest_path=dst)


# -- benign scenarios -----------------------------------------------------------
def idle(rng: Rng) -> list[FileEvent]:
    events = []
    for t in _ts(rng, int(rng.integers(1, 4))):
        before = _entropy(rng, "text")
        events.append(_m(t, _name(rng, "text"), before, before + rng.normal(0, 0.02)))
    return events


def doc_editing(rng: Rng) -> list[FileEvent]:
    events: list[FileEvent] = []
    for t in _ts(rng, int(rng.integers(1, 7))):
        ftype = str(rng.choice(["text", "office", "legacy_office", "code"]))
        before = _entropy(rng, ftype)
        after = float(np.clip(before + rng.normal(0, 0.08), 0, 8))
        events.append(_m(t, _name(rng, ftype), before, after))
    return events


def office_save(rng: Rng) -> list[FileEvent]:
    """Word/Excel safe-save: write temp file, delete original, rename temp -> original."""
    events: list[FileEvent] = []
    for t in _ts(rng, int(rng.integers(1, 5))):
        target = _name(rng, "office", ["docs", "docs/reports", "docs/finance"])
        tmp = target.rsplit("/", 1)[0] + f"/~WRL{rng.integers(1000, 9999)}.tmp"
        events += [
            _c(t, tmp, _entropy(rng, "office")),
            _d(t + 0.01, target),
            _mv(t + 0.02, tmp, target),
        ]
    return events


def code_build(rng: Rng) -> list[FileEvent]:
    dirs = list(rng.choice(DIRS, size=int(rng.integers(1, 5)), replace=False))
    events = [
        _c(t, _name(rng, "binary", dirs), _entropy(rng, "binary"))
        for t in _ts(rng, int(rng.integers(20, 300)))
    ]
    events += [_d(t, _name(rng, "binary", dirs)) for t in _ts(rng, int(rng.integers(0, 40)))]
    return events


def photo_import(rng: Rng) -> list[FileEvent]:
    dirs = list(rng.choice(["photos/2026", "photos/2025", "downloads"], size=2, replace=False))
    return [
        _c(t, _name(rng, "image", dirs), _entropy(rng, "image"))
        for t in _ts(rng, int(rng.integers(10, 150)))
    ]


def zip_and_cleanup(rng: Rng) -> list[FileEvent]:
    """Archive a folder then delete the originals: looks like 'encrypt + delete'."""
    d = str(rng.choice(DIRS))
    events = [_c(rng.uniform(0, 3), f"{d}/archive.zip", 7.99)]
    events += [
        _d(t, _name(rng, str(rng.choice(VICTIM_TYPES)), [d]))
        for t in 3 + _ts(rng, int(rng.integers(5, 120))) * 0.7
    ]
    return events


def bulk_rename(rng: Rng) -> list[FileEvent]:
    """Photo-manager rename; ~10% also normalise .jpeg -> .jpg (an extension change)."""
    events: list[FileEvent] = []
    for i, t in enumerate(_ts(rng, int(rng.integers(10, 200)))):
        src_ext = ".jpeg" if rng.random() < 0.1 else ".jpg"
        events.append(_mv(t, f"photos/2026/IMG_{i}{src_ext}", f"photos/2026/trip_{i}.jpg"))
    return events


def log_rotation(rng: Rng) -> list[FileEvent]:
    """app.log -> app.log.1 (unknown suffix!), gzip old logs, create a fresh log."""
    events: list[FileEvent] = []
    for i, t in enumerate(_ts(rng, int(rng.integers(1, 8)))):
        base = f"projects/app/logs/svc{i}.log"
        events += [
            _mv(t, base, base + ".1"),
            _c(t + 0.05, base + ".2.gz", 7.95),
            _d(t + 0.06, base + ".2"),
            _c(t + 0.07, base, 4.0),
        ]
    return events


def package_install(rng: Rng) -> list[FileEvent]:
    """npm install / git checkout: many small text/code files across many dirs."""
    dirs = [f"projects/web/node_modules/pkg{j}" for j in range(int(rng.integers(5, 60)))]
    events = [
        _c(t, _name(rng, str(rng.choice(["code", "text"])), dirs), _entropy(rng, "code"))
        for t in _ts(rng, int(rng.integers(50, 500)))
    ]
    events += [_d(t, _name(rng, "code", dirs)) for t in _ts(rng, int(rng.integers(0, 50)))]
    return events


def legit_encrypted_backup(rng: Rng) -> list[FileEvent]:
    """gpg/7z-encrypted backup of documents: genuinely encrypted, but originals kept."""
    return [
        _c(t, f"backups/doc_{i}.gpg", _encrypted_entropy(rng))
        for i, t in enumerate(_ts(rng, int(rng.integers(1, 25))))
    ]


def browser_cache(rng: Rng) -> list[FileEvent]:
    """Chrome/Firefox cache: extensionless, compressed (high-entropy) blobs + evictions."""
    d = "appdata/browser/Cache/Cache_Data"
    events = [
        _c(t, f"{d}/f_{rng.integers(0, 10**6):06x}", float(rng.uniform(7.3, 7.99)))
        for t in _ts(rng, int(rng.integers(10, 200)))
    ]
    events += [
        _d(t, f"{d}/f_{rng.integers(0, 10**6):06x}") for t in _ts(rng, int(rng.integers(0, 80)))
    ]
    return events


def cloud_sync(rng: Rng) -> list[FileEvent]:
    """OneDrive/Dropbox download: write 'name.ext.partial' then rename to 'name.ext'."""
    events: list[FileEvent] = []
    for t in _ts(rng, int(rng.integers(3, 80))):
        ftype = str(rng.choice(VICTIM_TYPES + ["archive"]))
        final = _name(rng, ftype)
        tmp = final + str(rng.choice([".partial", ".!sync", ".crdownload"]))
        events += [_c(t, tmp, _entropy(rng, ftype)), _mv(t + 0.01, tmp, final)]
    return events


def ml_checkpoint(rng: Rng) -> list[FileEvent]:
    """Training job overwriting weights/arrays: unknown extensions, high entropy in place."""
    events: list[FileEvent] = []
    for t in _ts(rng, int(rng.integers(1, 30))):
        ext = str(rng.choice([".pt", ".ckpt", ".npy", ".safetensors", ".bin", ".parquet"]))
        path = f"projects/ml/runs/{rng.integers(0, 50)}/state{ext}"
        if rng.random() < 0.5:
            before = float(rng.uniform(7.0, 7.95))
            events.append(_m(t, path, before, float(rng.uniform(7.0, 7.95))))
        else:
            events.append(_c(t, path, float(rng.uniform(7.0, 7.95))))
    return events


def photo_editing(rng: Rng) -> list[FileEvent]:
    """Lightroom-style export/overwrite of images: high entropy before and after."""
    events: list[FileEvent] = []
    for t in _ts(rng, int(rng.integers(1, 60))):
        before = _entropy(rng, "image")
        path = _name(rng, "image", ["photos/2026"])
        events.append(_m(t, path, before, _entropy(rng, "image")))
    return events


# -- attack scenarios -----------------------------------------------------------
def _victim(rng: Rng) -> tuple[str, float]:
    ftype = str(rng.choice(VICTIM_TYPES))
    return _name(rng, ftype), _entropy(rng, ftype)


def fast_rename_encrypt(rng: Rng, n: tuple[int, int] = (20, 250)) -> list[FileEvent]:
    """Classic: overwrite with ciphertext, then rename with a family extension."""
    ext = str(rng.choice(RANSOM_EXTS))
    events: list[FileEvent] = []
    for t in _ts(rng, int(rng.integers(*n))):
        path, before = _victim(rng)
        events += [_m(t, path, before, _encrypted_entropy(rng)), _mv(t + 0.005, path, path + ext)]
    note_dir = events[0].path.rsplit("/", 1)[0]
    events.append(_c(WINDOW - 0.1, f"{note_dir}/README_RESTORE.txt", 4.5))
    return events


def encrypt_copy_delete(rng: Rng) -> list[FileEvent]:
    """Write an encrypted copy, delete the original (no rename event at all)."""
    ext = str(rng.choice(RANSOM_EXTS))
    events: list[FileEvent] = []
    for t in _ts(rng, int(rng.integers(15, 200))):
        path, _ = _victim(rng)
        events += [_c(t, path + ext, _encrypted_entropy(rng)), _d(t + 0.01, path)]
    return events


def inplace_stealth(rng: Rng) -> list[FileEvent]:
    """Overwrite in place, keep the name and extension, moderate rate."""
    events = []
    for t in _ts(rng, int(rng.integers(3, 40))):
        path, before = _victim(rng)
        events.append(_m(t, path, before, _encrypted_entropy(rng)))
    return events


def slow_attack(rng: Rng) -> list[FileEvent]:
    """Rate-limited to evade burst detection: 1-4 files per window."""
    return fast_rename_encrypt(rng, n=(1, 5))[:-1]  # drop the ransom note


def partial_encryption(rng: Rng) -> list[FileEvent]:
    """Intermittent encryption (e.g. only the first N KB): entropy rises, but not to 8."""
    ext = str(rng.choice(RANSOM_EXTS)) if rng.random() < 0.5 else None
    events: list[FileEvent] = []
    for t in _ts(rng, int(rng.integers(10, 200))):
        path, before = _victim(rng)
        after = float(before + (8.0 - before) * rng.uniform(0.3, 0.8))
        events.append(_m(t, path, before, after))
        if ext:
            events.append(_mv(t + 0.005, path, path + ext))
    return events


def random_extension(rng: Rng) -> list[FileEvent]:
    """Each victim gets a unique random suffix (several real families do this)."""
    events: list[FileEvent] = []
    for t in _ts(rng, int(rng.integers(10, 200))):
        path, before = _victim(rng)
        events += [
            _m(t, path, before, _encrypted_entropy(rng)),
            _mv(t + 0.005, path, path + _random_ext(rng)),
        ]
    return events


Scenario = Callable[[Rng], list[FileEvent]]
BENIGN: dict[str, tuple[Scenario, float]] = {
    "idle": (idle, 0.20),
    "doc_editing": (doc_editing, 0.18),
    "office_save": (office_save, 0.08),
    "code_build": (code_build, 0.10),
    "photo_import": (photo_import, 0.10),
    "zip_and_cleanup": (zip_and_cleanup, 0.07),
    "bulk_rename": (bulk_rename, 0.07),
    "log_rotation": (log_rotation, 0.06),
    "package_install": (package_install, 0.08),
    "legit_encrypted_backup": (legit_encrypted_backup, 0.06),
    "browser_cache": (browser_cache, 0.08),
    "cloud_sync": (cloud_sync, 0.07),
    "ml_checkpoint": (ml_checkpoint, 0.05),
    "photo_editing": (photo_editing, 0.06),
}
ATTACK: dict[str, tuple[Scenario, float]] = {
    "fast_rename_encrypt": (fast_rename_encrypt, 0.25),
    "encrypt_copy_delete": (encrypt_copy_delete, 0.15),
    "inplace_stealth": (inplace_stealth, 0.15),
    "slow_attack": (slow_attack, 0.15),
    "partial_encryption": (partial_encryption, 0.15),
    "random_extension": (random_extension, 0.15),
}


def _pick(rng: Rng, table: dict[str, tuple[Scenario, float]]) -> tuple[str, Scenario]:
    names = list(table)
    weights = np.array([table[n][1] for n in names])
    name = str(rng.choice(names, p=weights / weights.sum()))
    return name, table[name][0]


def _degrade(events: list[FileEvent], rng: Rng) -> list[FileEvent]:
    """Make simulated telemetry look like the real monitor's.

    * entropy is measured on a sample of the file -> small measurement noise
    * the monitor only knows ``entropy_before`` if it saw the file earlier
      (snapshot or previous event) -> ~25% of rewrites have it missing
    """
    out = []
    for ev in events:
        before, after = ev.entropy_before, ev.entropy_after
        if after is not None:
            after = float(np.clip(after + rng.normal(0, 0.03), 0, 8))
        if before is not None:
            if rng.random() < 0.25:
                before = None
            else:
                before = float(np.clip(before + rng.normal(0, 0.03), 0, 8))
        out.append(replace(ev, entropy_before=before, entropy_after=after))
    return out


def _truncate_onset(events: list[FileEvent], rng: Rng) -> list[FileEvent]:
    """Attack starts mid-window: only the tail of the burst lands in this window."""
    onset = rng.uniform(0, WINDOW)
    kept = [e for e in events if e.timestamp >= onset]
    return kept or events[-1:]


def generate(n_benign: int, n_attack: int, seed: int) -> pd.DataFrame:
    """Return a DataFrame of feature rows plus ``label`` (1=attack) and ``scenario``."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for _ in range(n_benign):
        name, fn = _pick(rng, BENIGN)
        events = fn(rng)
        if rng.random() < 0.3:  # real windows often overlap two benign activities
            other, fn2 = _pick(rng, BENIGN)
            events += fn2(rng)
            name = f"{name}+{other}"
        events = _degrade(events, rng)
        rows.append(compute_window_features(events, WINDOW) | {"label": 0, "scenario": name})

    for _ in range(n_attack):
        name, fn = _pick(rng, ATTACK)
        events = fn(rng)
        if rng.random() < 0.35:
            events = _truncate_onset(events, rng)
        if rng.random() < 0.5:  # attacks happen while the user is also working
            _, bg = _pick(rng, BENIGN)
            events += bg(rng)
        events = _degrade(events, rng)
        rows.append(compute_window_features(events, WINDOW) | {"label": 1, "scenario": name})

    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, default=Path("ml/data/dataset.csv"))
    parser.add_argument("--benign", type=int, default=8000)
    parser.add_argument("--attack", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate(args.benign, args.attack, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows -> {args.out}")
    print(df.groupby("label").size().rename({0: "benign", 1: "attack"}).to_string())


if __name__ == "__main__":
    main()
