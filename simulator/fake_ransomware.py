"""SAFE ransomware *simulator* for testing detection and restore.

This is a test tool, not malware. Guarantees (enforced in code, covered by tests):

* Runs ONLY inside ``<project>/sandbox`` (after resolving symlinks). It refuses
  the filesystem root, the user's home directory, and anything outside sandbox.
* Touches ONLY files it created itself with ``seed``: every seeded file is
  recorded with its SHA-256 in ``.sim_manifest.json``; ``attack`` skips any file
  that is not listed or whose content no longer matches the recorded hash.
* No real encryption: "encrypting" = overwriting the seeded copy with
  ``os.urandom`` bytes (high entropy, no key, nothing to decrypt).
* No network, no persistence, no spreading, no self-execution. The only "note"
  is a plain text file that says it's a simulation.

Usage:
    python -m simulator.fake_ransomware seed   sandbox/watched --files 150
    python -m simulator.fake_ransomware attack sandbox/watched --mode fast
    python -m simulator.fake_ransomware clean  sandbox/watched
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SANDBOX_ROOT = PROJECT_ROOT / "sandbox"
MANIFEST_NAME = ".sim_manifest.json"
NOTE_NAME = "README_SIMULATION.txt"
FAKE_EXT = ".locked"

MODES = ("fast", "slow", "partial", "inplace")


class SafetyError(RuntimeError):
    """Raised when the simulator is asked to do something outside its sandbox."""


def validate_target(target: Path, sandbox_root: Path = SANDBOX_ROOT) -> Path:
    """Resolve ``target`` and refuse anything that is not inside the sandbox."""
    resolved = target.expanduser().resolve()
    sandbox = sandbox_root.resolve()
    if resolved == Path(resolved.anchor):
        raise SafetyError(f"Refusing filesystem root: {resolved}")
    if resolved == Path.home().resolve():
        raise SafetyError(f"Refusing home directory: {resolved}")
    if sandbox in (Path(sandbox.anchor), Path.home().resolve()):
        raise SafetyError(f"Sandbox root is unsafe: {sandbox}")
    if not resolved.is_relative_to(sandbox):
        raise SafetyError(f"Refusing {resolved}: must be inside {sandbox}")
    return resolved


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -- seed -----------------------------------------------------------------------
_WORDS = [
    "quarterly", "revenue", "forecast", "customer", "meeting", "project", "roadmap",
    "budget", "review", "invoice", "summary", "design", "notes", "release", "plan",
    "team", "update", "analysis", "report", "draft",
]  # fmt: skip


def _text(rng: random.Random, n_words: int) -> bytes:
    lines = []
    for _ in range(max(1, n_words // 12)):
        lines.append(" ".join(rng.choice(_WORDS) for _ in range(12)).capitalize() + ".")
    return ("\n".join(lines) + "\n").encode()


def _csv(rng: random.Random, rows: int) -> bytes:
    out = ["date,account,amount,category"]
    for i in range(rows):
        out.append(f"2026-09-{1 + i % 28:02d},ACC{rng.randint(100, 999)},"
                   f"{rng.uniform(5, 5000):.2f},{rng.choice(_WORDS)}")  # fmt: skip
    return ("\n".join(out) + "\n").encode()


def _code(rng: random.Random, funcs: int) -> bytes:
    body = []
    for i in range(funcs):
        body.append(
            f"def {rng.choice(_WORDS)}_{i}(x: int) -> int:\n"
            f'    """Compute {rng.choice(_WORDS)}."""\n'
            f"    return x * {rng.randint(2, 99)} + {rng.randint(0, 9)}\n"
        )
    return "\n\n".join(body).encode()


def seed(
    target: Path, n_files: int = 150, rng_seed: int = 7, sandbox_root: Path = SANDBOX_ROOT
) -> list[str]:
    """Create test files under ``target`` and record them in the sim manifest."""
    target = validate_target(target, sandbox_root)
    rng = random.Random(rng_seed)
    folders = ["docs", "docs/finance", "projects/app", "notes", "photos"]
    created: dict[str, str] = {}
    for i in range(n_files):
        folder = folders[i % len(folders)]
        kind = rng.choice(["txt", "md", "csv", "py", "txt", "jpg"])
        rel = f"{folder}/sample_{i:04d}.{kind}"
        if kind == "csv":
            data = _csv(rng, rng.randint(80, 600))
        elif kind == "py":
            data = _code(rng, rng.randint(20, 120))
        elif kind == "jpg":
            # Benign high-entropy file (JPEG-like header + random payload).
            data = b"\xff\xd8\xff\xe0" + os.urandom(rng.randint(8_000, 40_000))
        else:
            data = _text(rng, rng.randint(400, 4000))
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        created[rel] = hashlib.sha256(data).hexdigest()
    manifest = {"created_by": "fake_ransomware.seed", "files": created}
    (target / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return sorted(created)


# -- attack ---------------------------------------------------------------------
def _load_manifest(target: Path) -> dict[str, str]:
    mpath = target / MANIFEST_NAME
    if not mpath.is_file():
        raise SafetyError(f"No {MANIFEST_NAME} in {target}; run `seed` first.")
    return json.loads(mpath.read_text())["files"]


def attack(
    target: Path,
    mode: str = "fast",
    limit: int | None = None,
    delay: float | None = None,
    rng_seed: int = 13,
    sandbox_root: Path = SANDBOX_ROOT,
) -> list[str]:
    """Overwrite seeded copies with random bytes (and rename) to mimic ransomware.

    Modes:
      fast     overwrite fully + rename to *.locked, no delay (burst)
      slow     same, but ``delay`` seconds (default 4s) between files
      partial  overwrite only the first 30-70% of each file + rename
      inplace  overwrite fully, keep the original name
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    target = validate_target(target, sandbox_root)
    seeded = _load_manifest(target)
    rng = random.Random(rng_seed)
    delay = (4.0 if mode == "slow" else 0.0) if delay is None else delay

    victims = sorted(seeded)
    rng.shuffle(victims)
    touched: list[str] = []
    for rel in victims[: limit or len(victims)]:
        path = (target / rel).resolve()
        # Defence in depth: re-check every path and only touch pristine seeded copies.
        if not path.is_relative_to(target) or not path.is_file() or path.is_symlink():
            continue
        if _sha256(path) != seeded[rel]:
            continue
        size = path.stat().st_size
        if mode == "partial":
            cut = int(size * rng.uniform(0.3, 0.7))
            data = os.urandom(cut) + path.read_bytes()[cut:]
        else:
            data = os.urandom(size)
        path.write_bytes(data)
        if mode != "inplace":
            path.rename(path.with_name(path.name + FAKE_EXT))
        touched.append(rel)
        if delay:
            time.sleep(delay)

    (target / NOTE_NAME).write_text(
        "THIS IS A SIMULATION by simulator/fake_ransomware.py.\n"
        "No real encryption was performed. Files were overwritten with random test bytes.\n"
        "Restore them with the recovery service (POST /restore).\n"
    )
    return touched


# -- clean ----------------------------------------------------------------------
def clean(target: Path, sandbox_root: Path = SANDBOX_ROOT) -> int:
    """Delete only simulator-created files (seeded, .locked variants, note, manifest)."""
    target = validate_target(target, sandbox_root)
    seeded = _load_manifest(target)
    removed = 0
    for rel in seeded:
        for candidate in (target / rel, target / (rel + FAKE_EXT)):
            p = candidate.resolve()
            if p.is_relative_to(target) and p.is_file() and not candidate.is_symlink():
                p.unlink()
                removed += 1
    for extra in (NOTE_NAME, MANIFEST_NAME):
        if (target / extra).is_file():
            (target / extra).unlink()
    return removed


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="SAFE ransomware simulator (sandbox only).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_seed = sub.add_parser("seed", help="create test files")
    p_seed.add_argument("target", type=Path)
    p_seed.add_argument("--files", type=int, default=150)
    p_att = sub.add_parser("attack", help="simulate an attack on seeded files")
    p_att.add_argument("target", type=Path)
    p_att.add_argument("--mode", choices=MODES, default="fast")
    p_att.add_argument("--limit", type=int, default=None)
    p_att.add_argument("--delay", type=float, default=None)
    p_clean = sub.add_parser("clean", help="remove simulator files")
    p_clean.add_argument("target", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.cmd == "seed":
            print(f"seeded {len(seed(args.target, args.files))} files in {args.target}")
        elif args.cmd == "attack":
            touched = attack(args.target, args.mode, args.limit, args.delay)
            print(f"simulated attack ({args.mode}) on {len(touched)} files")
        else:
            print(f"removed {clean(args.target)} files")
    except SafetyError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
