"""The simulator's safety guarantees are part of the spec, so they are tested."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

import simulator.fake_ransomware as sim
from app.backup.manifest import shannon_entropy


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    root = tmp_path / "sandbox"
    (root / "watched").mkdir(parents=True)
    return root


def test_refuses_root_home_and_outside(sandbox: Path, tmp_path: Path) -> None:
    for bad in (Path(Path.cwd().anchor), Path.home(), tmp_path / "elsewhere", sandbox / ".."):
        with pytest.raises(sim.SafetyError):
            sim.validate_target(bad, sandbox)
    assert sim.validate_target(sandbox / "watched", sandbox) == (sandbox / "watched").resolve()


def test_refuses_symlink_escape(sandbox: Path, tmp_path: Path) -> None:
    outside = tmp_path / "real"
    outside.mkdir()
    link = sandbox / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not permitted on this platform/user")
    with pytest.raises(sim.SafetyError):
        sim.validate_target(link, sandbox)


def test_attack_touches_only_pristine_seeded_files(sandbox: Path) -> None:
    target = sandbox / "watched"
    seeded = sim.seed(target, n_files=20, sandbox_root=sandbox)
    bystander = target / "docs" / "user_file.txt"
    bystander.write_text("not created by the simulator")
    edited = target / seeded[0]
    edited.write_text("user edited this after seeding")

    touched = sim.attack(target, "fast", sandbox_root=sandbox)

    assert seeded[0] not in touched and len(touched) == 19
    assert bystander.read_text() == "not created by the simulator"
    assert edited.read_text() == "user edited this after seeding"
    for rel in touched:
        locked = target / (rel + sim.FAKE_EXT)
        assert locked.exists() and not (target / rel).exists()
        if locked.stat().st_size > 4096:
            assert shannon_entropy(locked.read_bytes()) > 7.9
    assert "SIMULATION" in (target / sim.NOTE_NAME).read_text()


def test_inplace_and_partial_modes(sandbox: Path) -> None:
    target = sandbox / "watched"
    seeded = sim.seed(target, n_files=6, sandbox_root=sandbox)
    before = {r: (target / r).read_bytes() for r in seeded}
    touched = sim.attack(target, "inplace", limit=3, sandbox_root=sandbox)
    for rel in touched:
        assert (target / rel).exists()
        assert (target / rel).read_bytes() != before[rel]

    target2 = sandbox / "watched2"
    seeded2 = sim.seed(target2, n_files=6, sandbox_root=sandbox)
    original = {r: (target2 / r).read_bytes() for r in seeded2}
    for rel in sim.attack(target2, "partial", sandbox_root=sandbox):
        data = (target2 / (rel + sim.FAKE_EXT)).read_bytes()
        assert data[-10:] == original[rel][-10:]  # tail untouched
        assert hashlib.sha256(data).hexdigest() != hashlib.sha256(original[rel]).hexdigest()


def test_attack_without_seed_refuses(sandbox: Path) -> None:
    with pytest.raises(sim.SafetyError):
        sim.attack(sandbox / "watched", sandbox_root=sandbox)


def test_clean_removes_only_simulator_files(sandbox: Path) -> None:
    target = sandbox / "watched"
    sim.seed(target, n_files=10, sandbox_root=sandbox)
    sim.attack(target, "fast", limit=5, sandbox_root=sandbox)
    keep = target / "mine.txt"
    keep.write_text("keep me")
    assert sim.clean(target, sandbox_root=sandbox) == 10
    assert keep.exists()
    assert not (target / sim.MANIFEST_NAME).exists()


def test_simulator_has_no_network_or_process_capabilities() -> None:
    tree = ast.parse(Path(sim.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {"socket", "urllib", "http", "requests", "httpx", "subprocess", "ctypes",
                 "smtplib", "ftplib", "winreg", "cryptography", "Crypto"}  # fmt: skip
    assert not imported & forbidden


def test_cli_refusal_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    assert sim.main(["seed", str(Path.home())]) == 2
    assert "REFUSED" in capsys.readouterr().err
