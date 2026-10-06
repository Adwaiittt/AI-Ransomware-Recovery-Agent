"""Restore: last-clean rule, path safety, end-to-end restore + verification."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.db.models import SnapshotState
from app.restore.restorer import RestoreError, resolve_target, safe_join, select_last_clean
from app.storage.s3_client import S3Storage, blob_key

T0 = datetime(2026, 10, 6, 9, 0, tzinfo=UTC)


def _snap(id_: str, minutes: int, state: SnapshotState = SnapshotState.clean) -> SimpleNamespace:
    return SimpleNamespace(id=id_, created_at=T0 + timedelta(minutes=minutes), state=state)


# -- pure rule ------------------------------------------------------------------
def test_last_clean_is_newest_clean_strictly_before_incident() -> None:
    snaps = [
        _snap("a", 0),
        _snap("b", 10),
        _snap("c", 20),  # exactly at incident start -> not "before"
        _snap("d", 30, SnapshotState.suspect),
    ]
    assert select_last_clean(snaps, T0 + timedelta(minutes=20)).id == "b"


def test_last_clean_skips_suspect_and_infected_even_if_before() -> None:
    snaps = [
        _snap("a", 0),
        _snap("b", 10, SnapshotState.suspect),
        _snap("c", 15, SnapshotState.infected),
    ]
    assert select_last_clean(snaps, T0 + timedelta(minutes=20)).id == "a"


def test_last_clean_none_when_nothing_qualifies() -> None:
    assert select_last_clean([_snap("a", 30)], T0) is None
    assert select_last_clean([], None) is None


def test_last_clean_without_incident_is_newest_clean() -> None:
    snaps = [_snap("a", 0), _snap("b", 10), _snap("c", 20, SnapshotState.suspect)]
    assert select_last_clean(snaps, None).id == "b"


# -- path safety ----------------------------------------------------------------
def test_resolve_target_rules(tmp_path: Path) -> None:
    watch, rdir = tmp_path / "watch", tmp_path / "restore"
    watch.mkdir()
    target, in_place = resolve_target(None, "snap-1", watch, rdir)
    assert target == (rdir / "snap-1").resolve() and not in_place
    assert resolve_target(str(watch), "snap-1", watch, rdir) == (watch.resolve(), True)
    assert resolve_target(str(rdir / "x" / "y"), "s", watch, rdir)[1] is False
    for bad in (tmp_path / "elsewhere", rdir, watch / "sub", Path.home()):
        with pytest.raises(RestoreError):
            resolve_target(str(bad), "snap-1", watch, rdir)


def test_safe_join_blocks_traversal(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    assert safe_join(root, "a/b.txt") == root / "a" / "b.txt"
    for bad in ("../escape.txt", "a/../../escape.txt", "."):
        with pytest.raises(RestoreError):
            safe_join(root, bad)


# -- API end-to-end -------------------------------------------------------------
def _attack(files: list[Path]) -> None:
    for p in files:
        p.write_bytes(os.urandom(p.stat().st_size))
        p.rename(p.with_name(p.name + ".locked"))


def test_full_recovery_flow(
    client_with_model: TestClient, watch_dir: Path, settings, make_tree
) -> None:  # type: ignore[no-untyped-def]
    c = client_with_model
    files = make_tree(watch_dir, 30)
    originals = {
        p.relative_to(watch_dir).as_posix(): p.read_bytes()
        for p in watch_dir.rglob("*")
        if p.is_file()
    }
    clean_id = c.post("/backups").json()["id"]

    _attack(files)
    incident = c.post("/detection/scan").json()["incident"]
    suspect_id = c.post("/backups").json()["id"]

    cand = c.get("/restore/candidates").json()
    assert cand["reference_incident_id"] == incident["id"]
    assert cand["last_clean_snapshot_id"] == clean_id
    by_id = {x["id"]: x for x in cand["candidates"]}
    assert by_id[clean_id]["recommended"] and by_id[clean_id]["before_incident"]
    assert by_id[suspect_id]["state"] == "suspect" and not by_id[suspect_id]["recommended"]

    # Untrusted snapshot needs force.
    r = c.post("/restore", json={"snapshot_id": suspect_id})
    assert r.status_code == 422 and "force" in r.json()["detail"]

    # Dry run (default) to a separate dir: nothing written.
    r = c.post("/restore", json={"snapshot_id": clean_id})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job"]["dry_run"] and body["job"]["status"] == "planned"
    assert body["summary"]["create"] == len(originals)
    assert not (settings.restore_dir / clean_id).exists()

    # Real restore to the separate dir, verified.
    body = c.post("/restore", json={"snapshot_id": clean_id, "dry_run": False}).json()
    assert body["job"]["status"] == "succeeded"
    assert body["job"]["files_verified"] == len(originals) == body["job"]["files_restored"]
    out = settings.restore_dir / clean_id
    for rel, data in originals.items():
        assert (out / rel).read_bytes() == data

    # In-place dry run sees encrypted copies as extras and originals as missing.
    body = c.post("/restore", json={"snapshot_id": clean_id, "target_path": str(watch_dir)}).json()
    assert body["job"]["in_place"]
    assert body["summary"]["create"] == 30
    assert sum(1 for x in body["extra_files"] if x.endswith(".locked")) == 30

    # In-place restore with quarantine: originals back, .locked files moved aside.
    body = c.post(
        "/restore",
        json={
            "snapshot_id": clean_id,
            "target_path": str(watch_dir),
            "dry_run": False,
            "quarantine_extras": True,
        },
    ).json()
    assert body["job"]["status"] == "succeeded"
    assert len(body["quarantined"]) == 30
    assert not list(watch_dir.rglob("*.locked"))
    assert len(list(Path(body["quarantine_dir"]).rglob("*.locked"))) == 30
    for rel, data in originals.items():
        assert (watch_dir / rel).read_bytes() == data

    # Second in-place restore is a no-op.
    body = c.post("/restore", json={"snapshot_id": clean_id, "target_path": str(watch_dir)}).json()
    assert body["summary"]["unchanged"] == len(originals)

    jobs = c.get("/restore/jobs").json()
    assert len(jobs) == 5 and jobs[0]["id"] > jobs[-1]["id"]


def test_corrupted_blob_is_detected_and_not_written(
    client: TestClient, storage: S3Storage, settings
) -> None:  # type: ignore[no-untyped-def]
    snap = client.post("/backups").json()
    detail = client.get(f"/backups/{snap['id']}").json()
    victim = detail["files"][0]
    storage._client.put_object(
        Bucket=storage.bucket, Key=blob_key(victim["sha256"]), Body=b"tampered"
    )
    body = client.post("/restore", json={"snapshot_id": snap["id"], "dry_run": False}).json()
    assert body["job"]["status"] == "failed"
    assert body["job"]["files_failed"] == 1
    assert "hash mismatch" in body["failures"][0]["error"]
    out = settings.restore_dir / snap["id"]
    assert not (out / victim["path"]).exists()
    assert not list(out.rglob("*.tmp"))  # temp file cleaned up


def test_restore_errors(client: TestClient, tmp_path: Path) -> None:
    assert client.post("/restore", json={"snapshot_id": "snap-nope"}).status_code == 404
    sid = client.post("/backups").json()["id"]
    r = client.post("/restore", json={"snapshot_id": sid, "target_path": str(tmp_path / "x")})
    assert r.status_code == 422
    r = client.post("/restore", json={"snapshot_id": sid, "quarantine_extras": True})
    assert r.status_code == 422
    assert client.get("/restore/candidates", params={"incident_id": 999}).status_code == 404
