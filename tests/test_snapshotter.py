"""Snapshot creation, dedup, and diff classification."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.backup.snapshotter import SnapshotError, create_snapshot, diff_snapshots
from app.storage.s3_client import S3Storage, blob_key


def test_first_snapshot_uploads_everything(
    db: Session, storage: S3Storage, watch_dir: Path
) -> None:
    snap = create_snapshot(db, storage, watch_dir)
    assert snap.file_count == 3
    assert snap.uploaded_files == 3
    assert len(storage.list_keys("objects/")) == 3
    manifest = storage.get_json(snap.manifest_key)
    assert manifest["snapshot_id"] == snap.id
    assert {f["path"] for f in manifest["files"]} == {
        "docs/report.docx",
        "docs/notes.txt",
        "code.py",
    }
    assert all(
        {"sha256", "entropy", "size", "mtime", "extension"} <= f.keys() for f in manifest["files"]
    )


def test_unchanged_files_are_deduplicated(db: Session, storage: S3Storage, watch_dir: Path) -> None:
    create_snapshot(db, storage, watch_dir)
    (watch_dir / "code.py").write_text("print('changed')\n")
    second = create_snapshot(db, storage, watch_dir)
    assert second.file_count == 3
    assert second.uploaded_files == 1  # only the changed file
    assert len(storage.list_keys("objects/")) == 4


def test_identical_content_in_two_paths_stored_once(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    root = tmp_path / "dup"
    root.mkdir()
    (root / "a.txt").write_text("same")
    (root / "b.txt").write_text("same")
    snap = create_snapshot(db, storage, root)
    assert snap.file_count == 2
    assert snap.uploaded_files == 1


def test_dedup_falls_back_to_bucket_when_db_is_empty(
    db: Session, storage: S3Storage, watch_dir: Path
) -> None:
    # Simulate a wiped DB: blob is already in the bucket, DB knows nothing.
    data = (watch_dir / "code.py").read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    storage._client.put_object(Bucket=storage.bucket, Key=blob_key(sha), Body=data)
    snap = create_snapshot(db, storage, watch_dir)
    assert snap.uploaded_bytes == sum(f.size for f in snap.files if f.path != "code.py")


def test_missing_source_dir_raises(db: Session, storage: S3Storage, tmp_path: Path) -> None:
    with pytest.raises(SnapshotError):
        create_snapshot(db, storage, tmp_path / "nope")


def test_diff_classifies_changes(db: Session, storage: S3Storage, watch_dir: Path) -> None:
    base = create_snapshot(db, storage, watch_dir)

    # modified in place
    (watch_dir / "code.py").write_text("print('v2')\n")
    # plain rename (same content)
    (watch_dir / "docs" / "notes.txt").rename(watch_dir / "docs" / "notes-old.txt")
    # ransomware-style: content replaced with random bytes + extension appended
    report = watch_dir / "docs" / "report.docx"
    report.unlink()
    (watch_dir / "docs" / "report.docx.locked").write_bytes(os.urandom(4096))
    # brand new file
    (watch_dir / "new.md").write_text("# new")

    target = create_snapshot(db, storage, watch_dir)
    diff = diff_snapshots(base, target)

    assert [m["path"] for m in diff.modified] == ["code.py"]
    assert [(r["old_path"], r["new_path"]) for r in diff.renamed] == [
        ("docs/notes.txt", "docs/notes-old.txt")
    ]
    assert len(diff.extension_changed) == 1
    ext = diff.extension_changed[0]
    assert ext["old_path"] == "docs/report.docx"
    assert ext["new_path"] == "docs/report.docx.locked"
    assert ext["new_extension"] == ".locked"
    assert ext["content_changed"] is True
    assert ext["entropy_after"] > 7.5 and ext["entropy_delta"] > 2.0
    assert [a["path"] for a in diff.added] == ["new.md"]
    assert diff.removed == []
    assert diff.summary()["unchanged"] == 0


def test_diff_detects_replaced_extension(db: Session, storage: S3Storage, tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / "photo.jpg").write_bytes(b"\xff\xd8" + b"jpegish" * 100)
    base = create_snapshot(db, storage, root)
    (root / "photo.jpg").unlink()
    (root / "photo.enc").write_bytes(os.urandom(800))
    target = create_snapshot(db, storage, root)
    diff = diff_snapshots(base, target)
    assert len(diff.extension_changed) == 1
    assert diff.extension_changed[0]["old_extension"] == ".jpg"
    assert diff.extension_changed[0]["new_extension"] == ".enc"


def test_snapshot_is_suspect_while_incident_open(
    db: Session, storage: S3Storage, watch_dir: Path
) -> None:
    from datetime import UTC, datetime

    from app.db.models import Incident, IncidentStatus, SnapshotState

    assert create_snapshot(db, storage, watch_dir).state is SnapshotState.clean
    now = datetime.now(UTC)
    db.add(
        Incident(
            created_at=now, started_at=now, ended_at=now, source="test",
            status=IncidentStatus.open, score=0.9, threshold=0.5, model_name="t",
        )
    )  # fmt: skip
    db.commit()
    assert create_snapshot(db, storage, watch_dir).state is SnapshotState.suspect
