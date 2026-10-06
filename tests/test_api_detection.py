"""API tests for /detection/*."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient


def test_status_and_scan_without_model(client: TestClient) -> None:
    body = client.get("/detection/status").json()
    assert body["model_loaded"] is False
    assert body["monitor"]["alive"] is False
    assert client.post("/detection/scan").status_code == 503


def test_scan_requires_baseline(client_with_model: TestClient) -> None:
    r = client_with_model.post("/detection/scan")
    assert r.status_code == 409


def test_scan_unknown_snapshot_404(client_with_model: TestClient) -> None:
    client_with_model.post("/backups")
    r = client_with_model.post("/detection/scan", json={"base_snapshot_id": "snap-nope"})
    assert r.status_code == 404


def test_benign_change_scans_clean(client_with_model: TestClient, watch_dir: Path) -> None:
    client_with_model.post("/backups")
    notes = watch_dir / "docs" / "notes.txt"
    notes.write_text(notes.read_text() + "follow-up item\n")
    body = client_with_model.post("/detection/scan").json()
    assert body["verdict"] == "clean"
    assert body["incident"] is None
    assert body["diff_summary"]["modified"] == 1


def test_attack_end_to_end(client_with_model: TestClient, watch_dir: Path, make_tree) -> None:  # type: ignore[no-untyped-def]
    c = client_with_model
    files = make_tree(watch_dir, 40)
    clean_id = c.post("/backups").json()["id"]

    for p in files:  # ransomware-like: ciphertext-looking bytes + new extension
        p.write_bytes(os.urandom(p.stat().st_size))
        p.rename(p.with_name(p.name + ".locked"))

    r = c.post("/detection/scan")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] == "alert"
    assert body["diff_summary"]["extension_changed"] == 40
    incident = body["incident"]
    assert incident["status"] == "open" and incident["source"] == "scan"
    assert incident["top_features"]

    listed = c.get("/detection/incidents", params={"status": "open"}).json()
    assert [i["id"] for i in listed] == [incident["id"]]
    assert c.get("/detection/status").json()["open_incidents"] == 1

    # Snapshots taken while the incident is open are not trusted...
    assert c.post("/backups").json()["state"] == "suspect"
    assert c.get(f"/backups/{clean_id}").json()["state"] == "clean"

    # ...until it is resolved.
    assert c.post(f"/detection/incidents/{incident['id']}/resolve").json()["status"] == "resolved"
    assert c.post("/backups").json()["state"] == "clean"
    assert c.post("/detection/incidents/9999/resolve").status_code == 404


def test_scan_without_recording(client_with_model: TestClient, watch_dir: Path, make_tree) -> None:  # type: ignore[no-untyped-def]
    c = client_with_model
    files = make_tree(watch_dir, 30)
    c.post("/backups")
    for p in files:
        p.write_bytes(os.urandom(p.stat().st_size))
    body = c.post("/detection/scan", json={"record_incident": False}).json()
    assert body["verdict"] == "alert" and body["incident"] is None
    assert c.get("/detection/incidents").json() == []
