"""API tests for /backups and /health."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "database": True,
        "storage": True,
        "detection_model": False,
    }


def test_create_list_get_backup(client: TestClient) -> None:
    r = client.post("/backups", json={"label": "first"})
    assert r.status_code == 201, r.text
    snap = r.json()
    assert snap["id"].startswith("snap-")
    assert snap["label"] == "first"
    assert snap["state"] == "clean"
    assert snap["file_count"] == 3
    assert snap["created_at"].endswith("Z")  # timezone-aware UTC

    r = client.get("/backups")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["items"][0]["id"] == snap["id"]

    r = client.get(f"/backups/{snap['id']}")
    assert r.status_code == 200
    assert {f["path"] for f in r.json()["files"]} == {
        "docs/report.docx",
        "docs/notes.txt",
        "code.py",
    }


def test_post_without_body_is_allowed(client: TestClient) -> None:
    assert client.post("/backups").status_code == 201


def test_list_is_newest_first_and_paginates(client: TestClient) -> None:
    ids = [client.post("/backups").json()["id"] for _ in range(3)]
    r = client.get("/backups", params={"limit": 2, "offset": 0}).json()
    assert r["total"] == 3
    assert [i["id"] for i in r["items"]] == list(reversed(ids))[:2]


def test_get_unknown_snapshot_404(client: TestClient) -> None:
    assert client.get("/backups/snap-nope").status_code == 404


def test_diff_endpoint(client: TestClient, watch_dir: Path) -> None:
    a = client.post("/backups").json()["id"]
    (watch_dir / "code.py").write_text("changed")
    b = client.post("/backups").json()["id"]
    r = client.get(f"/backups/{a}/diff/{b}")
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["modified"] == 1
    assert body["summary"]["unchanged"] == 2
    assert client.get(f"/backups/{a}/diff/snap-nope").status_code == 404


def test_invalid_pagination_rejected(client: TestClient) -> None:
    assert client.get("/backups", params={"limit": 0}).status_code == 422


def test_request_id_generated_and_echoed(client: TestClient) -> None:
    r = client.get("/health")
    assert len(r.headers["x-request-id"]) == 32
    r = client.get("/health", headers={"X-Request-ID": "demo-123"})
    assert r.headers["x-request-id"] == "demo-123"
    # Unsafe values are never reflected back (header/log injection).
    r = client.get("/health", headers={"X-Request-ID": "bad id\r\nx: y"})
    assert r.headers["x-request-id"] != "bad id\r\nx: y"


def test_json_log_lines_carry_request_id(capsys) -> None:  # type: ignore[no-untyped-def]
    import json
    import logging

    from app.logging_config import JsonFormatter, request_id_var

    token = request_id_var.set("abc")
    try:
        rec = logging.makeLogRecord({"msg": "hello", "levelname": "INFO", "name": "t"})
        rec.snapshot_id = "snap-1"
        line = json.loads(JsonFormatter().format(rec))
    finally:
        request_id_var.reset(token)
    assert line["request_id"] == "abc" and line["snapshot_id"] == "snap-1"
    assert line["msg"] == "hello"
