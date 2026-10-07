"""Dashboard serving, /info, and the Lab endpoints that drive the simulator."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import simulator.fake_ransomware as sim
from app.config import Settings
from app.main import create_app
from app.storage.s3_client import S3Storage


def test_root_redirects_to_dashboard(client: TestClient) -> None:
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/ui/"


def test_dashboard_served_with_strict_csp(client: TestClient) -> None:
    r = client.get("/ui/")
    assert r.status_code == 200 and "Recovery Console" in r.text
    csp = r.headers["content-security-policy"]
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp
    assert r.headers["x-content-type-options"] == "nosniff"
    for asset in ("app.js", "styles.css"):
        assert client.get(f"/ui/{asset}").status_code == 200
    # API responses are not affected by the UI CSP.
    assert "content-security-policy" not in client.get("/health").headers


def test_dashboard_js_never_uses_innerhtml() -> None:
    js = (Path(__file__).parents[1] / "app" / "ui" / "app.js").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in js.splitlines() if not line.strip().startswith(("*", "/*", "//"))
    )
    assert "innerHTML" not in code and "insertAdjacentHTML" not in code


def test_info_has_no_secrets(client: TestClient, settings: Settings) -> None:
    body = client.get("/info").json()
    assert body["watch_dir"] == str(settings.watch_dir.resolve())
    assert body["lab_enabled"] is False
    assert set(body) == {
        "watch_dir", "restore_dir", "storage_mode", "bucket", "detection_window_seconds",
        "lab_enabled", "agent_model", "embedding_model",
    }  # fmt: skip
    assert "testing" not in body.values()  # the AWS key/secret set by the test env


def test_lab_disabled_by_default(client: TestClient) -> None:
    for path in ("/lab/seed", "/lab/benign", "/lab/attack", "/lab/clean"):
        assert client.post(path).status_code == 404


@pytest.fixture
def lab_client(
    tmp_path: Path, settings: Settings, storage: S3Storage, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    sandbox = tmp_path / "sandbox"
    watch = sandbox / "watched"
    watch.mkdir(parents=True)
    monkeypatch.setattr(sim, "SANDBOX_ROOT", sandbox)
    lab_settings = settings.model_copy(update={"enable_lab": True, "watch_dir": watch})
    with TestClient(create_app(settings=lab_settings, storage=storage)) as c:
        yield c


def test_lab_flow(lab_client: TestClient, tmp_path: Path) -> None:
    c = lab_client
    watch = tmp_path / "sandbox" / "watched"
    assert c.post("/lab/attack").status_code == 409  # nothing seeded yet
    assert c.post("/lab/seed", json={"files": 20}).json() == {"seeded": 20}
    touched = c.post("/lab/benign").json()["touched"]
    assert any(t.endswith(".zip") for t in touched)

    r = c.post("/lab/attack", json={"mode": "fast"})
    assert r.status_code == 202
    time.sleep(0.2)  # background task runs after the response in TestClient
    locked = list(watch.rglob("*.locked"))
    edited = sum(1 for t in touched if t.endswith(".txt"))
    assert len(locked) == 20 - edited  # files edited by benign activity are skipped

    assert c.post("/lab/clean").json()["removed"] >= 20
    assert not list(watch.rglob("*.locked")) and not list(watch.rglob("*.zip"))


def test_lab_refuses_watch_dir_outside_sandbox(
    tmp_path: Path, settings: Settings, storage: S3Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sim, "SANDBOX_ROOT", tmp_path / "sandbox")  # watch_dir is NOT inside
    s = settings.model_copy(update={"enable_lab": True})
    with TestClient(create_app(settings=s, storage=storage)) as c:
        r = c.post("/lab/seed")
        assert r.status_code == 400 and "Refused" in r.json()["detail"]
