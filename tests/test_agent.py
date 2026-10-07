"""RAG agent: time parsing, embeddings, index, tools, and the Claude loop (mocked)."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.agent.claude_agent import RecoveryAgent
from app.agent.embeddings import HashingEmbedder
from app.agent.indexer import VectorIndex
from app.agent.retriever import Retriever, parse_time_range
from app.agent.tools import ToolContext, execute_tool, tool_definitions
from app.backup.snapshotter import create_snapshot
from app.config import Settings
from app.detection.model import Detector
from app.main import create_app
from app.storage.s3_client import S3Storage

# Wednesday 2026-10-07 15:30 UTC
NOW = datetime(2026, 10, 7, 15, 30, tzinfo=UTC)


# -- time parsing ---------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("what changed today?", "2026-10-07T00:00", "2026-10-07T15:30"),
        ("anything yesterday", "2026-10-06T00:00", "2026-10-07T00:00"),
        ("What changed on Tuesday?", "2026-10-06T00:00", "2026-10-07T00:00"),
        ("on wednesday", "2026-10-07T00:00", "2026-10-08T00:00"),  # today counts
        ("last wednesday", "2026-09-30T00:00", "2026-10-01T00:00"),
        ("last week", "2026-09-28T00:00", "2026-10-05T00:00"),
        ("this week", "2026-10-05T00:00", "2026-10-07T15:30"),
        ("in the past 3 hours", "2026-10-07T12:30", "2026-10-07T15:30"),
        ("last 2 days", "2026-10-05T15:30", "2026-10-07T15:30"),
        ("on 2026-10-01", "2026-10-01T00:00", "2026-10-02T00:00"),
        ("last month", "2026-09-01T00:00", "2026-10-01T00:00"),
    ],
)
def test_parse_time_range(text: str, start: str, end: str) -> None:
    tr = parse_time_range(text, NOW)
    assert tr is not None
    assert tr.start == datetime.fromisoformat(start).replace(tzinfo=UTC)
    assert tr.end == datetime.fromisoformat(end).replace(tzinfo=UTC)


def test_parse_time_range_respects_timezone() -> None:
    # 15:30 UTC is 21:00 in Kolkata; "today" starts at local midnight = 18:30 UTC prev day.
    tr = parse_time_range("today", NOW, "Asia/Kolkata")
    assert tr is not None and tr.start == datetime(2026, 10, 6, 18, 30, tzinfo=UTC)


def test_no_time_expression() -> None:
    assert parse_time_range("is it safe to restore?", NOW) is None


# -- embeddings + index -----------------------------------------------------------
def test_hashing_embedder_normalised_and_meaningful() -> None:
    e = HashingEmbedder()
    v = e.embed(["incident ransomware locked files", "incident ransomware locked", "photo zip"])
    assert v.shape == (3, 1024)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    assert v[0] @ v[1] > v[0] @ v[2]
    assert np.array_equal(e.embed(["same"]), e.embed(["same"]))


def _attack(files: list[Path]) -> None:
    for p in files:
        p.write_bytes(os.urandom(p.stat().st_size))
        p.rename(p.with_name(p.name + ".locked"))


@pytest.fixture
def populated(
    db: Session,
    storage: S3Storage,
    watch_dir: Path,
    detector: Detector,
    settings: Settings,
    make_tree,
):  # type: ignore[no-untyped-def]
    """Clean snapshot -> attack -> incident (via scan) -> suspect snapshot."""
    from app.detection.scan import run_scan

    files = make_tree(watch_dir, 30)
    clean = create_snapshot(db, storage, watch_dir)
    _attack(files)
    result = run_scan(
        db, detector, watch_dir=watch_dir, exclude_patterns=[], window_seconds=10,
        cooldown_seconds=60,
    )  # fmt: skip
    suspect = create_snapshot(db, storage, watch_dir)
    return SimpleNamespace(clean=clean, suspect=suspect, incident=result.incident)


def test_index_sync_is_incremental(db: Session, populated) -> None:  # type: ignore[no-untyped-def]
    index = VectorIndex(HashingEmbedder())
    first = index.sync(db)
    assert first["chunks"] == 4  # 2 snapshots + 1 diff + 1 incident
    assert first["embedded"] == 4
    assert index.sync(db)["embedded"] == 0  # nothing changed
    assert index.ensure_fresh(db) is False


def test_index_search_with_time_filter(db: Session, populated) -> None:  # type: ignore[no-untyped-def]
    index = VectorIndex(HashingEmbedder())
    # Words that only the incident chunk contains: a keyword-level embedder must rank it first.
    hits = index.search(db, "incident detected score threshold windows", k=4)
    assert hits[0].key == f"incident:{populated.incident.id}"

    past = datetime(2020, 1, 1, tzinfo=UTC)
    assert index.search(db, "incident", 4, past, past + timedelta(days=1)) == []
    only_inc = index.search(db, "anything", 4, kinds=["incident"])
    assert [h.kind for h in only_inc] == ["incident"]


def test_minilm_ranks_incident_first_for_natural_questions(db: Session, populated) -> None:  # type: ignore[no-untyped-def]
    """Semantic check with the real model; skipped when it isn't cached locally (CI)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never download inside tests
    from app.agent.embeddings import SentenceTransformerEmbedder

    emb = SentenceTransformerEmbedder()
    try:
        emb.embed(["warm-up"])
    except Exception as exc:  # model not cached / sentence-transformers unavailable
        pytest.skip(f"MiniLM not available offline: {exc}")
    index = VectorIndex(emb)
    for q in ("ransomware incident locked files", "was there an attack?"):
        assert index.search(db, q, k=4)[0].kind == "incident", q


def test_index_picks_up_changes_from_elsewhere(
    db: Session, storage: S3Storage, watch_dir: Path, populated
) -> None:  # type: ignore[no-untyped-def]
    index = VectorIndex(HashingEmbedder())
    index.sync(db)
    create_snapshot(db, storage, watch_dir)  # e.g. written by another process
    assert index.ensure_fresh(db) is True


# -- tools ----------------------------------------------------------------------
def test_tool_definitions_have_schemas() -> None:
    defs = tool_definitions()
    assert [d["name"] for d in defs] == [
        "search_metadata", "get_snapshot_diff", "list_incidents", "get_restore_candidates",
    ]  # fmt: skip
    assert all(d["input_schema"]["type"] == "object" for d in defs)
    assert not any(d["name"] == "restore" for d in defs)  # read-only by construction


def test_tools_execute(db: Session, populated) -> None:  # type: ignore[no-untyped-def]
    ctx = ToolContext(db, VectorIndex(HashingEmbedder()))
    out, err = execute_tool(ctx, "get_restore_candidates", {})
    data = json.loads(out)
    assert not err and data["last_clean_snapshot_id"] == populated.clean.id

    out, err = execute_tool(
        ctx,
        "get_snapshot_diff",
        {"base_snapshot_id": populated.clean.id, "target_snapshot_id": populated.suspect.id},
    )
    assert not err and json.loads(out)["summary"]["extension_changed"] == 30

    out, err = execute_tool(ctx, "list_incidents", {"status": "open"})
    assert not err and len(json.loads(out)["incidents"]) == 1

    out, err = execute_tool(ctx, "search_metadata", {"query": "locked", "kinds": ["diff"]})
    assert not err and json.loads(out)["results"][0]["key"].startswith("diff:")


def test_tool_errors_are_reported_not_raised(db: Session, populated) -> None:  # type: ignore[no-untyped-def]
    ctx = ToolContext(db, VectorIndex(HashingEmbedder()))
    assert execute_tool(ctx, "nope", {})[1] is True
    assert execute_tool(ctx, "search_metadata", {"query": ""})[1] is True
    out, err = execute_tool(
        ctx, "get_snapshot_diff", {"base_snapshot_id": "x", "target_snapshot_id": "y"}
    )
    assert err and "not found" in out


# -- the Claude loop with a scripted fake client ---------------------------------
def _text(t: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=t)


def _tool(id_: str, name: str, inp: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


def _msg(stop: str, *blocks: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop, content=list(blocks), model="claude-opus-5-5")


class FakeClient:
    """Stands in for anthropic.Anthropic: replays scripted responses, records requests."""

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        # snapshot the message list: the agent keeps appending to the same list
        self.requests.append(kwargs | {"messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def _agent(client: FakeClient, **kw: Any) -> RecoveryAgent:
    return RecoveryAgent(
        client, Retriever(VectorIndex(HashingEmbedder())), model="claude-opus-5-5", **kw
    )


def test_agent_tool_loop_and_validation(db: Session, populated) -> None:  # type: ignore[no-untyped-def]
    clean, suspect, inc = populated.clean.id, populated.suspect.id, populated.incident.id
    final = (
        f"## What changed\n30 files were encrypted (incident #{inc}).\n"
        f"## Evidence\nDiff {clean} -> {suspect}; also snap-20990101T000000Z-abcdef.\n"
        f"## Risk assessment\ncritical\n## Recommendation\nRestore {clean}.\n"
        f"RECOMMENDED_SNAPSHOT: {clean}"
    )
    client = FakeClient([
        _msg("tool_use", _text("Checking."), _tool("t1", "get_restore_candidates", {}),
             _tool("t2", "list_incidents", {"status": "open"})),
        _msg("end_turn", _text(final)),
    ])  # fmt: skip
    ans = _agent(client).ask(db, "What changed today, and is it safe to restore?", now=NOW)

    assert ans.recommended_snapshot_id == clean and ans.recommendation_warning is None
    assert set(ans.cited_snapshot_ids) == {clean, suspect}
    assert ans.cited_incident_ids == [inc]
    assert ans.unverified_ids == ["snap-20990101T000000Z-abcdef"]  # hallucination flagged
    assert "RECOMMENDED_SNAPSHOT" not in ans.answer
    assert [c.name for c in ans.tool_calls] == ["get_restore_candidates", "list_incidents"]

    first, second = client.requests
    assert first["model"] == "claude-opus-5-5"
    assert first["betas"] == ["server-side-fallback-2026-07-01"]
    assert first["fallbacks"] == "default"
    assert "tool_choice" not in first  # auto: forced tool choice is a 400 on this model
    assert "<retrieved_context>" in first["messages"][0]["content"]
    # Second request = history + assistant turn (unchanged) + ONE user msg with both results.
    roles = [m["role"] for m in second["messages"]]
    assert roles == ["user", "assistant", "user"]
    results = second["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert json.loads(results[0]["content"])["last_clean_snapshot_id"] == clean


def test_agent_rejects_unsafe_recommendation(db: Session, populated) -> None:  # type: ignore[no-untyped-def]
    client = FakeClient(
        [_msg("end_turn", _text(f"x\nRECOMMENDED_SNAPSHOT: {populated.suspect.id}"))]
    )
    ans = _agent(client, use_fallbacks=False).ask(db, "restore?", now=NOW)
    assert ans.recommended_snapshot_id is None
    assert "suspect" in ans.recommendation_warning
    assert "betas" not in client.requests[0]  # plain messages.create when fallbacks off


def test_agent_turn_cap_forces_final_answer(db: Session, populated) -> None:  # type: ignore[no-untyped-def]
    loop = _msg("tool_use", _tool("t", "list_incidents", {}))
    client = FakeClient([loop, loop, _msg("end_turn", _text("done\nRECOMMENDED_SNAPSHOT: NONE"))])
    ans = _agent(client, max_turns=2).ask(db, "anything?", now=NOW)
    assert len(client.requests) == 3
    assert client.requests[-1]["tool_choice"] == {"type": "none"}
    assert ans.recommended_snapshot_id is None and ans.recommendation_warning is None


def test_agent_handles_refusal(db: Session, populated) -> None:  # type: ignore[no-untyped-def]
    client = FakeClient([_msg("refusal")])
    ans = _agent(client).ask(db, "hm", now=NOW)
    assert "declined" in ans.answer and ans.stop_reason == "refusal"


# -- API ------------------------------------------------------------------------
def test_agent_api(
    settings: Settings, storage: S3Storage, detector: Detector, watch_dir: Path
) -> None:
    final = "## What changed\nNothing notable.\nRECOMMENDED_SNAPSHOT: NONE"
    fake = FakeClient([_msg("end_turn", _text(final))])
    app = create_app(
        settings=settings, storage=storage, detector=detector, agent_client=fake,
        embedder=HashingEmbedder(),
    )  # fmt: skip
    with TestClient(app) as c:
        sid = c.post("/backups").json()["id"]
        r = c.get("/agent/search", params={"q": "snapshot files today"})
        assert r.status_code == 200
        assert any(h["key"] == f"snapshot:{sid}" for h in r.json()["hits"])
        assert c.post("/agent/reindex").json()["chunks"] == 1

        r = c.post("/agent/ask", json={"question": "What changed today?"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["answer"].startswith("## What changed")
        assert body["recommended_snapshot_id"] is None
        assert body["time_range"].startswith("today")
        assert c.post("/agent/ask", json={"question": "x"}).status_code == 422


def test_agent_without_credentials_is_503_not_500(
    settings: Settings, storage: S3Storage, watch_dir: Path
) -> None:
    class NoCreds(FakeClient):
        def _create(self, **kwargs: Any) -> SimpleNamespace:
            # What anthropic 1.x raises when no API key / profile can be resolved.
            raise TypeError('"Could not resolve authentication method. Expected one of api_key"')

    app = create_app(
        settings=settings, storage=storage, agent_client=NoCreds([]), embedder=HashingEmbedder()
    )
    with TestClient(app) as c:
        r = c.post("/agent/ask", json={"question": "What changed today?"})
        assert r.status_code == 503
        assert "not configured" in r.json()["detail"]


def test_write_during_sync_is_not_masked_as_fresh(
    db: Session, storage: S3Storage, watch_dir: Path, settings: Settings
) -> None:
    """Regression: a snapshot written while the index is embedding (e.g. first-call
    model load) must trigger another sync, not be hidden behind a stale fingerprint."""
    from app.db.session import make_engine, make_session_factory

    other = make_session_factory(make_engine(settings.database_url))
    create_snapshot(db, storage, watch_dir)

    class SlowEmbedder(HashingEmbedder):
        fired = False

        def embed(self, texts: list[str]) -> np.ndarray:
            if not SlowEmbedder.fired:  # another process writes mid-sync
                SlowEmbedder.fired = True
                with other() as s:
                    create_snapshot(s, storage, watch_dir, label="concurrent")
            return super().embed(texts)

    index = VectorIndex(SlowEmbedder())
    assert index.sync(db)["chunks"] == 1  # saw only the first snapshot
    assert index.ensure_fresh(db) is True  # must notice the concurrent write
    keys = [h.key for h in index.search(db, "snapshot", k=10)]
    assert sum(k.startswith("snapshot:") for k in keys) == 2


def test_dry_runs_are_not_indexed(
    db: Session, storage: S3Storage, watch_dir: Path, settings: Settings
) -> None:
    from app.restore.restorer import run_restore

    snap = create_snapshot(db, storage, watch_dir)
    common = dict(
        snapshot_id=snap.id, target_path=None, watch_dir=watch_dir, restore_dir=settings.restore_dir
    )
    run_restore(db, storage, dry_run=True, **common)
    run_restore(db, storage, dry_run=False, **common)
    index = VectorIndex(HashingEmbedder())
    index.sync(db)
    keys = [h.key for h in index.search(db, "restore job", k=10, kinds=["restore"])]
    assert keys == ["restore:2"]  # only the real restore
