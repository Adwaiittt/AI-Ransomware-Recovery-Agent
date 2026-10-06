"""Tools exposed to Claude. All are READ-ONLY.

There is deliberately no restore tool: the agent may only *recommend* a
snapshot. Performing a restore requires an explicit ``POST /restore`` by a
human. That boundary is enforced by the tool list itself, not by prompting.

Each tool's input is validated with a Pydantic model before running (model
output is untrusted input); failures go back to Claude as ``is_error`` results
so it can correct itself. Outputs are compact JSON with long lists truncated,
keeping the conversation small.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.indexer import VectorIndex
from app.backup.snapshotter import SnapshotNotFoundError, diff_snapshots, get_snapshot
from app.db.models import Incident, IncidentStatus
from app.restore.restorer import IncidentNotFoundForRestoreError, restore_candidates

MAX_LIST = 15


@dataclass
class ToolContext:
    """Per-request dependencies handed to tool functions."""

    session: Session
    index: VectorIndex


# -- input models ---------------------------------------------------------------
class SearchMetadataInput(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    start: datetime | None = None
    end: datetime | None = None
    kinds: list[Literal["snapshot", "diff", "incident", "restore"]] | None = None
    k: int = Field(default=8, ge=1, le=20)


class SnapshotDiffInput(BaseModel):
    base_snapshot_id: str
    target_snapshot_id: str


class ListIncidentsInput(BaseModel):
    status: Literal["open", "resolved"] | None = None
    start: datetime | None = None
    end: datetime | None = None


class RestoreCandidatesInput(BaseModel):
    incident_id: int | None = None


# -- implementations ------------------------------------------------------------
def _search_metadata(ctx: ToolContext, args: SearchMetadataInput) -> dict[str, Any]:
    hits = ctx.index.search(ctx.session, args.query, args.k, args.start, args.end, args.kinds)
    return {
        "results": [
            {"key": h.key, "timestamp": h.timestamp.isoformat(), "score": h.score, "text": h.text}
            for h in hits
        ]
    }


def _get_snapshot_diff(ctx: ToolContext, args: SnapshotDiffInput) -> dict[str, Any]:
    base = get_snapshot(ctx.session, args.base_snapshot_id)
    target = get_snapshot(ctx.session, args.target_snapshot_id)
    d = diff_snapshots(base, target)

    def cap(items: list[dict]) -> dict[str, Any]:
        return {"count": len(items), "items": items[:MAX_LIST], "truncated": len(items) > MAX_LIST}

    return {
        "base": {"id": base.id, "created_at": base.created_at.isoformat(), "state": base.state},
        "target": {
            "id": target.id,
            "created_at": target.created_at.isoformat(),
            "state": target.state,
        },
        "summary": d.summary(),
        "extension_changed": cap(d.extension_changed),
        "modified": cap(sorted(d.modified, key=lambda m: -m["entropy_delta"])),
        "added": cap(d.added),
        "removed": cap(d.removed),
        "renamed": cap(d.renamed),
    }


def _list_incidents(ctx: ToolContext, args: ListIncidentsInput) -> dict[str, Any]:
    stmt = select(Incident).order_by(Incident.started_at)
    if args.status:
        stmt = stmt.where(Incident.status == IncidentStatus(args.status))
    if args.start:
        stmt = stmt.where(Incident.ended_at >= args.start)
    if args.end:
        stmt = stmt.where(Incident.started_at < args.end)
    return {
        "incidents": [
            {
                "id": i.id,
                "status": i.status,
                "source": i.source,
                "started_at": i.started_at.isoformat(),
                "ended_at": i.ended_at.isoformat(),
                "score": i.score,
                "threshold": i.threshold,
                "windows": i.windows,
                "top_features": i.top_features,
                "affected_file_count": i.affected_file_count,
                "affected_files_sample": i.affected_files[:MAX_LIST],
            }
            for i in ctx.session.scalars(stmt)
        ]
    }


def _get_restore_candidates(ctx: ToolContext, args: RestoreCandidatesInput) -> dict[str, Any]:
    c = restore_candidates(ctx.session, args.incident_id)
    started = c.incident.started_at if c.incident else None
    return {
        "reference_incident": (
            {"id": c.incident.id, "started_at": started.isoformat()} if c.incident else None
        ),
        "last_clean_snapshot_id": c.last_clean.id if c.last_clean else None,
        "rule": "newest snapshot in state 'clean' created strictly before the incident start",
        "snapshots": [
            {
                "id": s.id,
                "created_at": s.created_at.isoformat(),
                "state": s.state,
                "file_count": s.file_count,
                "before_incident": (s.created_at < started) if started else None,
            }
            for s in c.snapshots[:MAX_LIST]
        ],
    }


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    model: type[BaseModel]
    fn: Callable[[ToolContext, Any], dict[str, Any]]


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "search_metadata",
        "Semantic search over backup metadata: snapshot summaries, diffs between consecutive "
        "snapshots, detector incidents, and restore jobs. Optionally restrict to an ISO-8601 "
        "time window [start, end) and/or chunk kinds. Use this to find what happened when.",
        SearchMetadataInput,
        _search_metadata,
    ),
    ToolSpec(
        "get_snapshot_diff",
        "Exact file-level diff between two snapshots: counts plus examples of extension "
        "changes (e.g. .docx -> .docx.locked), modified files sorted by entropy increase, "
        "added/removed/renamed files. Use for concrete evidence.",
        SnapshotDiffInput,
        _get_snapshot_diff,
    ),
    ToolSpec(
        "list_incidents",
        "List ransomware-detector incidents (optionally by status and time window) with "
        "score, threshold, top contributing features and affected files.",
        ListIncidentsInput,
        _list_incidents,
    ),
    ToolSpec(
        "get_restore_candidates",
        "Snapshots with their trust state relative to an incident (default: earliest open "
        "incident) and the system's computed last clean snapshot.",
        RestoreCandidatesInput,
        _get_restore_candidates,
    ),
)
_BY_NAME = {t.name: t for t in TOOLS}


def tool_definitions() -> list[dict[str, Any]]:
    """Tool schemas for the Messages API, in a fixed order (keeps the prompt cacheable)."""
    defs = []
    for t in TOOLS:
        schema = t.model.model_json_schema()
        schema.pop("title", None)
        defs.append({"name": t.name, "description": t.description, "input_schema": schema})
    return defs


def execute_tool(ctx: ToolContext, name: str, raw_input: Any) -> tuple[str, bool]:
    """Run a tool; returns (JSON string, is_error)."""
    spec = _BY_NAME.get(name)
    if spec is None:
        return json.dumps({"error": f"unknown tool {name!r}"}), True
    try:
        args = spec.model.model_validate(raw_input or {})
        result = spec.fn(ctx, args)
        return json.dumps(result, default=str), False
    except ValidationError as exc:
        return json.dumps({"error": "invalid input", "details": exc.errors(include_url=False)},
                          default=str), True  # fmt: skip
    except (SnapshotNotFoundError, IncidentNotFoundForRestoreError) as exc:
        return json.dumps({"error": f"not found: {exc}"}), True
