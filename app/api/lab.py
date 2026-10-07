"""Lab endpoints: drive the SAFE simulator from the dashboard.

Disabled unless ``ENABLE_LAB=true`` (docker compose turns it on for the local
demo). Even when enabled, every call goes through the simulator's own guards:
the target is ``WATCH_DIR``, which must resolve inside ``<project>/sandbox``,
and only files the simulator seeded itself are ever modified.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field

import simulator.fake_ransomware as sim
from app.api.deps import get_app_settings
from app.config import Settings

router = APIRouter(prefix="/lab", tags=["lab"])


def require_lab(settings: Settings = Depends(get_app_settings)) -> Settings:
    """404 (not 403) when disabled, so the feature is invisible in production."""
    if not settings.enable_lab:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Lab is disabled (ENABLE_LAB=false).")
    return settings


class SeedRequest(BaseModel):
    files: int = Field(default=120, ge=5, le=1000)


class AttackRequest(BaseModel):
    mode: Literal["fast", "slow", "partial", "inplace"] = "fast"
    limit: int | None = Field(default=None, ge=1, le=1000)


def _guard(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
    try:
        return fn(*args, **kwargs)
    except sim.SafetyError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Refused by simulator: {exc}") from exc


@router.post("/seed")
def seed(body: SeedRequest | None = None, settings: Settings = Depends(require_lab)) -> dict:
    """Remove previous simulator files, then seed fresh test files into WATCH_DIR."""
    body = body or SeedRequest()
    target = settings.watch_dir
    if (target / sim.MANIFEST_NAME).is_file():
        _guard(sim.clean, target, sandbox_root=sim.SANDBOX_ROOT)
    files = _guard(sim.seed, target, body.files, sandbox_root=sim.SANDBOX_ROOT)
    return {"seeded": len(files)}


@router.post("/benign")
def benign(settings: Settings = Depends(require_lab)) -> dict:
    """Normal user activity (edit notes, create a zip) — should NOT alert."""
    touched = _guard(sim.benign_activity, settings.watch_dir, sandbox_root=sim.SANDBOX_ROOT)
    return {"touched": touched}


@router.post("/attack", status_code=status.HTTP_202_ACCEPTED)
def attack(
    background_tasks: BackgroundTasks,
    body: AttackRequest | None = None,
    settings: Settings = Depends(require_lab),
) -> dict:
    """Start a simulated attack in the background ('slow' mode takes minutes)."""
    body = body or AttackRequest()
    # Validate synchronously so a refusal is reported to the caller, not lost.
    _guard(sim.validate_target, settings.watch_dir, sim.SANDBOX_ROOT)
    if not (settings.watch_dir / sim.MANIFEST_NAME).is_file():
        raise HTTPException(status.HTTP_409_CONFLICT, "Nothing seeded yet; call /lab/seed first.")
    limit = body.limit or (10 if body.mode == "slow" else None)
    background_tasks.add_task(
        sim.attack, settings.watch_dir, body.mode, limit, None, 13, sim.SANDBOX_ROOT
    )
    return {"started": True, "mode": body.mode, "limit": limit}


@router.post("/clean")
def clean(settings: Settings = Depends(require_lab)) -> dict:
    """Remove everything the simulator created (seeded files, .locked copies, note)."""
    if not (settings.watch_dir / sim.MANIFEST_NAME).is_file():
        return {"removed": 0}
    return {"removed": _guard(sim.clean, settings.watch_dir, sandbox_root=sim.SANDBOX_ROOT)}
