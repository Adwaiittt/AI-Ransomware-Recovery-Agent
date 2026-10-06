"""Agent endpoints: ask (Claude), search (retrieval only), reindex."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.agent.claude_agent import (
    AgentNotConfiguredError,
    AgentUnavailableError,
    RecoveryAgent,
    tool_calls_json,
)
from app.agent.retriever import Retriever
from app.api.deps import get_agent, get_db, get_retriever
from app.schemas.agent import AskRequest, AskResponse, SearchHitOut, SearchResponse

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    db: Session = Depends(get_db),
    agent: RecoveryAgent = Depends(get_agent),
) -> AskResponse:
    """Ask about backup history. The agent only RECOMMENDS a restore point."""
    try:
        a = agent.ask(db, body.question)
    except AgentNotConfiguredError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"Claude is not configured: {exc}"
        ) from exc
    except AgentUnavailableError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Claude request failed: {exc}") from exc
    return AskResponse(
        answer=a.answer,
        cited_snapshot_ids=a.cited_snapshot_ids,
        cited_incident_ids=a.cited_incident_ids,
        unverified_ids=a.unverified_ids,
        recommended_snapshot_id=a.recommended_snapshot_id,
        recommendation_warning=a.recommendation_warning,
        time_range=a.time_range,
        retrieved_keys=a.retrieved_keys,
        tool_calls=tool_calls_json(a.tool_calls),
        model=a.model,
        stop_reason=a.stop_reason,
    )


@router.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(min_length=1, max_length=500),
    k: int = Query(8, ge=1, le=20),
    db: Session = Depends(get_db),
    retriever: Retriever = Depends(get_retriever),
) -> SearchResponse:
    """Run only the retrieval step (date parsing + vector search); no Claude call."""
    r = retriever.retrieve(db, q, k)
    return SearchResponse(
        time_range=r.time_range.describe() if r.time_range else None,
        widened=r.widened,
        hits=[
            SearchHitOut(key=h.key, kind=h.kind, timestamp=h.timestamp, score=h.score, text=h.text)
            for h in r.hits
        ],
    )


@router.post("/reindex")
def reindex(
    db: Session = Depends(get_db),
    retriever: Retriever = Depends(get_retriever),
) -> dict[str, int]:
    """Force a full sync of the RAG index with the metadata DB."""
    return retriever.index.sync(db)
