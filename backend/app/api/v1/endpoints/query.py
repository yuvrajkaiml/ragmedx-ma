"""Query endpoint — POST /api/v1/query/ask.

The RetrievalService singleton is built once at startup and stored on
app.state.  It is injected here via get_retrieval_service() so that every
request reuses the same loaded FAISS index, embedding model, and cross-
encoder rather than deserialising them from disk on each call.
"""
from fastapi import APIRouter, Depends, Request

from app.api.deps import get_current_user
from app.db.models import User
from app.schemas.query import MedicalQueryRequest
from app.schemas.response import MedicalResponse
from app.services.agent_service import AgentService
from app.services.query_service import QueryService
from app.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/query", tags=["Query"])


def get_retrieval_service(request: Request) -> RetrievalService | None:
    """FastAPI dependency — returns the app-lifetime RetrievalService singleton.

    Returns ``None`` when ``app.state`` has not been populated (test context
    without a running lifespan).  In that case ``MultiAgentOrchestrator``
    falls back to building its own instance — which is the correct behaviour
    for tests that don't need a real index.
    """
    return getattr(request.app.state, "retrieval_service", None)


def get_llm_wrapper(request: Request):
    """FastAPI dependency — returns the eagerly-loaded LLM wrapper, or None.

    None in mock mode or tests; the orchestrator falls back to constructing
    its own wrapper when this is None.
    """
    return getattr(request.app.state, "llm_wrapper", None)


@router.post("/ask", response_model=MedicalResponse)
async def ask_medical_question(
    request: MedicalQueryRequest,
    _: User = Depends(get_current_user),
    retrieval_service: RetrievalService | None = Depends(get_retrieval_service),
    llm_wrapper=Depends(get_llm_wrapper),
):
    agent_service = AgentService(
        llm_wrapper=llm_wrapper,  # None in mock/test → orchestrator builds its own
        retrieval_provider=(
            retrieval_service.retrieve_evidence if retrieval_service is not None else None
        ),
    )
    return await QueryService(agent_service=agent_service).ask(request)
