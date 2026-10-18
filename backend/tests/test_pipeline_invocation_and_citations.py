"""
Pipeline-invocation + citation-integrity + no-bypass tests (MOCKED — no network,
no paid API, no FAISS rebuild, no real LLM).

These assert the FE/BE integration cannot shortcut the real pipeline, that
invalid citation markers never become valid, that abstention survives
serialization to the FE shape, and that the production secret guard fires.
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

import app.main as main_module
from app.agents.generator import GeneratorAgent
from app.api.deps import get_current_user
from app.db import get_session
from app.main import app, create_app
from app.rag.citations.markers import evidence_markers, extract_markers
from app.schemas.query import MedicalQueryRequest, QueryAcceptedResponse, QueryExecutionStatus
from app.schemas.response import AbstentionResponse, ConfidenceScoreOutput, EvidenceSnippet
from app.services.query_service import QueryService


class MockUser:
    id = uuid4()
    email = "physician@hospital.org"
    role = "physician"
    is_active = True


class FakeSession:
    async def commit(self):
        return None


async def _fake_session_dependency():
    yield FakeSession()


class FakeLLM:
    """Returns a fixed string for generate(); used for the generator path."""

    def __init__(self, response):
        self.response = response

    def generate(self, prompt, temperature=0.0):
        return self.response


# ---------------------------------------------------------------------------
# Pipeline invocation: /query/ask reaches QueryService.start exactly once and
# there is no alternate "direct answer" path.
# ---------------------------------------------------------------------------

async def test_query_ask_invokes_query_service_start_once(monkeypatch):
    """A FE POST /query/ask must trigger exactly one QueryService.start call
    with the parsed MedicalQueryRequest — the only answer path."""
    calls = []

    async def fake_start(self, request, user_id, session):
        calls.append((request, user_id))
        return QueryAcceptedResponse(
            query_id=uuid4(), status=QueryExecutionStatus.QUEUED
        )

    monkeypatch.setattr(QueryService, "start", fake_start)
    app.dependency_overrides[get_current_user] = lambda: MockUser()
    app.dependency_overrides[get_session] = _fake_session_dependency

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/query/ask",
                json={
                    "question": "What is the first-line therapy for HFrEF?",
                    "include_agent_trace": True,
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 202
    # Exactly one invocation of the real pipeline entry point.
    assert len(calls) == 1
    request_arg, user_id_arg = calls[0]
    assert isinstance(request_arg, MedicalQueryRequest)
    assert request_arg.question == "What is the first-line therapy for HFrEF?"
    assert request_arg.include_agent_trace is True
    assert user_id_arg == MockUser.id


# ---------------------------------------------------------------------------
# Invalid markers never become valid citations (reuse the real guards).
# ---------------------------------------------------------------------------

def test_marker_utility_distinguishes_hallucinated_marker():
    """A hallucinated [Doc 9] is extracted but is NOT in the evidence marker
    set, so it can never resolve to a valid citation."""
    answer = "ACE inhibitors reduce mortality [Doc 9]."
    evidence = [{"citation_marker": "[Doc 1]", "cited_text": "ACE inhibitors reduce mortality."}]

    used = extract_markers(answer)
    valid = set(evidence_markers(evidence))

    assert "[Doc 9]" in used
    assert "[Doc 9]" not in valid
    assert used - valid == {"[Doc 9]"}


async def test_generator_raises_on_unresolved_marker():
    """GeneratorAgent.run raises RuntimeError when the answer cites evidence
    absent from the evidence set — the FE can only render emitted markers, so
    an invalid marker can never be upgraded to a valid citation."""
    llm = FakeLLM("ACE inhibitors reduce mortality [Doc 9].")
    agent = GeneratorAgent(llm_wrapper=llm)
    state = {
        "original_question": "Do ACE inhibitors reduce mortality?",
        "evidence": [
            {"citation_marker": "[Doc 1]", "cited_text": "ACE inhibitors reduce mortality in HFrEF."}
        ],
    }
    with pytest.raises(RuntimeError):
        await agent.run(state)


# ---------------------------------------------------------------------------
# Abstention / insufficient-evidence survives serialization to the FE shape.
# ---------------------------------------------------------------------------

def test_abstention_renders_to_fe_abstained_shape():
    resp = AbstentionResponse(
        message="Insufficient high-quality evidence to answer confidently.",
        abstention_reason="insufficient_evidence",
        partial_findings=[
            EvidenceSnippet(
                chunk_id=uuid4(),
                document_id=uuid4(),
                source_type="pubmed",
                title="Partial source",
                cited_text="A partially relevant finding.",
                relevance_score=0.35,
            )
        ],
        confidence=ConfidenceScoreOutput(score=0.15, level="low"),
        processing_time_ms=800,
    )
    d = resp.model_dump(mode="json")

    # adaptBackendResponse maps this arm to state: 'abstained'
    assert d["response_type"] == "abstention"
    assert d["message"]
    assert d["abstention_reason"] == "insufficient_evidence"
    assert d["partial_findings"][0]["cited_text"] == "A partially relevant finding."
    assert d["confidence"]["score"] == 0.15


# ---------------------------------------------------------------------------
# Production secret guard (main.create_app).
# ---------------------------------------------------------------------------

def test_create_app_raises_with_default_secret_in_non_debug(monkeypatch):
    settings = main_module.get_settings()
    monkeypatch.setattr(settings, "secret_key", "change-me-before-production")
    monkeypatch.setattr(settings, "debug", False)
    with pytest.raises(RuntimeError):
        create_app()


def test_create_app_allows_default_secret_in_debug(monkeypatch):
    settings = main_module.get_settings()
    monkeypatch.setattr(settings, "secret_key", "change-me-before-production")
    monkeypatch.setattr(settings, "debug", True)
    built = create_app()
    assert built is not None


def test_create_app_allows_strong_secret_in_non_debug(monkeypatch):
    settings = main_module.get_settings()
    monkeypatch.setattr(settings, "secret_key", "a-strong-secret-key-value-123")
    monkeypatch.setattr(settings, "debug", False)
    built = create_app()
    assert built is not None
