"""
Cross-layer FE <-> BE contract tests (MOCKED — no network, no paid API).

These pin the single reconciled query contract against the *real* route handlers
and the *real* Pydantic schemas, using the actual request body the frontend
`api.ts:askQuestion` sends: {"question", "include_agent_trace": true, "document_id"?}.

Harness mirrors tests/test_api_query_integration.py:
- httpx.AsyncClient + ASGITransport(app=app)
- app.dependency_overrides for get_current_user / get_session
- QueryExecutionService._persist_status monkeypatched so nothing writes to the
  developer's local SQLite file.

Scope checked here:
- FE request body (exact fields askQuestion sends) is accepted -> 202, body
  validates against QueryAcceptedResponse (query_id is a UUID; status is a
  QueryExecutionStatus enum value the FE QueryExecutionStatus type lists).
- Invalid/empty question -> 422 canonical error body.
- Unauthenticated -> 401 canonical ErrorResponse carrying `message`.
- Each MedicalResponse arm serializes to exactly the keys api.ts reads, and
  UUID/datetime fields serialize to strings (citations survive serialization).
"""

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_current_user
from app.db import get_session
from app.main import app
from app.schemas.query import QueryAcceptedResponse, QueryExecutionStatus
from app.schemas.response import (
    AbstentionResponse,
    AgentTraceOutput,
    AgentTraceStep,
    CitationOutput,
    ConfidenceScoreOutput,
    ErrorResponse,
    EvidenceSnippet,
    VerifiedAnswerResponse,
)
from app.services.query_execution_service import QueryExecutionService


class MockUser:
    id = uuid4()
    email = "physician@hospital.org"
    role = "physician"
    is_active = True


class FakeSession:
    def add(self, record):
        self.record = record

    async def commit(self):
        return None

    async def refresh(self, record):
        record.created_at = datetime.now(timezone.utc)


async def _fake_session_dependency():
    yield FakeSession()


async def _skip_persistence(*_, **__):
    """Avoid writing background state to the developer's local SQLite file."""


def _override_auth_and_session(monkeypatch):
    app.dependency_overrides[get_current_user] = lambda: MockUser()
    app.dependency_overrides[get_session] = _fake_session_dependency
    monkeypatch.setattr(QueryExecutionService, "_persist_status", _skip_persistence)


# ---------------------------------------------------------------------------
# FE request contract matches the BE schema
# ---------------------------------------------------------------------------

async def test_fe_request_body_accepted_and_response_matches_schema(monkeypatch):
    """The exact body api.ts:askQuestion sends -> 202 + QueryAcceptedResponse."""
    _override_auth_and_session(monkeypatch)
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
    data = response.json()
    # Validates against the real schema the way the FE consumes it.
    accepted = QueryAcceptedResponse.model_validate(data)
    assert isinstance(accepted.query_id, UUID)
    # FE's QueryExecutionStatus union = {queued, running, completed, failed}.
    assert accepted.status in {s for s in QueryExecutionStatus}
    assert data["status"] in {"queued", "running", "completed", "failed"}


async def test_fe_request_body_with_document_id_accepted(monkeypatch):
    """askQuestion optionally includes document_id; the schema accepts it."""
    _override_auth_and_session(monkeypatch)
    # document_id must reference an indexed doc; a random one makes QueryService
    # raise ValueError -> endpoint maps to 404 (not 422), proving the field is a
    # valid, parsed part of the contract (not rejected at validation).
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/query/ask",
                json={
                    "question": "Summarize the attached trial's primary endpoint.",
                    "include_agent_trace": True,
                    "document_id": str(uuid4()),
                },
            )
    finally:
        app.dependency_overrides.clear()

    # Either accepted (202) or a 404 for an unknown/unindexed doc — never a 422,
    # which would mean document_id is not part of the accepted contract.
    assert response.status_code in (202, 404)
    assert response.status_code != 422


# ---------------------------------------------------------------------------
# Invalid / empty query
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_question", ["", "Hi?"])
async def test_invalid_question_returns_422_canonical_error(monkeypatch, bad_question):
    """Empty and <5-char questions -> 422 with canonical error body."""
    _override_auth_and_session(monkeypatch)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/query/ask",
                json={"question": bad_question},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    body = response.json()
    assert body["response_type"] == "error"
    assert body["error_code"] == "VALIDATION_ERROR"


# ---------------------------------------------------------------------------
# Auth failure surfaces a canonical ErrorResponse with `message`
# ---------------------------------------------------------------------------

async def test_unauthenticated_returns_401_with_message():
    """No bearer token -> 401 and a `message` field FE's err.message can read."""
    app.dependency_overrides.clear()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/query/ask",
            json={"question": "What is the treatment for hypertension?"},
        )

    assert response.status_code == 401
    body = response.json()
    assert body["response_type"] == "error"
    assert isinstance(body.get("message"), str) and body["message"]


# ---------------------------------------------------------------------------
# Response-schema correctness: each arm serializes to exactly the keys api.ts
# reads, and UUID/datetime fields become strings (citations survive).
# ---------------------------------------------------------------------------

def _confidence() -> ConfidenceScoreOutput:
    return ConfidenceScoreOutput(score=0.82, level="high")


def test_verified_answer_serializes_to_fe_shape():
    now = datetime.now(timezone.utc)
    resp = VerifiedAnswerResponse(
        answer="ACE inhibitors reduce mortality in HFrEF [Doc 1].",
        confidence=_confidence(),
        citations=[
            CitationOutput(
                document_id=uuid4(),
                citation_marker="[Doc 1]",
                title="HFrEF Guideline",
                journal="Circulation",
                publication_year=2023,
                doi="10.1000/xyz",
                url="https://example.org/x",
                cited_text="ACE inhibitors reduce mortality in HFrEF.",
                relevance_score=0.91,
            )
        ],
        agent_trace=AgentTraceOutput(
            steps=[
                AgentTraceStep(
                    agent_type="generator",
                    status="completed",
                    started_at=now,
                    completed_at=now,
                    duration_ms=12,
                    details={"action": "generated answer"},
                )
            ],
            total_duration_ms=12,
        ),
        processing_time_ms=1234,
        total_tokens_used=256,
    )
    d = resp.model_dump(mode="json")

    # adaptBackendResponse keys
    assert d["response_type"] == "verified_answer"
    assert set(["answer", "confidence", "citations", "agent_trace"]).issubset(d.keys())
    assert d["confidence"]["score"] == 0.82 and d["confidence"]["level"] == "high"

    # adaptEvidence / citationObjects keys on each citation
    c = d["citations"][0]
    for key in ("document_id", "citation_marker", "title", "journal",
                "publication_year", "doi", "url", "cited_text", "relevance_score"):
        assert key in c
    # UUID serialized to str so the citation survives JSON transport.
    assert isinstance(c["document_id"], str)
    UUID(c["document_id"])  # parses -> valid UUID string

    # agent_trace steps carry the keys adaptTraceSteps reads.
    step = d["agent_trace"]["steps"][0]
    for key in ("agent_type", "status", "started_at", "duration_ms", "details"):
        assert key in step
    assert isinstance(step["started_at"], str)


def test_abstention_serializes_to_fe_shape():
    resp = AbstentionResponse(
        message="Insufficient evidence to answer confidently.",
        abstention_reason="insufficient_evidence",
        partial_findings=[
            EvidenceSnippet(
                chunk_id=uuid4(),
                document_id=uuid4(),
                source_type="bioasq",
                title="Partial source",
                cited_text="A partially relevant finding.",
                relevance_score=0.4,
            )
        ],
        suggested_followup="Try narrowing the question.",
        confidence=ConfidenceScoreOutput(score=0.2, level="low"),
        processing_time_ms=900,
    )
    d = resp.model_dump(mode="json")

    assert d["response_type"] == "abstention"
    for key in ("message", "abstention_reason", "partial_findings", "suggested_followup", "confidence"):
        assert key in d
    # adaptBackendResponse maps partial_findings[].cited_text
    assert d["partial_findings"][0]["cited_text"] == "A partially relevant finding."
    assert isinstance(d["partial_findings"][0]["document_id"], str)


def test_error_serializes_to_fe_shape():
    resp = ErrorResponse(message="Pipeline failed.", error_code="PIPELINE_ERROR")
    d = resp.model_dump(mode="json")
    assert d["response_type"] == "error"
    assert d["message"] == "Pipeline failed."
    assert d["error_code"] == "PIPELINE_ERROR"
