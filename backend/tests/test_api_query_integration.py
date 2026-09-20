import pytest
from httpx import ASGITransport, AsyncClient
from uuid import uuid4

import app.main as main_module
from app.api.deps import get_current_user
from app.db import get_session
from app.db.models import User
from app.main import app
from app.services.query_execution_service import QueryExecutionService


class MockUser:
    """Mock User instance satisfying get_current_user dependency."""
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
        from datetime import datetime, timezone

        record.created_at = datetime.now(timezone.utc)


async def _fake_session_dependency():
    yield FakeSession()


async def _skip_persistence(*_, **__):
    """Avoid writing background state to the developer's local SQLite file."""


class MockSequenceLLM:
    """Mock LLM returning sequence of responses for API integration testing."""
    def __init__(self, responses: list):
        self.responses = list(responses)
        self.call_count = 0

    def generate(self, prompt: str, **kwargs):
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp, {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20, "is_mock": True}
        return "Default answer [E1]", {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20, "is_mock": True}


# ==============================================================================
# API Integration Tests for Mounted /api/v1/query/ask Endpoint
# ==============================================================================
async def test_query_ask_unauthenticated_returns_401():
    """Verify that unauthenticated request to /api/v1/query/ask returns 401."""
    # Ensure no dependency overrides
    app.dependency_overrides.clear()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/query/ask",
            json={"question": "What is the treatment for hypertension?"},
        )

    assert response.status_code == 401
async def test_query_ask_authenticated_creates_single_execution(monkeypatch):
    """HTTP starts a query once and returns its WebSocket subscription ID."""

    # Override current user dependency
    app.dependency_overrides[get_current_user] = lambda: MockUser()
    app.dependency_overrides[get_session] = _fake_session_dependency
    monkeypatch.setattr(QueryExecutionService, "_persist_status", _skip_persistence)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/query/ask",
            json={
                "question": "What is the first-line pharmacotherapy for hypertension?",
                "include_agent_trace": True,
            },
        )

    app.dependency_overrides.clear()

    assert response.status_code == 202
    data = response.json()
    assert data["status"] in ("queued", "running")
    assert data["query_id"]
async def test_query_ask_short_query_returns_422(monkeypatch):
    """Verify that query shorter than 5 characters fails Pydantic schema validation with 422."""
    app.dependency_overrides[get_current_user] = lambda: MockUser()
    app.dependency_overrides[get_session] = _fake_session_dependency
    monkeypatch.setattr(QueryExecutionService, "_persist_status", _skip_persistence)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/query/ask",
            json={"question": "Hi?"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 422
async def test_query_ask_accepts_trace_request_for_the_background_execution(monkeypatch):
    """Trace preference belongs to the query execution, not the 202 response."""
    app.dependency_overrides[get_current_user] = lambda: MockUser()
    app.dependency_overrides[get_session] = _fake_session_dependency
    monkeypatch.setattr(QueryExecutionService, "_persist_status", _skip_persistence)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/query/ask",
            json={
                "question": "What are symptoms of acute coronary syndrome?",
                "include_agent_trace": True,
            },
        )

    app.dependency_overrides.clear()

    assert response.status_code == 202
    data = response.json()
    assert data["query_id"]
    assert data["status"] in ("queued", "running")


async def test_readiness_reports_missing_runtime_resources():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/readiness")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "not_ready"
    assert payload["checks"]["llm"]["ready"] is False
    assert payload["checks"]["retrieval"]["ready"] is False


async def test_readiness_reports_ready_when_runtime_resources_exist(monkeypatch, tmp_path):
    monkeypatch.setattr(main_module, "FAISS_INDEX_DIR", tmp_path)
    (tmp_path / "index.faiss").touch()
    (tmp_path / "chunks.pkl").touch()
    monkeypatch.setattr(main_module.get_settings(), "llm_runtime_mode", "local")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/readiness")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["checks"]["llm"]["ready"] is True
    assert payload["checks"]["retrieval"]["ready"] is True
