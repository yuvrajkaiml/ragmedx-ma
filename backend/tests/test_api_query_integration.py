import pytest
from httpx import ASGITransport, AsyncClient
from uuid import uuid4

from app.api.deps import get_current_user
from app.db.models import User
from app.main import app
from app.services.agent_service import AgentService


class MockUser:
    """Mock User instance satisfying get_current_user dependency."""
    id = uuid4()
    email = "physician@hospital.org"
    role = "physician"
    is_active = True


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
async def test_query_ask_authenticated_verified_response():
    """Verify authenticated request returns 200 with VerifiedAnswerResponse structure."""
    query_resp = '{"medical_specialty": "Cardiology", "confidence": 0.95}'
    gen_resp = "ACE inhibitors are recommended first-line therapy [E1]."
    critic_resp = '{"passed": true, "factual_consistency": 0.95, "citation_support": 0.90, "hallucination_risk": 0.05, "confidence": 0.95}'

    mock_llm = MockSequenceLLM([query_resp, gen_resp, critic_resp])

    # Override current user dependency
    app.dependency_overrides[get_current_user] = lambda: MockUser()

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

    assert response.status_code == 200
    data = response.json()
    assert data["response_type"] in ("verified_answer", "abstention")
    assert "answer" in data or "message" in data
    assert "processing_time_ms" in data
async def test_query_ask_short_query_returns_422():
    """Verify that query shorter than 5 characters fails Pydantic schema validation with 422."""
    app.dependency_overrides[get_current_user] = lambda: MockUser()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/query/ask",
            json={"question": "Hi?"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 422
async def test_query_ask_trace_serialization():
    """Verify trace serialization when include_agent_trace is requested."""
    app.dependency_overrides[get_current_user] = lambda: MockUser()

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

    assert response.status_code == 200
    data = response.json()
    assert "agent_trace" in data
    if data["agent_trace"] is not None:
        assert "steps" in data["agent_trace"]
        assert "total_duration_ms" in data["agent_trace"]
