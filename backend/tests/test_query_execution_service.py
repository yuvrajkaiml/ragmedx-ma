import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from app.api.v1 import websockets as websocket_module
from app.db.models import QueryExecutionRecord
from app.schemas.query import MedicalQueryRequest, QueryExecutionStatus
from app.schemas.response import ErrorResponse
from app.schemas.websocket import WebSocketEventType
from app.services.query_execution_service import QueryExecutionService


class FakeAgentService:
    def __init__(self):
        self.calls = []

    async def execute_query(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(0)
        return ErrorResponse(message="mock result", error_code="MOCK"), None


class FakeWebSocket:
    def __init__(self):
        self.query_params = {"token": "test-token"}
        self.accepted = False
        self.closed = None
        self.messages = []

    async def accept(self):
        self.accepted = True

    async def close(self, code, reason):
        self.closed = (code, reason)

    async def send_json(self, payload):
        self.messages.append(payload)


class FakeSession:
    def __init__(self):
        self.records = {}

    def add(self, record):
        self.records[record.id] = record

    async def commit(self):
        return None

    async def refresh(self, record):
        record.created_at = record.created_at or datetime.now(timezone.utc)

    async def get(self, _, record_id):
        return self.records.get(record_id)


async def _skip_persistence(*_, **__):
    """Keep unit tests independent of the application SQLite database."""


async def test_http_created_execution_runs_pipeline_once_and_replays_events(monkeypatch):
    monkeypatch.setattr(QueryExecutionService, "_persist_status", _skip_persistence)
    agent_service = FakeAgentService()
    service = QueryExecutionService(agent_service=agent_service)
    user_id = uuid4()

    accepted = await service.start(
        MedicalQueryRequest(question="What is the treatment for hypertension?"),
        user_id,
        FakeSession(),
    )

    execution = service.get(accepted.query_id, user_id)
    assert execution is not None
    await execution.task

    assert execution.status == QueryExecutionStatus.COMPLETED
    assert len(agent_service.calls) == 1
    assert agent_service.calls[0]["query_id"] == accepted.query_id

    subscription = service.subscribe(accepted.query_id, user_id)
    assert subscription is not None
    backlog, queue = subscription
    assert queue.empty()
    assert [event.event_type for event in backlog] == [
        WebSocketEventType.AGENT_STATUS,
        WebSocketEventType.AGENT_STATUS,
        WebSocketEventType.RETRIEVAL_UPDATE,
        WebSocketEventType.FINAL_RESPONSE,
    ]
    assert backlog[-1].data["response"]["response_type"] == "error"


async def test_subscription_cannot_access_another_users_execution(monkeypatch):
    monkeypatch.setattr(QueryExecutionService, "_persist_status", _skip_persistence)
    service = QueryExecutionService(agent_service=FakeAgentService())
    owner_id = uuid4()
    accepted = await service.start(
        MedicalQueryRequest(question="What is the treatment for hypertension?"),
        owner_id,
        FakeSession(),
    )

    assert service.get(accepted.query_id, uuid4()) is None
    assert service.subscribe(accepted.query_id, uuid4()) is None

    execution = service.get(accepted.query_id, owner_id)
    assert execution is not None
    await execution.task


async def test_persisted_query_status_is_available_to_its_owner():
    service = QueryExecutionService(agent_service=FakeAgentService())
    user_id = uuid4()
    query_id = uuid4()
    session = FakeSession()
    session.add(
        QueryExecutionRecord(
            id=query_id,
            user_id=user_id,
            question="What is the treatment for hypertension?",
            include_agent_trace=False,
            status=QueryExecutionStatus.COMPLETED.value,
            response_data={"response_type": "error", "message": "mock result", "error_code": "MOCK"},
            created_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )
    )

    persisted = await service.get_persisted(query_id, user_id, session)

    assert persisted is not None
    assert persisted.status == QueryExecutionStatus.COMPLETED
    assert persisted.response.response_type == "error"
    assert await service.get_persisted(query_id, uuid4(), session) is None


async def test_websocket_replays_existing_execution_without_running_it_again(monkeypatch):
    monkeypatch.setattr(QueryExecutionService, "_persist_status", _skip_persistence)
    agent_service = FakeAgentService()
    service = QueryExecutionService(agent_service=agent_service)
    user_id = uuid4()
    accepted = await service.start(
        MedicalQueryRequest(question="What is the treatment for hypertension?"),
        user_id,
        FakeSession(),
    )
    execution = service.get(accepted.query_id, user_id)
    assert execution is not None
    await execution.task

    async def fake_websocket_user(_):
        return SimpleNamespace(id=user_id)

    monkeypatch.setattr(websocket_module, "query_execution_service", service)
    monkeypatch.setattr(websocket_module, "_get_websocket_user", fake_websocket_user)

    websocket = FakeWebSocket()
    await websocket_module.query_stream(websocket, accepted.query_id)

    assert websocket.accepted
    assert websocket.closed is None
    assert len(agent_service.calls) == 1
    assert websocket.messages[-1]["event_type"] == WebSocketEventType.FINAL_RESPONSE.value
