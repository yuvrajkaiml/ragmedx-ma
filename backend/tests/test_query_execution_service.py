import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from app.api.v1 import websockets as websocket_module
from app.db.models import QueryExecutionRecord
from app.schemas.document import DocumentStatus, DocumentStatusResponse
from app.schemas.query import MedicalQueryRequest, QueryExecutionStatus
from app.schemas.response import ErrorResponse
from app.schemas.websocket import WebSocketEvent, WebSocketEventType
from app.services.query_execution_service import QueryExecutionService


class FakeAgentService:
    def __init__(self, retrieval_provider=None):
        self.calls = []
        # Mirror the real AgentService interface. A non-None provider models a
        # service wired by the FastAPI lifespan (the production KB-query path);
        # QueryExecutionService._run reads this to reuse the warm singleton.
        self.retrieval_provider = retrieval_provider if retrieval_provider is not None else _noop_retrieval_provider

    async def execute_query(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(0)
        return ErrorResponse(message="mock result", error_code="MOCK"), None


async def _noop_retrieval_provider(*_, **__):
    """Stand-in warm retrieval provider for unit tests (never invoked by the fake)."""
    return [], []


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


async def test_start_rejects_missing_conversation_before_persisting():
    service = QueryExecutionService(agent_service=FakeAgentService())
    session = FakeSession()

    request = MedicalQueryRequest(
        question="What is the treatment for hypertension?",
        conversation_id=uuid4(),
    )

    try:
        await service.start(request, uuid4(), session)
    except ValueError as exc:
        assert str(exc) == "Conversation not found"
    else:
        raise AssertionError("Expected an invalid conversation to be rejected")

    assert session.records == {}


async def test_persistence_failure_does_not_start_pipeline(monkeypatch):
    agent_service = FakeAgentService()
    service = QueryExecutionService(agent_service=agent_service)
    persistence_calls = 0

    async def fail_running_then_persist_terminal(execution, error_code=None):
        nonlocal persistence_calls
        persistence_calls += 1
        if error_code is None:
            raise RuntimeError("database unavailable")

    monkeypatch.setattr(service, "_persist_status", fail_running_then_persist_terminal)
    accepted = await service.start(
        MedicalQueryRequest(question="What is the treatment for hypertension?"),
        uuid4(),
        FakeSession(),
    )

    execution = service.get(accepted.query_id, uuid4())
    assert execution is None
    execution = service._executions[accepted.query_id]
    await execution.task

    assert execution.status == QueryExecutionStatus.FAILED
    assert execution.error == "Unable to persist query execution status"
    assert agent_service.calls == []
    assert persistence_calls == 2
    assert execution.events[-1].event_type == WebSocketEventType.ERROR


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


async def test_websocket_replays_durable_terminal_result_when_live_registry_is_gone(monkeypatch):
    service = QueryExecutionService(agent_service=FakeAgentService())
    user_id = uuid4()
    query_id = uuid4()

    async def fake_websocket_user(_):
        return SimpleNamespace(id=user_id)

    async def durable_terminal_event(_, __):
        return WebSocketEvent(
            query_id=query_id,
            event_type=WebSocketEventType.FINAL_RESPONSE,
            timestamp=datetime.now(timezone.utc),
            data={"response": {"response_type": "error", "message": "stored result"}},
        )

    monkeypatch.setattr(service, "subscribe", lambda *_: None)
    monkeypatch.setattr(service, "get_terminal_event", durable_terminal_event)
    monkeypatch.setattr(websocket_module, "query_execution_service", service)
    monkeypatch.setattr(websocket_module, "_get_websocket_user", fake_websocket_user)

    websocket = FakeWebSocket()
    await websocket_module.query_stream(websocket, query_id)

    assert websocket.accepted
    assert websocket.messages[-1]["data"]["response"]["message"] == "stored result"


async def test_kb_query_fails_loudly_when_warm_retrieval_provider_is_missing(monkeypatch):
    """A KB query must fail loudly (not silently rebuild) when the lifespan
    singleton was never wired — i.e. agent_service.retrieval_provider is None."""
    monkeypatch.setattr(QueryExecutionService, "_persist_status", _skip_persistence)
    agent_service = FakeAgentService()
    agent_service.retrieval_provider = None  # simulate bypassed lifespan wiring
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

    # The pipeline must NOT have run, and the terminal event must be a loud error.
    assert execution.status == QueryExecutionStatus.FAILED
    assert agent_service.calls == []
    assert execution.events[-1].event_type == WebSocketEventType.ERROR
    assert execution.events[-1].data["error_code"] == "RETRIEVAL_NOT_INITIALISED"


# ---------------------------------------------------------------------------
# Document-scope routing: a document_id must drive retrieval to the uploaded
# document's isolated index; absent document_id must behave EXACTLY as before
# (the warm shared-KB provider). These tests mock the agent and the scope
# validation so no real models or database are touched.
# ---------------------------------------------------------------------------


def _indexed_status(document_id, user_id):
    return DocumentStatusResponse(
        document_id=document_id,
        filename="report.pdf",
        status=DocumentStatus.INDEXED,
        uploaded_at=datetime.now(timezone.utc),
        error_message=None,
        chunk_count=3,
    )


class _FakeIndexService:
    """Stand-in for UploadedDocumentIndexService used in _run.

    ``evidence_provider`` returns a sentinel async callable so the test can
    assert the scoped provider (not the shared KB provider) was threaded into
    ``agent_service.execute_query``.
    """

    def __init__(self, *args, **kwargs):
        pass

    def evidence_provider(self, document_id):
        async def _scoped(*_, **__):  # pragma: no cover - never invoked by fake
            return [], []

        _scoped.__scoped_document_id__ = document_id
        return _scoped


def _patch_scope_indexed(monkeypatch, user_id):
    """Make DocumentService.get_status report an INDEXED doc for this user."""
    from app.services import document_service as ds_module

    def fake_init(self):
        pass

    monkeypatch.setattr(ds_module.DocumentService, "__init__", fake_init)
    monkeypatch.setattr(
        ds_module.DocumentService,
        "get_status",
        lambda self, document_id, uid: _indexed_status(document_id, uid),
    )


async def test_document_scoped_query_uses_isolated_index_provider(monkeypatch):
    monkeypatch.setattr(QueryExecutionService, "_persist_status", _skip_persistence)
    user_id = uuid4()
    document_id = uuid4()
    _patch_scope_indexed(monkeypatch, user_id)
    from app.services import uploaded_document_index as udi_module

    monkeypatch.setattr(udi_module, "UploadedDocumentIndexService", _FakeIndexService)

    agent_service = FakeAgentService()
    service = QueryExecutionService(agent_service=agent_service)

    accepted = await service.start(
        MedicalQueryRequest(question="What does my report say?", document_id=document_id),
        user_id,
        FakeSession(),
    )
    execution = service.get(accepted.query_id, user_id)
    assert execution is not None
    await execution.task

    assert execution.status == QueryExecutionStatus.COMPLETED
    assert len(agent_service.calls) == 1
    scoped = agent_service.calls[0]["retrieval_provider"]
    assert scoped is not None
    # Proves the provider came from the uploaded-document index, not the KB.
    assert getattr(scoped, "__scoped_document_id__", None) == document_id
    assert scoped is not agent_service.retrieval_provider


async def test_kb_query_without_document_id_uses_shared_provider(monkeypatch):
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
    # Unchanged KB path: the warm shared provider is reused verbatim.
    assert agent_service.calls[0]["retrieval_provider"] is agent_service.retrieval_provider


async def test_document_scope_rejects_unindexed_document(monkeypatch):
    user_id = uuid4()
    from app.services import document_service as ds_module

    monkeypatch.setattr(ds_module.DocumentService, "__init__", lambda self: None)

    def pending_status(self, document_id, uid):
        return DocumentStatusResponse(
            document_id=document_id,
            filename="report.pdf",
            status=DocumentStatus.PROCESSING,
            uploaded_at=datetime.now(timezone.utc),
        )

    monkeypatch.setattr(ds_module.DocumentService, "get_status", pending_status)

    service = QueryExecutionService(agent_service=FakeAgentService())

    try:
        await service.start(
            MedicalQueryRequest(question="What does my report say?", document_id=uuid4()),
            user_id,
            FakeSession(),
        )
    except ValueError as exc:
        assert "not ready" in str(exc)
    else:
        raise AssertionError("Expected an unindexed document to be rejected")


async def test_document_scope_rejects_unknown_document(monkeypatch):
    user_id = uuid4()
    from app.services import document_service as ds_module

    monkeypatch.setattr(ds_module.DocumentService, "__init__", lambda self: None)
    monkeypatch.setattr(
        ds_module.DocumentService,
        "get_status",
        lambda self, document_id, uid: None,
    )

    service = QueryExecutionService(agent_service=FakeAgentService())

    try:
        await service.start(
            MedicalQueryRequest(question="What does my report say?", document_id=uuid4()),
            user_id,
            FakeSession(),
        )
    except ValueError as exc:
        assert str(exc) == "Document not found"
    else:
        raise AssertionError("Expected an unknown document to be rejected")
