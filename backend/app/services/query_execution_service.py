"""In-process lifecycle manager for HTTP-created medical query executions.

The HTTP endpoint is the sole producer of executions.  WebSocket clients may
only subscribe to an existing ``query_id``; they never invoke the pipeline.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.db import AsyncSessionLocal
from app.db.models import QueryExecutionRecord
from app.schemas.query import (
    MedicalQueryRequest,
    QueryAcceptedResponse,
    QueryExecutionResponse,
    QueryExecutionStatus,
)
from app.schemas.response import MedicalResponse
from app.schemas.websocket import WebSocketEvent, WebSocketEventType
from app.services.agent_service import AgentService


@dataclass
class QueryExecution:
    """Server-side state for exactly one pipeline invocation."""

    query_id: UUID
    user_id: UUID
    request: MedicalQueryRequest
    status: QueryExecutionStatus = QueryExecutionStatus.QUEUED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    response: MedicalResponse | None = None
    error: str | None = None
    events: list[WebSocketEvent] = field(default_factory=list)
    subscribers: list[asyncio.Queue[WebSocketEvent]] = field(default_factory=list)
    task: asyncio.Task[None] | None = None


class QueryExecutionService:
    """Start and observe single-run medical query executions.

    The registry is intentionally process-local for the current single-worker
    deployment.  A Redis/pub-sub implementation can replace this service when
    the application is deployed with multiple API workers.
    """

    def __init__(self, agent_service: AgentService | None = None) -> None:
        self.agent_service = agent_service or AgentService()
        self._executions: dict[UUID, QueryExecution] = {}

    async def start(
        self,
        request: MedicalQueryRequest,
        user_id: UUID,
        session: AsyncSession,
    ) -> QueryAcceptedResponse:
        """Create an execution and schedule its one and only pipeline task."""
        query_id = uuid4()
        record = QueryExecutionRecord(
            id=query_id,
            user_id=user_id,
            conversation_id=request.conversation_id,
            question=request.question,
            include_agent_trace=request.include_agent_trace,
            status=QueryExecutionStatus.QUEUED.value,
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)

        execution = QueryExecution(
            query_id=query_id,
            user_id=user_id,
            request=request,
            created_at=record.created_at or datetime.now(timezone.utc),
        )
        self._executions[query_id] = execution
        self._publish(
            execution,
            WebSocketEventType.AGENT_STATUS,
            {"agent": "orchestrator", "status": "queued"},
        )
        execution.task = asyncio.create_task(self._run(execution))
        return QueryAcceptedResponse(query_id=query_id, status=execution.status)

    async def get_persisted(
        self,
        query_id: UUID,
        user_id: UUID,
        session: AsyncSession,
    ) -> QueryExecutionResponse | None:
        """Load a durable execution result for its owner."""
        record = await session.get(QueryExecutionRecord, query_id)
        if record is None or record.user_id != user_id:
            return None
        return QueryExecutionResponse(
            query_id=record.id,
            status=QueryExecutionStatus(record.status),
            created_at=record.created_at,
            completed_at=record.completed_at,
            response=record.response_data,
            error_code=record.error_code,
        )

    def get(self, query_id: UUID, user_id: UUID) -> QueryExecution | None:
        """Return an execution only to the user who started it."""
        execution = self._executions.get(query_id)
        if execution is None or execution.user_id != user_id:
            return None
        return execution

    def subscribe(
        self,
        query_id: UUID,
        user_id: UUID,
    ) -> tuple[list[WebSocketEvent], asyncio.Queue[WebSocketEvent]] | None:
        """Return the event backlog plus a queue for future execution events."""
        execution = self.get(query_id, user_id)
        if execution is None:
            return None
        queue: asyncio.Queue[WebSocketEvent] = asyncio.Queue()
        execution.subscribers.append(queue)
        return list(execution.events), queue

    def unsubscribe(self, query_id: UUID, queue: asyncio.Queue[WebSocketEvent]) -> None:
        execution = self._executions.get(query_id)
        if execution is not None and queue in execution.subscribers:
            execution.subscribers.remove(queue)

    async def _run(self, execution: QueryExecution) -> None:
        execution.status = QueryExecutionStatus.RUNNING
        await self._persist_status(execution)
        self._publish(
            execution,
            WebSocketEventType.AGENT_STATUS,
            {"agent": "orchestrator", "status": "running"},
        )
        self._publish(
            execution,
            WebSocketEventType.RETRIEVAL_UPDATE,
            {"status": "retrieval_requested"},
        )

        try:
            response, trace = await self.agent_service.execute_query(
                question=execution.request.question,
                query_id=execution.query_id,
                conversation_id=execution.request.conversation_id,
                include_trace=execution.request.include_agent_trace,
            )
            execution.response = response
            execution.status = QueryExecutionStatus.COMPLETED
            await self._persist_status(execution)

            if trace is not None:
                for step in trace.steps:
                    self._publish(
                        execution,
                        WebSocketEventType.AGENT_STATUS,
                        {
                            "agent": step.agent_type,
                            "status": step.status,
                            "duration_ms": step.duration_ms,
                            "details": step.details or {},
                        },
                    )

            self._publish(
                execution,
                WebSocketEventType.FINAL_RESPONSE,
                {"response": response.model_dump(mode="json")},
            )
        except Exception as exc:
            execution.status = QueryExecutionStatus.FAILED
            execution.error = str(exc)
            await self._persist_status(execution, error_code="QUERY_EXECUTION_ERROR")
            self._publish(
                execution,
                WebSocketEventType.ERROR,
                {"message": "Query execution failed", "error_code": "QUERY_EXECUTION_ERROR"},
            )

    def _publish(
        self,
        execution: QueryExecution,
        event_type: WebSocketEventType,
        data: dict[str, Any],
    ) -> None:
        event = WebSocketEvent(
            query_id=execution.query_id,
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            data=data,
        )
        execution.events.append(event)
        for subscriber in list(execution.subscribers):
            subscriber.put_nowait(event)

    async def _persist_status(
        self,
        execution: QueryExecution,
        error_code: str | None = None,
    ) -> None:
        """Persist lifecycle changes without sharing an HTTP request session."""
        async with AsyncSessionLocal() as session:
            record = await session.get(QueryExecutionRecord, execution.query_id)
            if record is None:
                return
            record.status = execution.status.value
            if execution.response is not None:
                record.response_data = execution.response.model_dump(mode="json")
            if error_code is not None:
                record.error_code = error_code
            if execution.status in {QueryExecutionStatus.COMPLETED, QueryExecutionStatus.FAILED}:
                record.completed_at = datetime.now(timezone.utc)
            await session.commit()


query_execution_service = QueryExecutionService()
