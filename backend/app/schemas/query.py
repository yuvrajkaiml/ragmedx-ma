# Owner: Pranav Sukale — not yet implemented, see Task_Assignment.md
from enum import Enum
from uuid import UUID

from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.response import MedicalResponse


class MedicalQueryRequest(BaseModel):
    question: str = Field(
        min_length=5,
        max_length=5_000,
        description="The medical question to answer.",
    )
    conversation_id: UUID | None = Field(
        default=None,
        description="Optional existing conversation ID.",
    )
    include_agent_trace: bool = Field(
        default=False,
        description="Whether to include the internal agent-processing trace.",
    )


class QueryExecutionStatus(str, Enum):
    """Lifecycle states for an HTTP-created pipeline execution."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class QueryAcceptedResponse(BaseModel):
    """Immediate acknowledgement returned after scheduling one pipeline run."""

    query_id: UUID
    status: QueryExecutionStatus


class QueryExecutionResponse(BaseModel):
    """Durable query state, available after HTTP or WebSocket reconnects."""

    query_id: UUID
    status: QueryExecutionStatus
    created_at: datetime
    completed_at: datetime | None = None
    response: MedicalResponse | None = None
    error_code: str | None = None
