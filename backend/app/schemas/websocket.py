# Owner: Pranav Sukale — not yet implemented, see Task_Assignment.md
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class WebSocketEventType(str, Enum):
    AGENT_STATUS = "agent_status"
    RETRIEVAL_UPDATE = "retrieval_update"
    FINAL_RESPONSE = "final_response"
    ERROR = "error"
    PING = "ping"


class QueryStreamSubscription(BaseModel):
    query_id: UUID


class WebSocketEvent(BaseModel):
    query_id: UUID
    event_type: WebSocketEventType
    timestamp: datetime
    data: dict[str, Any] = Field(default_factory=dict)