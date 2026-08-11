# Owner: Pranav Sukale — not yet implemented, see Task_Assignment.md
from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)


class ConversationUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    is_archived: bool | None = None


class ConversationMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    role: MessageRole
    content: str
    created_at: datetime


class ConversationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    is_archived: bool
    created_at: datetime
    updated_at: datetime
    message_count: int = Field(ge=0)


class ConversationDetail(ConversationSummary):
    messages: list[ConversationMessageResponse] = Field(default_factory=list)


class ConversationList(BaseModel):
    conversations: list[ConversationSummary]
    total: int = Field(ge=0)