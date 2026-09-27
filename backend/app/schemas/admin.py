# Owner: Pranav Sukale — not yet implemented, see Task_Assignment.md
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.user import UserRole


class AdminUserSummary(BaseModel):
    id: UUID
    email: str
    username: str
    role: UserRole
    is_active: bool
    created_at: datetime


class AdminMetricsResponse(BaseModel):
    total_users: int = Field(ge=0)
    active_users: int = Field(ge=0)
    total_conversations: int = Field(ge=0)
    total_messages: int = Field(ge=0)
    generated_at: datetime

class AdminUserStatusUpdate(BaseModel):
    is_active: bool


class AgentLatencySummary(BaseModel):
    agent_type: str
    sample_count: int = Field(ge=0)
    average_duration_ms: float = Field(ge=0)


class AdminLatencyResponse(BaseModel):
    sampled_queries: int = Field(ge=0)
    measured_queries: int = Field(ge=0)
    failed_queries: int = Field(ge=0)
    average_processing_time_ms: float | None = Field(default=None, ge=0)
    p50_processing_time_ms: int | None = Field(default=None, ge=0)
    p95_processing_time_ms: int | None = Field(default=None, ge=0)
    by_agent: list[AgentLatencySummary]
    generated_at: datetime


class KnowledgeBaseStatsResponse(BaseModel):
    index_loaded: bool
    index_directory: str
    required_files_present: dict[str, bool]
    total_index_bytes: int = Field(ge=0)