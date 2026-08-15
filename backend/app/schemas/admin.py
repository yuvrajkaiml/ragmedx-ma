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