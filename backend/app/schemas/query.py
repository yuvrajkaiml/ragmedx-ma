# Owner: Pranav Sukale — not yet implemented, see Task_Assignment.md
from uuid import UUID

from pydantic import BaseModel, Field


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