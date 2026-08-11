# Owner: Pranav Sukale — not yet implemented, see Task_Assignment.md
from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"


class DocumentUploadResponse(BaseModel):
    document_id: UUID
    filename: str
    status: DocumentStatus
    uploaded_at: datetime
    message: str


class DocumentStatusResponse(BaseModel):
    document_id: UUID
    filename: str
    status: DocumentStatus
    uploaded_at: datetime
    error_message: str | None = None
    chunk_count: int | None = Field(default=None, ge=0)