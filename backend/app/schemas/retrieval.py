"""Data contracts shared by document ingestion and evidence retrieval."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


SourceType = Literal["pubmed", "pmc", "medqa", "bioasq", "uploaded_document"]


class SourceDocument(BaseModel):
    """A normalized document ready to be split and indexed."""

    document_id: UUID
    source_type: SourceType
    title: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1)
    url: str | None = None
    publication_year: int | None = Field(default=None, ge=1900)
    metadata: dict[str, str] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """A traceable section of a source document stored in the retrieval index."""

    chunk_id: UUID
    document_id: UUID
    source_type: SourceType
    title: str
    text: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    url: str | None = None
    publication_year: int | None = Field(default=None, ge=1900)
    metadata: dict[str, str] = Field(default_factory=dict)


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=5, max_length=5_000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class RetrievalResult(BaseModel):
    chunk: DocumentChunk
    relevance_score: float = Field(ge=0.0, le=1.0)
