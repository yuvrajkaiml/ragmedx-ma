from datetime import datetime
from typing import Annotated, Any, Literal, Union
from uuid import UUID

from pydantic import BaseModel, Field


class ConfidenceScoreOutput(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    level: Literal["high", "moderate", "low"]
    retrieval_score: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    answer_consistency_score: float | None = Field(default=None, ge=0.0, le=1.0)
    inter_source_agreement: float | None = Field(default=None, ge=0.0, le=1.0)


class CitationOutput(BaseModel):
    document_id: UUID
    citation_marker: str
    title: str
    journal: str | None = None
    publication_year: int | None = Field(default=None, ge=1900)
    doi: str | None = None
    url: str | None = None
    cited_text: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class EvidenceSnippet(BaseModel):
    chunk_id: UUID
    document_id: UUID
    source_type: Literal["pubmed", "pmc", "medqa", "bioasq", "uploaded_document"]
    title: str
    cited_text: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class AgentTraceStep(BaseModel):
    agent_type: str
    status: Literal["running", "completed", "failed"]
    started_at: datetime
    completed_at: datetime | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    details: dict[str, Any] | None = None


class AgentTraceOutput(BaseModel):
    steps: list[AgentTraceStep]
    total_duration_ms: int = Field(ge=0)


class VerifiedAnswerResponse(BaseModel):
    response_type: Literal["verified_answer"] = "verified_answer"
    answer: str
    confidence: ConfidenceScoreOutput
    citations: list[CitationOutput]
    agent_trace: AgentTraceOutput | None = None
    processing_time_ms: int = Field(ge=0)
    total_tokens_used: int = Field(ge=0)


class AbstentionResponse(BaseModel):
    response_type: Literal["abstention"] = "abstention"
    message: str
    abstention_reason: str
    partial_findings: list[EvidenceSnippet] | None = None
    suggested_followup: str | None = None
    confidence: ConfidenceScoreOutput | None = None
    agent_trace: AgentTraceOutput | None = None
    processing_time_ms: int = Field(ge=0)


class ErrorResponse(BaseModel):
    response_type: Literal["error"] = "error"
    message: str
    error_code: str | None = None


MedicalResponse = Annotated[
    Union[VerifiedAnswerResponse, AbstentionResponse, ErrorResponse],
    Field(discriminator="response_type"),
]