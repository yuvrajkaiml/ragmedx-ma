from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import UUID, uuid5, NAMESPACE_DNS
from pydantic import BaseModel, Field


class AgentType(str, Enum):
    QUERY_UNDERSTANDING = "query_understanding"
    ITERATIVE_RETRIEVAL = "iterative_retrieval"
    EVIDENCE_RERANKING = "evidence_reranking"
    GENERATOR = "generator"
    CRITIC = "critic"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    ABSTENTION_DECISION = "abstention_decision"


class SourceType(str, Enum):
    PUBMED = "pubmed"
    PMC_OPEN_ACCESS = "pmc"
    MEDQA = "medqa"
    BIOASQ = "bioasq"
    UPLOADED_PDF = "uploaded_document"
    STATPEARLS = "statpearls"


class ConfidenceCategory(str, Enum):
    HIGH = "high"          # >= 0.80
    MODERATE = "moderate"  # 0.65–0.79
    LOW = "low"            # < 0.65


class CriticVerdictType(str, Enum):
    VERIFIED = "verified"
    REVISED = "revised"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class EvidenceSnippet(BaseModel):
    """
    Pipeline evidence snippet model.
    IDs must be explicitly provided by the data source/retriever (no automatic random UUIDs).

    Provenance fields (Phase 3, backward-compatible additions):
    - index_id: identifies which FAISS/retrieval index the chunk came from
    - retrieval_iteration: which iterative-retrieval pass surfaced this chunk (1-based)
    """
    chunk_id: UUID
    document_id: UUID
    source_type: str
    title: str
    journal: Optional[str] = None
    publication_year: Optional[int] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    cited_text: str
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    cross_encoder_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_marker: str
    # --- Provenance fields (Phase 3) — optional so existing code with no provenance still works ---
    index_id: Optional[str] = None
    retrieval_iteration: Optional[int] = None


class QueryUnderstandingOutput(BaseModel):
    original_question: str
    medical_specialty: Optional[str] = None
    intent: Optional[str] = None
    expanded_query: Optional[str] = None
    abbreviations: Dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class GapAnalysisOutput(BaseModel):
    sufficient: bool
    reasoning: str
    missing_information: List[str] = Field(default_factory=list)
    sub_query: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class AgentTraceStep(BaseModel):
    agent_type: str
    status: Literal["running", "completed", "failed"]
    started_at: str
    completed_at: Optional[str] = None
    duration_ms: Optional[int] = None
    details: Optional[Dict[str, Any]] = None


class AgentTraceOutput(BaseModel):
    steps: List[AgentTraceStep] = Field(default_factory=list)
    total_duration_ms: int = 0


class CriticVerdict(BaseModel):
    passed: bool
    verdict: Optional[str] = None
    factual_consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_support: float = Field(default=0.0, ge=0.0, le=1.0)
    hallucination_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)
    revision_instructions: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_verification: Dict[str, Any] = Field(default_factory=dict)


class ConfidenceCalibrationOutput(BaseModel):
    calibrated_confidence: float = Field(ge=0.0, le=1.0)
    retrieval_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    answer_consistency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    inter_source_agreement: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    factual_consistency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_support_score: float = Field(default=0.0, ge=0.0, le=1.0)
    hallucination_safety_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_band: str
    requires_abstention: bool = False
    reasoning: str = ""


class AbstentionDecisionOutput(BaseModel):
    decision: str
    should_abstain: bool
    reasons: List[str] = Field(default_factory=list)
    reasoning: str = ""


class AgentState(BaseModel):
    """
    Central Typed State Machine State for the RAGMedX-MA pipeline.
    Maintains all intermediate execution context, counters, and trace steps.
    """
    # Identity
    query_id: UUID
    conversation_id: Optional[UUID] = None
    original_question: str
    include_trace: bool = False
    # Provenance fields (Phase 3, backward-compatible — default None keeps old code working)
    index_id: Optional[str] = None
    pipeline_version: Optional[str] = None

    # Agent Artifacts
    processed_query: Optional[Dict[str, Any]] = None
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_iterations: List[Dict[str, Any]] = Field(default_factory=list)
    gap_analysis: Optional[Dict[str, Any]] = None
    # Graph-only routing state. These fields are internal and are never exposed
    # by response_builder as medical answer content.
    pending_retrieval_sub_query: Optional[str] = None
    evidence_reranking: Optional[Dict[str, Any]] = None

    # Generation & Revision
    answer: Optional[str] = None
    answer_citations: List[str] = Field(default_factory=list)
    critic_verdict: Optional[Dict[str, Any]] = None
    critic_reviews: List[Dict[str, Any]] = Field(default_factory=list)
    revision_instructions: Optional[str] = None

    # Confidence & Abstention
    confidence_calibration: Optional[Dict[str, Any]] = None
    abstention_decision: Optional[Dict[str, Any]] = None

    # Bounded Loop Counters
    retrieval_iteration_count: int = 0
    critic_round_count: int = 0

    # Diagnostic & Pipeline Status Flags
    status: str = "initialized"
    contradictory_evidence: bool = False
    out_of_domain: bool = False
    retrieval_failure: bool = False
    model_failure: bool = False
    force_abstention: bool = False
    error: Optional[str] = None
    error_code: Optional[str] = None

    # Trace & Accounting
    agent_trace: List[Dict[str, Any]] = Field(default_factory=list)
    total_tokens_used: int = 0
    processing_time_ms: int = 0


def create_deterministic_uuid(seed: str) -> UUID:
    """Helper to generate reproducible UUIDs for synthetic test fixtures."""
    return uuid5(NAMESPACE_DNS, seed)
