from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class EvidenceSnippet(BaseModel):
    chunk_id: UUID
    document_id: UUID

    source_type: str

    title: str
    journal: Optional[str] = None
    publication_year: Optional[int] = None

    cited_text: str

    relevance_score: float = 0.0
    cross_encoder_score: float = 0.0
    evidence_quality_score: float = 0.0

    citation_marker: str


class QueryUnderstandingOutput(BaseModel):
    original_question: str

    medical_specialty: Optional[str] = None
    intent: Optional[str] = None
    expanded_query: Optional[str] = None

    abbreviations: Dict[str, str] = Field(
        default_factory=dict
    )

    confidence: float = 0.0


class GapAnalysisOutput(BaseModel):
    sufficient: bool

    reasoning: str

    missing_information: List[str] = Field(
        default_factory=list
    )

    sub_query: Optional[str] = None

    confidence: float = 0.0


class AgentTraceStep(BaseModel):
    agent_type: str
    status: str

    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    duration_ms: Optional[int] = None

    details: Optional[Dict[str, Any]] = None


class AgentTraceOutput(BaseModel):
    steps: List[AgentTraceStep] = Field(
        default_factory=list
    )

    total_duration_ms: int = 0


class CriticVerdict(BaseModel):
    passed: bool

    factual_consistency: float = 0.0
    citation_support: float = 0.0
    hallucination_risk: float = 0.0

    issues: List[str] = Field(
        default_factory=list
    )

    revision_instructions: Optional[str] = None

    confidence: float = 0.0


class ConfidenceCalibrationOutput(BaseModel):
    calibrated_confidence: float

    evidence_score: float = 0.0
    factual_consistency_score: float = 0.0
    citation_support_score: float = 0.0
    hallucination_safety_score: float = 0.0

    confidence_band: str

    requires_abstention: bool = False

    reasoning: str = ""


class AbstentionDecisionOutput(BaseModel):
    decision: str

    should_abstain: bool

    reasons: List[str] = Field(
        default_factory=list
    )

    reasoning: str = ""