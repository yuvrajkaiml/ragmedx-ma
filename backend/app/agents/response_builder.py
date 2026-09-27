from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.agents.state import AgentState, create_deterministic_uuid
from app.schemas.response import (
    AbstentionResponse,
    AgentTraceOutput,
    AgentTraceStep as MasterTraceStep,
    CitationOutput,
    ConfidenceScoreOutput,
    ErrorResponse,
    EvidenceSnippet as MasterEvidenceSnippet,
    MedicalResponse,
    VerifiedAnswerResponse,
)

# Standard Exact Error and Abstention Messages
MSG_INVALID_QUERY = "Please enter a valid medical question."
MSG_RETRIEVAL_FAILURE = "No relevant medical literature found."
MSG_MODEL_FAILURE = "Unable to generate a response. Please try again."
MSG_CONTRADICTORY_EVIDENCE = "Sources contain conflicting information."
MSG_OUT_OF_DOMAIN = "Outside the scope of medical literature."
MSG_DEFAULT_ABSTENTION = "Unable to verify a medically safe answer based on the retrieved evidence."
MSG_DEFAULT_FOLLOWUP = "Please consult a qualified healthcare provider or provide more detailed clinical context."


def parse_datetime(val: Any) -> datetime:
    """Safely parse datetime from ISO string or datetime object."""
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val.replace("Z", "+00:00"))
        except Exception:
            pass
    return datetime.now(timezone.utc)


def build_trace_output(trace_steps_data: List[Dict[str, Any]], total_duration_ms: int) -> AgentTraceOutput:
    """Translate raw internal trace step dicts into Pranav's Master AgentTraceOutput schema."""
    master_steps: List[MasterTraceStep] = []

    for step in trace_steps_data:
        status_val = step.get("status", "completed")
        if status_val not in ("running", "completed", "failed"):
            status_val = "completed"

        started_at_dt = parse_datetime(step.get("started_at"))
        completed_at_dt = parse_datetime(step.get("completed_at")) if step.get("completed_at") else None

        master_steps.append(
            MasterTraceStep(
                agent_type=str(step.get("agent_type", "unknown")),
                status=status_val,
                started_at=started_at_dt,
                completed_at=completed_at_dt,
                duration_ms=step.get("duration_ms", 0),
                details=step.get("details"),
            )
        )

    return AgentTraceOutput(
        steps=master_steps,
        total_duration_ms=max(0, total_duration_ms),
    )


def build_confidence_output(calibration_dict: Optional[Dict[str, Any]]) -> ConfidenceScoreOutput:
    """Translate internal confidence calibration dict to Pranav's ConfidenceScoreOutput schema."""
    if not calibration_dict:
        return ConfidenceScoreOutput(
            score=0.0,
            level="low",
            retrieval_score=0.0,
            evidence_quality_score=0.0,
            answer_consistency_score=0.0,
            inter_source_agreement=0.0,
        )

    score = float(calibration_dict.get("calibrated_confidence", 0.0))
    score = max(0.0, min(1.0, score))

    level = "low"
    if score >= 0.80:
        level = "high"
    elif score >= 0.65:
        level = "moderate"

    return ConfidenceScoreOutput(
        score=score,
        level=level,
        retrieval_score=calibration_dict.get("retrieval_confidence"),
        evidence_quality_score=calibration_dict.get("evidence_quality_score"),
        answer_consistency_score=calibration_dict.get("answer_consistency_score"),
        inter_source_agreement=calibration_dict.get("inter_source_agreement"),
    )


def build_citations_output(
    evidence_list: List[Dict[str, Any]],
    used_citation_markers: Optional[List[str]] = None,
) -> List[CitationOutput]:
    """Translate evidence snippets to CitationOutput schemas matching cited markers."""
    citations: List[CitationOutput] = []
    used_set = set(used_citation_markers or [])

    for ev in evidence_list:
        marker = ev.get("citation_marker", "")
        # If specific markers were cited in answer, filter or include all if no markers specified
        if used_set and marker not in used_set and marker.strip("[]") not in used_set:
            continue

        doc_id_val = ev.get("document_id")
        if isinstance(doc_id_val, str):
            try:
                doc_id = UUID(doc_id_val)
            except ValueError:
                doc_id = create_deterministic_uuid(doc_id_val)
        elif isinstance(doc_id_val, UUID):
            doc_id = doc_id_val
        else:
            doc_id = create_deterministic_uuid(str(ev.get("title", "doc")))

        pub_year = ev.get("publication_year")
        if pub_year is not None:
            try:
                pub_year = int(pub_year)
                if pub_year < 1900:
                    pub_year = None
            except (ValueError, TypeError):
                pub_year = None

        relevance = float(ev.get("relevance_score", 0.0))
        relevance = max(0.0, min(1.0, relevance))

        citations.append(
            CitationOutput(
                document_id=doc_id,
                citation_marker=marker,
                title=str(ev.get("title", "Medical Evidence Source")),
                journal=ev.get("journal"),
                publication_year=pub_year,
                doi=ev.get("doi"),
                url=ev.get("url"),
                cited_text=str(ev.get("cited_text", "")),
                relevance_score=relevance,
            )
        )

    return citations


def build_evidence_snippets_output(evidence_list: List[Dict[str, Any]]) -> List[MasterEvidenceSnippet]:
    """Translate internal evidence items to Master EvidenceSnippet models for partial findings."""
    snippets: List[MasterEvidenceSnippet] = []
    for ev in evidence_list:
        chunk_id_val = ev.get("chunk_id")
        doc_id_val = ev.get("document_id")

        def _to_uuid(val: Any, seed: str) -> UUID:
            if isinstance(val, UUID):
                return val
            if isinstance(val, str):
                try:
                    return UUID(val)
                except ValueError:
                    return create_deterministic_uuid(val)
            return create_deterministic_uuid(seed)

        chunk_id = _to_uuid(chunk_id_val, "chunk")
        doc_id = _to_uuid(doc_id_val, "doc")

        source_type_val = ev.get("source_type", "pubmed")
        if source_type_val not in ("pubmed", "pmc", "medqa", "bioasq", "uploaded_document"):
            source_type_val = "pubmed"

        relevance = float(ev.get("relevance_score", 0.0))
        relevance = max(0.0, min(1.0, relevance))

        snippets.append(
            MasterEvidenceSnippet(
                chunk_id=chunk_id,
                document_id=doc_id,
                source_type=source_type_val,
                title=str(ev.get("title", "Evidence Snippet")),
                cited_text=str(ev.get("cited_text", "")),
                relevance_score=relevance,
            )
        )
    return snippets


def build_medical_response(state: AgentState) -> MedicalResponse:
    """
    Sole boundary function translating internal AgentState into Pranav's Master API response models.

    Returns one of:
    - VerifiedAnswerResponse
    - AbstentionResponse
    - ErrorResponse
    """
    # 1. Error Response: Invalid input or unrecoverable error
    if not state.original_question or not state.original_question.strip():
        return ErrorResponse(
            message=MSG_INVALID_QUERY,
            error_code="INVALID_QUERY",
        )

    if state.error_code in ("INVALID_QUERY", "UNRECOVERABLE_FATAL_ERROR"):
        return ErrorResponse(
            message=state.error or MSG_INVALID_QUERY,
            error_code=state.error_code,
        )

    # Prepare optional trace
    trace_output: Optional[AgentTraceOutput] = None
    if state.include_trace and state.agent_trace:
        trace_output = build_trace_output(state.agent_trace, state.processing_time_ms)

    # 2. Abstention Response: Low confidence, failed retrieval/model, or explicit abstention
    should_abstain = (
        state.status in ("abstained", "critic_rejected", "failed")
        or state.retrieval_failure
        or state.model_failure
        or (state.abstention_decision and state.abstention_decision.get("should_abstain"))
        or not state.answer
    )

    if should_abstain:
        # Determine exact message copy
        if state.out_of_domain:
            message_copy = MSG_OUT_OF_DOMAIN
        elif state.contradictory_evidence:
            message_copy = MSG_CONTRADICTORY_EVIDENCE
        elif state.retrieval_failure:
            message_copy = MSG_RETRIEVAL_FAILURE
        elif state.model_failure:
            message_copy = MSG_MODEL_FAILURE
        else:
            message_copy = MSG_DEFAULT_ABSTENTION

        # Format reason string
        reason_str = "Evidence confidence is below the safety threshold."
        if state.abstention_decision:
            reasons = state.abstention_decision.get("reasons", [])
            if reasons:
                reason_str = " ".join(reasons)
            elif state.abstention_decision.get("reasoning"):
                reason_str = state.abstention_decision.get("reasoning")
        elif state.retrieval_failure:
            reason_str = MSG_RETRIEVAL_FAILURE
        elif state.model_failure:
            reason_str = MSG_MODEL_FAILURE

        partial_findings = build_evidence_snippets_output(state.evidence) if state.evidence else None
        confidence_out = build_confidence_output(state.confidence_calibration) if state.confidence_calibration else None

        return AbstentionResponse(
            message=message_copy,
            abstention_reason=reason_str,
            partial_findings=partial_findings,
            suggested_followup=MSG_DEFAULT_FOLLOWUP,
            confidence=confidence_out,
            agent_trace=trace_output,
            processing_time_ms=max(0, state.processing_time_ms),
        )

    # 3. Verified Answer Response
    confidence_out = build_confidence_output(state.confidence_calibration)
    citations_out = build_citations_output(state.evidence, state.answer_citations)

    return VerifiedAnswerResponse(
        answer=state.answer or "",
        confidence=confidence_out,
        citations=citations_out,
        agent_trace=trace_output,
        processing_time_ms=max(0, state.processing_time_ms),
        total_tokens_used=max(0, state.total_tokens_used),
    )
