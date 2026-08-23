import pytest
from uuid import UUID

from app.agents import run_agent_pipeline
from app.agents.response_builder import (
    MSG_CONTRADICTORY_EVIDENCE,
    MSG_INVALID_QUERY,
    MSG_MODEL_FAILURE,
    MSG_OUT_OF_DOMAIN,
    MSG_RETRIEVAL_FAILURE,
    build_medical_response,
)
from app.agents.state import AgentState, create_deterministic_uuid
from app.schemas.response import (
    AbstentionResponse,
    ErrorResponse,
    MedicalResponse,
    VerifiedAnswerResponse,
)
from app.services.agent_service import AgentService


class SequenceFakeLLM:
    """Deterministic Mock LLM returning responses in a sequence."""
    def __init__(self, responses: list):
        self.responses = list(responses)
        self.call_count = 0

    def generate(self, prompt: str, **kwargs):
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp, {"prompt_tokens": 12, "completion_tokens": 15, "total_tokens": 27, "is_mock": True}
        return "Default answer [E1]", {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20, "is_mock": True}


# ==============================================================================
# Response Builder & Public Interface Contract Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_public_run_agent_pipeline_contract():
    """Verify run_agent_pipeline contract returns a valid MedicalResponse and trace."""
    query_resp = '{"medical_specialty": "Cardiology", "confidence": 0.95}'
    gen_resp = "ACE inhibitors are first-line agents for hypertension [E1]."
    critic_resp = '{"passed": true, "factual_consistency": 0.95, "citation_support": 0.90, "hallucination_risk": 0.05, "confidence": 0.95}'

    llm = SequenceFakeLLM([query_resp, gen_resp, critic_resp])

    response, trace = await run_agent_pipeline(
        original_question="What is the first-line medication for essential hypertension?",
        include_trace=True,
        llm_wrapper=llm,
    )

    assert isinstance(response, VerifiedAnswerResponse)
    assert response.response_type == "verified_answer"
    assert "ACE inhibitors" in response.answer
    assert response.confidence.score >= 0.65
    assert response.confidence.level in ("high", "moderate")
    assert len(response.citations) >= 1
    assert response.citations[0].citation_marker == "[E1]"
    assert trace is not None
    assert len(trace.steps) >= 5


@pytest.mark.asyncio
async def test_response_builder_verified_arm():
    """Verify build_medical_response constructs a VerifiedAnswerResponse when answer passes."""
    q_id = create_deterministic_uuid("query_test")
    doc_id = create_deterministic_uuid("doc_1")

    state = AgentState(
        query_id=q_id,
        original_question="How to treat uncomplicated hypertension?",
        status="verified",
        answer="Thiazide diuretics are effective first-line therapy [E1].",
        answer_citations=["[E1]"],
        evidence=[
            {
                "chunk_id": str(create_deterministic_uuid("chunk_1")),
                "document_id": str(doc_id),
                "source_type": "pubmed",
                "title": "Hypertension Study",
                "journal": "NEJM",
                "publication_year": 2024,
                "cited_text": "Thiazide diuretics are effective.",
                "relevance_score": 0.90,
                "citation_marker": "[E1]",
            }
        ],
        confidence_calibration={
            "calibrated_confidence": 0.88,
            "retrieval_confidence": 0.90,
            "evidence_quality_score": 0.90,
            "answer_consistency_score": 0.90,
            "inter_source_agreement": 0.85,
        },
        processing_time_ms=120,
        total_tokens_used=50,
    )

    response = build_medical_response(state)
    assert isinstance(response, VerifiedAnswerResponse)
    assert response.response_type == "verified_answer"
    assert response.confidence.score == 0.88
    assert response.confidence.level == "high"
    assert len(response.citations) == 1
    assert response.citations[0].document_id == doc_id
    assert response.processing_time_ms == 120
    assert response.total_tokens_used == 50


@pytest.mark.asyncio
async def test_response_builder_abstention_arm_with_exact_copy():
    """Verify build_medical_response constructs AbstentionResponse with exact standard copy."""
    # 1. Retrieval failure copy
    state_retrieval_fail = AgentState(
        query_id=create_deterministic_uuid("q1"),
        original_question="Rare unknown disease treatment?",
        status="abstained",
        retrieval_failure=True,
    )
    resp1 = build_medical_response(state_retrieval_fail)
    assert isinstance(resp1, AbstentionResponse)
    assert resp1.message == MSG_RETRIEVAL_FAILURE

    # 2. Contradictory evidence copy
    state_contradiction = AgentState(
        query_id=create_deterministic_uuid("q2"),
        original_question="Conflicting evidence query",
        status="abstained",
        contradictory_evidence=True,
    )
    resp2 = build_medical_response(state_contradiction)
    assert isinstance(resp2, AbstentionResponse)
    assert resp2.message == MSG_CONTRADICTORY_EVIDENCE

    # 3. Out-of-domain copy
    state_ood = AgentState(
        query_id=create_deterministic_uuid("q3"),
        original_question="How to change car engine oil?",
        status="abstained",
        out_of_domain=True,
    )
    resp3 = build_medical_response(state_ood)
    assert isinstance(resp3, AbstentionResponse)
    assert resp3.message == MSG_OUT_OF_DOMAIN


@pytest.mark.asyncio
async def test_response_builder_error_arm_empty_query():
    """Verify empty query produces ErrorResponse with exact required copy."""
    state_empty = AgentState(
        query_id=create_deterministic_uuid("q_empty"),
        original_question="   ",
    )
    resp = build_medical_response(state_empty)
    assert isinstance(resp, ErrorResponse)
    assert resp.response_type == "error"
    assert resp.message == MSG_INVALID_QUERY
    assert resp.error_code == "INVALID_QUERY"


@pytest.mark.asyncio
async def test_include_trace_flag_behavior():
    """Verify include_trace=False omits trace while include_trace=True includes trace."""
    query_resp = '{"medical_specialty": "Cardiology", "confidence": 0.90}'
    gen_resp = "Answer text [E1]."
    critic_resp = '{"passed": true, "factual_consistency": 0.90, "citation_support": 0.90, "hallucination_risk": 0.05, "confidence": 0.90}'

    # Run without trace
    llm_1 = SequenceFakeLLM([query_resp, gen_resp, critic_resp])
    resp_no_trace, trace_no_trace = await run_agent_pipeline(
        original_question="What is hypertension?",
        include_trace=False,
        llm_wrapper=llm_1,
    )
    assert trace_no_trace is None
    assert resp_no_trace.agent_trace is None

    # Run with trace
    llm_2 = SequenceFakeLLM([query_resp, gen_resp, critic_resp])
    resp_with_trace, trace_with_trace = await run_agent_pipeline(
        original_question="What is hypertension?",
        include_trace=True,
        llm_wrapper=llm_2,
    )
    assert trace_with_trace is not None
    assert resp_with_trace.agent_trace is not None
    assert len(trace_with_trace.steps) > 0


@pytest.mark.asyncio
async def test_agent_service_execution():
    """Verify AgentService boundary executes successfully."""
    query_resp = '{"medical_specialty": "Cardiology", "confidence": 0.90}'
    gen_resp = "Answer text [E1]."
    critic_resp = '{"passed": true, "factual_consistency": 0.90, "citation_support": 0.90, "hallucination_risk": 0.05, "confidence": 0.90}'

    service = AgentService(llm_wrapper=SequenceFakeLLM([query_resp, gen_resp, critic_resp]))
    response, trace = await service.execute_query(
        question="What is hypertension?",
        include_trace=True,
    )

    assert isinstance(response, VerifiedAnswerResponse)
    assert response.response_type == "verified_answer"
