import pytest
from uuid import UUID

from app.agents.orchestrator import (
    MAX_CRITIC_ROUNDS,
    MAX_RETRIEVAL_ITERATIONS,
    MultiAgentOrchestrator,
)
from app.agents.state import AgentState, AgentType, create_deterministic_uuid


class SequenceFakeLLM:
    """Mock LLM returning responses in a predefined sequence."""
    def __init__(self, responses: list):
        self.responses = list(responses)
        self.call_count = 0

    def generate(self, prompt: str, **kwargs):
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp, {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20, "is_mock": True}
        return "Default response [E1]", {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20, "is_mock": True}


# ==============================================================================
# Orchestrator Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_orchestrator_verified_answer_path():
    """Verify end-to-end verified path when evidence is strong and critic approves."""
    query_resp = (
        '{"medical_specialty": "Cardiology", "intent": "treatment", '
        '"expanded_query": "essential hypertension treatment guidelines", "abbreviations": {}, "confidence": 0.95}'
    )
    gen_resp = "First-line treatments for essential hypertension include ACE inhibitors and thiazide diuretics [E1]."
    critic_resp = (
        '{"passed": true, "factual_consistency": 0.95, "citation_support": 0.92, '
        '"hallucination_risk": 0.05, "issues": [], "revision_instructions": null, "confidence": 0.95}'
    )

    llm = SequenceFakeLLM([query_resp, gen_resp, critic_resp])
    orchestrator = MultiAgentOrchestrator(llm_wrapper=llm)

    state = AgentState(
        query_id=create_deterministic_uuid("test_query_1"),
        original_question="What is the first-line treatment for hypertension?",
        include_trace=True,
    )

    result_state = await orchestrator.run(state)

    assert result_state.status == "verified"
    assert result_state.answer is not None
    assert "[E1]" in result_state.answer
    assert result_state.confidence_calibration is not None
    assert result_state.confidence_calibration["calibrated_confidence"] >= 0.65
    assert result_state.abstention_decision["should_abstain"] is False
    assert len(result_state.agent_trace) >= 5


@pytest.mark.asyncio
async def test_orchestrator_revision_loop_stops_after_success():
    """Verify revision loop executes a second round when critic requests revision, then succeeds."""
    query_resp = '{"medical_specialty": "Cardiology", "intent": "treatment", "expanded_query": "hypertension", "confidence": 0.90}'
    # Round 1: draft with issue
    gen_resp_1 = "Take medication X [E1]."
    critic_resp_1 = (
        '{"passed": false, "factual_consistency": 0.60, "citation_support": 0.50, '
        '"hallucination_risk": 0.40, "issues": ["Clarify drug class"], '
        '"revision_instructions": "Specify ACE inhibitors or thiazide diuretics.", "confidence": 0.85}'
    )
    # Round 2: revised draft passes
    gen_resp_2 = "First-line agents include ACE inhibitors [E1]."
    critic_resp_2 = (
        '{"passed": true, "factual_consistency": 0.95, "citation_support": 0.92, '
        '"hallucination_risk": 0.05, "issues": [], "revision_instructions": null, "confidence": 0.95}'
    )

    llm = SequenceFakeLLM([query_resp, gen_resp_1, critic_resp_1, gen_resp_2, critic_resp_2])
    orchestrator = MultiAgentOrchestrator(llm_wrapper=llm)

    state = AgentState(
        query_id=create_deterministic_uuid("test_revision_query"),
        original_question="What drugs treat hypertension?",
        include_trace=True,
    )

    result_state = await orchestrator.run(state)

    assert result_state.critic_round_count == 2
    assert result_state.status == "verified"
    assert len(result_state.critic_reviews) == 2
    assert result_state.critic_reviews[0]["passed"] is False
    assert result_state.critic_reviews[1]["passed"] is True


@pytest.mark.asyncio
async def test_orchestrator_revision_loop_caps_at_two_rounds():
    """Verify revision loop terminates strictly at MAX_CRITIC_ROUNDS (2) when critic never passes."""
    query_resp = '{"medical_specialty": "Endocrinology", "confidence": 0.90}'
    gen_resp = "Unsupported draft [E1]."
    critic_fail_resp = (
        '{"passed": false, "factual_consistency": 0.40, "citation_support": 0.30, '
        '"hallucination_risk": 0.70, "issues": ["Unsupported claims"], '
        '"revision_instructions": "Rewrite with evidence support.", "confidence": 0.90}'
    )

    llm = SequenceFakeLLM([query_resp, gen_resp, critic_fail_resp, gen_resp, critic_fail_resp])
    orchestrator = MultiAgentOrchestrator(llm_wrapper=llm, max_critic_rounds=MAX_CRITIC_ROUNDS)

    state = AgentState(
        query_id=create_deterministic_uuid("test_cap_query"),
        original_question="What causes diabetes?",
        include_trace=True,
    )

    result_state = await orchestrator.run(state)

    assert result_state.critic_round_count == 2
    # Because confidence is low and hallucination risk is high, it should abstain
    assert result_state.status == "abstained"
    assert result_state.abstention_decision["should_abstain"] is True


@pytest.mark.asyncio
async def test_orchestrator_abstains_on_empty_retrieval():
    """Verify orchestrator triggers abstention when no evidence is retrieved."""
    query_resp = '{"medical_specialty": "General", "confidence": 0.90}'

    async def empty_retriever(**kwargs):
        return [], [{"status": "empty", "count": 0}]

    llm = SequenceFakeLLM([query_resp])
    orchestrator = MultiAgentOrchestrator(llm_wrapper=llm, retrieval_provider=empty_retriever)

    state = AgentState(
        query_id=create_deterministic_uuid("empty_retrieval_query"),
        original_question="Rare unknown condition?",
        include_trace=True,
    )

    result_state = await orchestrator.run(state)

    assert result_state.retrieval_failure is True
    assert result_state.status == "abstained"
    assert result_state.abstention_decision["should_abstain"] is True


@pytest.mark.asyncio
async def test_orchestrator_trace_steps_ordered_and_timed():
    """Verify trace steps have ordered sequence, valid timestamps, and durations."""
    query_resp = '{"medical_specialty": "General", "confidence": 0.90}'
    gen_resp = "Answer grounded in evidence [E1]."
    critic_resp = '{"passed": true, "factual_consistency": 0.95, "citation_support": 0.90, "hallucination_risk": 0.05, "confidence": 0.90}'

    llm = SequenceFakeLLM([query_resp, gen_resp, critic_resp])
    orchestrator = MultiAgentOrchestrator(llm_wrapper=llm)

    state = AgentState(
        query_id=create_deterministic_uuid("trace_test_query"),
        original_question="What is hypertension?",
        include_trace=True,
    )

    result_state = await orchestrator.run(state)

    trace = result_state.agent_trace
    assert len(trace) >= 5

    agent_types_visited = [step["agent_type"] for step in trace]
    assert AgentType.QUERY_UNDERSTANDING.value in agent_types_visited
    assert AgentType.ITERATIVE_RETRIEVAL.value in agent_types_visited
    assert AgentType.GENERATOR.value in agent_types_visited
    assert AgentType.CRITIC.value in agent_types_visited
    assert AgentType.CONFIDENCE_CALIBRATION.value in agent_types_visited
    assert AgentType.ABSTENTION_DECISION.value in agent_types_visited

    for step in trace:
        assert step["status"] in ("completed", "failed")
        assert "started_at" in step
        assert "duration_ms" in step
