from app.agents.orchestrator import MultiAgentOrchestrator
from app.agents.response_builder import build_medical_response
from app.agents.state import (
    AgentState,
    AgentType,
    ConfidenceCategory,
    CriticVerdictType,
    EvidenceSnippet,
    SourceType,
    create_deterministic_uuid,
)


async def run_agent_pipeline(
    original_question: str,
    query_id: str | None = None,
    processed_query: dict | None = None,
    evidence: list | None = None,
    include_trace: bool = False,
    conversation_id: str | None = None,
    llm_wrapper=None,
    retrieval_provider=None,
    generator_temperature: float | None = None,
    disable_critic: bool = False,
    disable_calibration: bool = False,
):
    """
    Canonical public pipeline entry point.
    Executes the multi-agent state machine and formats output into master response schemas.

    Returns:
        Tuple of (MedicalResponse, Optional[AgentTraceOutput])
    """
    from uuid import UUID

    q_uuid = UUID(query_id) if isinstance(query_id, str) else (query_id or create_deterministic_uuid(original_question or "query"))
    c_uuid = UUID(conversation_id) if isinstance(conversation_id, str) else conversation_id

    initial_state = AgentState(
        query_id=q_uuid,
        conversation_id=c_uuid,
        original_question=original_question,
        processed_query=processed_query,
        evidence=evidence or [],
        include_trace=include_trace,
    )

    orchestrator = MultiAgentOrchestrator(
        llm_wrapper=llm_wrapper,
        retrieval_provider=retrieval_provider,
        generator_temperature=generator_temperature,
        disable_critic=disable_critic,
        disable_calibration=disable_calibration,
    )

    final_state = await orchestrator.run(initial_state)
    response = build_medical_response(final_state)

    trace_output = response.agent_trace if include_trace else None
    return response, trace_output


__all__ = [
    "MultiAgentOrchestrator",
    "build_medical_response",
    "run_agent_pipeline",
    "AgentState",
    "AgentType",
    "SourceType",
    "ConfidenceCategory",
    "CriticVerdictType",
    "EvidenceSnippet",
    "create_deterministic_uuid",
]
