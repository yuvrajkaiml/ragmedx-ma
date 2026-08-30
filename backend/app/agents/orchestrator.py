import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import UUID

from app.agents.abstention_decision import AbstentionDecisionAgent
from app.agents.confidence_calibration import (
    CONFIDENCE_THRESHOLD,
    ConfidenceCalibrationAgent,
)
from app.agents.critic import CriticAgent
from app.agents.generator import GeneratorAgent
from app.agents.iterative_retrieval import IterativeRetrievalAgent
from app.agents.query_understanding import QueryUnderstandingAgent
from app.agents.state import (
    AgentState,
    AgentType,
    CriticVerdictType,
    create_deterministic_uuid,
)
from app.llm.local_llm import LocalLLMWrapper
from app.llm.temporary_retrieval_stub import retrieve_evidence as default_retrieval_provider

# Fixed Pipeline Constants
MAX_RETRIEVAL_ITERATIONS = 3
MAX_CRITIC_ROUNDS = 2


class MultiAgentOrchestrator:
    """
    Explicit Typed Async State Machine Orchestrator for RAGMedX-MA.

    Pipeline Flow:
    1. Query Understanding (Agent 1)
    2. Evidence Acquisition & Iterative Refinement (Agent 2 / Retrieval Provider, max 3 iters)
    3. Evidence Validation & Normalization
    4. Generation (Agent 5)
    5. Critic Review (Agent 6)
    6. Bounded Revision Loop (max 2 rounds)
    7. Confidence Calibration (Agent 7)
    8. Abstention Decision (Agent 8)
    """

    def __init__(
        self,
        llm_wrapper: Optional[Any] = None,
        retrieval_provider: Optional[Callable] = None,
        max_retrieval_iterations: int = MAX_RETRIEVAL_ITERATIONS,
        max_critic_rounds: int = MAX_CRITIC_ROUNDS,
    ):
        self.llm_wrapper = llm_wrapper or LocalLLMWrapper(use_mock_fallback=True)
        self.retrieval_provider = retrieval_provider or default_retrieval_provider
        self.max_retrieval_iterations = max(1, min(5, max_retrieval_iterations))
        self.max_critic_rounds = max(1, min(3, max_critic_rounds))

        # Initialize domain agents
        self.query_agent = QueryUnderstandingAgent(llm_wrapper=self.llm_wrapper)
        self.gap_agent = IterativeRetrievalAgent(llm_wrapper=self.llm_wrapper)
        self.generator_agent = GeneratorAgent(llm_wrapper=self.llm_wrapper)
        self.critic_agent = CriticAgent(llm_wrapper=self.llm_wrapper)
        self.confidence_agent = ConfidenceCalibrationAgent()
        self.abstention_agent = AbstentionDecisionAgent()

    async def run(
        self,
        state: Union[AgentState, Dict[str, Any]],
    ) -> AgentState:
        """
        Execute the multi-agent pipeline from initial state to final decision.

        Args:
            state: AgentState instance or dictionary representation.

        Returns:
            Fully populated AgentState instance.
        """
        start_wall_time = time.perf_counter()
        started_at = datetime.now(timezone.utc)

        # Convert dict to AgentState if necessary
        if isinstance(state, dict):
            agent_state = self._init_state_from_dict(state)
        else:
            agent_state = state

        # Ensure state has valid query_id
        if not agent_state.query_id:
            agent_state.query_id = create_deterministic_uuid(agent_state.original_question or "query")

        # Step 1: Query Understanding
        await self._step_query_understanding(agent_state)

        # Step 2: Evidence Acquisition & Retrieval Handoff
        await self._step_evidence_acquisition(agent_state)

        # Step 3: Evidence Validation and Boundary Normalization
        self._normalize_and_validate_evidence(agent_state)

        # Step 4: Generation & Bounded Critic Revision Loop
        await self._step_generation_and_critic_loop(agent_state)

        # Step 5: Confidence Calibration
        await self._step_confidence_calibration(agent_state)

        # Step 6: Abstention Decision
        await self._step_abstention_decision(agent_state)

        # Finalize timing and state status
        end_wall_time = time.perf_counter()
        agent_state.processing_time_ms = int((end_wall_time - start_wall_time) * 1000)

        return agent_state

    async def _step_query_understanding(self, state: AgentState) -> None:
        """Execute Agent 1 or reuse pre-supplied processed query."""
        if state.processed_query:
            # Query understanding was already supplied upstream
            step = {
                "agent_type": AgentType.QUERY_UNDERSTANDING.value,
                "status": "completed",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": 0,
                "details": {"action": "reused_supplied_processed_query"},
            }
            state.agent_trace.append(step)
            return

        try:
            result = await self.query_agent.run({
                "original_question": state.original_question,
            })
            state.processed_query = result.get("processed_query")
            for trace_step in result.get("agent_trace", []):
                state.agent_trace.append(trace_step)
        except Exception as e:
            # Query understanding failed
            state.error = str(e)
            state.error_code = "QUERY_UNDERSTANDING_ERROR"
            step = {
                "agent_type": AgentType.QUERY_UNDERSTANDING.value,
                "status": "failed",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": 0,
                "details": {"error": str(e)},
            }
            state.agent_trace.append(step)

    async def _step_evidence_acquisition(self, state: AgentState) -> None:
        """Acquire evidence from supplied items or via injected retrieval provider."""
        # If evidence was already supplied explicitly, skip retrieval
        if state.evidence:
            step = {
                "agent_type": AgentType.ITERATIVE_RETRIEVAL.value,
                "status": "completed",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": 0,
                "details": {
                    "action": "reused_supplied_evidence",
                    "evidence_count": len(state.evidence),
                },
            }
            state.agent_trace.append(step)
            return

        started_at = datetime.now(timezone.utc)
        try:
            state.retrieval_iteration_count += 1
            query_str = state.original_question
            optimized_str = None
            if state.processed_query:
                optimized_str = state.processed_query.get("expanded_query")

            retrieved_evidence, iteration_meta = await self.retrieval_provider(
                query=query_str,
                optimized_query=optimized_str,
                max_iterations=self.max_retrieval_iterations,
            )

            state.evidence = retrieved_evidence or []
            state.retrieval_iterations.extend(iteration_meta or [])

            if not state.evidence:
                state.retrieval_failure = True

            completed_at = datetime.now(timezone.utc)
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            step = {
                "agent_type": AgentType.ITERATIVE_RETRIEVAL.value,
                "status": "completed" if state.evidence else "failed",
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "duration_ms": duration_ms,
                "details": {
                    "iteration": state.retrieval_iteration_count,
                    "retrieved_count": len(state.evidence),
                    "retrieval_failure": state.retrieval_failure,
                },
            }
            state.agent_trace.append(step)

        except Exception as e:
            state.retrieval_failure = True
            state.error = str(e)
            state.error_code = "RETRIEVAL_ERROR"
            completed_at = datetime.now(timezone.utc)
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)
            step = {
                "agent_type": AgentType.ITERATIVE_RETRIEVAL.value,
                "status": "failed",
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "duration_ms": duration_ms,
                "details": {"error": str(e)},
            }
            state.agent_trace.append(step)

    def _normalize_and_validate_evidence(self, state: AgentState) -> None:
        """Deduplicate and normalize citation markers on evidence items."""
        if not state.evidence:
            return

        seen_chunk_ids = set()
        normalized_evidence = []

        for index, item in enumerate(state.evidence, start=1):
            if isinstance(item, dict):
                chunk_id = item.get("chunk_id")
                if not chunk_id:
                    chunk_id = str(create_deterministic_uuid(f"chunk_{index}_{state.original_question[:15]}"))
                    item["chunk_id"] = chunk_id

                if chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)

                if "citation_marker" not in item or not item["citation_marker"]:
                    item["citation_marker"] = f"[E{len(normalized_evidence) + 1}]"
                elif not item["citation_marker"].startswith("["):
                    item["citation_marker"] = f"[{item['citation_marker']}]"

                normalized_evidence.append(item)
            else:
                normalized_evidence.append(item)

        state.evidence = normalized_evidence

    async def _step_generation_and_critic_loop(self, state: AgentState) -> None:
        """Execute Generator (Agent 5) and Critic (Agent 6) within bounded revision loop."""
        # If retrieval failed or evidence is empty, skip generation
        if state.retrieval_failure or not state.evidence:
            return

        state.critic_round_count = 0
        revision_instructions: Optional[str] = None

        while state.critic_round_count < self.max_critic_rounds:
            state.critic_round_count += 1
            round_num = state.critic_round_count

            # 1. Generator Execution
            gen_state = {
                "original_question": state.original_question,
                "evidence": state.evidence,
                "revision_instructions": revision_instructions,
                "review_round": round_num,
            }

            try:
                gen_result = await self.generator_agent.run(gen_state)
                state.answer = gen_result.get("answer")
                state.answer_citations = gen_result.get("answer_citations", [])
                for trace_step in gen_result.get("agent_trace", []):
                    trace_step["round"] = round_num
                    state.agent_trace.append(trace_step)
            except Exception as e:
                state.model_failure = True
                state.error = str(e)
                state.error_code = "GENERATOR_ERROR"
                step = {
                    "agent_type": AgentType.GENERATOR.value,
                    "status": "failed",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": 0,
                    "details": {"error": str(e), "round": round_num},
                }
                state.agent_trace.append(step)
                break

            # 2. Critic Review Execution
            critic_state = {
                "original_question": state.original_question,
                "answer": state.answer,
                "evidence": state.evidence,
                "review_round": round_num,
            }

            try:
                critic_result = await self.critic_agent.run(critic_state)
                verdict_data = critic_result.get("critic_verdict", {})
                state.critic_verdict = verdict_data
                state.critic_reviews.append(verdict_data)

                for trace_step in critic_result.get("agent_trace", []):
                    trace_step["round"] = round_num
                    state.agent_trace.append(trace_step)

                # Evaluate Critic Routing
                passed = verdict_data.get("passed", False)
                verdict_str = verdict_data.get("verdict")

                if passed or verdict_str in (CriticVerdictType.VERIFIED.value, CriticVerdictType.REVISED.value):
                    # Answer is verified or successfully revised! Exit loop
                    break

                if verdict_str == CriticVerdictType.REJECTED.value:
                    # Explicit rejection: do not loop further
                    state.status = "critic_rejected"
                    break

                # Otherwise needs revision
                revision_instructions = verdict_data.get("revision_instructions")
                if state.critic_round_count >= self.max_critic_rounds:
                    # Cap reached
                    step = {
                        "agent_type": AgentType.CRITIC.value,
                        "status": "completed",
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "duration_ms": 0,
                        "details": {"action": "critic_revision_cap_reached", "max_rounds": self.max_critic_rounds},
                    }
                    state.agent_trace.append(step)
                    break

            except Exception as e:
                state.model_failure = True
                state.error = str(e)
                state.error_code = "CRITIC_ERROR"
                step = {
                    "agent_type": AgentType.CRITIC.value,
                    "status": "failed",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": 0,
                    "details": {"error": str(e), "round": round_num},
                }
                state.agent_trace.append(step)
                break

    async def _step_confidence_calibration(self, state: AgentState) -> None:
        """Execute Confidence Calibration (Agent 7)."""
        # Ensure a default critic verdict exists if generation/critic was skipped
        critic_data = state.critic_verdict or {
            "passed": False,
            "factual_consistency": 0.0,
            "citation_support": 0.0,
            "hallucination_risk": 1.0,
            "confidence": 0.0,
        }

        calib_state = {
            "evidence": state.evidence,
            "critic_verdict": critic_data,
        }

        try:
            calib_result = await self.confidence_agent.run(calib_state)
            state.confidence_calibration = calib_result.get("confidence_calibration")
            for trace_step in calib_result.get("agent_trace", []):
                state.agent_trace.append(trace_step)
        except Exception as e:
            state.error = str(e)
            state.error_code = "CONFIDENCE_CALIBRATION_ERROR"
            step = {
                "agent_type": AgentType.CONFIDENCE_CALIBRATION.value,
                "status": "failed",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": 0,
                "details": {"error": str(e)},
            }
            state.agent_trace.append(step)

    async def _step_abstention_decision(self, state: AgentState) -> None:
        """Execute Abstention Decision (Agent 8)."""
        calibration_data = state.confidence_calibration or {
            "calibrated_confidence": 0.0,
            "requires_abstention": True,
            "confidence_band": "low",
        }

        abstention_input_state = {
            "confidence_calibration": calibration_data,
            "contradictory_evidence": state.contradictory_evidence,
            "out_of_domain": state.out_of_domain,
            "retrieval_failure": state.retrieval_failure,
            "model_failure": state.model_failure,
        }

        try:
            abstention_result = await self.abstention_agent.run(abstention_input_state)
            state.abstention_decision = abstention_result.get("abstention_decision")
            for trace_step in abstention_result.get("agent_trace", []):
                state.agent_trace.append(trace_step)

            if state.abstention_decision and state.abstention_decision.get("should_abstain"):
                state.status = "abstained"
            elif state.status != "critic_rejected":
                state.status = "verified"

        except Exception as e:
            state.status = "abstained"
            state.error = str(e)
            state.error_code = "ABSTENTION_DECISION_ERROR"
            step = {
                "agent_type": AgentType.ABSTENTION_DECISION.value,
                "status": "failed",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": 0,
                "details": {"error": str(e)},
            }
            state.agent_trace.append(step)

    @staticmethod
    def _init_state_from_dict(d: Dict[str, Any]) -> AgentState:
        q_id = d.get("query_id") or create_deterministic_uuid(d.get("original_question", "query"))
        if isinstance(q_id, str):
            q_id = UUID(q_id)

        conv_id = d.get("conversation_id")
        if isinstance(conv_id, str):
            conv_id = UUID(conv_id)

        return AgentState(
            query_id=q_id,
            conversation_id=conv_id,
            original_question=str(d.get("original_question") or d.get("question") or ""),
            include_trace=bool(d.get("include_trace", False)),
            processed_query=d.get("processed_query"),
            evidence=d.get("evidence", []),
            retrieval_iterations=d.get("retrieval_iterations", []),
            status=d.get("status", "initialized"),
            total_tokens_used=d.get("total_tokens_used", 0),
        )
