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
from app.agents.evidence_reranking import EvidenceRerankingAgent
from app.agents.generator import GeneratorAgent
from app.agents.iterative_retrieval import IterativeRetrievalAgent
from app.agents.query_understanding import QueryUnderstandingAgent
from app.agents.state import (
    AgentState,
    AgentType,
    CriticVerdictType,
    create_deterministic_uuid,
)
from app.config import get_settings
from app.llm.local_llm import LLMRuntimeMode, LocalLLMWrapper
from app.services.retrieval_service import RetrievalService
from app.utils.logger import logger
from langgraph.graph import END, START, StateGraph

# Fixed Pipeline Constants
MAX_RETRIEVAL_ITERATIONS = 3
MAX_CRITIC_ROUNDS = 2


class MultiAgentOrchestrator:
    """
    LangGraph-backed typed async state machine for RAGMedX-MA.

    Pipeline Flow:
    1. Query Understanding (Agent 1)
    2. Evidence Acquisition & Iterative Refinement (Agent 2 / Retrieval Provider, max 3 iters)
    3. Evidence Re-Ranking & Validation (Agent 3)
    4. Generation (Agent 4)
    5. Critic Review (Agent 5)
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
        generator_temperature: Optional[float] = None,
        disable_critic: bool = False,
        disable_calibration: bool = False,
    ):
        if llm_wrapper is not None:
            self.llm_wrapper = llm_wrapper
        else:
            settings = get_settings()
            if settings.llm_mode == "real":
                # LOCAL mode: raises ModelLoadError on any load failure — no silent fallback.
                self.llm_wrapper = LocalLLMWrapper(
                    model_name_or_path=settings.llm_model_name,
                    mode=LLMRuntimeMode.LOCAL,
                    allow_fallback=False,
                )
            else:
                # mock mode (default): deterministic stub responses, safe for dev/CI.
                self.llm_wrapper = LocalLLMWrapper(use_mock_fallback=True)
        # The native retrieval service is the production default.  The temporary
        # retrieval stub remains available for tests only when callers inject it.
        self.retrieval_service = None
        if retrieval_provider is None:
            # Production path should inject retrieval_provider from app.state
            # via the query endpoint. If we reach here without one, either a
            # test is using the default wiring, or the endpoint DI is broken.
            logger.warning(
                "MultiAgentOrchestrator: no retrieval_provider injected — "
                "building a new RetrievalService. This is expected in tests; "
                "in production it means the app.state singleton is not being "
                "threaded through the query endpoint correctly."
            )
            self.retrieval_service = RetrievalService.build()
            self.retrieval_provider = self.retrieval_service.retrieve_evidence
        else:
            self.retrieval_provider = retrieval_provider
        self.max_retrieval_iterations = max(1, min(MAX_RETRIEVAL_ITERATIONS, max_retrieval_iterations))
        self.max_critic_rounds = max(1, min(MAX_CRITIC_ROUNDS, max_critic_rounds))
        self.disable_critic = disable_critic
        self.disable_calibration = disable_calibration

        # Initialize domain agents
        self.query_agent = QueryUnderstandingAgent(llm_wrapper=self.llm_wrapper)
        self.gap_agent = IterativeRetrievalAgent(llm_wrapper=self.llm_wrapper)
        self.reranking_agent = EvidenceRerankingAgent(llm_wrapper=self.llm_wrapper)
        # generator_temperature=None means GeneratorAgent uses its own DEFAULT_TEMPERATURE (0.1).
        # Pass an explicit value (e.g. 0.0) to override for deterministic benchmark runs.
        self.generator_agent = GeneratorAgent(
            llm_wrapper=self.llm_wrapper,
            temperature=generator_temperature,
        )
        self.critic_agent = CriticAgent(llm_wrapper=self.llm_wrapper)
        self.confidence_agent = ConfidenceCalibrationAgent()
        self.abstention_agent = AbstentionDecisionAgent()
        self.graph = self._build_graph()

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
        agent_state = self._init_state_from_dict(state) if isinstance(state, dict) else state
        if not agent_state.query_id:
            agent_state.query_id = create_deterministic_uuid(agent_state.original_question or "query")

        graph_result = await self.graph.ainvoke(agent_state.model_dump())
        final_state = AgentState.model_validate(graph_result)
        final_state.processing_time_ms = int((time.perf_counter() - start_wall_time) * 1000)
        return final_state

    def _build_graph(self):
        """Construct the seven-agent LangGraph workflow once per orchestrator."""
        workflow = StateGraph(AgentState)
        workflow.add_node("query_understanding", self._node_query_understanding)
        workflow.add_node("retrieval", self._node_retrieval)
        workflow.add_node("gap_analysis", self._node_gap_analysis)
        workflow.add_node("evidence_reranking", self._node_evidence_reranking)
        workflow.add_node("generator", self._node_generator)
        workflow.add_node("critic", self._node_critic)
        workflow.add_node("confidence", self._node_confidence)
        workflow.add_node("abstention", self._node_abstention)

        workflow.add_edge(START, "query_understanding")
        workflow.add_edge("query_understanding", "retrieval")
        workflow.add_edge("retrieval", "gap_analysis")
        workflow.add_conditional_edges(
            "gap_analysis",
            self._route_after_gap_analysis,
            {"retrieval": "retrieval", "rerank": "evidence_reranking", "confidence": "confidence"},
        )
        workflow.add_conditional_edges(
            "evidence_reranking",
            self._route_after_evidence_reranking,
            {"generator": "generator", "confidence": "confidence"},
        )
        workflow.add_edge("generator", "critic")
        workflow.add_conditional_edges(
            "critic",
            self._route_after_critic,
            {"generator": "generator", "confidence": "confidence"},
        )
        # Confidence never terminates the graph directly.  Both verified and
        # failed paths must be adjudicated by the Abstention Agent.
        workflow.add_edge("confidence", "abstention")
        workflow.add_edge("abstention", END)
        return workflow.compile()

    @staticmethod
    def _as_agent_state(payload: Union[AgentState, Dict[str, Any]]) -> AgentState:
        return payload if isinstance(payload, AgentState) else AgentState.model_validate(payload)

    @staticmethod
    def _state_update(state: AgentState) -> Dict[str, Any]:
        return state.model_dump()

    @staticmethod
    def _record_graph_node(state: AgentState, node: str) -> None:
        now = datetime.now(timezone.utc)
        state.agent_trace.append({
            "agent_type": "langgraph",
            "status": "completed",
            "started_at": now.isoformat(),
            "completed_at": now.isoformat(),
            "duration_ms": 0,
            "details": {"node": node},
        })

    async def _node_query_understanding(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = self._as_agent_state(payload)
        await self._step_query_understanding(state)
        self._record_graph_node(state, "query_understanding")
        return self._state_update(state)

    async def _node_retrieval(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = self._as_agent_state(payload)
        if state.evidence and state.retrieval_iteration_count == 0:
            state.retrieval_iteration_count = 1
            now = datetime.now(timezone.utc)
            state.agent_trace.append({
                "agent_type": AgentType.ITERATIVE_RETRIEVAL.value,
                "status": "completed",
                "started_at": now.isoformat(),
                "completed_at": now.isoformat(),
                "duration_ms": 0,
                "details": {"action": "reused_supplied_evidence", "iteration": 1, "evidence_count": len(state.evidence)},
            })
        else:
            await self._retrieve_evidence_pass(state, state.pending_retrieval_sub_query)
        state.pending_retrieval_sub_query = None
        self._normalize_and_validate_evidence(state)
        self._record_graph_node(state, "retrieval")
        return self._state_update(state)

    async def _node_gap_analysis(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = self._as_agent_state(payload)
        gap = None if state.retrieval_failure else await self._run_gap_analysis(state)
        if gap is None:
            state.retrieval_failure = True
        elif gap.get("sufficient", False) and state.evidence:
            pass
        elif state.retrieval_iteration_count >= self.max_retrieval_iterations:
            state.retrieval_failure = True
            state.error_code = "RETRIEVAL_ITERATION_CAP"
        elif gap.get("sub_query"):
            state.pending_retrieval_sub_query = gap["sub_query"]
        else:
            state.retrieval_failure = True
            state.error_code = "GAP_ANALYSIS_ERROR"
        self._record_graph_node(state, "gap_analysis")
        return self._state_update(state)

    @staticmethod
    def _route_after_gap_analysis(payload: Dict[str, Any]) -> str:
        state = MultiAgentOrchestrator._as_agent_state(payload)
        if state.retrieval_failure or not state.gap_analysis:
            return "confidence"
        if state.gap_analysis.get("sufficient", False) and state.evidence:
            return "rerank"
        return "retrieval"

    async def _node_evidence_reranking(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = self._as_agent_state(payload)
        try:
            result = await self.reranking_agent.run(state.model_dump())
            state.evidence = result["evidence"]
            state.contradictory_evidence = result["contradictory_evidence"]
            state.evidence_reranking = result["evidence_reranking"]
            state.agent_trace.extend(result.get("agent_trace", []))
        except Exception as exc:
            state.retrieval_failure = True
            state.error = str(exc)
            state.error_code = "EVIDENCE_RERANKING_ERROR"
        self._record_graph_node(state, "evidence_reranking")
        return self._state_update(state)

    @staticmethod
    def _route_after_evidence_reranking(payload: Dict[str, Any]) -> str:
        state = MultiAgentOrchestrator._as_agent_state(payload)
        return "confidence" if state.retrieval_failure or state.contradictory_evidence else "generator"

    async def _node_generator(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = self._as_agent_state(payload)
        state.critic_round_count += 1
        try:
            result = await self.generator_agent.run({
                "original_question": state.original_question,
                "evidence": state.evidence,
                "revision_instructions": state.revision_instructions,
                "review_round": state.critic_round_count,
            })
            state.answer = result.get("answer")
            state.answer_citations = result.get("answer_citations", [])
            for trace in result.get("agent_trace", []):
                trace["round"] = state.critic_round_count
                state.agent_trace.append(trace)
        except Exception as exc:
            state.model_failure = True
            state.error, state.error_code = str(exc), "GENERATOR_ERROR"
        self._record_graph_node(state, "generator")
        return self._state_update(state)

    async def _node_critic(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = self._as_agent_state(payload)
        if not self.disable_critic:
            if state.model_failure:
                self._record_graph_node(state, "critic")
                return self._state_update(state)
            try:
                result = await self.critic_agent.run({
                    "original_question": state.original_question,
                    "answer": state.answer,
                    "evidence": state.evidence,
                    "review_round": state.critic_round_count,
                })
                state.critic_verdict = result.get("critic_verdict", {})
                state.critic_reviews.append(state.critic_verdict)
                state.revision_instructions = state.critic_verdict.get("revision_instructions")
                if (
                    state.critic_verdict.get("verdict") == CriticVerdictType.REJECTED.value
                    or (
                        not state.critic_verdict.get("passed", False)
                        and state.critic_round_count >= self.max_critic_rounds
                    )
                ):
                    state.force_abstention = True
                for trace in result.get("agent_trace", []):
                    trace["round"] = state.critic_round_count
                    state.agent_trace.append(trace)
            except Exception as exc:
                state.model_failure = True
                state.error, state.error_code = str(exc), "CRITIC_ERROR"
        self._record_graph_node(state, "critic")
        return self._state_update(state)

    def _route_after_critic(self, payload: Dict[str, Any]) -> str:
        state = self._as_agent_state(payload)
        verdict = state.critic_verdict or {}
        if state.model_failure or verdict.get("verdict") == CriticVerdictType.REJECTED.value:
            return "confidence"
        if verdict.get("passed", False) or verdict.get("verdict") in (CriticVerdictType.VERIFIED.value, CriticVerdictType.REVISED.value):
            return "confidence"
        if state.critic_round_count < self.max_critic_rounds and state.revision_instructions:
            return "generator"
        return "confidence"

    async def _node_confidence(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = self._as_agent_state(payload)
        if self.disable_calibration:
            # Check retrieval/answer guard first
            if state.retrieval_failure or not state.answer:
                state.confidence_calibration = {
                    "requires_abstention": True,
                    "calibrated_confidence": 0.0,
                    "confidence_band": "low",
                }
            else:
                state.confidence_calibration = {
                    "requires_abstention": False,
                    "calibrated_confidence": 1.0,
                    "confidence_band": "high",
                    "factual_consistency": 1.0,
                    "citation_support": 1.0,
                    "hallucination_risk": 0.0,
                    "confidence": 1.0,
                }
        else:
            await self._step_confidence_calibration(state)
        self._record_graph_node(state, "confidence")
        return self._state_update(state)

    async def _node_abstention(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = self._as_agent_state(payload)
        if self.disable_calibration:
            # Guard: if no valid answer, still abstain
            if state.retrieval_failure or not state.answer:
                state.abstention_decision = {"decision": "ABSTAIN", "should_abstain": True, "reasons": ["no_answer_or_retrieval_failed"]}
                state.status = "abstained"
            else:
                state.abstention_decision = {"decision": "PROCEED", "should_abstain": False, "reasons": []}
                state.status = "verified"
        else:
            await self._step_abstention_decision(state)
        self._record_graph_node(state, "abstention")
        return self._state_update(state)

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
            # Query understanding failed — mirror generator/critic pattern so the
            # graph routes to abstention instead of continuing with a broken state.
            state.model_failure = True
            state.error, state.error_code = str(e), "QUERY_UNDERSTANDING_ERROR"
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
        """Run bounded retrieval → MGA → focused-retrieval iterations.

        A caller may supply initial evidence, but every evidence set still goes
        through Medical Gap Analysis.  An insufficient MGA result causes a new
        native retrieval request with its focused ``sub_query``.  The loop is
        deliberately capped at three retrieval passes.
        """
        next_query = None
        if state.evidence:
            state.retrieval_iteration_count = 1
            now = datetime.now(timezone.utc)
            state.agent_trace.append({
                "agent_type": AgentType.ITERATIVE_RETRIEVAL.value,
                "status": "completed",
                "started_at": now.isoformat(),
                "completed_at": now.isoformat(),
                "duration_ms": 0,
                "details": {
                    "action": "reused_supplied_evidence",
                    "iteration": 1,
                    "evidence_count": len(state.evidence),
                },
            })

        while state.retrieval_iteration_count < self.max_retrieval_iterations:
            if state.retrieval_iteration_count == 0 or next_query:
                await self._retrieve_evidence_pass(state, next_query)
                if state.retrieval_failure:
                    return

            gap = await self._run_gap_analysis(state)
            if gap is None:
                state.retrieval_failure = True
                return

            if gap.get("sufficient", False) and state.evidence:
                return

            if state.retrieval_iteration_count >= self.max_retrieval_iterations:
                break

            next_query = gap.get("sub_query")
            if not next_query:
                state.retrieval_failure = True
                state.error = "MGA reported insufficient evidence without a focused sub-query."
                state.error_code = "GAP_ANALYSIS_ERROR"
                return

        # Reaching the cap with insufficient evidence is a safety failure; do
        # not generate an answer from partial retrieval results.
        state.retrieval_failure = True
        now = datetime.now(timezone.utc)
        state.agent_trace.append({
            "agent_type": AgentType.ITERATIVE_RETRIEVAL.value,
            "status": "failed",
            "started_at": now.isoformat(),
            "completed_at": now.isoformat(),
            "duration_ms": 0,
            "details": {
                "action": "retrieval_iteration_cap_reached",
                "max_iterations": self.max_retrieval_iterations,
                "last_gap_analysis": state.gap_analysis,
            },
        })

    async def _retrieve_evidence_pass(self, state: AgentState, sub_query: Optional[str]) -> None:
        """Execute one provider pass and append its audit metadata to state."""
        started_at = datetime.now(timezone.utc)
        state.retrieval_iteration_count += 1
        iteration = state.retrieval_iteration_count
        try:
            optimized_query = sub_query
            if optimized_query is None and state.processed_query:
                optimized_query = state.processed_query.get("expanded_query")
            evidence, metadata = await self.retrieval_provider(
                query=state.original_question,
                optimized_query=optimized_query,
                max_iterations=self.max_retrieval_iterations,
            )
            state.evidence.extend(evidence or [])
            for item in metadata or []:
                entry = dict(item)
                entry["orchestrator_iteration"] = iteration
                entry["sub_query"] = sub_query
                state.retrieval_iterations.append(entry)
            completed_at = datetime.now(timezone.utc)
            state.agent_trace.append({
                "agent_type": AgentType.ITERATIVE_RETRIEVAL.value,
                "status": "completed",
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "duration_ms": int((completed_at - started_at).total_seconds() * 1000),
                "details": {
                    "action": "native_retrieval",
                    "iteration": iteration,
                    "sub_query": sub_query,
                    "retrieved_count": len(evidence or []),
                    "total_evidence_count": len(state.evidence),
                },
            })
        except Exception as exc:
            state.retrieval_failure = True
            state.error = str(exc)
            state.error_code = "RETRIEVAL_ERROR"
            completed_at = datetime.now(timezone.utc)
            state.agent_trace.append({
                "agent_type": AgentType.ITERATIVE_RETRIEVAL.value,
                "status": "failed",
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "duration_ms": int((completed_at - started_at).total_seconds() * 1000),
                "details": {"error": str(exc), "iteration": iteration, "sub_query": sub_query},
            })

    async def _run_gap_analysis(self, state: AgentState) -> Optional[Dict[str, Any]]:
        """Run MGA and preserve its result and trace for each retrieval pass."""
        try:
            result = await self.gap_agent.run({
                "original_question": state.original_question,
                "evidence": state.evidence,
                "retrieval_iteration": state.retrieval_iteration_count,
            })
            state.gap_analysis = result.get("gap_analysis")
            for trace_step in result.get("agent_trace", []):
                trace_step["retrieval_iteration"] = state.retrieval_iteration_count
                state.agent_trace.append(trace_step)
            return state.gap_analysis
        except Exception as exc:
            state.error = str(exc)
            state.error_code = "GAP_ANALYSIS_ERROR"
            now = datetime.now(timezone.utc)
            state.agent_trace.append({
                "agent_type": AgentType.ITERATIVE_RETRIEVAL.value,
                "status": "failed",
                "started_at": now.isoformat(),
                "completed_at": now.isoformat(),
                "duration_ms": 0,
                "details": {"action": "gap_analysis", "error": str(exc)},
            })
            return None

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
        critic_data = state.critic_verdict or (
            {"passed": True, "factual_consistency": 1.0, "citation_support": 1.0, "hallucination_risk": 0.0, "confidence": 1.0}
            if self.disable_critic
            else {"passed": False, "factual_consistency": 0.0, "citation_support": 0.0, "hallucination_risk": 1.0, "confidence": 0.0}
        )

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
            "force_abstention": state.force_abstention,
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
