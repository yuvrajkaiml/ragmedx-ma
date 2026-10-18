# Owner: Pranav Sukale
"""Ablation study runner for RAGMedX-MA.

Compares the full multi-agent pipeline against two controlled ablations:

    full_multi_agent   — 8-agent pipeline (Query Understanding → Retrieval →
                         Gap Analysis → Evidence Reranking → Generator →
                         Critic → Confidence Calibration → Abstention)

    no_critic_baseline — Same pipeline with the Critic agent disabled.
                         The generator output goes directly to confidence
                         calibration without review or revision.

    no_mga_baseline    — Same pipeline with Medical Gap Analysis (MGA)
                         disabled.  A single retrieval pass is performed;
                         the gap-analysis node immediately marks evidence
                         as sufficient and skips iterative retrieval.

Usage::

    from evaluation.ablation_studies import AblationStudyRunner
    from evaluation.metrics import evaluate_batch_full

    runner = AblationStudyRunner(llm_wrapper=llm, retrieval_provider=retrieve)
    result = await runner.run_study(samples, ground_truths=ground_truths)
    print(result)
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from app.agents import run_agent_pipeline
from app.agents.orchestrator import MultiAgentOrchestrator
from app.agents.state import AgentState
from evaluation.metrics import evaluate_batch_full


# ---------------------------------------------------------------------------
# Ablation orchestrators
# ---------------------------------------------------------------------------

class _NoCriticOrchestrator(MultiAgentOrchestrator):
    """Full pipeline with the Critic agent replaced by a pass-through.

    The generator output is accepted unconditionally — no review, no revision
    instructions, no force-abstention from critic failure.
    """

    async def _node_critic(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Pass-through critic: always accepts the generated answer."""
        state = self._as_agent_state(payload)
        if not state.model_failure and state.answer:
            # Inject a synthetic passing verdict so confidence calibration
            # has all the fields it expects.
            state.critic_verdict = {
                "passed": True,
                "verdict": "verified",
                "factual_consistency": 0.85,
                "citation_support": 0.85,
                "hallucination_risk": 0.05,
                "issues": [],
                "revision_instructions": None,
                "confidence": 0.85,
                "evidence_verification": {
                    "referenced_markers": [],
                    "mapped_markers": [],
                    "unresolved_markers": [],
                    "invalid_evidence_markers": [],
                    "blocking_issues": [],
                },
            }
            state.critic_reviews.append(state.critic_verdict)
            from datetime import datetime, timezone
            state.agent_trace.append({
                "agent_type": "critic",
                "status": "completed",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": 0,
                "details": {"action": "ablation_pass_through", "critic_disabled": True},
            })
        self._record_graph_node(state, "critic")
        return self._state_update(state)


class _NoMGAOrchestrator(MultiAgentOrchestrator):
    """Full pipeline with Medical Gap Analysis replaced by a pass-through.

    A single retrieval pass is performed; gap analysis always reports
    the evidence as sufficient so no iterative retrieval is attempted.
    """

    async def _node_gap_analysis(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Pass-through MGA: always marks evidence sufficient after first pass."""
        state = self._as_agent_state(payload)
        if not state.retrieval_failure:
            state.gap_analysis = {
                "sufficient": True,
                "reasoning": "MGA disabled for ablation — evidence accepted after single pass.",
                "missing_information": [],
                "sub_query": None,
                "confidence": 1.0,
            }
            from datetime import datetime, timezone
            state.agent_trace.append({
                "agent_type": "iterative_retrieval",
                "status": "completed",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": 0,
                "details": {
                    "action": "ablation_pass_through",
                    "mga_disabled": True,
                    "retrieval_iteration": state.retrieval_iteration_count,
                },
            })
        self._record_graph_node(state, "gap_analysis")
        return self._state_update(state)


# ---------------------------------------------------------------------------
# Pipeline runner helpers
# ---------------------------------------------------------------------------

async def _run_samples(
    samples: List[Dict[str, Any]],
    orchestrator_cls: type,
    llm_wrapper: Optional[Any],
    retrieval_provider: Optional[Callable],
) -> List[Dict[str, Any]]:
    """Run every sample through a given orchestrator class.

    Returns a list of prediction dicts compatible with ``evaluate_batch_full``.
    """
    # Instantiate once — model weights stay loaded across all samples.
    orchestrator = orchestrator_cls(
        llm_wrapper=llm_wrapper,
        retrieval_provider=retrieval_provider,
    )

    predictions: List[Dict[str, Any]] = []
    for sample in samples:
        question = sample.get("question", "")
        t_start = time.perf_counter()
        try:
            from app.agents.state import AgentState, create_deterministic_uuid
            initial_state = AgentState(
                query_id=create_deterministic_uuid(question),
                original_question=question,
                include_trace=False,
            )
            final_state = await orchestrator.run(initial_state)
            from app.agents.response_builder import build_medical_response
            response = build_medical_response(final_state)

            elapsed_ms = int((time.perf_counter() - t_start) * 1000)
            conf = getattr(response, "confidence", None)
            conf_score = 0.0
            if conf is not None:
                raw = getattr(conf, "score", None) or getattr(conf, "calibrated_confidence", 0.0)
                # Normalise: pipeline returns 0-1, UI uses 0-100
                conf_score = float(raw) / 100.0 if float(raw) > 1.0 else float(raw)

            predictions.append({
                "response_type": response.response_type,
                "answer": getattr(response, "answer", "") or "",
                "processing_time_ms": elapsed_ms,
                "citations": getattr(response, "citations", []) or [],
                "confidence_score": conf_score,
                "status": (
                    "abstained"
                    if response.response_type == "abstention"
                    else "answered"
                ),
            })
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = int((time.perf_counter() - t_start) * 1000)
            predictions.append({
                "response_type": "error",
                "answer": "",
                "processing_time_ms": elapsed_ms,
                "citations": [],
                "confidence_score": 0.0,
                "status": "error",
                "error": str(exc),
            })

    return predictions


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

class AblationStudyRunner:
    """Compare full pipeline against no-critic and no-MGA ablations.

    Parameters
    ----------
    llm_wrapper:
        LLM wrapper shared across all three arms. Passing a real wrapper
        keeps model weights in VRAM for all runs; passing ``None`` defaults
        to mock mode.
    retrieval_provider:
        Async retrieval callable shared across all arms. Uses the production
        FAISS index when injected; falls back to the empty stub otherwise.
    """

    def __init__(
        self,
        llm_wrapper: Optional[Any] = None,
        retrieval_provider: Optional[Callable] = None,
    ) -> None:
        self.llm_wrapper = llm_wrapper
        self.retrieval_provider = retrieval_provider

    async def run_study(
        self,
        samples: List[Dict[str, Any]],
        ground_truths: Optional[List[Dict[str, Any]]] = None,
        run_no_critic: bool = True,
        run_no_mga: bool = True,
    ) -> Dict[str, Any]:
        """Run all three arms and return a comparative metrics report.

        Parameters
        ----------
        samples:
            List of ``{"question": str, ...}`` dicts.
        ground_truths:
            Optional list of ``{"answer": str, "answerable": bool}`` dicts
            (same length as ``samples``).  When supplied, answer-quality
            metrics (EM, F1) and calibration metrics (ECE, Brier) are included.
        run_no_critic:
            Whether to run the no-critic arm. Set ``False`` to skip.
        run_no_mga:
            Whether to run the no-MGA arm. Set ``False`` to skip.

        Returns
        -------
        Dict[str, Any]
            ``{"status", "arms_run", "configurations": {arm: metrics}}``.
        """
        if not samples:
            return {"status": "NO_SAMPLES", "arms_run": [], "configurations": {}}

        results: Dict[str, Any] = {}
        arms_run: List[str] = []

        # ── Arm 1: Full multi-agent pipeline ──────────────────────────────
        full_preds = await _run_samples(
            samples,
            MultiAgentOrchestrator,
            self.llm_wrapper,
            self.retrieval_provider,
        )
        results["full_multi_agent"] = evaluate_batch_full(full_preds, ground_truths)
        arms_run.append("full_multi_agent")

        # ── Arm 2: No-critic baseline ──────────────────────────────────────
        if run_no_critic:
            no_critic_preds = await _run_samples(
                samples,
                _NoCriticOrchestrator,
                self.llm_wrapper,
                self.retrieval_provider,
            )
            results["no_critic_baseline"] = evaluate_batch_full(
                no_critic_preds, ground_truths
            )
            arms_run.append("no_critic_baseline")

        # ── Arm 3: No-MGA baseline ─────────────────────────────────────────
        if run_no_mga:
            no_mga_preds = await _run_samples(
                samples,
                _NoMGAOrchestrator,
                self.llm_wrapper,
                self.retrieval_provider,
            )
            results["no_mga_baseline"] = evaluate_batch_full(
                no_mga_preds, ground_truths
            )
            arms_run.append("no_mga_baseline")

        return {
            "status": "COMPLETED",
            "arms_run": arms_run,
            "samples_evaluated": len(samples),
            "configurations": results,
        }
