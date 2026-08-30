"""
Ablation study scaffold comparing full multi-agent pipeline vs ablations.
"""

from typing import Any, Callable, Dict, List, Optional
from app.agents import run_agent_pipeline
from evaluation.metrics import evaluate_batch


class AblationStudyRunner:
    """Scaffold comparing full pipeline vs ablations (no-critic, no-iterative-retrieval)."""

    def __init__(
        self,
        llm_wrapper: Optional[Any] = None,
        retrieval_provider: Optional[Callable] = None,
    ):
        self.llm_wrapper = llm_wrapper
        self.retrieval_provider = retrieval_provider

    async def run_study(
        self,
        samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Run full comparison across provided samples.
        """
        if not samples:
            return {"status": "NO_SAMPLES", "results": {}}

        full_pipeline_preds = []

        for s in samples:
            resp, _ = await run_agent_pipeline(
                original_question=s.get("question", ""),
                llm_wrapper=self.llm_wrapper,
                retrieval_provider=self.retrieval_provider,
            )
            full_pipeline_preds.append({
                "response_type": resp.response_type,
                "answer": getattr(resp, "answer", ""),
                "processing_time_ms": getattr(resp, "processing_time_ms", 0),
                "citations": getattr(resp, "citations", []),
            })

        metrics = evaluate_batch(full_pipeline_preds, samples)

        return {
            "status": "COMPLETED",
            "configurations": {
                "full_multi_agent": metrics,
                "no_critic_baseline": {"status": "NOT_RUN", "reason": "Ablation run deferred to full benchmark pass"},
                "no_mga_baseline": {"status": "NOT_RUN", "reason": "Ablation run deferred to full benchmark pass"},
            },
        }
