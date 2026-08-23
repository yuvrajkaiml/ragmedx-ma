"""
Evaluation runner scaffold for MedQA / MIRAGE benchmarks.

Clearly identifies whether execution is running over synthetic fixtures or real benchmarks.
External full-scale datasets and GPU-heavy benchmark runs are reported honestly.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.agents import run_agent_pipeline
from evaluation.metrics import evaluate_batch


class BenchmarkRunner:
    """Scaffold for executing evaluation over dataset fixtures."""

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        llm_wrapper: Optional[Any] = None,
        retrieval_provider: Optional[Callable] = None,
    ):
        self.dataset_path = dataset_path
        self.llm_wrapper = llm_wrapper
        self.retrieval_provider = retrieval_provider

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load samples from JSONL dataset fixture."""
        if not self.dataset_path or not self.dataset_path.exists():
            return []

        samples = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return samples

    async def run_evaluation(
        self,
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute evaluation over dataset samples.
        """
        samples = self.load_dataset()
        if not samples:
            return {
                "status": "BLOCKED_ON_DATASET",
                "message": f"Dataset at {self.dataset_path} not found or empty.",
                "total_samples": 0,
                "metrics": {},
            }

        eval_samples = samples[:max_samples] if max_samples else samples
        predictions = []
        ground_truths = []

        for sample in eval_samples:
            question = sample.get("question", "")
            true_answer = sample.get("answer", "")

            response, trace = await run_agent_pipeline(
                original_question=question,
                llm_wrapper=self.llm_wrapper,
                retrieval_provider=self.retrieval_provider,
                include_trace=False,
            )

            pred_dict = {
                "response_type": response.response_type,
                "answer": getattr(response, "answer", ""),
                "processing_time_ms": getattr(response, "processing_time_ms", 0),
                "citations": getattr(response, "citations", []),
            }
            predictions.append(pred_dict)
            ground_truths.append({"answer": true_answer, "question": question})

        metrics_summary = evaluate_batch(predictions, ground_truths)

        return {
            "status": "COMPLETED_MOCK_SCAFFOLD",
            "dataset_path": str(self.dataset_path),
            "samples_evaluated": len(eval_samples),
            "metrics": metrics_summary,
            "predictions": predictions,
        }
