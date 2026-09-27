"""
Evaluation runner for MedQA / MIRAGE benchmarks.

Two evaluation modes
--------------------
``run_evaluation``
    Original scaffold mode -- uses any injected LLM + retrieval provider.
    Returns ``status="COMPLETED_MOCK_SCAFFOLD"``.  All existing tests pass.

``run_rag_evaluation``
    Full RAG-aware evaluation -- builds a FAISS index from the dataset,
    constructs the real RAGPipeline, runs agent pipeline with real retrieval,
    and returns MIRAGE-style metrics.
"""

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.agents import run_agent_pipeline
from app.llm.local_llm import LocalLLMWrapper
from evaluation.metrics import (
    compute_answer_with_citation_rate,
    compute_retrieval_coverage,
    evaluate_batch,
)


class BenchmarkRunner:
    """Scaffold for executing evaluation over dataset fixtures.

    Parameters
    ----------
    dataset_path:
        Path to a MedQA-format JSONL file.
    llm_wrapper:
        Optional LLM wrapper. Defaults to LocalLLMWrapper(use_mock_fallback=True).
    retrieval_provider:
        Optional async retrieval callable. Defaults to the temporary stub.
    """

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        llm_wrapper: Optional[Any] = None,
        retrieval_provider: Optional[Callable] = None,
    ):
        self.dataset_path = dataset_path
        # Own exactly one wrapper for the complete evaluation.  Supplying a real
        # local wrapper makes its model residency span all questions; the default
        # remains explicit mock mode for lightweight existing scaffold callers.
        self.llm_wrapper = llm_wrapper or LocalLLMWrapper(use_mock_fallback=True)
        self.retrieval_provider = retrieval_provider

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load samples from a JSONL dataset file."""
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

    def _load_llm_once(self) -> int:
        """Warm an injected local wrapper once and return its load duration.

        A benchmark must pass the *same* wrapper to every graph invocation.
        ``LocalLLMWrapper.generate`` is lazy, so without this explicit warm-up the
        first question absorbs model initialization and timing becomes ambiguous.
        Wrappers already loaded by the caller are reused without reloading.
        """
        if self.llm_wrapper is None or not hasattr(self.llm_wrapper, "load_model"):
            return 0
        if getattr(self.llm_wrapper, "is_loaded", False):
            return 0
        started = time.perf_counter()
        self.llm_wrapper.load_model()
        return int((time.perf_counter() - started) * 1000)

    # ------------------------------------------------------------------
    # Original scaffold evaluation (API unchanged)
    # ------------------------------------------------------------------

    async def run_evaluation(
        self,
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute evaluation using the injected LLM and retrieval provider.

        Returns ``status="COMPLETED_MOCK_SCAFFOLD"`` to clearly label this
        as a scaffold run rather than a full MIRAGE benchmark.
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

    # ------------------------------------------------------------------
    # Full RAG-aware evaluation (new)
    # ------------------------------------------------------------------

    async def run_rag_evaluation(
        self,
        max_samples: Optional[int] = None,
        index_dir: Optional[Path] = None,
        top_k_retrieval: int = 10,
        top_k_final: int = 3,
        force_reindex: bool = False,
    ) -> Dict[str, Any]:
        """Run a full MIRAGE-style evaluation with real retrieval.

        Steps
        -----
        1. Load the dataset from ``self.dataset_path``.
        2. Build (or load) a FAISS index from the dataset via MedQAIndexBuilder.
        3. Construct a RAGPipeline over the index.
        4. Run ``run_agent_pipeline`` for each sample with the real pipeline.
        5. Collect per-sample traces, evidence, and aggregate MIRAGE metrics.

        Parameters
        ----------
        max_samples:
            Evaluate only the first N samples.
        index_dir:
            Where to save/load the FAISS index. Defaults to
            ``FAISS_INDEX_DIR/mirage_eval``.
        top_k_retrieval:
            Hybrid retriever candidate pool size.
        top_k_final:
            Final evidence count after reranking.
        force_reindex:
            Rebuild the index even if one already exists.
        """
        from app.config import FAISS_INDEX_DIR
        from app.rag.pipeline import RAGPipeline
        from evaluation.index_builder import MedQAIndexBuilder

        # 1. Warm exactly one injected local model before any question timing.
        model_load_ms = self._load_llm_once()

        # 2. Load dataset
        samples = self.load_dataset()
        if not samples:
            return {
                "status": "BLOCKED_ON_DATASET",
                "message": f"Dataset at {self.dataset_path} not found or empty.",
                "samples_evaluated": 0,
                "total_samples": 0,
                "metrics": {},
            }

        eval_samples = samples[:max_samples] if max_samples else samples

        # 3. Build / load FAISS index
        if index_dir is None:
            index_dir = FAISS_INDEX_DIR / "mirage_eval"
        index_dir = Path(index_dir)

        builder = MedQAIndexBuilder(
            dataset_path=self.dataset_path,
            index_dir=index_dir,
        )
        store = builder.build(force=force_reindex)

        # 4. Construct the async facade over Sana's native pipeline.
        pipeline = RAGPipeline.build(
            index_dir=index_dir,
            top_k_retrieval=top_k_retrieval,
            top_k_final=top_k_final,
        )

        # 5. Evaluation loop. The already-warmed self.llm_wrapper is passed into
        # every run_agent_pipeline call, so no model is reinstantiated per sample.
        predictions: List[Dict[str, Any]] = []
        ground_truths: List[Dict[str, Any]] = []
        per_sample_results: List[Dict[str, Any]] = []
        evidence_lists: List[List[str]] = []

        for sample in eval_samples:
            question = sample.get("question", "")
            true_answer = sample.get("answer", "")
            t_start = time.perf_counter()

            # Retrieve evidence for per-sample trace
            raw_evidence, _ = await pipeline.retrieve_evidence(question)
            passage_texts = [ev.get("cited_text", "") for ev in raw_evidence]
            evidence_lists.append(passage_texts)

            # Run full agent pipeline (retrieval happens again inside)
            response, _ = await run_agent_pipeline(
                original_question=question,
                llm_wrapper=self.llm_wrapper,
                retrieval_provider=pipeline.retrieve_evidence,
                include_trace=False,
            )

            elapsed_ms = int((time.perf_counter() - t_start) * 1000)
            pred_dict = {
                "response_type":      response.response_type,
                "answer":             getattr(response, "answer", "") or "",
                "processing_time_ms": elapsed_ms,
                "citations":          getattr(response, "citations", []) or [],
            }
            predictions.append(pred_dict)
            ground_truths.append({"answer": true_answer, "question": question})

            per_sample_results.append({
                "question":          question,
                "true_answer":       true_answer,
                "predicted_answer":  pred_dict["answer"],
                "response_type":     pred_dict["response_type"],
                "evidence_count":    len(raw_evidence),
                "evidence_titles":   [ev.get("title", "") for ev in raw_evidence],
                "elapsed_ms":        elapsed_ms,
            })

        # 6. Aggregate metrics
        base_metrics = evaluate_batch(predictions, ground_truths)

        keyword_sets = [
            set(str(g["answer"]).lower().split())
            for g in ground_truths
        ]
        retrieval_coverage = compute_retrieval_coverage(evidence_lists, keyword_sets)
        citation_rate = compute_answer_with_citation_rate(predictions)

        metrics = {
            **base_metrics,
            "retrieval_coverage":       retrieval_coverage,
            "answer_with_citation_rate": citation_rate,
            "index_size":               store.size,
        }

        return {
            "status":              "COMPLETED_RAG_EVALUATION",
            "dataset_path":        str(self.dataset_path),
            "index_dir":           str(index_dir),
            "samples_evaluated":   len(eval_samples),
            "timing": {
                "model_load_ms": model_load_ms,
                "questions_wall_clock_ms": sum(item["elapsed_ms"] for item in per_sample_results),
            },
            "metrics":             metrics,
            "per_sample_results":  per_sample_results,
        }
