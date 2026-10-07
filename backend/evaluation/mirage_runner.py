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
from app.utils.logger import logger
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
        generator_temperature: Optional[float] = None,
        disable_critic: bool = False,
        disable_calibration: bool = False,
        retrieval_provider: Optional[Callable] = None,
        max_retrieval_iterations: Optional[int] = None,
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
        generator_temperature:
            Override the GeneratorAgent's sampling temperature.  Pass 0.0 for
            deterministic, reproducible benchmark runs.  None (default) uses
            GeneratorAgent.DEFAULT_TEMPERATURE (0.1).
        disable_critic:
            If True, skip CriticAgent execution (see MultiAgentOrchestrator).
        disable_calibration:
            If True, skip ConfidenceCalibrationAgent execution (see MultiAgentOrchestrator).
        retrieval_provider:
            If provided, use this callable directly instead of building a new
            RAGPipeline.  The index guard still runs.  MedQAIndexBuilder.build
            and RAGPipeline.build are skipped when this is supplied.
        max_retrieval_iterations:
            Override the orchestrator's maximum retrieval iteration count.
            Pass 1 to disable iterative retrieval (single-pass).  None uses
            the orchestrator default (MAX_RETRIEVAL_ITERATIONS = 3).
        """
        from app.config import BASE_DIR, get_settings
        from app.rag.pipeline import RAGPipeline
        from evaluation.index_builder import MedQAIndexBuilder

        # ── Eval-split index guard ─────────────────────────────────────────
        # The benchmark MUST run against the eval-split FAISS index
        # (data/faiss_index/eval_split/), NOT the main production index
        # (data/faiss_index/).  The main index contains 1,916 of the 2,394
        # eval-question PMIDs, so using it would contaminate every metric.
        #
        # Resolution order (highest to lowest priority):
        #   1. Explicit ``index_dir`` argument passed by the caller.
        #   2. RAGMEDX_FAISS_INDEX_DIR env-var → settings.faiss_index_dir.
        #   3. Hard default: BASE_DIR / "data/faiss_index/eval_split".
        #
        # If the resolved path resolves to the *production* index directory
        # (BASE_DIR / "data/faiss_index" without the eval_split suffix) the
        # guard raises immediately rather than silently contaminating results.
        settings = get_settings()
        _eval_split_dir = (BASE_DIR / "data" / "faiss_index" / "eval_split").resolve()

        if index_dir is None:
            # Use the settings field if it differs from the bare production
            # path; otherwise fall back to the canonical eval_split dir.
            _settings_dir = settings.faiss_index_dir
            if not _settings_dir.is_absolute():
                _settings_dir = BASE_DIR / _settings_dir
            _settings_dir = _settings_dir.resolve()

            _production_dir = (BASE_DIR / "data" / "faiss_index").resolve()
            if _settings_dir == _production_dir:
                # Settings still point at the main index — use the eval_split
                # default and warn so the operator knows what happened.
                logger.warning(
                    "RAGMEDX_FAISS_INDEX_DIR resolves to the production index "
                    f"({_settings_dir}).  Overriding to eval_split index "
                    f"({_eval_split_dir}) to prevent leakage contamination.  "
                    "Set RAGMEDX_FAISS_INDEX_DIR=data/faiss_index/eval_split "
                    "explicitly to suppress this warning."
                )
                index_dir = _eval_split_dir
            else:
                index_dir = _settings_dir

        index_dir = Path(index_dir)
        if not index_dir.is_absolute():
            index_dir = BASE_DIR / index_dir
        index_dir = index_dir.resolve()

        # Hard guard: raise immediately if the resolved path is the production
        # index.  Matches the "fail loud, not silent" pattern in
        # RetrievalService.build() — better to abort than to produce
        # contaminated accuracy numbers.
        _production_dir = (BASE_DIR / "data" / "faiss_index").resolve()
        if index_dir == _production_dir:
            raise RuntimeError(
                f"run_rag_evaluation() resolved index_dir to the production FAISS "
                f"index ({index_dir}), which contains eval-question PMIDs and would "
                f"contaminate benchmark results.  "
                f"Pass index_dir explicitly pointing at the eval_split index, or set "
                f"RAGMEDX_FAISS_INDEX_DIR=data/faiss_index/eval_split before running."
            )

        logger.info(
            f"[run_rag_evaluation] FAISS index resolved to: {index_dir}  "
            f"(eval_split canonical: {_eval_split_dir})  "
            f"match={index_dir == _eval_split_dir}"
        )

        # 1. Warm exactly one injected local model before any question timing.
        #    _load_llm_once() raises ModelLoadError (propagates here) if real-mode
        #    loading fails — no silent fallback to mock is permitted here.
        model_load_ms = self._load_llm_once()

        # Guard: confirm the wrapper is genuinely in LOCAL mode after warming.
        # If it silently fell back to MOCK (e.g. allow_fallback=True was set),
        # raise immediately rather than produce mock answers labelled as real.
        from app.llm.local_llm import LLMRuntimeMode
        if (
            self.llm_wrapper is not None
            and hasattr(self.llm_wrapper, "active_runtime_mode")
            and self.llm_wrapper.active_runtime_mode != LLMRuntimeMode.MOCK
            and self.llm_wrapper.active_runtime_mode != LLMRuntimeMode.LOCAL
        ):
            raise RuntimeError(
                f"LocalLLMWrapper is in unexpected runtime mode "
                f"'{self.llm_wrapper.active_runtime_mode}' after load_model(). "
                "Expected LOCAL or MOCK. Aborting to avoid ambiguous results."
            )
        _llm_mode_label = (
            self.llm_wrapper.active_runtime_mode.value
            if (self.llm_wrapper is not None and hasattr(self.llm_wrapper, "active_runtime_mode"))
            else "unknown"
        )
        logger.info(f"[run_rag_evaluation] LLM active_runtime_mode after warm-up: {_llm_mode_label}")

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

        # 3. Build / load FAISS index (index_dir already resolved above)

        if retrieval_provider is None:
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
            _retrieval_provider = pipeline.retrieve_evidence
        else:
            _retrieval_provider = retrieval_provider
            store = None

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

            # Retrieve evidence exactly once.  The result is both recorded in
            # the per-sample trace and passed directly into run_agent_pipeline
            # via the ``evidence`` parameter so the orchestrator reuses it
            # instead of triggering a second retrieval call.
            raw_evidence, _ = await _retrieval_provider(question)
            passage_texts = [ev.get("cited_text", "") for ev in raw_evidence]
            evidence_lists.append(passage_texts)

            # Run full agent pipeline with pre-retrieved evidence.
            # Passing evidence= causes the orchestrator's _node_retrieval to
            # skip its own retrieval pass (see orchestrator._node_retrieval:
            # "if state.evidence and state.retrieval_iteration_count == 0").
            # retrieval_provider is also passed so the orchestrator constructor
            # does NOT fall into the "no retrieval_provider injected" branch and
            # rebuild RetrievalService (loading BGE-M3 + reranker) from scratch
            # for every question.  The provider will not actually be called
            # because evidence is already populated.
            response, _ = await run_agent_pipeline(
                original_question=question,
                evidence=raw_evidence,
                llm_wrapper=self.llm_wrapper,
                retrieval_provider=_retrieval_provider,
                generator_temperature=generator_temperature,
                disable_critic=disable_critic,
                disable_calibration=disable_calibration,
                max_retrieval_iterations=max_retrieval_iterations,
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
            "index_size":               store.size if store is not None else None,
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
