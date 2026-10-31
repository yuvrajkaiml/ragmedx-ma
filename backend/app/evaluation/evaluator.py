# Owner: Pranav Sukale
"""Evidence-grounded evaluation runner.

Provides ``Evaluator`` — the main entry point for running reproducible
evaluation experiments over the RAGMedX-MA pipeline.

Usage::

    from app.evaluation.evaluator import Evaluator
    from app.evaluation.experiments import ExperimentConfig

    config = ExperimentConfig(
        name="medqa_dev_full_pipeline",
        dataset_key="medqa_dev",
        pipeline_config="full",
        llm_mode="mock",
        max_samples=50,
    )
    result = await Evaluator().run(config)
    result.save()
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from app.evaluation.datasets import get_dataset
from app.evaluation.experiments import ExperimentConfig, ExperimentResult
from app.evaluation.metrics import evaluate_batch_full
from app.utils.logger import logger


class Evaluator:
    """Run one reproducible evaluation experiment end-to-end.

    Parameters
    ----------
    llm_wrapper:
        Optional LLM wrapper to inject into the pipeline. Defaults to
        ``LocalLLMWrapper(use_mock_fallback=True)`` (mock mode).
    retrieval_provider:
        Optional async retrieval callable. Defaults to the shared
        ``RetrievalService`` singleton built at app startup.
    """

    def __init__(
        self,
        llm_wrapper: Optional[Any] = None,
        retrieval_provider: Optional[Callable] = None,
    ) -> None:
        self.llm_wrapper = llm_wrapper
        self.retrieval_provider = retrieval_provider

    async def run(self, config: ExperimentConfig) -> ExperimentResult:
        """Execute the experiment described by ``config``.

        Steps
        -----
        1. Load the dataset from the registry.
        2. Delegate to the appropriate pipeline runner based on
           ``config.pipeline_config``.
        3. Aggregate metrics via ``evaluate_batch_full``.
        4. Return a serialisable ``ExperimentResult``.

        Parameters
        ----------
        config:
            Experiment configuration.

        Returns
        -------
        ExperimentResult
            Contains metrics, per-sample results, and timing information.
        """
        started_at = datetime.now(timezone.utc).isoformat()
        logger.info(f"Evaluator: starting experiment '{config.name}'")

        try:
            spec = get_dataset(config.dataset_key)
        except (KeyError, FileNotFoundError) as exc:
            logger.error(f"Evaluator: dataset error — {exc}")
            return ExperimentResult(
                config=config,
                metrics={},
                samples_evaluated=0,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc).isoformat(),
                status="failed",
                error=str(exc),
            )

        # Build the pipeline runner based on config.pipeline_config
        runner = self._build_runner(config)

        try:
            run_result = await runner.run_rag_evaluation(
                max_samples=config.max_samples,
                top_k_retrieval=config.top_k_retrieval,
                top_k_final=config.top_k_final,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Evaluator: pipeline error — {exc}")
            return ExperimentResult(
                config=config,
                metrics={},
                samples_evaluated=0,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc).isoformat(),
                status="failed",
                error=str(exc),
            )

        # Augment with full research metrics
        predictions = run_result.get("per_sample_results", [])
        ground_truth_records = [
            {"answer": p.get("true_answer", ""), "answerable": True}
            for p in predictions
        ]
        # Add confidence scores if present in pipeline outputs
        for pred, p in zip(predictions, predictions):
            pred["confidence_score"] = p.get("confidence_score", 0.0)
            pred["answer"] = p.get("predicted_answer", "")
            pred["response_type"] = p.get("response_type", "verified_answer")

        full_metrics = evaluate_batch_full(predictions, ground_truth_records)
        # Merge with runner metrics
        merged_metrics = {**run_result.get("metrics", {}), **full_metrics}

        completed_at = datetime.now(timezone.utc).isoformat()
        logger.info(
            f"Evaluator: experiment '{config.name}' completed — "
            f"{len(predictions)} samples, EM={merged_metrics.get('exact_match', 'n/a')}, "
            f"F1={merged_metrics.get('f1_score', 'n/a')}"
        )

        return ExperimentResult(
            config=config,
            metrics=merged_metrics,
            samples_evaluated=len(predictions),
            started_at=started_at,
            completed_at=completed_at,
            status=run_result.get("status", "completed"),
            per_sample_results=predictions,
        )

    def _build_runner(self, config: ExperimentConfig) -> Any:
        """Construct the appropriate BenchmarkRunner for the pipeline config."""
        from evaluation.mirage_runner import BenchmarkRunner
        from app.evaluation.datasets import get_dataset
        from app.llm.local_llm import LocalLLMWrapper, LLMRuntimeMode

        spec = get_dataset(config.dataset_key)

        llm_wrapper = self.llm_wrapper
        if llm_wrapper is None:
            if config.llm_mode == "real":
                from app.config import get_settings
                settings = get_settings()
                llm_wrapper = LocalLLMWrapper(
                    model_name_or_path=settings.llm_model_name,
                    mode=LLMRuntimeMode.LOCAL,
                    allow_fallback=False,
                )
            elif config.llm_mode == "ollama":
                from app.config import get_settings
                settings = get_settings()
                llm_wrapper = LocalLLMWrapper(
                    mode=LLMRuntimeMode.OLLAMA,
                    ollama_base_url=settings.ollama_base_url,
                    ollama_model=settings.ollama_model,
                    allow_fallback=False,
                )
            else:
                llm_wrapper = LocalLLMWrapper(use_mock_fallback=True)

        return BenchmarkRunner(
            dataset_path=spec.path,
            llm_wrapper=llm_wrapper,
            retrieval_provider=self.retrieval_provider,
        )
