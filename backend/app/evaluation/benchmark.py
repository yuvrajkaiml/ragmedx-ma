# Owner: Pranav Sukale
"""Reproducible benchmark run orchestration.

Defines the standard benchmark suite for RAGMedX-MA research evaluation.
Run via the evaluation API endpoint or directly from the command line:

    python -m app.evaluation.benchmark

Benchmark suite:
    1. MedQA dev  — full multi-agent pipeline (mock LLM)
    2. MedQA dev  — no-critic ablation
    3. MedQA dev  — no-MGA (Medical Gap Analysis) ablation
    4. BioASQ     — full pipeline (when dataset available)
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

from app.evaluation.evaluator import Evaluator
from app.evaluation.experiments import ExperimentConfig, ExperimentResult
from app.evaluation.datasets import list_available
from app.utils.logger import logger


# Standard benchmark suite definition
BENCHMARK_SUITE: List[ExperimentConfig] = [
    ExperimentConfig(
        name="medqa_dev_full_pipeline",
        dataset_key="medqa_dev",
        pipeline_config="full",
        llm_mode="mock",
        max_samples=100,
        tags=["standard", "full_pipeline"],
        notes="Full 8-agent pipeline over MedQA dev split.",
    ),
    ExperimentConfig(
        name="medqa_dev_no_critic",
        dataset_key="medqa_dev",
        pipeline_config="no_critic",
        llm_mode="mock",
        max_samples=100,
        tags=["ablation", "no_critic"],
        notes="Ablation: critic agent disabled.",
    ),
    ExperimentConfig(
        name="medqa_dev_no_mga",
        dataset_key="medqa_dev",
        pipeline_config="no_mga",
        llm_mode="mock",
        max_samples=100,
        tags=["ablation", "no_mga"],
        notes="Ablation: Medical Gap Analysis disabled (single retrieval pass).",
    ),
    ExperimentConfig(
        name="bioasq_full_pipeline",
        dataset_key="bioasq_train",
        pipeline_config="full",
        llm_mode="mock",
        max_samples=200,
        tags=["standard", "bioasq"],
        notes="Full pipeline over BioASQ 13b training set.",
    ),
]


async def run_benchmark_suite(
    llm_wrapper: Optional[Any] = None,
    retrieval_provider: Optional[Callable] = None,
    configs: Optional[List[ExperimentConfig]] = None,
    save_results: bool = True,
) -> Dict[str, ExperimentResult]:
    """Run the full benchmark suite (or a custom subset).

    Parameters
    ----------
    llm_wrapper:
        Optional shared LLM wrapper. Passed to every experiment so the
        model is loaded once for the entire suite.
    retrieval_provider:
        Optional async retrieval callable.
    configs:
        Custom list of experiment configs. Defaults to ``BENCHMARK_SUITE``.
    save_results:
        If True, persist each result to ``EXPERIMENTS_DIR`` as JSON.

    Returns
    -------
    Dict[str, ExperimentResult]
        Keyed by experiment ``name``.
    """
    available = set(list_available())
    suite = configs or BENCHMARK_SUITE
    evaluator = Evaluator(
        llm_wrapper=llm_wrapper,
        retrieval_provider=retrieval_provider,
    )
    results: Dict[str, ExperimentResult] = {}

    for config in suite:
        if config.dataset_key not in available:
            logger.warning(
                f"Benchmark: skipping '{config.name}' — "
                f"dataset '{config.dataset_key}' not available on disk."
            )
            continue

        logger.info(f"Benchmark: running '{config.name}'…")
        result = await evaluator.run(config)
        results[config.name] = result

        if save_results:
            path = result.save()
            logger.info(f"Benchmark: saved result to {path}")

        logger.info(
            f"Benchmark: '{config.name}' — "
            f"status={result.status} "
            f"samples={result.samples_evaluated} "
            f"EM={result.metrics.get('exact_match', 'n/a')} "
            f"F1={result.metrics.get('f1_score', 'n/a')}"
        )

    return results


if __name__ == "__main__":
    asyncio.run(run_benchmark_suite())
