# Owner: Pranav Sukale
"""Research evaluation package for RAGMedX-MA.

Public API
----------
    from app.evaluation import Evaluator, ExperimentConfig, ExperimentResult
    from app.evaluation import evaluate_batch_full, compute_f1_score
    from app.evaluation import DATASETS, get_dataset, list_available
    from app.evaluation import run_benchmark_suite
"""

from app.evaluation.metrics import (  # noqa: F401
    compute_accuracy,
    compute_exact_match,
    compute_f1_score,
    compute_token_f1,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_mean_ndcg_at_k,
    compute_ece,
    compute_brier_score,
    compute_abstention_rate,
    compute_abstention_precision,
    compute_abstention_recall,
    compute_selective_accuracy,
    compute_citation_support_rate,
    compute_latency_metrics,
    evaluate_batch,
    evaluate_batch_full,
)

from app.evaluation.datasets import (  # noqa: F401
    DatasetSpec,
    DATASETS,
    get_dataset,
    list_available,
)

from app.evaluation.experiments import (  # noqa: F401
    ExperimentConfig,
    ExperimentResult,
    list_experiments,
)

from app.evaluation.evaluator import Evaluator  # noqa: F401
from app.evaluation.benchmark import run_benchmark_suite, BENCHMARK_SUITE  # noqa: F401

__all__ = [
    # Metrics
    "compute_accuracy", "compute_exact_match", "compute_f1_score",
    "compute_token_f1", "compute_precision_at_k", "compute_recall_at_k",
    "compute_mrr", "compute_ndcg_at_k", "compute_mean_ndcg_at_k",
    "compute_ece", "compute_brier_score", "compute_abstention_rate",
    "compute_abstention_precision", "compute_abstention_recall",
    "compute_selective_accuracy", "compute_citation_support_rate",
    "compute_latency_metrics", "evaluate_batch", "evaluate_batch_full",
    # Datasets
    "DatasetSpec", "DATASETS", "get_dataset", "list_available",
    # Experiments
    "ExperimentConfig", "ExperimentResult", "list_experiments",
    # Runner
    "Evaluator", "run_benchmark_suite", "BENCHMARK_SUITE",
]
