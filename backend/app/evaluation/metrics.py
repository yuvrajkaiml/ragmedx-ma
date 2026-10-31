# Owner: Pranav Sukale
"""App-layer evaluation metrics.

Re-exports every metric from the canonical ``backend/evaluation/metrics.py``
implementation so the rest of the application imports from one stable path:

    from app.evaluation.metrics import compute_f1_score, evaluate_batch_full

Adding new metrics to ``backend/evaluation/metrics.py`` automatically makes
them available here — no duplicate maintenance required.
"""

from evaluation.metrics import (  # noqa: F401 — re-export intentional
    # Basic answer quality
    compute_accuracy,
    compute_exact_match,
    compute_token_f1,
    compute_f1_score,
    # Retrieval quality
    compute_precision_at_k,
    compute_recall_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_mean_ndcg_at_k,
    compute_retrieval_coverage,
    # Abstention quality
    compute_abstention_rate,
    compute_abstention_precision,
    compute_abstention_recall,
    compute_selective_accuracy,
    # Citation quality
    compute_citation_support_rate,
    compute_answer_with_citation_rate,
    # Confidence calibration
    compute_ece,
    compute_brier_score,
    # Latency
    compute_latency_metrics,
    # Batch evaluators
    evaluate_batch,
    evaluate_batch_full,
)

__all__ = [
    "compute_accuracy",
    "compute_exact_match",
    "compute_token_f1",
    "compute_f1_score",
    "compute_precision_at_k",
    "compute_recall_at_k",
    "compute_mrr",
    "compute_ndcg_at_k",
    "compute_mean_ndcg_at_k",
    "compute_retrieval_coverage",
    "compute_abstention_rate",
    "compute_abstention_precision",
    "compute_abstention_recall",
    "compute_selective_accuracy",
    "compute_citation_support_rate",
    "compute_answer_with_citation_rate",
    "compute_ece",
    "compute_brier_score",
    "compute_latency_metrics",
    "evaluate_batch",
    "evaluate_batch_full",
]
