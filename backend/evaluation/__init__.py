"""Evaluation package for the RAGMedX-MA medical pipeline.

Public API
----------
``BenchmarkRunner``                   -- MIRAGE / MedQA evaluation runner.
``MedQAIndexBuilder``                 -- Ingests MedQA JSONL into a FAISS index.
``compute_accuracy``                  -- Answer accuracy.
``compute_precision_at_k``            -- Retrieval precision@K.
``compute_recall_at_k``               -- Retrieval recall@K.
``compute_abstention_rate``           -- Pipeline abstention rate.
``compute_citation_support_rate``     -- Proportion of responses with citations.
``compute_answer_with_citation_rate`` -- Proportion of answered responses citing evidence.
``compute_retrieval_coverage``        -- Keyword coverage of retrieved passages.
``compute_latency_metrics``           -- avg/median/min/max latency stats.
``evaluate_batch``                    -- Consolidated batch metrics report.
"""

from evaluation.index_builder import MedQAIndexBuilder
from evaluation.metrics import (
    compute_abstention_rate,
    compute_accuracy,
    compute_answer_with_citation_rate,
    compute_citation_support_rate,
    compute_latency_metrics,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_retrieval_coverage,
    evaluate_batch,
)
from evaluation.mirage_runner import BenchmarkRunner

__all__ = [
    "BenchmarkRunner",
    "MedQAIndexBuilder",
    "compute_accuracy",
    "compute_precision_at_k",
    "compute_recall_at_k",
    "compute_abstention_rate",
    "compute_citation_support_rate",
    "compute_answer_with_citation_rate",
    "compute_retrieval_coverage",
    "compute_latency_metrics",
    "evaluate_batch",
]
