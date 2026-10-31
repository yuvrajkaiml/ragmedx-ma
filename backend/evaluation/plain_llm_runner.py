"""Plain LLM (no RAG) runner for ablation evaluation.

Provides run_plain_llm() and run_plain_llm_batch() for evaluating Qwen3-8B
without any retrieval, used in the ablation study as the 'plain-llm-no-rag' config.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.utils.logger import logger
from evaluation.metrics import evaluate_batch


def run_plain_llm(question: str, llm_wrapper, temperature: float = 0.0) -> dict:
    """Run a single question through the LLM without any RAG retrieval.

    Parameters
    ----------
    question:
        The medical question to answer.
    llm_wrapper:
        A loaded LocalLLMWrapper instance (must be in LOCAL mode).
    temperature:
        Sampling temperature. 0.0 for deterministic output.

    Returns
    -------
    dict with keys matching per_sample_results shape:
      {"question", "predicted_answer", "response_type", "evidence_count", "evidence_titles", "elapsed_ms"}
    """
    prompt = (
        "You are a medical knowledge assistant. Answer the following question "
        "concisely and accurately based on your training knowledge. "
        "Do not say you cannot answer — provide your best answer.\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    t_start = time.perf_counter()
    try:
        answer = llm_wrapper.generate(prompt, temperature=temperature)
    except Exception as exc:
        logger.warning(f"[plain_llm] generate() failed for question: {question[:60]!r}: {exc}")
        answer = ""
    elapsed_ms = int((time.perf_counter() - t_start) * 1000)

    return {
        "question": question,
        "predicted_answer": answer or "",
        "response_type": "verified_answer",
        "evidence_count": 0,
        "evidence_titles": [],
        "elapsed_ms": elapsed_ms,
    }


def run_plain_llm_batch(
    questions: List[Dict[str, Any]],
    llm_wrapper,
    temperature: float = 0.0,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a batch of BioASQ questions through the LLM without any RAG retrieval.

    Parameters
    ----------
    questions:
        List of dicts with 'question' and 'answer' keys (BioASQ-normalized format).
    llm_wrapper:
        A loaded LocalLLMWrapper instance.
    temperature:
        Sampling temperature. 0.0 for deterministic output.
    max_samples:
        Evaluate only the first N questions.

    Returns
    -------
    dict matching the structure of run_rag_evaluation() output:
      {"status", "samples_evaluated", "timing", "per_sample_results", "metrics"}
    """
    eval_samples = questions[:max_samples] if max_samples else questions
    per_sample_results: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []
    ground_truths: List[Dict[str, Any]] = []

    t_batch_start = time.perf_counter()
    for i, sample in enumerate(eval_samples):
        question = sample.get("question", "")
        true_answer = sample.get("answer", "")
        logger.info(f"[plain_llm_batch] Q{i+1}/{len(eval_samples)}: {question[:80]!r}")

        result = run_plain_llm(question, llm_wrapper, temperature=temperature)
        per_sample_results.append({
            **result,
            "true_answer": true_answer,
        })
        predictions.append({
            "response_type": result["response_type"],
            "answer": result["predicted_answer"],
            "processing_time_ms": result["elapsed_ms"],
            "citations": [],
        })
        ground_truths.append({"answer": true_answer, "question": question})

    total_wall_ms = int((time.perf_counter() - t_batch_start) * 1000)
    metrics = evaluate_batch(predictions, ground_truths)

    return {
        "status": "COMPLETED_PLAIN_LLM",
        "samples_evaluated": len(eval_samples),
        "timing": {
            "model_load_ms": 0,  # model loaded externally before batch
            "questions_wall_clock_ms": total_wall_ms,
        },
        "per_sample_results": per_sample_results,
        "metrics": metrics,
    }
