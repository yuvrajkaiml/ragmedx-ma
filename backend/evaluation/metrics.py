from typing import Any, Dict, List, Optional, Set, Union


def compute_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Compute classification/option-matching accuracy between predictions and ground truths.
    Normalizes strings by stripping whitespace and matching case-insensitively.
    """
    if not predictions or not ground_truths or len(predictions) != len(ground_truths):
        return 0.0

    correct = 0
    for pred, truth in zip(predictions, ground_truths):
        pred_clean = str(pred or "").strip().lower()
        truth_clean = str(truth or "").strip().lower()
        if truth_clean in pred_clean or pred_clean in truth_clean:
            correct += 1

    return round(correct / len(predictions), 4)


def compute_precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
    """
    Compute Precision@K: Proportion of top-K retrieved items that are relevant.
    """
    if not retrieved_ids or not relevant_ids or k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return round(hits / min(k, len(top_k)), 4)


def compute_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
    """
    Compute Recall@K: Proportion of relevant items that are retrieved in top-K.
    """
    if not retrieved_ids or not relevant_ids or k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return round(hits / len(relevant_ids), 4)


def compute_abstention_rate(responses: List[Dict[str, Any]]) -> float:
    """
    Compute abstention rate across a set of pipeline responses.
    """
    if not responses:
        return 0.0

    abstained = sum(
        1 for r in responses
        if r.get("response_type") == "abstention" or r.get("status") == "abstained"
    )
    return round(abstained / len(responses), 4)


def compute_citation_support_rate(responses: List[Dict[str, Any]]) -> float:
    """
    Compute the proportion of generated responses that contain valid evidence citations.
    """
    if not responses:
        return 0.0

    cited_count = 0
    for r in responses:
        citations = r.get("citations", []) or r.get("answer_citations", [])
        if len(citations) > 0:
            cited_count += 1

    return round(cited_count / len(responses), 4)


def compute_latency_metrics(latencies_ms: List[Union[int, float]]) -> Dict[str, float]:
    """
    Compute average, median, min, max latency in milliseconds.
    """
    if not latencies_ms:
        return {"avg_ms": 0.0, "median_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}

    sorted_l = sorted(latencies_ms)
    n = len(sorted_l)
    avg = sum(sorted_l) / n
    median = (sorted_l[n // 2] + sorted_l[(n - 1) // 2]) / 2.0

    return {
        "avg_ms": round(avg, 2),
        "median_ms": round(median, 2),
        "min_ms": round(sorted_l[0], 2),
        "max_ms": round(sorted_l[-1], 2),
    }


def evaluate_batch(
    predictions: List[Dict[str, Any]],
    ground_truths: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Compute a consolidated evaluation metrics report over a batch of query results.
    """
    total_samples = len(predictions)
    if total_samples == 0:
        return {"total_samples": 0, "status": "empty"}

    latencies = [p.get("processing_time_ms", 0) for p in predictions]
    abstention_rate = compute_abstention_rate(predictions)
    citation_rate = compute_citation_support_rate(predictions)
    latency_stats = compute_latency_metrics(latencies)

    report = {
        "total_samples": total_samples,
        "abstention_rate": abstention_rate,
        "citation_support_rate": citation_rate,
        "latency_stats": latency_stats,
    }

    if ground_truths and len(ground_truths) == total_samples:
        pred_answers = [str(p.get("answer", "")) for p in predictions]
        true_answers = [str(g.get("answer", "")) for g in ground_truths]
        report["accuracy"] = compute_accuracy(pred_answers, true_answers)

    return report


def compute_answer_with_citation_rate(responses: List[Dict[str, Any]]) -> float:
    """Proportion of non-abstention responses that contain at least one evidence citation.

    A response is considered cited when:
    - Its ``answer`` string contains a token matching ``[E<n>]``, OR
    - Its ``citations`` / ``answer_citations`` list is non-empty.

    Abstention responses are excluded from the denominator.
    """
    import re
    if not responses:
        return 0.0

    _MARKER_RE = re.compile(r"\[E\d+\]")
    answered = [
        r for r in responses
        if r.get("response_type") not in ("abstention",)
        and r.get("status") not in ("abstained",)
    ]
    if not answered:
        return 0.0

    cited = 0
    for r in answered:
        answer_text = str(r.get("answer", "") or "")
        citations = r.get("citations", []) or r.get("answer_citations", [])
        if _MARKER_RE.search(answer_text) or len(citations) > 0:
            cited += 1

    return round(cited / len(answered), 4)


def compute_retrieval_coverage(
    evidence_lists: List[List[str]],
    relevant_keyword_sets: List[Set[str]],
) -> float:
    """Proportion of samples where at least one retrieved passage contains
    a keyword from the expected relevant set (case-insensitive).

    Parameters
    ----------
    evidence_lists:
        One list of passage text strings per sample.
    relevant_keyword_sets:
        One set of expected keywords per sample.

    Returns
    -------
    float
        Coverage rate in ``[0, 1]``.
    """
    if not evidence_lists or not relevant_keyword_sets:
        return 0.0
    if len(evidence_lists) != len(relevant_keyword_sets):
        return 0.0

    covered = 0
    for passages, keywords in zip(evidence_lists, relevant_keyword_sets):
        combined = " ".join(passages).lower()
        if any(kw.lower() in combined for kw in keywords):
            covered += 1

    return round(covered / len(evidence_lists), 4)


# ============================================================
# Research-grade metrics (P1 — added by Pranav Sukale)
# ============================================================

import re as _re
import math as _math
from collections import Counter as _Counter


# ------------------------------------------------------------------
# Token-level helpers (for F1 / EM)
# ------------------------------------------------------------------

def _normalize_answer(text: str) -> str:
    """Lower, strip punctuation, collapse whitespace — standard NLP normalisation."""
    text = text.lower()
    text = _re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def _token_counts(text: str) -> "_Counter[str]":
    return _Counter(_normalize_answer(text).split())


# ------------------------------------------------------------------
# Answer metrics
# ------------------------------------------------------------------

def compute_exact_match(predictions: List[str], ground_truths: List[str]) -> float:
    """Exact-match accuracy after normalisation (lower, strip punctuation).

    Standard EM metric used in SQuAD, BioASQ, MedQA.

    Parameters
    ----------
    predictions:
        Generated answer strings.
    ground_truths:
        Reference answer strings (one per sample).

    Returns
    -------
    float
        Fraction of samples where normalised prediction == normalised truth.
    """
    if not predictions or not ground_truths or len(predictions) != len(ground_truths):
        return 0.0
    hits = sum(
        1 for p, t in zip(predictions, ground_truths)
        if _normalize_answer(str(p or "")) == _normalize_answer(str(t or ""))
    )
    return round(hits / len(predictions), 4)


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between a single prediction and a single reference.

    Standard token-overlap F1 as used in SQuAD / BioASQ answer evaluation.

    Parameters
    ----------
    prediction:
        Generated answer string.
    ground_truth:
        Reference answer string.

    Returns
    -------
    float
        F1 score in ``[0, 1]``.
    """
    pred_tokens = _token_counts(str(prediction or ""))
    truth_tokens = _token_counts(str(ground_truth or ""))

    if not pred_tokens or not truth_tokens:
        return 0.0

    common = pred_tokens & truth_tokens
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / sum(pred_tokens.values())
    recall = num_common / sum(truth_tokens.values())
    return round(2 * precision * recall / (precision + recall), 4)


def compute_f1_score(predictions: List[str], ground_truths: List[str]) -> float:
    """Mean token-level F1 across a batch of prediction / reference pairs.

    Parameters
    ----------
    predictions:
        Generated answer strings.
    ground_truths:
        Reference answer strings (one per sample).

    Returns
    -------
    float
        Mean F1 in ``[0, 1]``.
    """
    if not predictions or not ground_truths or len(predictions) != len(ground_truths):
        return 0.0
    scores = [
        compute_token_f1(str(p or ""), str(t or ""))
        for p, t in zip(predictions, ground_truths)
    ]
    return round(sum(scores) / len(scores), 4)


# ------------------------------------------------------------------
# Retrieval metrics
# ------------------------------------------------------------------

def compute_mrr(
    retrieved_id_lists: List[List[str]],
    relevant_id_sets: List[Set[str]],
) -> float:
    """Mean Reciprocal Rank.

    MRR = mean over queries of 1/rank_of_first_relevant_doc.
    If no relevant document is retrieved for a query, the reciprocal
    rank for that query is 0.

    Parameters
    ----------
    retrieved_id_lists:
        One ordered list of retrieved document IDs per query.
    relevant_id_sets:
        One set of relevant document IDs per query.

    Returns
    -------
    float
        MRR in ``[0, 1]``.
    """
    if not retrieved_id_lists or not relevant_id_sets:
        return 0.0
    if len(retrieved_id_lists) != len(relevant_id_sets):
        return 0.0

    reciprocal_ranks: List[float] = []
    for retrieved, relevant in zip(retrieved_id_lists, relevant_id_sets):
        rr = 0.0
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)


def compute_ndcg_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int = 5,
) -> float:
    """Normalised Discounted Cumulative Gain at K for a single query.

    Binary relevance: 1 if the document is in ``relevant_ids``, else 0.

    Parameters
    ----------
    retrieved_ids:
        Ordered list of retrieved document IDs.
    relevant_ids:
        Set of relevant document IDs for this query.
    k:
        Cutoff rank.

    Returns
    -------
    float
        nDCG@K in ``[0, 1]``.
    """
    if not retrieved_ids or not relevant_ids or k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]

    # DCG
    dcg = sum(
        1.0 / _math.log2(rank + 1)
        for rank, doc_id in enumerate(top_k, start=1)
        if doc_id in relevant_ids
    )

    # Ideal DCG — best possible ordering
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / _math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0

    return round(dcg / idcg, 4)


def compute_mean_ndcg_at_k(
    retrieved_id_lists: List[List[str]],
    relevant_id_sets: List[Set[str]],
    k: int = 5,
) -> float:
    """Mean nDCG@K across a batch of queries.

    Parameters
    ----------
    retrieved_id_lists:
        One ordered list of retrieved IDs per query.
    relevant_id_sets:
        One set of relevant IDs per query.
    k:
        Cutoff rank.

    Returns
    -------
    float
        Mean nDCG@K in ``[0, 1]``.
    """
    if not retrieved_id_lists or not relevant_id_sets:
        return 0.0
    if len(retrieved_id_lists) != len(relevant_id_sets):
        return 0.0

    scores = [
        compute_ndcg_at_k(retrieved, relevant, k)
        for retrieved, relevant in zip(retrieved_id_lists, relevant_id_sets)
    ]
    return round(sum(scores) / len(scores), 4)


# ------------------------------------------------------------------
# Confidence-calibration metrics
# ------------------------------------------------------------------

def compute_ece(
    confidences: List[float],
    correct_flags: List[bool],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Measures the average gap between predicted confidence and empirical
    accuracy across equal-width confidence bins.  A perfectly calibrated
    model has ECE = 0.

    Parameters
    ----------
    confidences:
        Predicted confidence scores in ``[0, 1]`` — one per sample.
    correct_flags:
        Whether each sample's prediction was correct (True/False).
    n_bins:
        Number of equal-width bins in ``[0, 1]``.

    Returns
    -------
    float
        ECE in ``[0, 1]`` (lower is better).
    """
    if not confidences or not correct_flags:
        return 0.0
    if len(confidences) != len(correct_flags):
        return 0.0

    n = len(confidences)
    bin_width = 1.0 / n_bins
    ece = 0.0

    for bin_idx in range(n_bins):
        lo = bin_idx * bin_width
        hi = lo + bin_width

        # Include hi==1.0 in the last bin
        indices = [
            i for i, c in enumerate(confidences)
            if lo <= c < hi or (bin_idx == n_bins - 1 and c == 1.0)
        ]
        if not indices:
            continue

        bin_confidence = sum(confidences[i] for i in indices) / len(indices)
        bin_accuracy = sum(1 for i in indices if correct_flags[i]) / len(indices)
        ece += (len(indices) / n) * abs(bin_confidence - bin_accuracy)

    return round(ece, 4)


def compute_brier_score(
    confidences: List[float],
    correct_flags: List[bool],
) -> float:
    """Brier Score for probabilistic confidence calibration.

    Brier = mean squared error between predicted confidence and binary outcome.
    Range ``[0, 1]``; lower is better.  A random classifier scores 0.25.

    Parameters
    ----------
    confidences:
        Predicted confidence scores in ``[0, 1]``.
    correct_flags:
        Binary outcome per sample (True = correct answer).

    Returns
    -------
    float
        Brier score in ``[0, 1]``.
    """
    if not confidences or not correct_flags:
        return 0.0
    if len(confidences) != len(correct_flags):
        return 0.0

    brier = sum(
        (float(c) - float(y)) ** 2
        for c, y in zip(confidences, correct_flags)
    ) / len(confidences)

    return round(brier, 4)


# ------------------------------------------------------------------
# Abstention quality metrics
# ------------------------------------------------------------------

def compute_abstention_precision(
    responses: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
) -> float:
    """Abstention precision: fraction of abstentions that were warranted.

    A sample is "should abstain" when the ground truth marks it as
    unanswerable/out-of-scope (``ground_truth["answerable"] == False``).

    Parameters
    ----------
    responses:
        Pipeline responses, each with ``"response_type"`` field.
    ground_truths:
        Ground truth records with ``"answerable": bool`` field.

    Returns
    -------
    float
        Abstention precision in ``[0, 1]``.
    """
    if not responses or not ground_truths:
        return 0.0
    if len(responses) != len(ground_truths):
        return 0.0

    abstained_indices = [
        i for i, r in enumerate(responses)
        if r.get("response_type") == "abstention"
        or r.get("status") == "abstained"
    ]
    if not abstained_indices:
        return 0.0

    true_positives = sum(
        1 for i in abstained_indices
        if not ground_truths[i].get("answerable", True)
    )
    return round(true_positives / len(abstained_indices), 4)


def compute_abstention_recall(
    responses: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
) -> float:
    """Abstention recall: fraction of unanswerable questions that were abstained on.

    Parameters
    ----------
    responses:
        Pipeline responses.
    ground_truths:
        Ground truth records with ``"answerable": bool`` field.

    Returns
    -------
    float
        Abstention recall in ``[0, 1]``.
    """
    if not responses or not ground_truths:
        return 0.0
    if len(responses) != len(ground_truths):
        return 0.0

    unanswerable_indices = [
        i for i, g in enumerate(ground_truths)
        if not g.get("answerable", True)
    ]
    if not unanswerable_indices:
        return 0.0

    caught = sum(
        1 for i in unanswerable_indices
        if responses[i].get("response_type") == "abstention"
        or responses[i].get("status") == "abstained"
    )
    return round(caught / len(unanswerable_indices), 4)


def compute_selective_accuracy(
    predictions: List[str],
    ground_truths: List[str],
    responses: List[Dict[str, Any]],
) -> float:
    """Selective accuracy: accuracy on the subset of questions that were answered.

    Also known as "risk-coverage accuracy" — the model's accuracy when it
    chooses to answer rather than abstain.  Abstained samples are excluded
    from both numerator and denominator.

    Parameters
    ----------
    predictions:
        Generated answer strings (empty string for abstentions).
    ground_truths:
        Reference answer strings.
    responses:
        Pipeline response dicts (used to identify abstentions).

    Returns
    -------
    float
        Selective accuracy in ``[0, 1]``, or 0.0 if everything was abstained.
    """
    if not predictions or not ground_truths or not responses:
        return 0.0
    if len(predictions) != len(ground_truths) or len(predictions) != len(responses):
        return 0.0

    answered_indices = [
        i for i, r in enumerate(responses)
        if r.get("response_type") != "abstention"
        and r.get("status") != "abstained"
    ]
    if not answered_indices:
        return 0.0

    correct = sum(
        1 for i in answered_indices
        if _normalize_answer(str(predictions[i] or ""))
        == _normalize_answer(str(ground_truths[i] or ""))
    )
    return round(correct / len(answered_indices), 4)


# ------------------------------------------------------------------
# Updated evaluate_batch — now includes all research metrics
# ------------------------------------------------------------------

def evaluate_batch_full(
    predictions: List[Dict[str, Any]],
    ground_truths: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Full research-grade evaluation report over a batch of query results.

    Extends :func:`evaluate_batch` with F1, EM, abstention quality, and
    confidence-calibration metrics when ground-truth data is available.

    Parameters
    ----------
    predictions:
        Pipeline response dicts, each containing at minimum:
        ``response_type``, ``answer``, ``processing_time_ms``,
        ``citations``, and optionally ``confidence_score``.
    ground_truths:
        Optional reference records.  When provided, each record may contain:
        ``answer`` (str), ``answerable`` (bool), ``confidence`` (float).

    Returns
    -------
    Dict[str, Any]
        Consolidated metrics report.
    """
    # Start with the basic report
    from evaluation.metrics import evaluate_batch  # base implementation
    report = evaluate_batch(predictions, ground_truths)

    if not ground_truths or len(ground_truths) != len(predictions):
        return report

    pred_answers = [str(p.get("answer", "") or "") for p in predictions]
    true_answers = [str(g.get("answer", "") or "") for g in ground_truths]

    # --- Answer quality ---
    report["exact_match"] = compute_exact_match(pred_answers, true_answers)
    report["f1_score"] = compute_f1_score(pred_answers, true_answers)

    # --- Abstention quality ---
    report["abstention_precision"] = compute_abstention_precision(predictions, ground_truths)
    report["abstention_recall"] = compute_abstention_recall(predictions, ground_truths)
    report["selective_accuracy"] = compute_selective_accuracy(pred_answers, true_answers, predictions)

    # --- Confidence calibration (when predictions carry a confidence score) ---
    confidences = [
        float(p.get("confidence_score", 0.0) or 0.0)
        for p in predictions
    ]
    correct_flags = [
        _normalize_answer(str(p.get("answer", "") or ""))
        == _normalize_answer(str(g.get("answer", "") or ""))
        for p, g in zip(predictions, ground_truths)
    ]

    if any(c > 0.0 for c in confidences):
        report["ece"] = compute_ece(confidences, correct_flags)
        report["brier_score"] = compute_brier_score(confidences, correct_flags)

    return report
