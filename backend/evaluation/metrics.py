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
