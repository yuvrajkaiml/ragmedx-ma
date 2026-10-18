"""
Evaluation metrics for RAGMedX-MA benchmark runs.

Public API
----------
compute_rouge_l                  -- ROUGE-L F1 for a single (pred, truth) pair
compute_batch_rouge_l            -- mean ROUGE-L over a list of (pred, truth) pairs
compute_exact_containment_rate   -- renamed from the old compute_accuracy; bidirectional
                                    substring check, useful as a secondary signal for
                                    simple factual questions but NOT the primary metric
compute_abstention_rate          -- fraction of responses that are abstentions
compute_citation_support_rate    -- fraction of non-abstained responses carrying citations
compute_latency_metrics          -- avg / median / min / max latency in ms
evaluate_batch                   -- consolidated report; uses ROUGE-L as the primary
                                    accuracy metric, keeps exact-containment as a
                                    secondary signal
compute_answer_with_citation_rate
compute_retrieval_coverage

Phase 2 additions (Workstream B)
---------------------------------
accuracy                         -- fraction correct (abstentions count as wrong)
exact_match                      -- token-normalized exact string match
rouge_l                          -- LCS-based F1 (summary level, single pair)
recall_at_k                      -- |retrieved[:k] ∩ relevant| / |relevant|
precision_at_k                   -- |retrieved[:k] ∩ relevant| / k
mrr                              -- mean reciprocal rank
ndcg_at_k                        -- normalized DCG
ece                              -- Expected Calibration Error
brier_score                      -- mean squared error of confidence vs outcome
risk_coverage_curve              -- (coverage, risk) pairs over confidence thresholds
abstention_precision             -- TP_abstain / (TP_abstain + FP_abstain)
abstention_recall                -- TP_abstain / (TP_abstain + FN_abstain)
false_abstention_rate            -- abstained on answerable / total answerable
false_acceptance_rate            -- not abstained on unanswerable / total unanswerable
"""

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ---------------------------------------------------------------------------
# ROUGE-L (LCS-based, matching NurRAG methodology)
# ---------------------------------------------------------------------------

def _lcs_length(a: List[str], b: List[str]) -> int:
    """Return the length of the Longest Common Subsequence of token lists a and b.

    Uses a space-optimised DP approach (two rows only) so it scales to
    typical answer/reference lengths without materialising the full matrix.
    """
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0
    prev = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr = [0] * (lb + 1)
        for j in range(1, lb + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[lb]


def compute_rouge_l(prediction: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-L precision, recall, and F1 for a single (prediction, reference) pair.

    Methodology (matches NurRAG):
      RA (recall)    = LCS / len(reference_tokens)
      PA (precision) = LCS / len(prediction_tokens)
      F1             = 2 * RA * PA / (RA + PA)   [0 if both are 0]

    Abstentions naturally score 0: an empty prediction has 0 LCS with any reference,
    so RA = PA = F1 = 0 with no special casing needed.

    Parameters
    ----------
    prediction:
        The model's generated answer string.
    reference:
        The ground-truth answer string.

    Returns
    -------
    dict with keys ``rouge_l_precision``, ``rouge_l_recall``, ``rouge_l_f1``
    """
    pred_tokens = str(prediction or "").lower().split()
    ref_tokens  = str(reference  or "").lower().split()

    if not pred_tokens or not ref_tokens:
        return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_f1": 0.0}

    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall    = lcs / len(ref_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "rouge_l_precision": round(precision, 4),
        "rouge_l_recall":    round(recall,    4),
        "rouge_l_f1":        round(f1,        4),
    }


def compute_batch_rouge_l(
    predictions: List[str],
    references:  List[str],
) -> Dict[str, float]:
    """Compute mean ROUGE-L precision, recall, and F1 over a batch.

    Abstentions (empty ``predictions`` entries) naturally score 0.

    Returns
    -------
    dict with keys ``mean_rouge_l_precision``, ``mean_rouge_l_recall``,
    ``mean_rouge_l_f1``, ``scores_per_sample`` (list of per-sample dicts)
    """
    if not predictions or len(predictions) != len(references):
        return {
            "mean_rouge_l_precision": 0.0,
            "mean_rouge_l_recall":    0.0,
            "mean_rouge_l_f1":        0.0,
            "scores_per_sample":      [],
        }

    per_sample = [compute_rouge_l(p, r) for p, r in zip(predictions, references)]
    n = len(per_sample)
    return {
        "mean_rouge_l_precision": round(sum(s["rouge_l_precision"] for s in per_sample) / n, 4),
        "mean_rouge_l_recall":    round(sum(s["rouge_l_recall"]    for s in per_sample) / n, 4),
        "mean_rouge_l_f1":        round(sum(s["rouge_l_f1"]        for s in per_sample) / n, 4),
        "scores_per_sample":      per_sample,
    }


# ---------------------------------------------------------------------------
# Exact containment (renamed from the old compute_accuracy)
# ---------------------------------------------------------------------------

def compute_exact_containment_rate(
    predictions: List[str],
    ground_truths: List[str],
) -> float:
    """Bidirectional substring containment rate.

    Returns the fraction of samples where the cleaned prediction string contains
    the cleaned truth string, OR the truth string contains the prediction string.
    Useful as a secondary signal for simple single-token/phrase factual answers.

    IMPORTANT: this function is NOT the primary accuracy metric.  An empty
    prediction (abstention) always returns False because ``if not pred_clean``
    short-circuits before the containment check, fixing the original bug where
    ``"" in truth`` evaluated to True and counted every abstention as correct.

    The previous function ``compute_accuracy`` had that bug; it is replaced by
    this function.  Any callers that used ``compute_accuracy`` and expected the
    old (broken) behaviour should be updated.
    """
    if not predictions or not ground_truths or len(predictions) != len(ground_truths):
        return 0.0

    correct = 0
    for pred, truth in zip(predictions, ground_truths):
        pred_clean  = str(pred  or "").strip().lower()
        truth_clean = str(truth or "").strip().lower()
        # Require a non-empty prediction — abstentions are never "correct" here
        if pred_clean and (truth_clean in pred_clean or pred_clean in truth_clean):
            correct += 1

    return round(correct / len(predictions), 4)


# ---------------------------------------------------------------------------
# Keep the old name as a deprecated alias so existing callers don't break
# ---------------------------------------------------------------------------

def compute_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """Deprecated alias for compute_exact_containment_rate.

    The original implementation counted every abstention (empty prediction) as
    correct because ``"" in truth_string`` is always True in Python.  That bug
    is fixed here: this function now delegates to compute_exact_containment_rate
    which requires a non-empty prediction.

    Prefer compute_rouge_l / compute_batch_rouge_l for the primary accuracy
    metric in new code.
    """
    return compute_exact_containment_rate(predictions, ground_truths)


# ---------------------------------------------------------------------------
# Abstention rate
# ---------------------------------------------------------------------------

def compute_abstention_rate(responses: List[Dict[str, Any]]) -> float:
    """Fraction of pipeline responses that are abstentions."""
    if not responses:
        return 0.0
    abstained = sum(
        1 for r in responses
        if r.get("response_type") == "abstention" or r.get("status") == "abstained"
    )
    return round(abstained / len(responses), 4)


# ---------------------------------------------------------------------------
# Citation support rate — fixed to handle stored result format
# ---------------------------------------------------------------------------

# Matches citation markers embedded in predicted_answer text:
#   [Doc 1], [E1], [E12], [Doc 12]
_CITATION_MARKER_RE = re.compile(r"\[(?:Doc|E)\s*\d+\]", re.IGNORECASE)


def compute_citation_support_rate(responses: List[Dict[str, Any]]) -> float:
    """Fraction of responses that carry at least one evidence citation.

    Handles two storage formats:
    1. Structured: ``r["citations"]`` or ``r["answer_citations"]`` is a non-empty list.
    2. Inline: citation markers (``[Doc N]``, ``[EN]``) are embedded in
       ``r["predicted_answer"]`` or ``r["answer"]``.  This is the format used
       by ``mirage_runner.run_rag_evaluation``'s ``per_sample_results`` records.

    Abstentions (empty predicted_answer) correctly return no citations.
    """
    if not responses:
        return 0.0

    cited_count = 0
    for r in responses:
        # Format 1: explicit citations list
        citations = r.get("citations", []) or r.get("answer_citations", [])
        if citations:
            cited_count += 1
            continue
        # Format 2: inline markers in the answer text
        answer_text = str(r.get("predicted_answer") or r.get("answer") or "")
        if _CITATION_MARKER_RE.search(answer_text):
            cited_count += 1

    return round(cited_count / len(responses), 4)


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

def compute_latency_metrics(latencies_ms: List[Union[int, float]]) -> Dict[str, float]:
    """Compute average, median, min, max latency in milliseconds."""
    if not latencies_ms:
        return {"avg_ms": 0.0, "median_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}

    sorted_l = sorted(latencies_ms)
    n = len(sorted_l)
    avg    = sum(sorted_l) / n
    median = (sorted_l[n // 2] + sorted_l[(n - 1) // 2]) / 2.0

    return {
        "avg_ms":    round(avg,         2),
        "median_ms": round(median,      2),
        "min_ms":    round(sorted_l[0], 2),
        "max_ms":    round(sorted_l[-1],2),
    }


# ---------------------------------------------------------------------------
# Consolidated batch evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(
    predictions:   List[Dict[str, Any]],
    ground_truths: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute a consolidated evaluation metrics report over a batch of query results.

    Primary accuracy metric: ROUGE-L F1 (mean over all samples; abstentions score 0).
    Secondary signal:        exact_containment_rate (non-empty predictions only).
    Both are reported alongside abstention_rate so they can be interpreted
    independently — a high abstention_rate with a moderate ROUGE-L on answered
    questions tells a different story than a low abstention_rate with low ROUGE-L.
    """
    total_samples = len(predictions)
    if total_samples == 0:
        return {"total_samples": 0, "status": "empty"}

    latencies      = [p.get("processing_time_ms", 0) for p in predictions]
    abstention_rate = compute_abstention_rate(predictions)
    citation_rate   = compute_citation_support_rate(predictions)
    latency_stats   = compute_latency_metrics(latencies)

    report: Dict[str, Any] = {
        "total_samples":        total_samples,
        "abstention_rate":      abstention_rate,
        "citation_support_rate": citation_rate,
        "latency_stats":        latency_stats,
    }

    if ground_truths and len(ground_truths) == total_samples:
        pred_answers = [str(p.get("answer", "")) for p in predictions]
        true_answers = [str(g.get("answer", "")) for g in ground_truths]

        rouge = compute_batch_rouge_l(pred_answers, true_answers)
        report["rouge_l_f1"]        = rouge["mean_rouge_l_f1"]
        report["rouge_l_precision"] = rouge["mean_rouge_l_precision"]
        report["rouge_l_recall"]    = rouge["mean_rouge_l_recall"]

        # Secondary signal — non-empty predictions only (abstentions score False)
        report["exact_containment_rate"] = compute_exact_containment_rate(
            pred_answers, true_answers
        )

        # Keep the old "accuracy" key populated for backwards compatibility with
        # any downstream code that reads it, but point it at ROUGE-L F1 now.
        # Callers that want the old broken behaviour must use the raw function.
        report["accuracy"] = rouge["mean_rouge_l_f1"]

    return report


# ---------------------------------------------------------------------------
# Answer-with-citation rate (non-abstained answers only)
# ---------------------------------------------------------------------------

def compute_answer_with_citation_rate(responses: List[Dict[str, Any]]) -> float:
    """Proportion of non-abstained responses that contain at least one evidence citation.

    Uses the same dual-format detection as compute_citation_support_rate.
    Abstention responses are excluded from the denominator.
    """
    if not responses:
        return 0.0

    answered = [
        r for r in responses
        if r.get("response_type") not in ("abstention",)
        and r.get("status") not in ("abstained",)
    ]
    if not answered:
        return 0.0

    cited = 0
    for r in answered:
        citations = r.get("citations", []) or r.get("answer_citations", [])
        if citations:
            cited += 1
            continue
        answer_text = str(r.get("predicted_answer") or r.get("answer") or "")
        if _CITATION_MARKER_RE.search(answer_text):
            cited += 1

    return round(cited / len(answered), 4)


# ---------------------------------------------------------------------------
# Precision / Recall @ K
# ---------------------------------------------------------------------------

def compute_precision_at_k(
    retrieved_ids: List[str],
    relevant_ids:  Set[str],
    k: int = 5,
) -> float:
    """Precision@K: proportion of top-K retrieved items that are relevant."""
    if not retrieved_ids or not relevant_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits  = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return round(hits / min(k, len(top_k)), 4)


def compute_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids:  Set[str],
    k: int = 5,
) -> float:
    """Recall@K: proportion of relevant items that appear in top-K."""
    if not retrieved_ids or not relevant_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits  = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return round(hits / len(relevant_ids), 4)


# ---------------------------------------------------------------------------
# Retrieval coverage
# ---------------------------------------------------------------------------

def compute_retrieval_coverage(
    evidence_lists:       List[List[str]],
    relevant_keyword_sets: List[Set[str]],
) -> float:
    """Proportion of samples where at least one retrieved passage contains
    a keyword from the expected relevant set (case-insensitive)."""
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


# ===========================================================================
# Phase 2 — Workstream B additions
# ===========================================================================

def accuracy(predictions: List[str], references: List[str]) -> float:
    """Fraction of questions where predicted answer matches reference.

    Definition: For each sample, a match is declared when the stripped,
    lowercased prediction equals the stripped, lowercased reference.
    Abstentions (empty-string predictions) never match and therefore count
    as wrong.

    Denominator: total questions (including abstentions counted as wrong).
    i.e. denominator = len(predictions).

    Parameters
    ----------
    predictions : list of str
        Model predictions; use empty string "" for abstentions.
    references : list of str
        Ground-truth answers; must be same length as predictions.

    Returns
    -------
    float in [0, 1]
    """
    if not predictions or not references or len(predictions) != len(references):
        return 0.0
    correct = sum(
        1 for p, r in zip(predictions, references)
        if str(p or "").strip().lower() == str(r or "").strip().lower()
        and str(p or "").strip() != ""
    )
    return round(correct / len(predictions), 4)


def exact_match(prediction: str, reference: str) -> bool:
    """Token-normalised exact string match for a single (prediction, reference) pair.

    Normalisation steps (applied to both strings):
      1. Convert to lowercase.
      2. Strip leading/trailing whitespace.
      3. Collapse internal whitespace runs to a single space.
      4. Strip punctuation at word boundaries (keeps letters, digits, spaces).

    An empty (abstained) prediction never matches.

    Parameters
    ----------
    prediction : str
        Model output; empty string for abstentions.
    reference : str
        Ground-truth answer.

    Returns
    -------
    bool — True iff normalised strings are equal and prediction is non-empty.
    """
    def _normalise(s: str) -> str:
        s = str(s or "").lower().strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^\w\s]", "", s)  # remove non-alphanumeric/space
        return s.strip()

    pred_norm = _normalise(prediction)
    ref_norm  = _normalise(reference)
    return bool(pred_norm) and pred_norm == ref_norm


def rouge_l(prediction: str, reference: str) -> Dict[str, float]:
    """LCS-based ROUGE-L F1 at the *summary* level for a single pair.

    Level: summary-level (treats entire string as one sequence of tokens;
    not sentence-by-sentence). This matches the NurRAG convention already
    used throughout this module.

    Gain formula:  LCS count over whitespace-split tokens.
    Discount:      none (pure LCS, not position-weighted).

    precision = LCS / len(prediction_tokens)
    recall    = LCS / len(reference_tokens)
    F1        = 2 * precision * recall / (precision + recall)  [0 if denom=0]

    Denominator for precision: len(prediction_tokens).
    Denominator for recall:    len(reference_tokens).

    Parameters
    ----------
    prediction : str
    reference  : str

    Returns
    -------
    dict with keys ``rouge_l_precision``, ``rouge_l_recall``, ``rouge_l_f1``
    """
    return compute_rouge_l(prediction, reference)


def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """Recall@K for a single query.

    Definition: number of relevant items in the top-K retrieved list divided
    by the total number of relevant items.

    Denominator: |relevant_ids| (total relevant items for the query).

    Handles the edge case where |relevant_ids| > K: Recall@K is capped at
    |relevant_ids ∩ retrieved[:k]| / |relevant_ids| which may be < 1 even if
    all K slots are relevant.

    Parameters
    ----------
    retrieved_ids : list of str
        Ordered list of retrieved item IDs (ranked from most to least relevant).
    relevant_ids  : list of str
        Ground-truth relevant item IDs for the query (order irrelevant).
    k             : int
        Cutoff rank (must be >= 1).

    Returns
    -------
    float in [0, 1]; 0.0 if relevant_ids is empty or k < 1.
    """
    if not relevant_ids or k < 1:
        return 0.0
    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return round(hits / len(relevant_set), 4)


def precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """Precision@K for a single query.

    Definition: number of relevant items in the top-K retrieved list divided
    by K.

    Denominator: k (always k, even if fewer than k items were retrieved).
    If fewer than k items are available the missing slots are treated as
    non-relevant (precision is penalised).

    Parameters
    ----------
    retrieved_ids : list of str
        Ordered list of retrieved item IDs.
    relevant_ids  : list of str
        Ground-truth relevant item IDs.
    k             : int
        Cutoff rank (must be >= 1).

    Returns
    -------
    float in [0, 1]; 0.0 if k < 1 or relevant_ids is empty.
    """
    if not relevant_ids or k < 1:
        return 0.0
    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    # Denominator is always k, not len(top_k)
    return round(hits / k, 4)


def mrr(
    retrieved_ids_list: List[List[str]],
    relevant_ids_list: List[List[str]],
) -> float:
    """Mean Reciprocal Rank across multiple queries.

    Definition: MRR = (1/|Q|) * sum_q [ 1 / rank_q ]
    where rank_q is the rank (1-based) of the first relevant item retrieved
    for query q.  If no relevant item is retrieved, the reciprocal rank for
    that query is 0.

    Denominator: number of queries (len(retrieved_ids_list)).

    Parameters
    ----------
    retrieved_ids_list : list of list of str
        Per-query ordered lists of retrieved item IDs.
    relevant_ids_list  : list of list of str
        Per-query ground-truth relevant item ID lists.
        Must be same length as retrieved_ids_list.

    Returns
    -------
    float in [0, 1]; 0.0 if list is empty.
    """
    if not retrieved_ids_list or len(retrieved_ids_list) != len(relevant_ids_list):
        return 0.0
    total_rr = 0.0
    for retrieved, relevant in zip(retrieved_ids_list, relevant_ids_list):
        relevant_set = set(relevant)
        rr = 0.0
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant_set:
                rr = 1.0 / rank
                break
        total_rr += rr
    return round(total_rr / len(retrieved_ids_list), 4)


def ndcg_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at K for a single query.

    Gain formula: binary relevance — gain(i) = 1 if retrieved_ids[i] is
    relevant else 0.

    Discount formula: log2(rank + 1) where rank is 1-based position.
    So DCG = sum_{i=1}^{k} gain(i) / log2(i + 1)

    Ideal DCG (IDCG): places all relevant items at the top ranks, up to
    min(k, |relevant_ids|) positions.

    nDCG@K = DCG@K / IDCG@K.  Returns 0.0 when IDCG = 0 (no relevant items).

    Parameters
    ----------
    retrieved_ids : list of str
        Ordered list of retrieved item IDs.
    relevant_ids  : list of str
        Ground-truth relevant item IDs (binary relevance: in set = 1, else 0).
    k             : int
        Cutoff rank (must be >= 1).

    Returns
    -------
    float in [0, 1]; 0.0 if relevant_ids is empty or k < 1.
    """
    if not relevant_ids or k < 1:
        return 0.0
    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]

    # Actual DCG
    dcg = sum(
        (1.0 / math.log2(rank + 1))
        for rank, doc_id in enumerate(top_k, start=1)
        if doc_id in relevant_set
    )

    # Ideal DCG: best possible placement of min(k, |relevant|) relevant items
    ideal_hits = min(k, len(relevant_set))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0
    return round(dcg / idcg, 4)


def ece(
    confidences: List[float],
    correct_flags: List[bool],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Definition: ECE = sum_{b=1}^{B} (|B_b| / n) * |acc(B_b) - conf(B_b)|

    where B_b is the set of samples whose confidence falls in bin b,
    acc(B_b) is the fraction of those samples that are correct,
    and conf(B_b) is the mean confidence of those samples.

    Bins are equal-width over [0, 1].

    Denominator: total samples n (counting all samples in the weighted sum).

    Parameters
    ----------
    confidences   : list of float in [0, 1]
        Model-reported confidence per sample.
    correct_flags : list of bool
        Whether each prediction was correct.
    n_bins        : int
        Number of equal-width calibration bins. Default 10.

    Returns
    -------
    float in [0, 1].  0.0 on empty input.
    """
    if not confidences or len(confidences) != len(correct_flags) or n_bins < 1:
        return 0.0
    n = len(confidences)
    bin_width = 1.0 / n_bins

    total_ece = 0.0
    for b in range(n_bins):
        lo = b * bin_width
        hi = lo + bin_width
        # Last bin is closed on right
        if b == n_bins - 1:
            indices = [i for i, c in enumerate(confidences) if lo <= c <= hi]
        else:
            indices = [i for i, c in enumerate(confidences) if lo <= c < hi]
        if not indices:
            continue
        bin_acc  = sum(1 for i in indices if correct_flags[i]) / len(indices)
        bin_conf = sum(confidences[i] for i in indices) / len(indices)
        total_ece += (len(indices) / n) * abs(bin_acc - bin_conf)

    return round(total_ece, 6)


def brier_score(
    confidences: List[float],
    correct_flags: List[bool],
) -> float:
    """Brier Score — mean squared error of confidence vs binary outcome.

    Definition: BS = (1/n) * sum_{i=1}^{n} (confidence_i - outcome_i)^2

    where outcome_i = 1 if correct_flags[i] is True else 0.

    Denominator: total samples n.

    Perfect calibration with all correct → BS = 0.
    Worst case (all wrong with confidence 1) → BS = 1.
    Lower is better.

    Parameters
    ----------
    confidences   : list of float in [0, 1]
    correct_flags : list of bool

    Returns
    -------
    float in [0, 1].  0.0 on empty input.
    """
    if not confidences or len(confidences) != len(correct_flags):
        return 0.0
    n = len(confidences)
    total = sum(
        (conf - (1.0 if flag else 0.0)) ** 2
        for conf, flag in zip(confidences, correct_flags)
    )
    return round(total / n, 6)


def risk_coverage_curve(
    confidences: List[float],
    correct_flags: List[bool],
) -> List[Tuple[float, float]]:
    """Compute (coverage, risk) pairs sweeping confidence thresholds.

    Coverage: fraction of samples with confidence >= threshold that the model
    answers (i.e. does not abstain by filtering below the threshold).
    Equivalent to: |{i : confidence_i >= threshold}| / n.

    Risk (error rate on the answered subset): fraction of answered samples
    that are incorrect.
    risk = |{i : confidence_i >= threshold AND NOT correct_flags[i]}|
           / |{i : confidence_i >= threshold}|
    Risk = 0.0 when all answered samples are correct; 1.0 when all are wrong.

    Thresholds are drawn from the sorted unique confidence values plus a
    sentinel above 1.0 (which yields coverage=0).

    Parameters
    ----------
    confidences   : list of float in [0, 1]
    correct_flags : list of bool

    Returns
    -------
    List of (coverage, risk) tuples sorted by descending coverage (i.e.
    from lowest threshold to highest threshold).
    0.0 on empty input returns [].
    """
    if not confidences or len(confidences) != len(correct_flags):
        return []
    n = len(confidences)
    thresholds = sorted(set(confidences))
    # Add a sentinel so that the last point represents zero coverage
    thresholds.append(max(thresholds) + 1e-9)

    curve: List[Tuple[float, float]] = []
    for thresh in thresholds:
        answered_idx = [i for i, c in enumerate(confidences) if c >= thresh]
        coverage = len(answered_idx) / n
        if not answered_idx:
            curve.append((0.0, 0.0))
            continue
        errors = sum(1 for i in answered_idx if not correct_flags[i])
        risk = errors / len(answered_idx)
        curve.append((round(coverage, 4), round(risk, 4)))

    return curve


def abstention_precision(
    abstained_flags: List[bool],
    should_abstain_flags: List[bool],
) -> float:
    """Abstention precision — fraction of model abstentions that were correct.

    Definition: TP_abstain / (TP_abstain + FP_abstain)

    TP_abstain: model abstained AND should have abstained.
    FP_abstain: model abstained AND should have answered (false abstention).

    Denominator: total model abstentions (TP_abstain + FP_abstain).
    Returns 0.0 if model never abstained.

    Parameters
    ----------
    abstained_flags     : list of bool  — True if model abstained
    should_abstain_flags: list of bool  — True if ground truth requires abstention

    Returns
    -------
    float in [0, 1]; 0.0 when no abstentions were made.
    """
    if not abstained_flags or len(abstained_flags) != len(should_abstain_flags):
        return 0.0
    tp = sum(1 for a, s in zip(abstained_flags, should_abstain_flags) if a and s)
    fp = sum(1 for a, s in zip(abstained_flags, should_abstain_flags) if a and not s)
    denom = tp + fp
    return round(tp / denom, 4) if denom > 0 else 0.0


def abstention_recall(
    abstained_flags: List[bool],
    should_abstain_flags: List[bool],
) -> float:
    """Abstention recall — fraction of cases that required abstention where the model did abstain.

    Definition: TP_abstain / (TP_abstain + FN_abstain)

    TP_abstain: model abstained AND should have abstained.
    FN_abstain: model answered AND should have abstained (false acceptance).

    Denominator: total cases requiring abstention (TP_abstain + FN_abstain).
    Returns 0.0 if no cases require abstention.

    Parameters
    ----------
    abstained_flags     : list of bool  — True if model abstained
    should_abstain_flags: list of bool  — True if ground truth requires abstention

    Returns
    -------
    float in [0, 1]; 0.0 when no cases require abstention.
    """
    if not abstained_flags or len(abstained_flags) != len(should_abstain_flags):
        return 0.0
    tp = sum(1 for a, s in zip(abstained_flags, should_abstain_flags) if a and s)
    fn = sum(1 for a, s in zip(abstained_flags, should_abstain_flags) if not a and s)
    denom = tp + fn
    return round(tp / denom, 4) if denom > 0 else 0.0


def false_abstention_rate(
    abstained_flags: List[bool],
    should_abstain_flags: List[bool],
) -> float:
    """False abstention rate — fraction of answerable questions where model abstained.

    Definition: FP_abstain / total_answerable

    FP_abstain: model abstained AND should have answered (should_abstain=False).
    total_answerable: total questions where should_abstain=False.

    Denominator: total answerable questions.
    Returns 0.0 if no answerable questions exist.

    Parameters
    ----------
    abstained_flags     : list of bool  — True if model abstained
    should_abstain_flags: list of bool  — True if ground truth requires abstention

    Returns
    -------
    float in [0, 1]; 0.0 when no answerable questions exist.
    """
    if not abstained_flags or len(abstained_flags) != len(should_abstain_flags):
        return 0.0
    fp = sum(1 for a, s in zip(abstained_flags, should_abstain_flags) if a and not s)
    total_answerable = sum(1 for s in should_abstain_flags if not s)
    return round(fp / total_answerable, 4) if total_answerable > 0 else 0.0


def false_acceptance_rate(
    abstained_flags: List[bool],
    should_abstain_flags: List[bool],
) -> float:
    """False acceptance rate — fraction of unanswerable questions where model answered.

    Definition: FN_abstain / total_unanswerable

    FN_abstain: model did NOT abstain AND should have abstained (should_abstain=True).
    total_unanswerable: total questions where should_abstain=True.

    Denominator: total unanswerable questions.
    Returns 0.0 if no unanswerable questions exist.

    Parameters
    ----------
    abstained_flags     : list of bool  — True if model abstained
    should_abstain_flags: list of bool  — True if ground truth requires abstention

    Returns
    -------
    float in [0, 1]; 0.0 when no unanswerable questions exist.
    """
    if not abstained_flags or len(abstained_flags) != len(should_abstain_flags):
        return 0.0
    fn = sum(1 for a, s in zip(abstained_flags, should_abstain_flags) if not a and s)
    total_unanswerable = sum(1 for s in should_abstain_flags if s)
    return round(fn / total_unanswerable, 4) if total_unanswerable > 0 else 0.0


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
