"""
Phase 2 Workstream B — Unit tests for metrics.py and statistics.py.

All expected values are hand-computable; computations are shown inline.
Evidence tier: SOFTWARE TEST (no model execution, no FAISS, no LLM).
"""

import math
import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from backend.evaluation.metrics import (
    accuracy,
    exact_match,
    rouge_l,
    recall_at_k,
    precision_at_k,
    mrr,
    ndcg_at_k,
    ece,
    brier_score,
    risk_coverage_curve,
    abstention_precision,
    abstention_recall,
    false_abstention_rate,
    false_acceptance_rate,
)
from backend.evaluation.statistics import (
    wilson_ci,
    mcnemar_test,
    paired_bootstrap,
    holm_correction,
    format_result,
)


# ===========================================================================
# 1. accuracy
# ===========================================================================

class TestAccuracy:
    """accuracy(predictions, references)

    Definition: fraction where strip().lower() prediction == strip().lower() reference.
    Abstentions (empty string) never match.
    Denominator: total questions (including abstentions).
    """

    def test_normal_case(self):
        # 2 correct out of 4 (0.5)
        preds = ["yes", "no", "maybe", ""]
        refs  = ["yes", "yes", "maybe", "no"]
        # "yes"=="yes" ✓, "no"!="yes" ✗, "maybe"=="maybe" ✓, ""≠any ✗
        # correct=2, total=4 → 0.5
        assert accuracy(preds, refs) == pytest.approx(0.5, abs=1e-4)

    def test_abstentions_counted_wrong(self):
        # All abstentions should give 0.0
        preds = ["", "", ""]
        refs  = ["yes", "no", "maybe"]
        assert accuracy(preds, refs) == pytest.approx(0.0, abs=1e-4)

    def test_all_correct(self):
        preds = ["aspirin", "ibuprofen"]
        refs  = ["aspirin", "ibuprofen"]
        assert accuracy(preds, refs) == pytest.approx(1.0, abs=1e-4)

    def test_empty_input(self):
        assert accuracy([], []) == 0.0

    def test_case_insensitive(self):
        preds = ["Aspirin"]
        refs  = ["aspirin"]
        assert accuracy(preds, refs) == pytest.approx(1.0, abs=1e-4)

    def test_whitespace_stripped(self):
        preds = ["  yes  "]
        refs  = ["yes"]
        assert accuracy(preds, refs) == pytest.approx(1.0, abs=1e-4)


# ===========================================================================
# 2. exact_match
# ===========================================================================

class TestExactMatch:
    """exact_match(prediction, reference) → bool

    Normalise: lowercase, strip, collapse whitespace, strip punctuation.
    Empty prediction → False always.
    """

    def test_simple_match(self):
        assert exact_match("aspirin", "aspirin") is True

    def test_case_normalised(self):
        assert exact_match("ASPIRIN", "aspirin") is True

    def test_punctuation_stripped(self):
        assert exact_match("aspirin.", "aspirin") is True

    def test_whitespace_collapsed(self):
        assert exact_match("  heart   disease  ", "heart disease") is True

    def test_mismatch(self):
        assert exact_match("ibuprofen", "aspirin") is False

    def test_empty_prediction(self):
        assert exact_match("", "aspirin") is False

    def test_both_empty(self):
        assert exact_match("", "") is False


# ===========================================================================
# 3. rouge_l
# ===========================================================================

class TestRougeL:
    """rouge_l(prediction, reference) → dict with rouge_l_f1 etc.

    Summary-level ROUGE-L: single LCS over full token sequences.
    precision = LCS / len(pred_tokens)
    recall    = LCS / len(ref_tokens)
    F1        = 2 * P * R / (P + R)
    """

    def test_identical_strings(self):
        # LCS = 3 tokens, P=1, R=1, F1=1
        result = rouge_l("the cat sat", "the cat sat")
        assert result["rouge_l_f1"] == pytest.approx(1.0, abs=1e-4)

    def test_known_values(self):
        # pred="the cat" (2 tokens), ref="the cat sat" (3 tokens)
        # LCS("the","cat") vs ("the","cat","sat") = ["the","cat"] → LCS=2
        # P = 2/2 = 1.0, R = 2/3 = 0.6667, F1 = 2*1.0*0.6667/(1.0+0.6667)
        # F1 = 1.3333/1.6667 = 0.8000
        result = rouge_l("the cat", "the cat sat")
        assert result["rouge_l_precision"] == pytest.approx(1.0, abs=1e-3)
        assert result["rouge_l_recall"]    == pytest.approx(0.6667, abs=1e-3)
        assert result["rouge_l_f1"]        == pytest.approx(0.8000, abs=1e-3)

    def test_empty_prediction(self):
        result = rouge_l("", "the cat sat")
        assert result["rouge_l_f1"] == pytest.approx(0.0, abs=1e-4)

    def test_empty_reference(self):
        result = rouge_l("some answer", "")
        assert result["rouge_l_f1"] == pytest.approx(0.0, abs=1e-4)

    def test_no_overlap(self):
        result = rouge_l("dog runs fast", "cat sleeps slowly")
        assert result["rouge_l_f1"] == pytest.approx(0.0, abs=1e-4)


# ===========================================================================
# 4. recall_at_k
# ===========================================================================

class TestRecallAtK:
    """recall_at_k(retrieved_ids, relevant_ids, k) → float

    Definition: |retrieved[:k] ∩ relevant| / |relevant|
    Denominator: |relevant|
    """

    def test_normal_case(self):
        # retrieved=["a","b","c","d"], relevant=["a","c"], k=3
        # top_3=["a","b","c"], hits=2 (a and c), |relevant|=2 → 2/2=1.0
        assert recall_at_k(["a","b","c","d"], ["a","c"], 3) == pytest.approx(1.0, abs=1e-4)

    def test_partial_recall(self):
        # retrieved=["a","b","c","d"], relevant=["a","c"], k=1
        # top_1=["a"], hits=1, |relevant|=2 → 1/2=0.5
        assert recall_at_k(["a","b","c","d"], ["a","c"], 1) == pytest.approx(0.5, abs=1e-4)

    def test_empty_relevant(self):
        assert recall_at_k(["a","b"], [], 5) == pytest.approx(0.0, abs=1e-4)

    def test_k_zero(self):
        assert recall_at_k(["a","b"], ["a"], 0) == pytest.approx(0.0, abs=1e-4)

    def test_none_retrieved_in_top_k(self):
        # retrieved=["d","e","f"], relevant=["a","b"], k=3 → hits=0 → 0/2=0.0
        assert recall_at_k(["d","e","f"], ["a","b"], 3) == pytest.approx(0.0, abs=1e-4)


# ===========================================================================
# 5. precision_at_k
# ===========================================================================

class TestPrecisionAtK:
    """precision_at_k(retrieved_ids, relevant_ids, k) → float

    Definition: |retrieved[:k] ∩ relevant| / k
    Denominator: k (always k, not len(top_k))
    """

    def test_normal_case(self):
        # retrieved=["a","b","c","d"], relevant=["a","c"], k=3
        # top_3=["a","b","c"], hits=2 (a and c), k=3 → 2/3=0.6667
        result = precision_at_k(["a","b","c","d"], ["a","c"], 3)
        assert result == pytest.approx(2/3, abs=1e-4)

    def test_all_relevant(self):
        # retrieved=["a","b"], relevant=["a","b"], k=2 → 2/2=1.0
        assert precision_at_k(["a","b"], ["a","b"], 2) == pytest.approx(1.0, abs=1e-4)

    def test_denominator_is_k_not_retrieved(self):
        # retrieved=["a"], relevant=["a","b"], k=5
        # Only 1 item retrieved but k=5 → 1/5=0.2 (penalised for missing items)
        assert precision_at_k(["a"], ["a","b"], 5) == pytest.approx(0.2, abs=1e-4)

    def test_empty_relevant(self):
        assert precision_at_k(["a","b"], [], 3) == pytest.approx(0.0, abs=1e-4)

    def test_no_hits(self):
        assert precision_at_k(["d","e"], ["a","b"], 2) == pytest.approx(0.0, abs=1e-4)


# ===========================================================================
# 6. mrr
# ===========================================================================

class TestMRR:
    """mrr(retrieved_ids_list, relevant_ids_list) → float

    Definition: mean over queries of 1/rank_q where rank_q = position of
    first relevant item (1-based). 0 if no relevant item retrieved.
    Denominator: number of queries.
    """

    def test_single_query_first(self):
        # retrieved=["a","b","c"], relevant=["a"] → rank=1 → RR=1.0
        assert mrr([["a","b","c"]], [["a"]]) == pytest.approx(1.0, abs=1e-4)

    def test_single_query_second(self):
        # retrieved=["b","a","c"], relevant=["a"] → rank=2 → RR=0.5
        assert mrr([["b","a","c"]], [["a"]]) == pytest.approx(0.5, abs=1e-4)

    def test_two_queries_mixed(self):
        # Q1: retrieved=["a","b"], relevant=["a"] → RR=1.0
        # Q2: retrieved=["b","c"], relevant=["a"] → RR=0.0 (not found)
        # MRR = (1.0 + 0.0) / 2 = 0.5
        retrieved_list = [["a","b"], ["b","c"]]
        relevant_list  = [["a"],     ["a"]]
        assert mrr(retrieved_list, relevant_list) == pytest.approx(0.5, abs=1e-4)

    def test_empty_input(self):
        assert mrr([], []) == pytest.approx(0.0, abs=1e-4)

    def test_no_relevant_found(self):
        assert mrr([["d","e","f"]], [["a","b"]]) == pytest.approx(0.0, abs=1e-4)

    def test_three_queries(self):
        # Q1: rank=1 → RR=1.0
        # Q2: rank=3 → RR=1/3
        # Q3: rank=2 → RR=0.5
        # MRR = (1.0 + 1/3 + 0.5)/3 = 1.8333/3 = 0.6111
        retrieved_list = [["a","b","c"], ["b","c","a"], ["b","a","c"]]
        relevant_list  = [["a"],         ["a"],          ["a"]]
        expected = (1.0 + 1/3 + 0.5) / 3
        assert mrr(retrieved_list, relevant_list) == pytest.approx(expected, abs=1e-3)


# ===========================================================================
# 7. ndcg_at_k
# ===========================================================================

class TestNDCGAtK:
    """ndcg_at_k(retrieved_ids, relevant_ids, k) → float

    Gain: binary (1 if relevant else 0).
    Discount: 1/log2(rank+1), rank is 1-based.
    DCG = sum gain(i)/log2(i+1) for i=1..k
    IDCG = sum 1/log2(i+1) for i=1..min(k,|relevant|)
    nDCG = DCG / IDCG
    """

    def test_perfect_ranking(self):
        # retrieved=["a","b","c"], relevant=["a","b"], k=3
        # DCG = 1/log2(2) + 1/log2(3) + 0 = 1.0 + 0.6309 = 1.6309
        # IDCG (2 relevant, k=3) = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309 = 1.6309
        # nDCG = 1.0
        result = ndcg_at_k(["a","b","c"], ["a","b"], 3)
        assert result == pytest.approx(1.0, abs=1e-3)

    def test_imperfect_ranking(self):
        # retrieved=["b","a","c"], relevant=["a","b"], k=3
        # DCG = 1/log2(2) + 1/log2(3) + 0 = 1.0 + 0.6309 = 1.6309 (same as above)
        # IDCG = 1.6309 → nDCG = 1.0
        # Both relevant items are in top-3, just swapped positions
        result = ndcg_at_k(["b","a","c"], ["a","b"], 3)
        assert result == pytest.approx(1.0, abs=1e-3)

    def test_k1_hit(self):
        # retrieved=["a","b","c"], relevant=["a"], k=1
        # DCG = 1/log2(2) = 1.0
        # IDCG = 1/log2(2) = 1.0
        # nDCG = 1.0
        result = ndcg_at_k(["a","b","c"], ["a"], 1)
        assert result == pytest.approx(1.0, abs=1e-3)

    def test_k1_miss(self):
        # retrieved=["b","a","c"], relevant=["a"], k=1
        # DCG = 0 (b is not relevant), IDCG=1.0 → nDCG=0.0
        result = ndcg_at_k(["b","a","c"], ["a"], 1)
        assert result == pytest.approx(0.0, abs=1e-3)

    def test_degraded_ranking(self):
        # retrieved=["c","a","b"], relevant=["a","b"], k=3
        # DCG = 0 + 1/log2(3) + 1/log2(4) = 0 + 0.6309 + 0.5 = 1.1309
        # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309 = 1.6309
        # nDCG = 1.1309 / 1.6309 = 0.6933
        dcg = 1.0/math.log2(3) + 1.0/math.log2(4)
        idcg = 1.0/math.log2(2) + 1.0/math.log2(3)
        expected = dcg / idcg
        result = ndcg_at_k(["c","a","b"], ["a","b"], 3)
        assert result == pytest.approx(expected, abs=1e-3)

    def test_empty_relevant(self):
        assert ndcg_at_k(["a","b"], [], 3) == pytest.approx(0.0, abs=1e-4)


# ===========================================================================
# 8. ece
# ===========================================================================

class TestECE:
    """ece(confidences, correct_flags, n_bins=10)

    ECE = sum_b (|B_b|/n) * |acc(B_b) - conf(B_b)|
    Denominator: total samples n.
    """

    def test_perfectly_calibrated(self):
        # One bin [0.6, 0.7): 3 samples all correct (acc=1.0), conf=0.65
        # ECE = (3/3) * |1.0 - 0.65| = 0.35 ... no: n=3, single bin
        # ECE = (3/3) * |1.0 - 0.65| = 0.35
        # Let's use perfectly calibrated: all bins acc==conf → ECE=0
        # 10 samples, confidence=0.6, all correct: acc(bin)=1.0 ≠ 0.6 → not 0
        # Perfect calibration: conf_i == P(correct) for every bin
        # Simplest: confidence=0.0, all wrong → single bin: acc=0.0, conf=0.0 → ECE=0
        confs = [0.0] * 5
        flags = [False] * 5
        result = ece(confs, flags, n_bins=5)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_known_miscalibration(self):
        # 4 samples: confidences=[0.9, 0.9, 0.9, 0.9], flags=[T, T, F, F]
        # All fall in bin [0.8, 1.0] (bin 9 of 10)
        # acc=2/4=0.5, conf=0.9, |acc-conf|=0.4
        # ECE = (4/4)*0.4 = 0.4
        confs = [0.9, 0.9, 0.9, 0.9]
        flags = [True, True, False, False]
        result = ece(confs, flags, n_bins=10)
        assert result == pytest.approx(0.4, abs=1e-4)

    def test_empty_input(self):
        assert ece([], [], n_bins=10) == pytest.approx(0.0, abs=1e-6)

    def test_all_correct_high_confidence(self):
        # confs=[0.95]*4, all correct → single bin [0.9,1.0]
        # acc=1.0, conf=0.95 → ECE=(4/4)*|1.0-0.95|=0.05
        confs = [0.95] * 4
        flags = [True] * 4
        result = ece(confs, flags, n_bins=10)
        assert result == pytest.approx(0.05, abs=1e-4)


# ===========================================================================
# 9. brier_score
# ===========================================================================

class TestBrierScore:
    """brier_score(confidences, correct_flags)

    BS = (1/n) * sum (conf_i - outcome_i)^2
    Denominator: total samples n.
    """

    def test_perfect_score(self):
        # conf=1.0, correct=True → (1-1)^2=0; conf=0.0, correct=False → (0-0)^2=0
        confs = [1.0, 0.0]
        flags = [True, False]
        assert brier_score(confs, flags) == pytest.approx(0.0, abs=1e-6)

    def test_worst_score(self):
        # conf=1.0, correct=False → (1-0)^2=1; conf=0.0, correct=True → (0-1)^2=1
        # BS = (1+1)/2 = 1.0
        confs = [1.0, 0.0]
        flags = [False, True]
        assert brier_score(confs, flags) == pytest.approx(1.0, abs=1e-6)

    def test_known_values(self):
        # confs=[0.8, 0.2, 0.6], flags=[True, False, True]
        # (0.8-1)^2 + (0.2-0)^2 + (0.6-1)^2 = 0.04 + 0.04 + 0.16 = 0.24
        # BS = 0.24/3 = 0.08
        confs = [0.8, 0.2, 0.6]
        flags = [True, False, True]
        assert brier_score(confs, flags) == pytest.approx(0.08, abs=1e-6)

    def test_empty_input(self):
        assert brier_score([], []) == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# 10. risk_coverage_curve
# ===========================================================================

class TestRiskCoverageCurve:
    """risk_coverage_curve(confidences, correct_flags)

    Returns list of (coverage, risk) pairs.
    coverage = |{i: conf_i >= thresh}| / n
    risk = |{i: conf_i >= thresh AND NOT correct}| / |{i: conf_i >= thresh}|
    """

    def test_basic_structure(self):
        confs = [0.9, 0.7, 0.5]
        flags = [True, True, False]
        curve = risk_coverage_curve(confs, flags)
        # Should return non-empty list of tuples
        assert len(curve) > 0
        for coverage, risk in curve:
            assert 0.0 <= coverage <= 1.0
            assert 0.0 <= risk <= 1.0

    def test_full_coverage_all_correct(self):
        confs = [0.5, 0.6, 0.7]
        flags = [True, True, True]
        curve = risk_coverage_curve(confs, flags)
        # At threshold=0.5, all 3 answered: coverage=1.0, risk=0.0
        first = curve[0]
        assert first[0] == pytest.approx(1.0, abs=1e-4)
        assert first[1] == pytest.approx(0.0, abs=1e-4)

    def test_empty_input(self):
        assert risk_coverage_curve([], []) == []

    def test_zero_coverage_sentinel(self):
        confs = [0.8, 0.9]
        flags = [True, False]
        curve = risk_coverage_curve(confs, flags)
        # Last point should be coverage=0.0 (sentinel above max threshold)
        last = curve[-1]
        assert last[0] == pytest.approx(0.0, abs=1e-4)


# ===========================================================================
# 11. abstention_precision
# ===========================================================================

class TestAbstentionPrecision:
    """abstention_precision(abstained_flags, should_abstain_flags)

    = TP_abstain / (TP_abstain + FP_abstain)
    Denominator: total model abstentions.
    """

    def test_all_correct_abstentions(self):
        # Model abstained on all 3 that should abstain → precision=1.0
        abstained = [True, True, True, False]
        should    = [True, True, True, False]
        assert abstention_precision(abstained, should) == pytest.approx(1.0, abs=1e-4)

    def test_mixed(self):
        # abstained=[T,T,F,F], should=[T,F,T,F]
        # TP=1 (pos0: T,T), FP=1 (pos1: T,F), FN=1 (pos2: F,T)
        # precision = 1/(1+1) = 0.5
        abstained = [True, True, False, False]
        should    = [True, False, True, False]
        assert abstention_precision(abstained, should) == pytest.approx(0.5, abs=1e-4)

    def test_no_abstentions(self):
        # Model never abstained → denominator=0 → 0.0
        abstained = [False, False]
        should    = [True, True]
        assert abstention_precision(abstained, should) == pytest.approx(0.0, abs=1e-4)

    def test_empty_input(self):
        assert abstention_precision([], []) == pytest.approx(0.0, abs=1e-4)


# ===========================================================================
# 12. abstention_recall
# ===========================================================================

class TestAbstentionRecall:
    """abstention_recall(abstained_flags, should_abstain_flags)

    = TP_abstain / (TP_abstain + FN_abstain)
    Denominator: total cases requiring abstention.
    """

    def test_all_caught(self):
        # All 3 should-abstain cases correctly abstained → recall=1.0
        abstained = [True, True, True, False]
        should    = [True, True, True, False]
        assert abstention_recall(abstained, should) == pytest.approx(1.0, abs=1e-4)

    def test_partial_recall(self):
        # abstained=[T,F,T,F], should=[T,T,F,F]
        # TP=1 (pos0: T,T), FN=1 (pos1: F,T)
        # recall = 1/(1+1) = 0.5
        abstained = [True, False, True, False]
        should    = [True, True, False, False]
        assert abstention_recall(abstained, should) == pytest.approx(0.5, abs=1e-4)

    def test_no_should_abstain_cases(self):
        # No cases require abstention → 0.0
        abstained = [True, False]
        should    = [False, False]
        assert abstention_recall(abstained, should) == pytest.approx(0.0, abs=1e-4)

    def test_empty_input(self):
        assert abstention_recall([], []) == pytest.approx(0.0, abs=1e-4)


# ===========================================================================
# 13. false_abstention_rate
# ===========================================================================

class TestFalseAbstentionRate:
    """false_abstention_rate(abstained_flags, should_abstain_flags)

    = FP_abstain / total_answerable
    Denominator: total questions where should_abstain=False.
    """

    def test_no_false_abstentions(self):
        # Model only abstains when it should → FAR=0.0
        abstained = [True, True, False, False]
        should    = [True, True, False, False]
        assert false_abstention_rate(abstained, should) == pytest.approx(0.0, abs=1e-4)

    def test_known_rate(self):
        # abstained=[T,T,F,F], should=[T,F,T,F]
        # FP (abstained on answerable): pos1 (T,F) → FP=1
        # total_answerable (should=False): pos1 and pos3 → 2
        # FAR = 1/2 = 0.5
        abstained = [True, True, False, False]
        should    = [True, False, True, False]
        assert false_abstention_rate(abstained, should) == pytest.approx(0.5, abs=1e-4)

    def test_no_answerable_questions(self):
        # All should abstain → denominator=0 → 0.0
        abstained = [True, True]
        should    = [True, True]
        assert false_abstention_rate(abstained, should) == pytest.approx(0.0, abs=1e-4)

    def test_empty_input(self):
        assert false_abstention_rate([], []) == pytest.approx(0.0, abs=1e-4)


# ===========================================================================
# 14. false_acceptance_rate
# ===========================================================================

class TestFalseAcceptanceRate:
    """false_acceptance_rate(abstained_flags, should_abstain_flags)

    = FN_abstain / total_unanswerable
    Denominator: total questions where should_abstain=True.
    """

    def test_no_false_acceptances(self):
        # Model correctly abstains on all unanswerable → FAcR=0.0
        abstained = [True, True, False, False]
        should    = [True, True, False, False]
        assert false_acceptance_rate(abstained, should) == pytest.approx(0.0, abs=1e-4)

    def test_known_rate(self):
        # abstained=[T,F,T,F], should=[T,T,F,F]
        # FN (not abstained on unanswerable): pos1 (F,T) → FN=1
        # total_unanswerable (should=True): pos0 and pos1 → 2
        # FAcR = 1/2 = 0.5
        abstained = [True, False, True, False]
        should    = [True, True, False, False]
        assert false_acceptance_rate(abstained, should) == pytest.approx(0.5, abs=1e-4)

    def test_no_unanswerable_questions(self):
        # All answerable → denominator=0 → 0.0
        abstained = [True, False]
        should    = [False, False]
        assert false_acceptance_rate(abstained, should) == pytest.approx(0.0, abs=1e-4)

    def test_empty_input(self):
        assert false_acceptance_rate([], []) == pytest.approx(0.0, abs=1e-4)


# ===========================================================================
# Statistics tests
# ===========================================================================

class TestWilsonCI:
    """wilson_ci(k, n, alpha=0.05) → (lower, upper)"""

    def test_known_case(self):
        # k=5, n=10 → proportion=0.5; Wilson 95% CI should straddle 0.5
        lower, upper = wilson_ci(5, 10)
        assert lower < 0.5 < upper
        assert 0.0 <= lower < upper <= 1.0

    def test_zero_successes(self):
        # k=0, n=10 → CI should start at 0.0 and not be negative
        lower, upper = wilson_ci(0, 10)
        assert lower == pytest.approx(0.0, abs=1e-4)
        assert upper > 0.0

    def test_all_successes(self):
        # k=n → CI should end at 1.0
        lower, upper = wilson_ci(10, 10)
        assert upper == pytest.approx(1.0, abs=1e-4)
        assert lower < 1.0

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            wilson_ci(0, 0)

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            wilson_ci(-1, 10)

    def test_k_greater_than_n(self):
        with pytest.raises(ValueError):
            wilson_ci(11, 10)

    def test_ci_ordering(self):
        lower, upper = wilson_ci(3, 20)
        assert lower < upper

    def test_wider_ci_smaller_n(self):
        lo1, hi1 = wilson_ci(5, 10)
        lo2, hi2 = wilson_ci(50, 100)
        # Larger n → narrower CI
        assert (hi1 - lo1) > (hi2 - lo2)


class TestMcNemarTest:
    """mcnemar_test(table, exact_threshold=25)"""

    def test_exact_binomial_used_for_small_discordant(self):
        # b=3, c=1 → discordant=4 < 25 → exact
        table = ((10, 3), (1, 10))
        stat, p, method = mcnemar_test(table, exact_threshold=25)
        assert method == "exact_binomial"
        assert 0.0 <= p <= 1.0

    def test_chi_squared_used_for_large_discordant(self):
        # b=20, c=10 → discordant=30 >= 25 → chi-squared
        table = ((50, 20), (10, 50))
        stat, p, method = mcnemar_test(table, exact_threshold=25)
        assert method == "chi_squared_continuity_corrected"
        assert 0.0 <= p <= 1.0

    def test_equal_discordant_high_p(self):
        # b=c=10 → no preference for either model; p should be high (near 1)
        table = ((20, 10), (10, 20))
        stat, p, method = mcnemar_test(table, exact_threshold=25)
        # p > 0.5 expected (no significant difference)
        assert p > 0.1

    def test_very_different_discordant_low_p(self):
        # b=0, c=30 → strong preference; p should be very small
        table = ((10, 0), (30, 10))
        stat, p, method = mcnemar_test(table, exact_threshold=25)
        assert p < 0.05

    def test_malformed_table(self):
        with pytest.raises(ValueError):
            mcnemar_test(((1, 2, 3), (4, 5, 6)))


class TestPairedBootstrap:
    """paired_bootstrap(metric_fn, y_a, y_b, n_bootstrap, seed)"""

    def test_observed_diff_correct(self):
        # metric_fn = mean; y_a = [1,2,3], y_b = [0,1,2]
        # mean(y_a)=2.0, mean(y_b)=1.0, diff=1.0
        def mean_fn(xs): return sum(xs) / len(xs)
        obs, lo, hi = paired_bootstrap(mean_fn, [1.0, 2.0, 3.0], [0.0, 1.0, 2.0], n_bootstrap=1000, seed=42)
        assert obs == pytest.approx(1.0, abs=1e-6)

    def test_ci_contains_observed_when_large_n_bootstrap(self):
        def mean_fn(xs): return sum(xs) / len(xs)
        obs, lo, hi = paired_bootstrap(mean_fn, [1.0]*10, [0.5]*10, n_bootstrap=5000, seed=0)
        # The diff is 0.5; CI should surround 0.5
        assert lo <= obs <= hi

    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError):
            paired_bootstrap(lambda x: sum(x)/len(x), [1.0, 2.0], [3.0], n_bootstrap=100)

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            paired_bootstrap(lambda x: 0.0, [], [])

    def test_seeded_reproducibility(self):
        def mean_fn(xs): return sum(xs) / len(xs)
        res1 = paired_bootstrap(mean_fn, [1.0, 2.0, 3.0], [0.5, 1.5, 2.5], n_bootstrap=500, seed=7)
        res2 = paired_bootstrap(mean_fn, [1.0, 2.0, 3.0], [0.5, 1.5, 2.5], n_bootstrap=500, seed=7)
        assert res1 == res2


class TestHolmCorrection:
    """holm_correction(p_values, alpha=0.05)"""

    def test_all_significant(self):
        # Very small p-values → all rejected
        p_values = [0.001, 0.002, 0.003]
        rejected, adj = holm_correction(p_values, alpha=0.05)
        assert all(rejected)
        assert len(adj) == 3

    def test_none_significant(self):
        # Large p-values → none rejected
        p_values = [0.5, 0.6, 0.7]
        rejected, adj = holm_correction(p_values, alpha=0.05)
        assert not any(rejected)

    def test_partial_rejection(self):
        # p_values = [0.001, 0.04, 0.60]
        # Sorted: 0.001 (thresh=0.05/3=0.0167 ✓), 0.04 (thresh=0.05/2=0.025 ✗)
        # Rejection stops at second → only first rejected
        p_values = [0.001, 0.04, 0.60]
        rejected, adj = holm_correction(p_values, alpha=0.05)
        assert rejected[0] is True     # 0.001 < 0.05/3
        assert rejected[1] is False    # step-down stops
        assert rejected[2] is False

    def test_output_length_matches_input(self):
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        rejected, adj = holm_correction(p_values)
        assert len(rejected) == 5
        assert len(adj) == 5

    def test_adjusted_p_nondecreasing_in_sorted_order(self):
        # Adjusted p-values in Holm are non-decreasing in sorted order
        p_values = [0.01, 0.02, 0.04]
        rejected, adj = holm_correction(p_values)
        sorted_adj = [adj[i] for i in sorted(range(3), key=lambda i: p_values[i])]
        for i in range(len(sorted_adj) - 1):
            assert sorted_adj[i] <= sorted_adj[i + 1] + 1e-9

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            holm_correction([])


class TestFormatResult:
    """format_result(n, numerator, denominator, diff, ci, p_value)"""

    def test_basic_format(self):
        result = format_result(100, 75, 100, 0.05, (0.01, 0.09), 0.0034)
        assert "n=100" in result
        assert "75/100" in result
        assert "75.0%" in result
        assert "+0.050" in result
        assert "0.010" in result
        assert "0.090" in result
        assert "0.0034" in result

    def test_zero_denominator_raises(self):
        with pytest.raises(ValueError):
            format_result(10, 5, 0, 0.0, (0.0, 0.0), 0.5)

    def test_negative_diff_formatted_with_sign(self):
        result = format_result(50, 20, 50, -0.1, (0.0, 0.2), 0.1)
        assert "-0.100" in result
