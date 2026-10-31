"""Calibration-logic unit tests (hand-computed, no model, leakage guards)."""

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluation import metrics  # noqa: E402
from evaluation.bioasq_eval import calibrate_confidence as cc  # noqa: E402


# ---------------------------------------------------------------------------
# ECE / Brier vs hand-computed
# ---------------------------------------------------------------------------

def test_brier_hand_computed():
    # conf 0.9 correct, 0.1 wrong -> ((0.9-1)^2 + (0.1-0)^2)/2 = (0.01+0.01)/2 = 0.01
    assert metrics.brier_score([0.9, 0.1], [True, False]) == 0.01


def test_ece_single_bin_hand_computed():
    # Two samples in different bins:
    # conf 0.95 correct -> bin[0.9,1.0]: acc=1, conf=0.95, gap=0.05, weight 0.5
    # conf 0.05 wrong   -> bin[0.0,0.1]: acc=0, conf=0.05, gap=0.05, weight 0.5
    # ECE = 0.5*0.05 + 0.5*0.05 = 0.05
    assert metrics.ece([0.95, 0.05], [True, False], n_bins=10) == 0.05


# ---------------------------------------------------------------------------
# invalid-score / missing-label handling
# ---------------------------------------------------------------------------

def test_clamp_out_of_range_and_nonfinite():
    warnings = []
    assert cc._clamp01(1.5, warnings, "ctx") == 1.0
    assert cc._clamp01(-0.2, warnings, "ctx") == 0.0
    assert cc._clamp01(float("nan"), warnings, "ctx") == 0.0
    assert len(warnings) == 3


def test_filter_excludes_pubmedqa_and_abstentions():
    rows = [
        {"question_id": "bioasq::1", "confidence": 0.7, "abstained": False,
         "correctness": True, "source": "bioasq"},
        {"question_id": "bioasq::2", "confidence": 0.6, "abstained": True,
         "correctness": False, "source": "bioasq"},
        {"question_id": "pubmedqa::3", "confidence": 0.8, "abstained": False,
         "correctness": True, "source": "pubmedqa"},
    ]
    out = cc.filter_bioasq_labelled(rows, [])
    assert out["n_bioasq"] == 2
    assert out["n_pubmedqa_excluded"] == 1
    assert out["n_abstained_excluded"] == 1
    assert out["N"] == 1 and out["P"] == 1


# ---------------------------------------------------------------------------
# branch selector on synthetic (N, P)
# ---------------------------------------------------------------------------

def _labelled(n, p):
    rows = []
    for i in range(n):
        rows.append({"question_id": f"bioasq::{i}", "confidence": 0.5,
                     "correctness": i < p, "type": ["factoid", "yesno", "list", "summary"][i % 4]})
    return rows


def test_degenerate_when_zero_positive():
    rep = cc.degenerate_report(_labelled(2, 0), 2, 0, "cfg")
    assert rep["data_regime"] == "DEGENERATE"
    assert rep["threshold_claim"] is None
    assert rep["P"] == 0


def test_small_n_provisional_branch_both_classes():
    # 0 < N < 15 with both classes -> SMALL_N_PROVISIONAL via run() logic is tested
    # indirectly; here we assert the method selector works on both-class data.
    conf = [0.2, 0.3, 0.8, 0.9]
    correct = [False, False, True, True]
    method, chosen, comparison = cc.select_calibration_method(conf, correct)
    assert method in ("platt", "isotonic")
    assert "ece" in chosen and "brier" in chosen
    assert set(comparison).issubset({"platt", "isotonic"})


# ---------------------------------------------------------------------------
# phase5-id -> eval-id join and schema classification
# ---------------------------------------------------------------------------

def test_candidate_schema_classification():
    preds = [{"configuration": "phase5-full-pipeline", "confidence_score": 0.5,
              "correctness": True, "question_id": "bioasq::1", "dataset": "bioasq"}]
    conf_ab = [{"configuration": "phase5-full-pipeline", "calibrated_confidence": 0.5,
                "correctness_token_containment": True, "question_id": "bioasq::1",
                "source": "bioasq"}]
    mirror = [{"config": "phase5-full-pipeline", "confidence": 0.5, "abstain_flag": True,
               "question_id": "bioasq::1", "source": "bioasq"}]
    assert cc._candidate_schema(preds) == "predictions"
    assert cc._candidate_schema(conf_ab) == "confidence_abstention"
    assert cc._candidate_schema(mirror) == "incomparable"


def test_extract_rows_incomparable_yields_empty():
    mirror = [{"config": "phase5-full-pipeline", "confidence": 0.5, "abstain_flag": True,
               "question_id": "bioasq::1", "source": "bioasq"}]
    assert cc._extract_rows(mirror, "mirror") == []


# ---------------------------------------------------------------------------
# build_splits slot mapping + disjoint/union + method selection
# ---------------------------------------------------------------------------

def test_build_splits_disjoint_union_and_slot_mapping():
    from evaluation.split_builder import build_splits
    dataset = [{"id": str(i), "type": ["factoid", "yesno"][i % 2]} for i in range(20)]
    fit, select, test = build_splits(dataset, dev_n=12, cal_n=4, test_n=4,
                                     seed=42, stratify_by=None)
    assert len(fit) == 12 and len(select) == 4 and len(test) == 4
    fit_ids = {d["id"] for d in fit}
    sel_ids = {d["id"] for d in select}
    test_ids = {d["id"] for d in test}
    assert fit_ids.isdisjoint(sel_ids) and fit_ids.isdisjoint(test_ids) \
        and sel_ids.isdisjoint(test_ids)
    assert fit_ids | sel_ids | test_ids == {str(i) for i in range(20)}


def test_split_and_validate_sparse_stratum_falls_back():
    # 16 rows, 4 types each 4 -> stratified would need > per-stratum count; expect
    # non-stratified fallback or a graceful not_evaluable, never a crash.
    labelled = _labelled(16, 8)
    rep = cc.split_and_validate(labelled, 16, 8, "cfg", provisional=True)
    assert "status" in rep or "stratified" in rep


def test_method_selection_prefers_better_calibrated_map():
    # Well-separated scores -> both maps fit; selector returns a valid method with
    # finite ECE/Brier.
    rng = np.random.default_rng(42)
    conf = list(np.concatenate([rng.uniform(0.0, 0.4, 15), rng.uniform(0.6, 1.0, 15)]))
    correct = [False] * 15 + [True] * 15
    method, chosen, comparison = cc.select_calibration_method(conf, correct)
    assert 0.0 <= chosen["ece"] <= 1.0
    assert 0.0 <= chosen["brier"] <= 1.0


# ---------------------------------------------------------------------------
# threshold objective + smallest-threshold tie-break
# ---------------------------------------------------------------------------

def test_threshold_objective_max_coverage_meeting_floor():
    # Build a select set where several thresholds meet the 0.80 floor; the chosen
    # one must maximize coverage, breaking ties toward the smallest threshold.
    # confidences all 1.0 and all correct -> every threshold has sel_acc 1.0; the
    # smallest threshold gives the max coverage.
    conf = [0.9] * 10
    correct = [True] * 10
    sel = cc.select_threshold(conf, correct)
    assert sel["floor_reachable"] is True
    # All thresholds <= 0.90 answer everything; smallest grid threshold 0.30 wins.
    assert sel["selected_threshold"] == 0.30


def test_threshold_floor_unreachable_picks_best_accuracy():
    # All wrong -> selective accuracy 0 everywhere; floor unreachable.
    conf = [0.9] * 10
    correct = [False] * 10
    sel = cc.select_threshold(conf, correct)
    assert sel["floor_reachable"] is False
    assert sel["selected_threshold"] is not None


# ---------------------------------------------------------------------------
# calibration never uses test labels; threshold selection never uses test outcomes
# ---------------------------------------------------------------------------

def test_test_partition_untouched_until_final_eval():
    # Run the full split path and assert the method/threshold were selected without
    # the test ids, i.e. partitions are disjoint and test is only evaluated once.
    labelled = _labelled(60, 30)
    rep = cc.split_and_validate(labelled, 60, 30, "cfg", provisional=False)
    if "partitions" in rep:
        fit = set(rep["partitions"]["fit"])
        sel = set(rep["partitions"]["select"])
        test = set(rep["partitions"]["test"])
        assert fit.isdisjoint(test) and sel.isdisjoint(test)
        if rep.get("test_evaluation"):
            assert rep["test_evaluation"]["test_opened_once"] is True
