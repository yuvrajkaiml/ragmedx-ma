"""
Phase D/E — Confidence calibration & threshold validation.

Reuses prior phase5 artifacts (never runs the LLM by default). Treats the raw
Agent-7 confidence as UNCALIBRATED, fits a SEPARATE calibration map on calibration
data only, and reports ECE/Brier BEFORE and AFTER. The on-disk data is DEGENERATE
(N approx 2 non-abstained BioASQ, 0 correct), so the expected primary outcome is the
degenerate-data path (design 8.4): report raw ECE/Brier + confidence distribution +
0.65 behaviour, clearly labelled provisional, with NO fabricated calibration claim.

The split + method-selection + threshold protocol (design 8.5-8.6) is reached only
when N >= 15 and both classes are present. Its logic is exercised by the unit tests.

Run from ``backend/``::

    d:\\Ragmed_2\\.venv\\Scripts\\python.exe -m evaluation.bioasq_eval.calibrate_confidence
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluation import metrics  # noqa: E402
from evaluation.bioasq_eval.common import (  # noqa: E402
    SEED, atomic_write_json, atomic_write_text, write_manifest,
)

EVAL_QUESTIONS = BACKEND_DIR / "data" / "datasets" / "bioasq_eval_split" / "eval_questions.json"
RESULTS_DIR = BACKEND_DIR / "results" / "bioasq_eval_20261018"
CALIB_DIR = RESULTS_DIR / "calibration"

INCUMBENT_THRESHOLD = 0.65
SELECTIVE_ACC_FLOOR = 0.80
COVERAGE_FLOOR = 0.50
N_SPLIT_MIN = 15
N_FULL_MIN = 40
N_BOOTSTRAP_MIN = 30
THRESHOLD_GRID = [round(0.30 + 0.05 * i, 2) for i in range(13)]  # 0.30..0.90

# Calibration source candidates, in fixed precedence (anchored, never cwd).
CANDIDATES = [
    REPO_ROOT / "RAGMedXMA_Chapter5_Validated_Results" / "predictions.jsonl",
    REPO_ROOT / "RAGMedXMA_Chapter5_Validated_Results" / "confidence_abstention.jsonl",
    BACKEND_DIR / "results" / "phase5_benchmark" / "per_question" / "phase5-full-pipeline.jsonl",
]
TARGET_CONFIG = "phase5-full-pipeline"


# ---------------------------------------------------------------------------
# Part 1 — source loader + label/join logic
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _clamp01(x: float, warnings: List[str], ctx: str) -> float:
    if not np.isfinite(x):
        warnings.append(f"non-finite confidence in {ctx}; set to 0.0")
        return 0.0
    if x < 0.0:
        warnings.append(f"confidence {x} < 0 in {ctx}; clamped")
        return 0.0
    if x > 1.0:
        warnings.append(f"confidence {x} > 1 in {ctx}; clamped")
        return 1.0
    return float(x)


def _candidate_schema(records: List[Dict[str, Any]]) -> str:
    """Classify a candidate's schema so we only equivalence-check comparable ones.

    - ``predictions``: has ``configuration`` + ``confidence_score`` + ``correctness``.
    - ``confidence_abstention``: has ``configuration`` + ``calibrated_confidence`` +
      ``correctness_token_containment``.
    - ``incomparable``: anything else (e.g. the in-backend per-question mirror, which
      uses ``config``/``confidence``/``abstain_flag`` and carries NO correctness field).
    """
    if not records:
        return "incomparable"
    sample = records[0]
    if "configuration" in sample and "confidence_score" in sample and "correctness" in sample:
        return "predictions"
    if "configuration" in sample and "calibrated_confidence" in sample \
            and "correctness_token_containment" in sample:
        return "confidence_abstention"
    return "incomparable"


def _extract_rows(records: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
    """Normalize a comparable candidate's rows to the common calibration schema.

    Returns rows ``{question_id, confidence, abstained, correctness, source}`` filtered
    to the TARGET_CONFIG. Only ``predictions``/``confidence_abstention`` schemas yield
    usable rows; ``incomparable`` schemas (the reduced in-backend mirror, which lacks a
    ``correctness`` field) return an empty list and are handled by schema class, not by
    pretending they are equivalent.
    """
    schema = _candidate_schema(records)
    if schema == "incomparable":
        return []
    rows = []
    for r in records:
        if r.get("configuration") != TARGET_CONFIG:
            continue
        qid = r.get("question_id")
        if qid is None:
            continue
        if schema == "predictions":
            conf = r.get("confidence_score")
            abstained = bool(r.get("abstained"))
            correctness = bool(r.get("correctness"))
            src = r.get("dataset")
        else:  # confidence_abstention
            conf = r.get("calibrated_confidence")
            withheld = r.get("withheld")
            abstained = bool(withheld) if withheld is not None else False
            correctness = bool(r.get("correctness_token_containment"))
            src = r.get("source")
        if conf is None:
            continue
        rows.append({
            "question_id": qid,
            "confidence": float(conf),
            "abstained": abstained,
            "correctness": correctness,
            "source": src,
            "_from": source_name,
        })
    return rows


def load_calibration_source() -> Dict[str, Any]:
    """Resolve candidates in precedence, select the first, cross-check the rest.

    Returns a dict with the selected source rows and a source_consistency verdict.
    Raises SystemExit on CALIBRATION_SOURCE_MISSING / CALIBRATION_SOURCE_MISMATCH.
    """
    probed = []
    comparable: List[Tuple[Path, List[Dict[str, Any]]]] = []
    incomparable: List[str] = []
    for cand in CANDIDATES:
        exists = cand.exists()
        schema = "not-found"
        if exists:
            records = _read_jsonl(cand)
            schema = _candidate_schema(records)
            rows = _extract_rows(records, cand.name)
            if schema == "incomparable":
                incomparable.append(str(cand))
            else:
                comparable.append((cand, rows))
        probed.append({"path": str(cand), "exists": exists, "schema": schema})
        print(f"[calib] probe {'FOUND    ' if exists else 'not-found'} [{schema}] {cand}")

    if not comparable:
        print("CALIBRATION_SOURCE_MISSING: no comparable calibration source resolved "
              "(the only resolvable source has an incomparable schema).", file=sys.stderr)
        raise SystemExit("CALIBRATION_SOURCE_MISSING")

    selected_path, selected_rows = comparable[0]

    # Cross-check every OTHER comparable candidate for equivalence on the config.
    # Incomparable candidates (the reduced in-backend mirror with no correctness field)
    # are NOT equivalence-checked — they cannot supply the labels the check needs, so
    # treating a structural schema difference as a data "mismatch" would be wrong.
    consistency = "single_source" if len(comparable) == 1 else "ok"
    mismatch_detail = None
    sel_by_id = {r["question_id"]: r for r in selected_rows}
    for other_path, other_rows in comparable[1:]:
        other_by_id = {r["question_id"]: r for r in other_rows}
        if len(other_rows) != len(selected_rows) or set(other_by_id) != set(sel_by_id):
            consistency = "mismatch"
            mismatch_detail = (f"{other_path.name}: n_rows/id-set differs "
                               f"({len(other_rows)} vs {len(selected_rows)})")
            break
        for qid, sr in sel_by_id.items():
            orr = other_by_id[qid]
            if orr["correctness"] != sr["correctness"] or orr["abstained"] != sr["abstained"]:
                consistency = "mismatch"
                mismatch_detail = f"{other_path.name}: correctness/abstained differs for {qid}"
                break
        if consistency == "mismatch":
            break

    if consistency == "mismatch":
        print(f"CALIBRATION_SOURCE_MISMATCH: {mismatch_detail}", file=sys.stderr)
        raise SystemExit("CALIBRATION_SOURCE_MISMATCH")

    return {
        "selected_source": str(selected_path),
        "probed": probed,
        "source_consistency": consistency,
        "incomparable_sources": incomparable,
        "rows": selected_rows,
    }


def filter_bioasq_labelled(rows: List[Dict[str, Any]],
                           warnings: List[str]) -> Dict[str, Any]:
    """Exclude pubmedqa + abstentions; clamp confidences; return the labelled set."""
    bioasq = [r for r in rows if r.get("source") == "bioasq"]
    non_abstained = [r for r in bioasq if not r["abstained"]]
    for r in non_abstained:
        r["confidence"] = _clamp01(r["confidence"], warnings, r["question_id"])
    n = len(non_abstained)
    p = sum(1 for r in non_abstained if r["correctness"])
    return {
        "n_total_config_rows": len(rows),
        "n_bioasq": len(bioasq),
        "n_pubmedqa_excluded": sum(1 for r in rows if r.get("source") == "pubmedqa"),
        "n_abstained_excluded": len(bioasq) - n,
        "labelled": non_abstained,
        "N": n,
        "P": p,
    }


# ---------------------------------------------------------------------------
# Part 2 — calibration metrics, degenerate policy, graded evaluator gate
# ---------------------------------------------------------------------------

def _confidence_histogram(confidences: List[float], n_bins: int = 10) -> List[Dict[str, Any]]:
    bins = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        if b == n_bins - 1:
            idx = [i for i, c in enumerate(confidences) if lo <= c <= hi]
        else:
            idx = [i for i, c in enumerate(confidences) if lo <= c < hi]
        bins.append({"bin": [round(lo, 2), round(hi, 2)], "count": len(idx)})
    return bins


def _ece_with_bins(confidences: List[float], correct: List[bool],
                   n_bins: int = 10) -> Dict[str, Any]:
    ece_val = metrics.ece(confidences, correct, n_bins=n_bins)
    bins = []
    n = len(confidences)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        if b == n_bins - 1:
            idx = [i for i, c in enumerate(confidences) if lo <= c <= hi]
        else:
            idx = [i for i, c in enumerate(confidences) if lo <= c < hi]
        if idx:
            acc = sum(1 for i in idx if correct[i]) / len(idx)
            conf = sum(confidences[i] for i in idx) / len(idx)
        else:
            acc = conf = None
        bins.append({"bin": [round(lo, 2), round(hi, 2)], "count": len(idx),
                     "accuracy": acc, "mean_confidence": conf})
    return {"ece": ece_val, "n_bins": n_bins, "bins": bins, "n": n}


def _selective_accuracy_at(threshold: float, confidences: List[float],
                           correct: List[bool]) -> Dict[str, Any]:
    answered = [i for i, c in enumerate(confidences) if c >= threshold]
    n = len(confidences)
    if not answered:
        return {"threshold": threshold, "coverage": 0.0, "selective_accuracy": None,
                "error_rate_answered": None, "n_answered": 0, "abstention_rate": 1.0}
    correct_answered = sum(1 for i in answered if correct[i])
    sel_acc = correct_answered / len(answered)
    return {
        "threshold": threshold,
        "coverage": round(len(answered) / n, 6),
        "selective_accuracy": round(sel_acc, 6),
        "error_rate_answered": round(1 - sel_acc, 6),
        "n_answered": len(answered),
        "abstention_rate": round(1 - len(answered) / n, 6),
    }


def degenerate_report(labelled: List[Dict[str, Any]], N: int, P: int,
                      source: str) -> Dict[str, Any]:
    """Design 8.4 DEGENERATE / single-class path. No calibration/threshold claim."""
    confidences = [r["confidence"] for r in labelled]
    correct = [r["correctness"] for r in labelled]
    label = "DEGENERATE — single-class / insufficient data, provisional, not a calibration claim"
    report: Dict[str, Any] = {
        "data_regime": "DEGENERATE",
        "label": label,
        "N": N,
        "P": P,
        "n_negative": N - P,
        "source_config": source,
        "raw_ece": _ece_with_bins(confidences, correct) if N > 0 else None,
        "raw_brier": metrics.brier_score(confidences, correct) if N > 0 else None,
        "confidence_histogram": _confidence_histogram(confidences) if N > 0 else [],
        "at_incumbent_0_65": _selective_accuracy_at(INCUMBENT_THRESHOLD, confidences, correct)
        if N > 0 else None,
        "threshold_claim": None,
        "explanation": (
            f"The correctness labeller found P={P} correct answers among N={N} non-abstained "
            f"BioASQ questions in config '{source}'. With no positive class (or no negative "
            f"class), ECE/Brier are degenerate and a selective-accuracy threshold cannot be "
            f"meaningfully optimized, so the 0.65 proposal can be neither confirmed nor refuted "
            f"from this data. No calibration map is fit; all reported numbers are the RAW "
            f"heuristic over the labelled rows, provisional."
        ),
    }
    return report


# ---------------------------------------------------------------------------
# Part 3 — calibration map fitting + method selection + split + threshold
# ---------------------------------------------------------------------------

def _fit_platt(x: np.ndarray, y: np.ndarray):
    """Platt (logistic) calibration. sklearn if importable, else NumPy fallback."""
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
        clf.fit(x.reshape(-1, 1), y)

        def _map(v: np.ndarray) -> np.ndarray:
            return clf.predict_proba(v.reshape(-1, 1))[:, 1]
        return _map, "sklearn_logistic"
    except Exception:
        # NumPy gradient-descent logistic fallback.
        w, b = 0.0, 0.0
        lr = 0.1
        for _ in range(5000):
            z = w * x + b
            p = 1.0 / (1.0 + np.exp(-z))
            grad_w = np.mean((p - y) * x)
            grad_b = np.mean(p - y)
            w -= lr * grad_w
            b -= lr * grad_b

        def _map(v: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-(w * v + b)))
        return _map, "numpy_logistic"


def _fit_isotonic(x: np.ndarray, y: np.ndarray):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(x, y)

    def _map(v: np.ndarray) -> np.ndarray:
        return np.clip(ir.predict(v), 0.0, 1.0)
    return _map, "sklearn_isotonic"


def select_calibration_method(fit_conf: List[float], fit_correct: List[bool]):
    """Fit Platt + isotonic on fit; pick lower fit-ECE (tie-break lower fit-Brier)."""
    x = np.asarray(fit_conf, dtype=float)
    y = np.asarray([1.0 if c else 0.0 for c in fit_correct], dtype=float)

    platt_map, platt_name = _fit_platt(x, y)
    results = {}
    platt_mapped = list(platt_map(x))
    results["platt"] = {
        "name": platt_name, "map": platt_map,
        "ece": metrics.ece(platt_mapped, fit_correct),
        "brier": metrics.brier_score(platt_mapped, fit_correct),
    }
    try:
        iso_map, iso_name = _fit_isotonic(x, y)
        iso_mapped = list(iso_map(x))
        results["isotonic"] = {
            "name": iso_name, "map": iso_map,
            "ece": metrics.ece(iso_mapped, fit_correct),
            "brier": metrics.brier_score(iso_mapped, fit_correct),
        }
    except Exception:
        pass

    def _key(item):
        return (item[1]["ece"], item[1]["brier"])
    chosen = min(results.items(), key=_key)
    return chosen[0], results[chosen[0]], {k: {"ece": v["ece"], "brier": v["brier"],
                                               "name": v["name"]}
                                           for k, v in results.items()}


def select_threshold(select_conf: List[float], select_correct: List[bool]) -> Dict[str, Any]:
    """Max coverage s.t. selective accuracy >= floor on select; smallest-threshold tie-break."""
    rows = [_selective_accuracy_at(t, select_conf, select_correct) for t in THRESHOLD_GRID]
    feasible = [r for r in rows
                if r["selective_accuracy"] is not None
                and r["selective_accuracy"] >= SELECTIVE_ACC_FLOOR]
    if feasible:
        max_cov = max(r["coverage"] for r in feasible)
        tied = [r for r in feasible if r["coverage"] == max_cov]
        chosen = min(tied, key=lambda r: r["threshold"])
        floor_reachable = True
    else:
        scored = [r for r in rows if r["selective_accuracy"] is not None]
        if not scored:
            return {"grid": rows, "floor_reachable": False, "selected_threshold": None}
        max_acc = max(r["selective_accuracy"] for r in scored)
        tied = [r for r in scored if r["selective_accuracy"] == max_acc]
        chosen = min(tied, key=lambda r: r["threshold"])
        floor_reachable = False
    return {"grid": rows, "floor_reachable": floor_reachable,
            "selected_threshold": chosen["threshold"], "selected_on_select": chosen}


def _bootstrap_ci(confidences: List[float], correct: List[bool], threshold: float,
                  n_resamples: int = 1000) -> Dict[str, Any]:
    rng = np.random.default_rng(SEED)
    n = len(confidences)
    sel_accs, covs = [], []
    conf = np.asarray(confidences)
    corr = np.asarray(correct)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        c, k = conf[idx], corr[idx]
        answered = c >= threshold
        if answered.sum() == 0:
            sel_accs.append(0.0)
            covs.append(0.0)
            continue
        sel_accs.append(float(k[answered].mean()))
        covs.append(float(answered.mean()))
    return {
        "selective_accuracy_ci95": [round(float(np.percentile(sel_accs, 2.5)), 6),
                                    round(float(np.percentile(sel_accs, 97.5)), 6)],
        "coverage_ci95": [round(float(np.percentile(covs, 2.5)), 6),
                          round(float(np.percentile(covs, 97.5)), 6)],
        "n_resamples": n_resamples,
    }


def split_and_validate(labelled: List[Dict[str, Any]], N: int, P: int,
                       source: str, provisional: bool) -> Dict[str, Any]:
    """Design 8.5-8.6 split + method-selection + threshold protocol (N >= 15, both classes)."""
    from evaluation.split_builder import build_splits

    dataset = [{"id": r["question_id"], "type": r.get("type", "__unknown__"),
                "confidence": r["confidence"], "correctness": r["correctness"]}
               for r in labelled]
    dev_n = round(0.6 * N)
    cal_n = round(0.2 * N)
    test_n = N - dev_n - cal_n

    report: Dict[str, Any] = {
        "data_regime": "PROVISIONAL" if provisional else "FULL",
        "N": N, "P": P, "source_config": source,
        "split_sizes": {"fit": dev_n, "select": cal_n, "test": test_n},
    }

    # Per-stratum sparsity pre-check; fall back to non-stratified if too sparse.
    #
    # build_splits' internal _adjust() can loop indefinitely when a stratum's
    # per-split allocation is forced to the floor of 1 and the running total still
    # needs to be reduced. To avoid ever triggering that fragile path, we only
    # stratify when EVERY stratum has strict margin over its summed floor-allocation
    # (i.e. no split's per-stratum allocation sits at the minimum of 1 that _adjust
    # cannot reduce further). Otherwise we fall back to the non-stratified path,
    # which is deterministic and loop-free. The design makes the pre-check advisory
    # and the authoritative guard a try/except around build_splits; we keep both.
    stratify = "type"
    types = {}
    for d in dataset:
        types[d["type"]] = types.get(d["type"], 0) + 1
    fallback_reason = None
    for t, cnt in types.items():
        alloc = [max(1, round(cnt / N * dev_n)),
                 max(1, round(cnt / N * cal_n)),
                 max(1, round(cnt / N * test_n))]
        need = sum(alloc)
        # Any allocation at the floor of 1, or a stratum without strict margin,
        # risks the _adjust loop -> fall back to non-stratified.
        if cnt < need or min(alloc) <= 1 or cnt <= need:
            stratify = None
            fallback_reason = f"sparse/tight stratum {t}={cnt} (alloc={alloc})"
            break

    try:
        fit_split, select_split, test_split = build_splits(
            dataset, dev_n=dev_n, cal_n=cal_n, test_n=test_n, seed=SEED, stratify_by=stratify)
    except ValueError as exc:
        report["status"] = "not_evaluable"
        report["reason"] = f"build_splits ValueError: {exc}"
        return report

    # Disjoint + union assertion.
    fit_ids = {d["id"] for d in fit_split}
    sel_ids = {d["id"] for d in select_split}
    test_ids = {d["id"] for d in test_split}
    assert fit_ids.isdisjoint(sel_ids) and fit_ids.isdisjoint(test_ids) \
        and sel_ids.isdisjoint(test_ids), "splits not disjoint"
    report["stratified"] = stratify is not None
    report["fallback_reason"] = fallback_reason
    report["partitions"] = {"fit": sorted(fit_ids), "select": sorted(sel_ids),
                            "test": sorted(test_ids)}

    fit_conf = [d["confidence"] for d in fit_split]
    fit_correct = [bool(d["correctness"]) for d in fit_split]
    sel_conf = [d["confidence"] for d in select_split]
    sel_correct = [bool(d["correctness"]) for d in select_split]
    test_conf = [d["confidence"] for d in test_split]
    test_correct = [bool(d["correctness"]) for d in test_split]

    # Method selection (requires both classes in fit).
    if len(set(fit_correct)) < 2:
        report["status"] = "not_evaluable"
        report["reason"] = "fit partition single-class"
        return report
    method, chosen, all_methods = select_calibration_method(fit_conf, fit_correct)
    cal_map = chosen["map"]
    report["calibration_method"] = method
    report["method_comparison"] = all_methods

    def _mapped(cs):
        return list(cal_map(np.asarray(cs, dtype=float)))

    report["ece_brier_before_after"] = {
        "fit": {"raw_ece": metrics.ece(fit_conf, fit_correct),
                "mapped_ece": metrics.ece(_mapped(fit_conf), fit_correct),
                "raw_brier": metrics.brier_score(fit_conf, fit_correct),
                "mapped_brier": metrics.brier_score(_mapped(fit_conf), fit_correct)},
        "select": {"raw_ece": metrics.ece(sel_conf, sel_correct),
                   "mapped_ece": metrics.ece(_mapped(sel_conf), sel_correct),
                   "raw_brier": metrics.brier_score(sel_conf, sel_correct),
                   "mapped_brier": metrics.brier_score(_mapped(sel_conf), sel_correct)},
        "test": {"raw_ece": metrics.ece(test_conf, test_correct),
                 "mapped_ece": metrics.ece(_mapped(test_conf), test_correct),
                 "raw_brier": metrics.brier_score(test_conf, test_correct),
                 "mapped_brier": metrics.brier_score(_mapped(test_conf), test_correct)},
    }

    # Per-partition single-class guards.
    if len(set(sel_correct)) < 2:
        report["select_partition"] = "single-class — threshold not selectable"
        report["status"] = "provisional_select_single_class"
    if len(set(test_correct)) < 2:
        report["test_partition"] = "single-class — threshold not evaluable"
        report["status"] = "not_evaluable_test_single_class"
        return report

    # Threshold selection on calibrated select confidence.
    sel_cal = _mapped(sel_conf)
    test_cal = _mapped(test_conf)
    if report.get("status") != "provisional_select_single_class":
        thr = select_threshold(sel_cal, sel_correct)
        report["threshold_selection"] = {k: v for k, v in thr.items() if k != "grid"}
        report["threshold_grid"] = thr["grid"]
        selected = thr["selected_threshold"]
    else:
        selected = None

    # Evaluate selected + incumbent 0.65 ONCE on untouched test.
    report["test_evaluation"] = {
        "incumbent_0_65": _selective_accuracy_at(INCUMBENT_THRESHOLD, test_cal, test_correct),
        "selected": (_selective_accuracy_at(selected, test_cal, test_correct)
                     if selected is not None else None),
        "risk_coverage_curve": metrics.risk_coverage_curve(test_cal, test_correct),
        "test_opened_once": True,
    }
    inc = report["test_evaluation"]["incumbent_0_65"]
    report["threshold_0_65_validated"] = bool(
        inc["selective_accuracy"] is not None
        and inc["selective_accuracy"] >= SELECTIVE_ACC_FLOOR
        and inc["coverage"] >= COVERAGE_FLOOR)

    if len(test_cal) >= N_BOOTSTRAP_MIN:
        report["bootstrap_ci"] = _bootstrap_ci(test_cal, test_correct, INCUMBENT_THRESHOLD)
    else:
        report["bootstrap_ci"] = "n insufficient for CI"

    report["status"] = report.get("status", "evaluated")
    return report


def _attach_types(labelled: List[Dict[str, Any]]) -> None:
    """Join question type from eval_questions by stripping the <source>:: prefix."""
    try:
        with open(EVAL_QUESTIONS, "r", encoding="utf-8") as fh:
            questions = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return
    by_id = {q["id"]: q for q in questions}
    for r in labelled:
        bare = r["question_id"].split("::", 1)[-1]
        q = by_id.get(bare)
        if q:
            r["type"] = q.get("type", "__unknown__")


def run() -> Dict[str, Any]:
    warnings: List[str] = []
    src = load_calibration_source()
    filt = filter_bioasq_labelled(src["rows"], warnings)
    _attach_types(filt["labelled"])
    N, P = filt["N"], filt["P"]
    source_name = TARGET_CONFIG

    base: Dict[str, Any] = {
        "seed": SEED,
        "selected_source": src["selected_source"],
        "source_consistency": src["source_consistency"],
        "incomparable_sources": src.get("incomparable_sources", []),
        "probed_sources": src["probed"],
        "counts": {k: filt[k] for k in
                   ("n_total_config_rows", "n_bioasq", "n_pubmedqa_excluded",
                    "n_abstained_excluded", "N", "P")},
        "warnings": warnings,
        "incumbent_threshold": INCUMBENT_THRESHOLD,
        "n_thresholds_predeclared": {"split_min": N_SPLIT_MIN, "full_min": N_FULL_MIN,
                                     "bootstrap_min": N_BOOTSTRAP_MIN},
        "graded_evaluator_gate": {
            "status": "provisional_by_default",
            "note": ("Automated-vs-manual agreement gate (>=0.85) requires --manual-labels; "
                     "unattended run supplies none, so the graded secondary view is "
                     "provisional regardless of n."),
        },
    }

    # Deterministic branch on (N, P).
    if P == 0 or (N - P) == 0:
        detail = degenerate_report(filt["labelled"], N, P, source_name)
    elif 0 < N < N_SPLIT_MIN:
        confidences = [r["confidence"] for r in filt["labelled"]]
        correct = [r["correctness"] for r in filt["labelled"]]
        method, chosen, all_methods = select_calibration_method(confidences, correct)
        mapped = list(chosen["map"](np.asarray(confidences, dtype=float)))
        detail = {
            "data_regime": "SMALL_N_PROVISIONAL",
            "label": "0 < N < 15 — provisional, no threshold claim, no CIs",
            "N": N, "P": P, "source_config": source_name,
            "calibration_method": method,
            "method_comparison": all_methods,
            "raw_ece": _ece_with_bins(confidences, correct),
            "mapped_ece": _ece_with_bins(mapped, correct),
            "raw_brier": metrics.brier_score(confidences, correct),
            "mapped_brier": metrics.brier_score(mapped, correct),
            "threshold_claim": None,
        }
    else:
        provisional = N < N_FULL_MIN
        detail = split_and_validate(filt["labelled"], N, P, source_name, provisional)
        # Write partitions.json for auditability.
        if "partitions" in detail:
            atomic_write_json(CALIB_DIR / "partitions.json", {
                "seed": SEED, "N": N, "split_sizes": detail["split_sizes"],
                "partitions": detail["partitions"]})
        # Write threshold_evaluation.json
        atomic_write_json(RESULTS_DIR / "threshold_evaluation.json", {
            "seed": SEED, "N": N, "P": P, "source_config": source_name,
            "status": detail.get("status"),
            "threshold_0_65_validated": detail.get("threshold_0_65_validated"),
            "test_evaluation": detail.get("test_evaluation"),
            "threshold_selection": detail.get("threshold_selection"),
        })

    report = {**base, **detail}

    # If we did not reach the split path, write a not_evaluable threshold_evaluation.
    if detail.get("data_regime") in ("DEGENERATE", "SMALL_N_PROVISIONAL"):
        atomic_write_json(RESULTS_DIR / "threshold_evaluation.json", {
            "seed": SEED, "N": N, "P": P, "source_config": source_name,
            "status": "not_evaluable",
            "reason": detail.get("explanation") or detail.get("label"),
            "threshold_0_65_validated": None,
        })

    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(CALIB_DIR / "calibration_report.json", report)
    atomic_write_text(CALIB_DIR / "calibration_report.md", _write_md(report))

    # Manual review sample (seed-42) for the evaluator gate.
    _write_manual_review_sample(filt["labelled"])

    write_manifest("calibration",
                   cfg={"source": source_name, "seed": SEED,
                        "incumbent_threshold": INCUMBENT_THRESHOLD},
                   inputs={"selected_source": src["selected_source"]},
                   extra={"N": N, "P": P, "data_regime": report["data_regime"],
                          "source_consistency": src["source_consistency"]})
    return report


def _write_manual_review_sample(labelled: List[Dict[str, Any]]) -> None:
    import random
    rng = random.Random(SEED)
    sample = labelled[:]
    rng.shuffle(sample)
    sample = sample[:40]
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    with open(CALIB_DIR / "manual_review_sample.jsonl", "w", encoding="utf-8") as fh:
        for r in sample:
            fh.write(json.dumps({"question_id": r["question_id"], "type": r.get("type"),
                                 "automated_correctness": r["correctness"],
                                 "confidence": r["confidence"]}) + "\n")


def _write_md(report: Dict[str, Any]) -> str:
    c = report["counts"]
    lines = [
        "# Phase D — Confidence Calibration (reused phase5 artifacts)",
        "",
        f"- Selected source: `{report['selected_source']}`",
        f"- Source consistency: **{report['source_consistency']}**",
        f"- Config rows: {c['n_total_config_rows']}, BioASQ: {c['n_bioasq']}, "
        f"pubmedqa excluded: {c['n_pubmedqa_excluded']}, abstained excluded: "
        f"{c['n_abstained_excluded']}",
        f"- **N (non-abstained BioASQ) = {c['N']}**, **P (correct) = {c['P']}**",
        f"- Data regime: **{report['data_regime']}**",
        "",
    ]
    if report["data_regime"] == "DEGENERATE":
        lines += [f"> {report['label']}", "", report["explanation"], ""]
        if report.get("raw_ece"):
            lines.append(f"- Raw ECE: {report['raw_ece']['ece']}, "
                         f"Raw Brier: {report['raw_brier']}")
        inc = report.get("at_incumbent_0_65")
        if inc:
            lines.append(f"- At incumbent 0.65: coverage={inc['coverage']}, "
                         f"selective_accuracy={inc['selective_accuracy']}, "
                         f"n_answered={inc['n_answered']}")
        lines += ["", "**The 0.65 threshold can be neither confirmed nor refuted from this data.**", ""]
    else:
        lines += [f"- Calibration method: {report.get('calibration_method')}",
                  f"- Threshold 0.65 validated: {report.get('threshold_0_65_validated')}", ""]
    if report.get("warnings"):
        lines += ["## Warnings", ""] + [f"- {w}" for w in report["warnings"]] + [""]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase D calibration")
    parser.add_argument("--manual-labels", type=str, default=None,
                        help="Optional path to human labels for the evaluator gate.")
    parser.parse_args(argv)
    report = run()
    print(f"data_regime={report['data_regime']} N={report['counts']['N']} "
          f"P={report['counts']['P']} source_consistency={report['source_consistency']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
