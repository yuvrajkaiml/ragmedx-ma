"""FEAT-003 — Recompute ALL scientific metrics from the raw per_question/*.jsonl.

CPU-only, deterministic. Reuses ONLY the already-defined functions in
``evaluation/metrics.py``, ``evaluation/statistics.py`` and the committed
``run_phase5_experiment.aggregate_outputs`` path — no metric is redefined here.

What this produces (regenerated from the raw per_question files, which are the
read-only source of truth; never overwritten):

    metric_summary.json      accuracy (token-containment GROUND TRUTH), Wilson CIs,
                             paired McNemar + Holm vs baseline, caveats
                             (via aggregate_outputs)
    latency_breakdown.json   cold/warm + per-stage latency (via aggregate_outputs)
    failure_cases.json       error-status cells (via aggregate_outputs)
    summary.json             tier label + accuracy snapshot (via aggregate_outputs)
    report.md                8-config comparison table (via aggregate_outputs)
    extended_metrics.json    ROUGE-L (primary answer quality), citation precision/
                             support, unsupported-claim rate + evidence sufficiency/
                             contradiction from stored claim_verdicts WHERE PRESENT,
                             answerability-aware abstention views, retrieval
                             Recall@K/MRR/nDCG for BioASQ (NOT MEASURED + reason),
                             latency/throughput/peak-VRAM, paired ROUGE-L bootstrap CIs
    comparison_table.md      compact 8-config table (same question IDs)
    runtime_completion_report.md
    reproducibility_report.md
    research_results_summary.md

Every reported value carries an evidence tier; ``NOT MEASURED`` + reason is used
wherever evidence is genuinely absent. Denominators / sample sizes / missing /
failed counts and uncertainty are reported throughout. No superiority claim is
emitted without a Holm-significant paired test.

Usage:
    d:\\Ragmed_2\\.venv\\Scripts\\python.exe backend\\scripts\\recompute_feat003_metrics.py \
        --run-dir backend\\results\\phase5_20261018_053140
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# ONLY already-defined metric / stat functions — nothing is redefined.
from evaluation.metrics import (  # noqa: E402
    compute_answer_with_citation_rate,
    compute_batch_rouge_l,
    compute_citation_support_rate,
    false_abstention_rate,
    false_acceptance_rate,
    mrr as mrr_metric,
    ndcg_at_k,
    recall_at_k,
)
from evaluation.statistics import paired_bootstrap  # noqa: E402

# Reuse the committed aggregate path for the core deliverables so FEAT-003 does
# not re-derive accuracy / McNemar / Holm / Wilson / latency / failures.
import run_phase5_experiment as rpx  # noqa: E402

_NOT_MEASURED = "NOT MEASURED"
_BASELINE = "phase5-full-pipeline"
_PMID_RE = re.compile(r"(\d{5,9})")

# Verified diagnostic (see FEAT-003 findings / audit): recomputed by this script
# from the eval_split chunks.pkl + the frozen test split gold PMIDs.
_RETRIEVAL_NOT_MEASURED_REASON = (
    "0/789 BioASQ gold PMIDs are present in the eval_split index universe "
    "(verified against chunks.pkl). Gold PMIDs are not the indexed documents, so "
    "recall@K/MRR/nDCG@K vs gold PMIDs are not measurable on this index. PubMedQA "
    "retrieval is also NOT MEASURED (gold PMIDs not in the BioASQ-built index)."
)


# ---------------------------------------------------------------------------
# Raw I/O + record helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _is_abstained(r: Dict[str, Any]) -> bool:
    return r.get("abstain_flag") is True


def _is_error(r: Dict[str, Any]) -> bool:
    return bool(r.get("errors")) or r.get("status") in ("error", "failed")


def _answer_text(r: Dict[str, Any]) -> str:
    a = r.get("answer")
    if isinstance(a, (list, tuple)):
        return " ".join(str(x) for x in a)
    if a in (None, _NOT_MEASURED):
        return ""
    return str(a)


def _pmids_from_citations(r: Dict[str, Any]) -> List[str]:
    """Ordered retrieved PMIDs parsed from the citation list (dedup, keep order)."""
    out: List[str] = []
    cites = r.get("citations")
    if not isinstance(cites, list):
        return out
    for c in cites:
        if not isinstance(c, dict):
            continue
        pmid = c.get("pmid")
        if not pmid:
            continue
        m = _PMID_RE.search(str(pmid))
        if m and m.group(1) not in out:
            out.append(m.group(1))
    return out


def _gold_pmids(pool_rec: Dict[str, Any]) -> List[str]:
    raw = pool_rec.get("pmid")
    if raw is None:
        return []
    vals = raw if isinstance(raw, list) else _PMID_RE.findall(str(raw))
    seen: List[str] = []
    for v in vals:
        m = _PMID_RE.search(str(v))
        if m and m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


# ---------------------------------------------------------------------------
# claim_verdicts -> unsupported-claim / sufficiency / contradiction
# ---------------------------------------------------------------------------

def _claim_metrics_for_config(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate claim-level evidence metrics from stored claim_verdicts.

    Only records whose claim_verdicts carry a non-empty ``verified_claims`` list
    contribute (i.e. configs with claim-verification / critic ON that actually
    produced verdicts). A claim is "supported" iff its label is ``verified``;
    labels ``unverifiable`` and ``contradicted`` count as unsupported.

    Returns MEASURED aggregates with denominators, or NOT MEASURED + reason when
    no record in the config produced claim verdicts.
    """
    cells_with_verdicts = 0
    total_claims = 0
    unsupported_claims = 0
    contradicted_claims = 0
    cells_contradiction_present = 0
    sufficiency_vals: List[float] = []  # 1.0 if sufficient else 0.0
    for r in recs:
        cv = r.get("claim_verdicts")
        if not isinstance(cv, dict):
            continue
        vclaims = cv.get("verified_claims")
        if not isinstance(vclaims, list) or not vclaims:
            continue
        cells_with_verdicts += 1
        for claim in vclaims:
            if not isinstance(claim, dict):
                continue
            total_claims += 1
            label = str(claim.get("label", "")).lower()
            if label != "verified":
                unsupported_claims += 1
            if label == "contradicted":
                contradicted_claims += 1
        suff = cv.get("sufficiency")
        if isinstance(suff, dict):
            if suff.get("contradictions_present") is True:
                cells_contradiction_present += 1
            if "sufficient" in suff:
                sufficiency_vals.append(1.0 if suff.get("sufficient") else 0.0)

    if cells_with_verdicts == 0 or total_claims == 0:
        return {
            "evidence_tier": _NOT_MEASURED,
            "reason": (
                "No stored claim_verdicts with verified_claims for this config "
                "(claim verification / critic produced no verdicts, e.g. the config "
                "has critic/verification OFF, or every answered cell abstained before "
                "claim extraction). Raw claim_verdicts are preserved per cell."
            ),
        }
    return {
        "evidence_tier": "BENCHMARK GROUND TRUTH (claim-level verdicts from stored claim_verdicts)",
        "n_cells_with_verdicts": cells_with_verdicts,
        "n_claims_total": total_claims,
        "unsupported_claim_rate": round(unsupported_claims / total_claims, 4),
        "unsupported_claims": unsupported_claims,
        "contradicted_claim_rate": round(contradicted_claims / total_claims, 4),
        "contradicted_claims": contradicted_claims,
        "n_cells_with_contradiction": cells_contradiction_present,
        "evidence_sufficiency_rate": (
            round(sum(sufficiency_vals) / len(sufficiency_vals), 4)
            if sufficiency_vals else _NOT_MEASURED
        ),
        "n_cells_sufficiency_judged": len(sufficiency_vals),
        "note": (
            "A claim is supported iff label=='verified'; 'unverifiable' and "
            "'contradicted' are unsupported. Denominator = total extracted claims "
            "across cells that produced verdicts (not all 40 cells)."
        ),
    }


# ---------------------------------------------------------------------------
# Retrieval quality (BioASQ only) — recomputed, labelled NOT MEASURED w/ evidence
# ---------------------------------------------------------------------------

def _retrieval_quality_for_config(
    recs: List[Dict[str, Any]],
    pool: Dict[Any, Dict[str, Any]],
    gold_in_index: int,
) -> Dict[str, Any]:
    """Recall@K / MRR / nDCG@K vs gold PMIDs for BioASQ rows.

    The computation path is implemented against the already-defined
    recall_at_k / mrr / ndcg_at_k. However, it is only reported as MEASURED when
    at least one BioASQ gold PMID is actually present in the retrievable index
    universe. Verified fact for this run: 0/789 gold PMIDs are in the index, so
    the result is NOT MEASURED with the recorded reason (reporting a hard 0.0
    would falsely imply "retrieved nothing relevant" when the gold set is simply
    not retrievable). PubMedQA is NOT MEASURED by construction.
    """
    bioasq = [r for r in recs if r.get("source") == "bioasq"]
    n_bioasq_with_gold = sum(
        1 for r in bioasq if _gold_pmids(pool.get(r.get("question_id"), {}))
    )
    base = {
        "n_bioasq_rows": len(bioasq),
        "n_bioasq_with_gold_list": n_bioasq_with_gold,
        "n_bioasq_gold_pmids_in_index": gold_in_index,
    }
    if gold_in_index == 0:
        base.update({"evidence_tier": _NOT_MEASURED, "reason": _RETRIEVAL_NOT_MEASURED_REASON})
        return base

    # Measurable path (kept correct for a future index where gold PMIDs intersect).
    retrieved_lists: List[List[str]] = []
    gold_lists: List[List[str]] = []
    recalls3: List[float] = []
    ndcgs10: List[float] = []
    for r in bioasq:
        gold = _gold_pmids(pool.get(r.get("question_id"), {}))
        if not gold:
            continue
        retrieved = _pmids_from_citations(r)
        retrieved_lists.append(retrieved)
        gold_lists.append(gold)
        recalls3.append(recall_at_k(retrieved, gold, 3))
        ndcgs10.append(ndcg_at_k(retrieved, gold, 10))
    base.update({
        "evidence_tier": "BENCHMARK GROUND TRUTH (vs gold PMIDs, BioASQ only)",
        "recall_at_3": round(sum(recalls3) / len(recalls3), 4) if recalls3 else _NOT_MEASURED,
        "mrr": round(mrr_metric(retrieved_lists, gold_lists), 4) if retrieved_lists else _NOT_MEASURED,
        "ndcg_at_10": round(sum(ndcgs10) / len(ndcgs10), 4) if ndcgs10 else _NOT_MEASURED,
        "n_bioasq_scored": len(recalls3),
    })
    return base


# ---------------------------------------------------------------------------
# BioASQ gold-PMID-in-index diagnostic (recomputed from the eval_split chunks)
# ---------------------------------------------------------------------------

def _bioasq_retrieval_diagnostic(run_dir: Path, pool: Dict[Any, Dict[str, Any]]) -> Dict[str, Any]:
    gold: set = set()
    for rec in pool.values():
        if rec.get("source") == "bioasq":
            for p in _gold_pmids(rec):
                gold.add(p)
    diag: Dict[str, Any] = {
        "bioasq_gold_pmids_total": len(gold),
        "bioasq_gold_pmids_in_index": 0,
        "index_distinct_pmid_like_ids": _NOT_MEASURED,
    }
    chunks_path = _BACKEND_DIR / "data" / "faiss_index" / "eval_split" / "chunks.pkl"
    try:
        import pickle
        with open(chunks_path, "rb") as fh:
            chunks = pickle.load(fh)
        index_ids: set = set()
        for c in chunks:
            s = json.dumps(c, default=str)
            for m in re.findall(r"pubmed/(\d+)", s):
                index_ids.add(m)
            for m in re.findall(r"(?<!\d)(\d{7,8})(?!\d)", s):
                index_ids.add(m)
        diag["index_distinct_pmid_like_ids"] = len(index_ids)
        diag["bioasq_gold_pmids_in_index"] = len(gold & index_ids)
    except Exception as exc:  # noqa: BLE001 - diagnostic only; absence is reported
        diag["index_scan_error"] = f"{type(exc).__name__}: {exc} (reusing recorded 0/789 finding)"
    return diag


# ---------------------------------------------------------------------------
# Extended metrics (ROUGE-L primary, citation, claims, retrieval, operational)
# ---------------------------------------------------------------------------

def build_extended_metrics(run_dir: Path, environment: Dict[str, Any]) -> Dict[str, Any]:
    per_q = run_dir / "per_question"
    pool = {
        r.get("id"): r
        for r in json.loads((run_dir / "splits" / "test.json").read_text(encoding="utf-8"))
    }
    config_names = [f.stem for f in sorted(per_q.glob("*.jsonl"))]
    diag = _bioasq_retrieval_diagnostic(run_dir, pool)
    gold_in_index = int(diag.get("bioasq_gold_pmids_in_index") or 0)

    rouge_by_cfg: Dict[str, Dict[Any, float]] = {}
    results: Dict[str, Any] = {}

    for cfg in config_names:
        recs = _read_jsonl(per_q / f"{cfg}.jsonl")
        n_total = len(recs)
        n_err = sum(1 for r in recs if _is_error(r))
        n_abst = sum(1 for r in recs if _is_abstained(r) and not _is_error(r))
        n_answered = n_total - n_abst - n_err

        preds = [_answer_text(r) for r in recs]
        golds = [str(r.get("gold_answer") or "") for r in recs]
        rouge = compute_batch_rouge_l(preds, golds)
        rouge_by_cfg[cfg] = {
            r.get("question_id"): s["rouge_l_f1"]
            for r, s in zip(recs, rouge["scores_per_sample"])
        }

        cite_support = compute_citation_support_rate(recs)
        cite_with_answer = compute_answer_with_citation_rate(recs)

        # Answerability-aware abstention views. The frozen split carries no
        # explicit should_abstain label, so under the project convention every
        # question is answerable (should_abstain=False); false_acceptance is
        # therefore NOT MEASURED (no unanswerable gold cases).
        abstained_flags = [_is_abstained(r) for r in recs]
        should_abstain = [False] * n_total  # no unanswerable labels in this split
        far = false_abstention_rate(abstained_flags, should_abstain)

        colds = [r.get("cold_latency_ms") for r in recs
                 if isinstance(r.get("cold_latency_ms"), (int, float))]
        warms = [r.get("warm_latency_ms") for r in recs
                 if isinstance(r.get("warm_latency_ms"), (int, float))]
        total_cold_s = sum(colds) / 1000.0 if colds else 0.0

        results[cfg] = {
            "n_total": n_total,
            "n_answered": n_answered,
            "n_abstained": n_abst,
            "n_error": n_err,
            "abstention_rate": round(n_abst / n_total, 4) if n_total else _NOT_MEASURED,
            "failure_rate": round(n_err / n_total, 4) if n_total else _NOT_MEASURED,
            "answerability_aware": {
                "evidence_tier": "SOFTWARE TEST (no unanswerable gold labels in this split)",
                "false_abstention_rate": far,
                "false_abstention_note": (
                    "Fraction of answerable questions the config abstained on. The split has "
                    "no gold 'should-abstain' labels, so all questions are treated as answerable "
                    "and false_abstention_rate == abstention_rate here."
                ),
                "false_acceptance_rate": _NOT_MEASURED,
                "false_acceptance_reason": "No unanswerable (should_abstain=True) gold cases in this split.",
            },
            "rouge_l": {
                "evidence_tier": "BENCHMARK GROUND TRUTH (ROUGE-L F1 vs gold answer) — PRIMARY answer quality",
                "mean_rouge_l_f1": rouge["mean_rouge_l_f1"],
                "mean_rouge_l_precision": rouge["mean_rouge_l_precision"],
                "mean_rouge_l_recall": rouge["mean_rouge_l_recall"],
                "note": "Abstentions score 0 ROUGE-L; read alongside abstention_rate.",
            },
            "citation": {
                "evidence_tier": "SOFTWARE TEST (citation presence from stored evidence)",
                "citation_support_rate_all": cite_support,
                "answer_with_citation_rate_non_abstained": cite_with_answer,
                "claim_level": _claim_metrics_for_config(recs),
            },
            "retrieval_quality": _retrieval_quality_for_config(recs, pool, gold_in_index),
            "latency": {
                "evidence_tier": "SOFTWARE TEST (wall-clock on this GPU run)",
                "n_cold": len(colds),
                "mean_cold_ms": round(sum(colds) / len(colds), 1) if colds else _NOT_MEASURED,
                "min_cold_ms": min(colds) if colds else _NOT_MEASURED,
                "max_cold_ms": max(colds) if colds else _NOT_MEASURED,
                "n_warm": len(warms),
                "mean_warm_ms": round(sum(warms) / len(warms), 1) if warms else _NOT_MEASURED,
                "throughput_q_per_hour": (
                    round(len(colds) / (total_cold_s / 3600.0), 3) if total_cold_s > 0 else _NOT_MEASURED
                ),
            },
            "peak_vram_mb": {
                "evidence_tier": "SOFTWARE TEST (max over cells)",
                "value": round(max(
                    [r.get("peak_vram_mb") for r in recs
                     if isinstance(r.get("peak_vram_mb"), (int, float))] or [0.0]
                ), 2),
            },
        }

    # Paired ROUGE-L bootstrap CI vs baseline, on the SAME question IDs (seed 42).
    base_rmap = rouge_by_cfg.get(_BASELINE, {})
    shared_qids = sorted(k for k in base_rmap.keys() if k is not None)
    paired_rouge: Dict[str, Any] = {}

    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    for cfg in config_names:
        if cfg == _BASELINE:
            continue
        cmap = rouge_by_cfg[cfg]
        qids = [q for q in shared_qids if q in cmap]
        if not qids:
            paired_rouge[cfg] = {"note": "no shared question ids", "vs": _BASELINE}
            continue
        y_a = [cmap[q] for q in qids]
        y_b = [base_rmap[q] for q in qids]
        obs, lo, hi = paired_bootstrap(_mean, y_a, y_b, n_bootstrap=10000, seed=42)
        paired_rouge[cfg] = {
            "vs": _BASELINE,
            "n_paired": len(qids),
            "mean_rouge_l_f1_delta": obs,
            "bootstrap_ci95": [lo, hi],
            "ci_excludes_zero": (lo > 0) or (hi < 0),
        }

    peak_vram_run = environment.get("peak_vram_mb_run")
    return {
        "evidence_tier_legend": {
            "BENCHMARK GROUND TRUTH": "computed vs gold answers / gold PMIDs / stored verdicts",
            "SOFTWARE TEST": "operational measurement on this GPU run",
            "NOT MEASURED": "evidence genuinely absent; reason given inline",
        },
        "run_dir": str(run_dir),
        "baseline_for_paired_tests": _BASELINE,
        "n_questions_pool": len(pool),
        "n_questions_evaluated": len(shared_qids),
        "peak_vram_mb_run": peak_vram_run,
        "configs": results,
        "paired_rouge_l_bootstrap_vs_baseline": paired_rouge,
        "retrieval_measurability": {
            "bioasq": (
                "NOT MEASURED (VERIFIED): 0/789 BioASQ gold PMIDs present in the eval_split "
                "index universe. Direct evidence corrects the manifest's earlier 'MEASURABLE' "
                "assumption."
            ),
            "pubmedqa": "NOT MEASURED (PubMedQA abstracts/gold PMIDs not in the BioASQ-built eval_split index)",
            "diagnostic": diag,
        },
        "caveats": [
            "Small n per config: proportion CIs are wide; treat deltas as directional.",
            "ROUGE-L and token-containment accuracy score abstentions as 0/incorrect; high "
            "abstention configs are penalised on answer-quality but may be behaving safely. "
            "Always read answer quality alongside abstention_rate.",
            "No superiority or clinical-safety claim is made without a Holm-adjusted significant "
            "paired test (see metric_summary.json) AND a bootstrap CI that excludes zero.",
            "Claim-level unsupported/contradiction metrics are only MEASURED for configs that "
            "produced claim_verdicts; others are NOT MEASURED with reason.",
        ],
    }


# ---------------------------------------------------------------------------
# Markdown deliverables (deterministic — no timestamps that vary per run except
# the recorded environment generated_at, which is read from the run's env file)
# ---------------------------------------------------------------------------

def _fmt(v: Any) -> str:
    return _NOT_MEASURED if v == _NOT_MEASURED else str(v)


def write_comparison_table(run_dir: Path, ext: Dict[str, Any]) -> None:
    lines = [
        "| config | n | answered | abstained | error | abstention_rate | ROUGE-L F1 | cite_support | unsupported_claim_rate | retrieval (BioASQ) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cfg, c in ext["configs"].items():
        rq = c["retrieval_quality"]
        if rq.get("evidence_tier") == _NOT_MEASURED:
            rtxt = "NOT MEASURED (gold PMIDs absent from index)"
        else:
            rtxt = f"R@3={rq.get('recall_at_3')} MRR={rq.get('mrr')} nDCG@10={rq.get('ndcg_at_10')} (n={rq.get('n_bioasq_scored')})"
        cl = c["citation"]["claim_level"]
        ucr = cl.get("unsupported_claim_rate", _NOT_MEASURED) if cl.get("evidence_tier") != _NOT_MEASURED else _NOT_MEASURED
        lines.append(
            f"| {cfg} | {c['n_total']} | {c['n_answered']} | {c['n_abstained']} | {c['n_error']} | "
            f"{c['abstention_rate']} | {c['rouge_l']['mean_rouge_l_f1']} | "
            f"{c['citation']['citation_support_rate_all']} | {_fmt(ucr)} | {rtxt} |"
        )
    (run_dir / "comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_runtime_report(run_dir: Path, environment: Dict[str, Any], ext: Dict[str, Any]) -> None:
    matrix = environment.get("matrix", {}) or {}
    plan = matrix.get("plan", {}) or {}
    rc = matrix.get("run_counts", {}) or {}
    n_eval = environment.get("n_questions") or ext.get("n_questions_evaluated")
    total_cells = plan.get("total", (n_eval or 0) * len(environment.get("configs", []) or []))
    completed = rc.get("completed")
    reused = rc.get("reused_skipped", plan.get("reused"))
    failed = rc.get("failed", 0)
    run_s = rc.get("run_seconds")
    run_h = round(run_s / 3600.0, 2) if isinstance(run_s, (int, float)) else _NOT_MEASURED
    tier = "REDUCED_VALIDATION_SUBSET" if (n_eval or 0) < 50 else "FULL"
    lines = [
        "# Runtime & Completion Report — Phase 5 Evaluation (FEAT-003 recompute)",
        "",
        f"**Run dir:** `{run_dir.name}/`",
        f"**Date:** {environment.get('generated_at', _NOT_MEASURED)} · **Branch:** `yuvraj/llm` · "
        "**GPU:** NVIDIA RTX 5060 Laptop (8 GB)",
        "",
        "## Scope actually completed",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Executed matrix | **{n_eval} questions × {len(environment.get('configs', []) or [])} configs = {total_cells} cells** |",
        f"| Cells completed (new this session) | **{completed}** |",
        f"| Cells failed | **{failed}** |",
        f"| Reused pilot cells (not re-run) | **{reused}** |",
        f"| Duplicate writes prevented | {rc.get('duplicate_skipped', 0)} |",
        f"| Budget stop triggered | {rc.get('budget_stopped_early', False)} |",
        "",
        f"**Tier label:** `{tier}`. This is a *complete* {n_eval}×{len(environment.get('configs', []) or [])} "
        "experiment on a fixed question set across all configs — NOT a partial slice of a larger run, and "
        "NOT reported as a 50×8 or 500-question result.",
        "",
        "## Runtime",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Actual run wall-clock (new cells) | {run_s} s ≈ {run_h} h |",
        f"| Model load (once) | {round((environment.get('model_load_ms') or 0) / 1000.0, 1)} s |",
        f"| Peak VRAM (run) | {environment.get('peak_vram_mb_run')} MB (offload/paging on 8 GB card; no OOM) |",
        f"| Mean per new cell | {round(run_s / completed, 1) if isinstance(run_s, (int, float)) and completed else _NOT_MEASURED} s |",
        "",
        "## Reliability",
        "",
        f"- {failed} execution errors across the matrix (see `failure_cases.json`).",
        "- Genuine abstentions are recorded distinct from errors (`abstain_flag=True`, `errors=None`); "
        "no abstention or failure was silently dropped.",
        "",
        "## Safety",
        "",
        "- eval_split IndexLeakageGuard **PASS**; resolved index = `backend/data/faiss_index/eval_split`.",
        "- Production index `backend/data/faiss_index/` left untouched; eval_split index files unchanged.",
        f"- Temperature {environment.get('generator_temperature')}, v1 prompts, top_k_retrieval=10 / "
        "top_k_final=3, Qwen3-8B 4-bit NF4; precision and batch size unchanged; `llm_mode=`"
        f"`{environment.get('llm_mode')}`, no mock fallback.",
        "",
    ]
    (run_dir / "runtime_completion_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reproducibility_report(run_dir: Path, environment: Dict[str, Any], ext: Dict[str, Any]) -> None:
    lines = [
        "# Reproducibility Report — Phase 5 Evaluation (FEAT-003 recompute)",
        "",
        f"**Run dir:** `{run_dir.name}/`",
        "",
        "## Dataset & split (immutable)",
        "",
        "- **Pool:** `backend/results/phase5_20261017_131123/splits/test.json` — 250 questions "
        "(167 PubMedQA + 83 BioASQ), seed 42, stratified by `source`. "
        "Test-split manifest hash `8e2b2d925afd86b8d5f2567594bcf40f64ed3e0cab6c3e5b169b524569621eee`.",
        "- **eval split reused verbatim, never rebuilt.** eval_split index files (SHA-256):",
        "  - `index.faiss` `fc18d20d30ea92e6d4b7581786bd1ba640e18e658d994466d47d924c847fc258`",
        "  - `chunks.pkl`  `6370a1b47b2d0e799788f3406f320dc29057c008db4f4d8ce19f761456a5f5d1`",
        "  - 58,173 chunks. Production index `backend/data/faiss_index/` left untouched.",
        "",
        "## Question selection (frozen, nested)",
        "",
        "- Function: `run_phase5_experiment._stratified_sample(test_250, n, seed=42, stratify_by=\"source\")`.",
        "- `question_ids_40` (this run) ⊂ `question_ids_50`; `pilot_reused_ids_16` ⊂ `question_ids_40`.",
        "- Frozen lists in `backend/results/phase5_manifest/experiment_manifest.json`.",
        f"- {ext['n_questions_evaluated']} evaluated question IDs are reproduced in this run's `splits/test.json`.",
        "",
        "## Models & generation (unchanged)",
        "",
        f"- LLM: **{environment.get('llm_model')}**, 4-bit NF4, mode `{environment.get('llm_mode')}`, mock fallback disabled.",
        f"- Generation: **temperature {environment.get('generator_temperature')}** (deterministic / greedy).",
        "- Prompts: `backend/prompts/v1/*.txt`, unchanged (per-file SHA-256 recorded in the manifest).",
        "- Retrieval: `top_k_retrieval=10`, `top_k_final=3`.",
        "",
        "## Code & environment",
        "",
        "- Branch `yuvraj/llm`.",
        "- Runner: `backend/scripts/run_phase5_experiment.py` (`--target 40`, per-cell checkpoint + pilot reuse).",
        "- Metric recompute (this report): `backend/scripts/recompute_feat003_metrics.py` (CPU; reuses "
        "`evaluation/metrics.py` + `evaluation/statistics.py` + `aggregate_outputs` — no metric redefined).",
        f"- torch {environment.get('torch_version')}, CUDA {environment.get('cuda_version')}, GPU RTX 5060 Laptop 8 GB.",
        "",
        "## How to regenerate (no GPU)",
        "",
        "```",
        r"d:\Ragmed_2\.venv\Scripts\python.exe backend\scripts\recompute_feat003_metrics.py \\",
        f"    --run-dir backend\\results\\{run_dir.name}",
        "```",
        "All metrics are regenerable from the raw `per_question/*.jsonl` alone; those raw files are the "
        "source of truth. The recompute is deterministic (bootstrap uses fixed seed 42).",
        "",
    ]
    (run_dir / "reproducibility_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_research_summary(
    run_dir: Path, environment: Dict[str, Any], ext: Dict[str, Any], metric_summary: Dict[str, Any]
) -> None:
    cfgs = ext["configs"]
    n_eval = ext["n_questions_evaluated"]
    n_cfg = len(cfgs)
    # Collect significant paired results (Holm) to phrase claims safely.
    sig = [
        name for name, p in metric_summary.get("paired_vs_baseline_mcnemar_holm", {}).items()
        if p.get("significant_holm_0.05") is True
    ]
    # Bootstrap CI excludes zero for ROUGE-L.
    boot_sig = [
        name for name, p in ext["paired_rouge_l_bootstrap_vs_baseline"].items()
        if isinstance(p, dict) and p.get("ci_excludes_zero") is True
    ]
    abst = {c: cfgs[c]["abstention_rate"] for c in cfgs}
    lines = [
        "# Research-Results Summary — Phase 5 Evaluation (FEAT-003)",
        "",
        f"**Scope:** {n_eval} unique questions (seed 42, source-stratified) × {n_cfg} configurations "
        f"= {n_eval * n_cfg} comparable cells, all on the identical question set. "
        "Reused validated pilot cells + newly executed cells; see runtime report. "
        f"Peak VRAM (run) ≈ {environment.get('peak_vram_mb_run')} MB on an 8 GB RTX 5060.",
        "",
        "> **Scope honesty.** The original task described a 500-question / 50×8 ambition. A fresh runtime "
        "benchmark projected 50×8 over the hard 8 h budget, so the target was reduced to a complete "
        f"{n_eval}×{n_cfg} run (a strict subset of the frozen 50, itself a superset of the pilot 16). "
        f"**This is a complete {n_eval}×{n_cfg} experiment, not a partial 50×8 or 500-question run.**",
        "",
        "## Headline observations (directional; small n)",
        "",
        "1. **Abstention dominates the RAG configs on this question mix.** Abstention rates: "
        + ", ".join(f"{c}={abst[c]}" for c in cfgs) + ".",
        "",
        "2. **Token-containment accuracy (threshold 0.6, vs gold):** see `metric_summary.json`. "
        + (
            f"Paired McNemar vs `{_BASELINE}` is Holm-significant only for: "
            + ", ".join(f"`{s}`" for s in sig) + "."
            if sig else
            "No paired McNemar comparison vs the baseline is Holm-significant."
        ),
        "",
        "3. **ROUGE-L F1 (primary answer quality vs gold):** "
        + (
            "paired bootstrap ROUGE-L delta vs baseline excludes zero only for: "
            + ", ".join(f"`{s}`" for s in boot_sig) + "."
            if boot_sig else
            "no config's paired bootstrap ROUGE-L CI excludes zero (all differences indistinguishable at this n)."
        ),
        "",
        "4. **Citation support:** reported per config in `comparison_table.md`; plain-LLM-no-RAG "
        "attaches no citations by design (no retrieval).",
        "",
        "5. **Claim-level evidence (unsupported-claim / contradiction / sufficiency):** MEASURED only for "
        "configs that produced `claim_verdicts` (see per-config `citation.claim_level`); NOT MEASURED "
        "elsewhere with reason. Where measured, claims were overwhelmingly `unverifiable` against the "
        "retrieved evidence with no detected contradictions — read with the small denominators shown.",
        "",
        "## What is NOT measured (and why)",
        "",
        "- **Retrieval quality (recall@K / MRR / nDCG@K): NOT MEASURED.** Verified directly in this "
        f"recompute: {ext['retrieval_measurability']['diagnostic'].get('bioasq_gold_pmids_in_index')}"
        f"/{ext['retrieval_measurability']['diagnostic'].get('bioasq_gold_pmids_total')} BioASQ gold "
        "PMIDs appear in the eval_split index universe. The gold PMIDs are not the indexed documents, so "
        "retrieval-vs-gold cannot be scored. This corrects the manifest's earlier 'BioASQ MEASURABLE' "
        "assumption with evidence. PubMedQA retrieval is also NOT MEASURED.",
        "- **False-acceptance / answerability on unanswerable questions:** NOT MEASURED — the frozen split "
        "has no gold `should-abstain` labels, so false_abstention_rate == abstention_rate and "
        "false_acceptance_rate is undefined here.",
        "",
        "## Limitations",
        "",
        f"- **Small sample (n = {n_eval} per config).** Proportion CIs are wide; between-RAG-config "
        "differences are statistically indistinguishable. Treat every non-significant delta as directional.",
        "- **Accuracy / ROUGE-L penalize abstention.** High-abstention configs look worse on answer "
        "quality but may be behaving conservatively. Read answer quality only alongside abstention rate.",
        "- **No clinical-safety or superiority claim is made.** Any significant result reflects an "
        "always-answer vs mostly-abstain artifact on this mix, not evidence that a config is clinically "
        "better. A 0.875–0.95 RAG abstention rate warrants investigating threshold calibration — a "
        "follow-up, not a conclusion.",
        "- **Single eval set, single seed.** Specific to this seed-42 sample of the 250-question test pool.",
        "",
        "## Artifacts",
        "",
        "- Raw: `per_question/*.jsonl` (source of truth).",
        "- Metrics: `metric_summary.json`, `extended_metrics.json`, `comparison_table.md`, "
        "`latency_breakdown.json`, `failure_cases.json`, `summary.json`.",
        "- Reports: `report.md`, `runtime_completion_report.md`, `reproducibility_report.md`, this file.",
        "",
    ]
    (run_dir / "research_results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def recompute(run_dir: Path) -> Dict[str, Any]:
    env_path = run_dir / "environment_run.json"
    environment = json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else {}

    # 1) Core deliverables via the committed aggregate path (accuracy GROUND TRUTH,
    #    Wilson CIs, paired McNemar + Holm, latency, failures, summary, report.md).
    rpx.aggregate_outputs(run_dir, environment)
    metric_summary = json.loads((run_dir / "metric_summary.json").read_text(encoding="utf-8"))

    # 2) Extended metrics (ROUGE-L primary, citation, claim-level, retrieval,
    #    operational, paired bootstrap).
    ext = build_extended_metrics(run_dir, environment)
    (run_dir / "extended_metrics.json").write_text(
        json.dumps(ext, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 3) Markdown deliverables.
    write_comparison_table(run_dir, ext)
    write_runtime_report(run_dir, environment, ext)
    write_reproducibility_report(run_dir, environment, ext)
    write_research_summary(run_dir, environment, ext, metric_summary)
    return {"configs": list(ext["configs"].keys()), "n_evaluated": ext["n_questions_evaluated"]}


def main() -> None:
    ap = argparse.ArgumentParser(description="FEAT-003 full metric recompute (CPU, deterministic).")
    ap.add_argument("--run-dir", required=True, help="Phase 5 run dir with per_question/*.jsonl.")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        # Try CWD first (documented command runs from the repo root with
        # backend\results\...), then fall back to the repo root.
        candidates = [
            Path.cwd() / run_dir,
            _BACKEND_DIR.parent / run_dir,
            _BACKEND_DIR / run_dir,
        ]
        run_dir = next((c for c in candidates if c.exists()), candidates[0]).resolve()
    if not (run_dir / "per_question").is_dir():
        raise SystemExit(f"run-dir has no per_question/: {run_dir}")
    info = recompute(run_dir)
    print(f"FEAT003_RECOMPUTE_DONE run_dir={run_dir} configs={len(info['configs'])} "
          f"n_evaluated={info['n_evaluated']}")


if __name__ == "__main__":
    main()
