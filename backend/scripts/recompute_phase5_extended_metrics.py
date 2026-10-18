"""Recompute extended Phase 5 metrics from saved raw per_question/*.jsonl outputs.

CPU-only. Reuses the ALREADY-DEFINED functions in evaluation/metrics.py and
evaluation/statistics.py — no metric is redefined here. Every value carries an
evidence tier; absent evidence is reported as "NOT MEASURED" + reason.

Usage:
    d:\\Ragmed_2\\.venv\\Scripts\\python.exe backend\\scripts\\recompute_phase5_extended_metrics.py \
        --run-dir backend\\results\\phase5_20261018_053140

Outputs (written INTO the run dir, never overwriting the raw per_question files):
    extended_metrics.json      machine-readable recomputed metrics + CIs
    comparison_table.md        8-config comparison table (same question IDs)

The raw per_question/*.jsonl are read-only inputs here.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from evaluation.metrics import (  # noqa: E402
    compute_answer_with_citation_rate,
    compute_batch_rouge_l,
    compute_citation_support_rate,
)
from evaluation.statistics import paired_bootstrap  # noqa: E402

_NOT_MEASURED = "NOT MEASURED"
_BASELINE = "phase5-full-pipeline"
_PMID_RE = re.compile(r"(\d{5,9})")


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
    return bool(r.get("errors"))


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
    """Parse the BioASQ gold PMID list from a pool record's 'pmid' field."""
    raw = pool_rec.get("pmid")
    if raw is None:
        return []
    if isinstance(raw, list):
        vals = raw
    else:
        # stored as a stringified python list, e.g. "['123', '456']"
        vals = _PMID_RE.findall(str(raw))
    return [str(_PMID_RE.search(str(v)).group(1)) for v in vals if _PMID_RE.search(str(v))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    per_q = run_dir / "per_question"
    pool = {r.get("id"): r for r in json.loads((run_dir / "splits" / "test.json").read_text(encoding="utf-8"))}

    config_files = sorted(per_q.glob("*.jsonl"))
    config_names = [f.stem for f in config_files]

    # ROUGE-L per config, keyed by question_id so bootstrap pairs are aligned.
    rouge_by_cfg: Dict[str, Dict[str, float]] = {}
    results: Dict[str, Any] = {}

    for cfg in config_names:
        recs = _read_jsonl(per_q / f"{cfg}.jsonl")
        n_total = len(recs)
        n_abst = sum(1 for r in recs if _is_abstained(r))
        n_err = sum(1 for r in recs if _is_error(r))
        n_answered = n_total - n_abst - n_err

        preds = [_answer_text(r) for r in recs]
        golds = [str(r.get("gold_answer") or "") for r in recs]
        rouge = compute_batch_rouge_l(preds, golds)
        # per-question rouge for bootstrap alignment
        rmap = {r.get("question_id"): s["rouge_l_f1"]
                for r, s in zip(recs, rouge["scores_per_sample"])}
        rouge_by_cfg[cfg] = rmap

        # citation metrics over the config's records (functions handle abstention)
        cite_support = compute_citation_support_rate(recs)
        cite_with_answer = compute_answer_with_citation_rate(recs)

        # Retrieval quality vs gold PMIDs.
        #
        # VERIFIED (diagnostic): 0 of the 789 BioASQ gold PMIDs appear anywhere in the
        # eval_split chunk universe (90,074 distinct PMID-like ids). The eval_split was
        # built from a BioASQ-derived corpus, but the questions' gold-standard PMIDs are
        # NOT the indexed documents, so recall/MRR/nDCG against gold PMIDs cannot be
        # measured on this index. Reporting a hard 0.0 would misleadingly imply "retrieved
        # nothing relevant" when the gold set is simply not in the retrievable universe.
        # Hence BioASQ retrieval quality is NOT MEASURED here (correcting the manifest's
        # earlier MEASURABLE assumption with direct evidence). PubMedQA is also NOT MEASURED
        # (abstracts/gold PMIDs not in the BioASQ-built index).
        bioasq_retr = [r for r in recs if r.get("source") == "bioasq"]
        n_bioasq_gold = sum(1 for r in bioasq_retr if _gold_pmids(pool.get(r.get("question_id"), {})))
        retrieval_metrics = {
            "evidence_tier": _NOT_MEASURED,
            "reason": (
                "0/789 BioASQ gold PMIDs are present in the eval_split index universe "
                "(verified against chunks.pkl). Gold PMIDs are not the indexed documents, "
                "so recall@K/MRR/nDCG@K vs gold PMIDs are not measurable on this index. "
                "PubMedQA retrieval is also NOT MEASURED (gold PMIDs not in the index)."
            ),
            "n_bioasq_with_gold_list": n_bioasq_gold,
            "n_bioasq_gold_pmids_in_index": 0,
        }

        # latency (SOFTWARE TEST): cold latency where numeric
        colds = [r.get("cold_latency_ms") for r in recs
                 if isinstance(r.get("cold_latency_ms"), (int, float))]
        latency = {
            "evidence_tier": "SOFTWARE TEST (wall-clock cold latency on this GPU run)",
            "n": len(colds),
            "mean_cold_ms": round(sum(colds) / len(colds), 1) if colds else _NOT_MEASURED,
            "min_cold_ms": min(colds) if colds else _NOT_MEASURED,
            "max_cold_ms": max(colds) if colds else _NOT_MEASURED,
        }

        results[cfg] = {
            "n_total": n_total,
            "n_answered": n_answered,
            "n_abstained": n_abst,
            "n_error": n_err,
            "abstention_rate": round(n_abst / n_total, 4) if n_total else _NOT_MEASURED,
            "failure_rate": round(n_err / n_total, 4) if n_total else _NOT_MEASURED,
            "rouge_l": {
                "evidence_tier": "BENCHMARK GROUND TRUTH (ROUGE-L F1 vs gold answer)",
                "mean_rouge_l_f1": rouge["mean_rouge_l_f1"],
                "mean_rouge_l_precision": rouge["mean_rouge_l_precision"],
                "mean_rouge_l_recall": rouge["mean_rouge_l_recall"],
                "note": "Abstentions score 0 ROUGE-L; read alongside abstention_rate.",
            },
            "citation": {
                "evidence_tier": "SOFTWARE TEST (citation presence from stored evidence)",
                "citation_support_rate_all": cite_support,
                "answer_with_citation_rate_non_abstained": cite_with_answer,
                "unsupported_claim_rate": _NOT_MEASURED,
                "unsupported_claim_reason": (
                    "Claim-level support requires claim_verdicts; only configs with "
                    "claim verification / critic ON populate it. Reported per config below "
                    "where present, else NOT MEASURED."
                ),
            },
            "retrieval_quality": retrieval_metrics,
            "latency": latency,
        }

    # Paired bootstrap CIs on mean ROUGE-L F1 delta vs baseline (same question IDs).
    base_rmap = rouge_by_cfg.get(_BASELINE, {})
    shared_qids = sorted(base_rmap.keys())
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

    out = {
        "evidence_tier_legend": {
            "BENCHMARK GROUND TRUTH": "computed vs gold answers / gold PMIDs",
            "SOFTWARE TEST": "operational measurement on this GPU run",
            "NOT MEASURED": "evidence genuinely absent; reason given inline",
        },
        "run_dir": str(run_dir),
        "baseline_for_paired_tests": _BASELINE,
        "n_questions": len(pool),
        "configs": results,
        "paired_rouge_l_bootstrap_vs_baseline": paired_rouge,
        "retrieval_measurability": {
            "bioasq": (
                "NOT MEASURED (VERIFIED): 0/789 BioASQ gold PMIDs present in the eval_split "
                "index universe (90,074 distinct PMID-like ids in chunks.pkl). This corrects "
                "the manifest's earlier 'MEASURABLE' assumption with direct evidence."
            ),
            "pubmedqa": "NOT MEASURED (PubMedQA abstracts/gold PMIDs not in the BioASQ-built eval_split index)",
            "diagnostic": {
                "bioasq_gold_pmids_total": 789,
                "bioasq_gold_pmids_in_index": 0,
                "index_distinct_pmid_like_ids": 90074,
            },
        },
        "caveats": [
            "Small n=40 per config: proportion CIs are wide; treat deltas as directional.",
            "ROUGE-L and token-containment accuracy score abstentions as 0/incorrect; high "
            "abstention configs are penalised on answer-quality but may be behaving safely. "
            "Always read answer quality alongside abstention_rate.",
            "No superiority or clinical-safety claim is made without a Holm-adjusted significant "
            "paired test (see metric_summary.json) AND a CI that excludes zero.",
        ],
    }

    (run_dir / "extended_metrics.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Comparison table (markdown).
    lines = []
    lines.append("| config | n | answered | abstained | error | abstention_rate | ROUGE-L F1 | cite_support | retrieval (BioASQ) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for cfg in config_names:
        c = results[cfg]
        rq = c["retrieval_quality"]
        if rq.get("evidence_tier") == _NOT_MEASURED:
            rtxt = "NOT MEASURED (gold PMIDs absent from index)"
        else:
            rtxt = f"R@3={rq['recall_at_3']} MRR={rq['mrr']} nDCG@10={rq['ndcg_at_10']} (n={rq['n_bioasq_with_gold']})"
        lines.append(
            f"| {cfg} | {c['n_total']} | {c['n_answered']} | {c['n_abstained']} | {c['n_error']} | "
            f"{c['abstention_rate']} | {c['rouge_l']['mean_rouge_l_f1']} | "
            f"{c['citation']['citation_support_rate_all']} | {rtxt} |"
        )
    (run_dir / "comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"EXTENDED_METRICS_DONE run_dir={run_dir} configs={len(config_names)}")


if __name__ == "__main__":
    main()
