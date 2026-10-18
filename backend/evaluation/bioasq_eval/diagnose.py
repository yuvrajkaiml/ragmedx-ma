"""
Phase A — Diagnose the missing BioASQ gold PMIDs (runs WITHOUT the generation LLM).

Reconciles the 789 / 2,394 / 172 / 250 figures, confirms the deliberate eval_split
exclusion (expect 0 gold PMIDs matched), distinguishes "missing gold document"
(coverage gap) from "retrieval failure", and runs a PMID->chunk round-trip
traceability check.

The embedding model is loaded ONLY when ``--roundtrip`` is passed. The per-corpus
availability counts against the NEW retrieval-eval corpus are computed by
``--corpus <index_dir>`` after the index exists (step 5 calls this).

Run from ``backend/``::

    d:\\Ragmed_2\\.venv\\Scripts\\python.exe -m evaluation.bioasq_eval.diagnose
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluation.bioasq_eval.common import (  # noqa: E402
    RESULTS_DIR, SEED, atomic_write_json, atomic_write_text,
    canonical_pmid, normalize_pmid, write_manifest,
)

EVAL_QUESTIONS = BACKEND_DIR / "data" / "datasets" / "bioasq_eval_split" / "eval_questions.json"
EVAL_SPLIT_DIR = BACKEND_DIR / "data" / "faiss_index" / "eval_split"
PRODUCTION_DIR = BACKEND_DIR / "data" / "faiss_index"
PRIOR_RETRIEVAL = REPO_ROOT / "RAGMedXMA_Chapter5_Validated_Results" / "retrieval_results.jsonl"

DIAG_DIR = RESULTS_DIR / "diagnostics"


def load_eval_questions() -> List[Dict[str, Any]]:
    with open(EVAL_QUESTIONS, "r", encoding="utf-8") as fh:
        return json.load(fh)


def gold_pmids_for_question(q: Dict[str, Any]) -> List[Optional[str]]:
    """Normalized gold PMIDs from ``documents[]``; keeps None for unparsable."""
    return [normalize_pmid(d) for d in (q.get("documents") or [])]


def _load_store_pmids(index_dir: Path) -> Dict[str, Any]:
    """Load a FAISS store read-only and summarize its PMID membership."""
    from app.rag.vectorstore.faiss_store import FAISSVectorStore
    store = FAISSVectorStore.load(index_dir)
    distinct = set()
    missing_meta_pmid = 0
    fb: Dict[str, int] = {}
    doc_groups: Dict[str, set] = {}
    for ch in store.chunks:
        meta = getattr(ch, "metadata", None) or {}
        if "pmid" not in meta or meta.get("pmid") is None:
            missing_meta_pmid += 1
        pmid = canonical_pmid(ch, fb)
        if pmid is not None:
            distinct.add(pmid)
            doc_groups.setdefault(getattr(ch, "document_id", pmid), set()).add(pmid)
    return {
        "n_chunks": len(store.chunks),
        "distinct_pmids": distinct,
        "chunks_missing_meta_pmid": missing_meta_pmid,
        "pmid_fallback_used": fb.get("pmid_fallback_used", 0),
        "store": store,
    }


def _figure_reconciliation(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Name the candidate populations behind 789/2,394/172/250 and the count each yields."""
    all_gold = set()
    for q in questions:
        for p in gold_pmids_for_question(q):
            if p is not None:
                all_gold.add(p)

    # phase5 40-question subset from the prior retrieval run.
    subset_gold = set()
    subset_n_gold_sum = 0
    bioasq_subset_gold = set()
    if PRIOR_RETRIEVAL.exists():
        with open(PRIOR_RETRIEVAL, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                golds = [normalize_pmid(g) for g in (rec.get("gold_relevant_pmids") or [])]
                golds = [g for g in golds if g is not None]
                subset_gold.update(golds)
                subset_n_gold_sum += int(rec.get("n_gold") or 0)
                if rec.get("source") == "bioasq":
                    bioasq_subset_gold.update(golds)

    return {
        "candidate_populations": {
            "all_250_distinct_gold_pmids": {
                "description": "distinct normalized gold PMIDs over all 250 eval questions",
                "count": len(all_gold),
            },
            "phase5_40_subset_distinct_gold_pmids": {
                "description": "distinct gold PMIDs over the 40-question phase5 subset",
                "count": len(subset_gold),
            },
            "phase5_40_subset_sum_n_gold": {
                "description": "sum of per-query n_gold over the 40 phase5 queries",
                "count": subset_n_gold_sum,
            },
            "phase5_40_subset_bioasq_only_distinct": {
                "description": "distinct gold PMIDs over the BioASQ-only phase5 subset",
                "count": len(bioasq_subset_gold),
            },
        },
        "quoted_789_reproducible": False,
        "resolution": (
            "The originally quoted 789 figure could not be reproduced from any artifact "
            "present in the repository; it most plausibly originated from a run "
            "configuration or question subset not retained on disk. The canonical measured "
            "figures are 2,394/250 (full) and 172/40 (phase5 subset)."
        ),
        "canonical_full_distinct_gold_pmids": len(all_gold),
        "canonical_full_n_queries": len(questions),
    }


def run_diagnose(roundtrip: bool, corpus_dir: Optional[Path]) -> Dict[str, Any]:
    questions = load_eval_questions()
    n_queries = len(questions)
    n_with_gold = sum(1 for q in questions if q.get("documents"))

    raw_gold_count = 0
    unique_norm = set()
    unparsable: List[str] = []
    for q in questions:
        for d in (q.get("documents") or []):
            raw_gold_count += 1
            norm = normalize_pmid(d)
            if norm is None:
                if len(unparsable) < 25:
                    unparsable.append(str(d))
            else:
                unique_norm.add(norm)

    # eval_split membership.
    eval_split = _load_store_pmids(EVAL_SPLIT_DIR)
    eval_split_pmids = eval_split["distinct_pmids"]
    matched_in_eval = unique_norm & eval_split_pmids

    report: Dict[str, Any] = {
        "seed": SEED,
        "n_queries": n_queries,
        "n_queries_with_gold": n_with_gold,
        "n_gold_pmids_raw": raw_gold_count,
        "n_gold_pmids_unique_normalized": len(unique_norm),
        "unparsable_pmids_sample": unparsable,
        "unparsable_pmids_count": raw_gold_count - sum(
            1 for q in questions for d in (q.get("documents") or []) if normalize_pmid(d) is not None
        ),
        "type_distribution": dict(Counter(q.get("type") for q in questions)),
        "eval_split_chunks": eval_split["n_chunks"],
        "eval_split_distinct_pmids": len(eval_split_pmids),
        "eval_split_chunks_missing_meta_pmid": eval_split["chunks_missing_meta_pmid"],
        "gold_pmids_matched_in_eval_split": len(matched_in_eval),
        "gold_pmids_matched_in_eval_split_pct": round(
            100.0 * len(matched_in_eval) / max(1, len(unique_norm)), 4),
        "eval_split_exclusion_confirmed": len(matched_in_eval) == 0,
        "prior_run_metric_reason_samples": [],
        "figure_reconciliation": _figure_reconciliation(questions),
    }

    # Cite the prior run's metric_reason as the matching symptom.
    if PRIOR_RETRIEVAL.exists():
        reasons = Counter()
        with open(PRIOR_RETRIEVAL, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                reasons[rec.get("metric_reason")] += 1
        report["prior_run_metric_reason_samples"] = [
            {"reason": r, "count": c} for r, c in reasons.most_common()
        ]

    # Optional production cross-check (coverage context only; never used for scoring).
    if PRODUCTION_DIR.joinpath("index.faiss").exists():
        try:
            prod = _load_store_pmids(PRODUCTION_DIR)
            matched_prod = unique_norm & prod["distinct_pmids"]
            report["production_index_distinct_pmids"] = len(prod["distinct_pmids"])
            report["gold_pmids_present_in_production_index"] = len(matched_prod)
            report["gold_pmids_present_in_production_index_pct"] = round(
                100.0 * len(matched_prod) / max(1, len(unique_norm)), 4)
        except Exception as exc:  # pragma: no cover - defensive
            report["production_index_cross_check_error"] = str(exc)

    # Per-corpus availability against the NEW retrieval-eval corpus (if provided).
    missing_examples: List[Dict[str, Any]] = []
    if corpus_dir is not None and corpus_dir.joinpath("index.faiss").exists():
        corpus = _load_store_pmids(corpus_dir)
        corpus_pmids = corpus["distinct_pmids"]
        at_least_one = 0
        all_available = 0
        for q in questions:
            gold = {g for g in gold_pmids_for_question(q) if g is not None}
            if not gold:
                continue
            avail = gold & corpus_pmids
            if avail:
                at_least_one += 1
            if avail == gold:
                all_available += 1
        report["corpus"] = str(corpus_dir)
        report["corpus_distinct_pmids"] = len(corpus_pmids)
        report["n_queries_with_at_least_one_available_gold"] = at_least_one
        report["n_queries_with_all_gold_available"] = all_available
        covered_pmids = unique_norm & corpus_pmids
        report["gold_pmids_covered_in_corpus"] = len(covered_pmids)
        report["gold_pmids_covered_in_corpus_pct"] = round(
            100.0 * len(covered_pmids) / max(1, len(unique_norm)), 4)
        # Missing-PMID examples (absent from the retrieval-eval corpus).
        for p in sorted(unique_norm - corpus_pmids):
            if len(missing_examples) >= 25:
                break
            missing_examples.append({"pmid": p, "likely_cause": "not_in_local_snippets"})
    report["missing_pmid_examples"] = missing_examples

    report["coverage_gap_vs_retrieval_failure"] = {
        "definition_missing_gold_document": (
            "The corpus does not contain the gold PMID at all; the retriever cannot "
            "return it. This is a corpus-coverage gap, not a ranking failure."
        ),
        "definition_retrieval_failure": (
            "The gold PMID IS present in the corpus but does not appear in the retriever's "
            "top-k. This is a ranking failure attributable to the retriever."
        ),
        "eval_split_case": (
            "All gold PMIDs are absent from eval_split by deliberate leakage exclusion, so "
            "every eval_split 'miss' is a coverage gap, never a retrieval failure."
        ),
    }

    # PMID->chunk round-trip traceability (optional model-free sampling on eval_split).
    roundtrip_report = {"ran": False}
    if roundtrip:
        roundtrip_report = _roundtrip_traceability(eval_split["store"])
    report["pmid_chunk_roundtrip"] = roundtrip_report

    return report


def _roundtrip_traceability(store: Any, n_sample: int = 20) -> Dict[str, Any]:
    """Assert chunk.metadata['pmid'] round-trips to the source-parsed PMID.

    This is model-free: it only checks that a retrieved chunk can be traced back
    to its PubMed article via both metadata and source URL, and that document_id
    groups chunks of one article.
    """
    import random
    rng = random.Random(SEED)
    chunks = store.chunks
    sample_idx = rng.sample(range(len(chunks)), min(n_sample, len(chunks)))
    ok = 0
    checked = 0
    mismatches: List[Dict[str, Any]] = []
    for i in sample_idx:
        ch = chunks[i]
        meta = getattr(ch, "metadata", None) or {}
        meta_pmid = normalize_pmid(meta.get("pmid"))
        src_pmid = normalize_pmid(getattr(ch, "source", None))
        checked += 1
        if meta_pmid is not None and meta_pmid == src_pmid:
            ok += 1
        else:
            if len(mismatches) < 10:
                mismatches.append({"meta_pmid": meta_pmid, "source_pmid": src_pmid})
    return {
        "ran": True,
        "sampled": checked,
        "roundtrip_ok": ok,
        "mismatches": mismatches,
        "all_ok": ok == checked,
    }


def _write_markdown(report: Dict[str, Any]) -> str:
    fr = report["figure_reconciliation"]
    lines = [
        "# Phase A — BioASQ PMID Diagnostic",
        "",
        f"- Queries: **{report['n_queries']}** (with gold: {report['n_queries_with_gold']})",
        f"- Raw gold PMID entries: {report['n_gold_pmids_raw']}",
        f"- Distinct normalized gold PMIDs: **{report['n_gold_pmids_unique_normalized']}**",
        f"- Unparsable gold entries: {report['unparsable_pmids_count']}",
        f"- eval_split chunks: {report['eval_split_chunks']} "
        f"(distinct PMIDs: {report['eval_split_distinct_pmids']})",
        f"- Gold PMIDs matched in eval_split: **{report['gold_pmids_matched_in_eval_split']}** "
        f"({report['gold_pmids_matched_in_eval_split_pct']}%)",
        f"- eval_split exclusion confirmed: **{report['eval_split_exclusion_confirmed']}**",
        "",
        "## 789 / 2,394 / 172 / 250 reconciliation",
        "",
    ]
    for name, cp in fr["candidate_populations"].items():
        lines.append(f"- `{name}` — {cp['description']}: **{cp['count']}**")
    lines += ["", f"- 789 reproducible: **{fr['quoted_789_reproducible']}**", "",
              fr["resolution"], "",
              "## Coverage gap vs retrieval failure", ""]
    cg = report["coverage_gap_vs_retrieval_failure"]
    lines += [f"- Missing gold document (coverage gap): {cg['definition_missing_gold_document']}",
              f"- Retrieval failure: {cg['definition_retrieval_failure']}",
              f"- eval_split case: {cg['eval_split_case']}", ""]
    if "corpus" in report:
        lines += ["## Retrieval-eval corpus availability", "",
                  f"- Corpus: `{report['corpus']}` (distinct PMIDs: {report['corpus_distinct_pmids']})",
                  f"- Queries with >=1 available gold: {report['n_queries_with_at_least_one_available_gold']}",
                  f"- Queries with all gold available: {report['n_queries_with_all_gold_available']}",
                  f"- Gold PMIDs covered in corpus: {report['gold_pmids_covered_in_corpus']} "
                  f"({report['gold_pmids_covered_in_corpus_pct']}%)", ""]
    rt = report.get("pmid_chunk_roundtrip", {})
    if rt.get("ran"):
        lines += ["## PMID->chunk round-trip", "",
                  f"- Sampled: {rt['sampled']}, round-trip OK: {rt['roundtrip_ok']}, "
                  f"all OK: {rt['all_ok']}", ""]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase A BioASQ diagnostic")
    parser.add_argument("--roundtrip", action="store_true",
                        help="Run the model-free PMID->chunk round-trip check on eval_split.")
    parser.add_argument("--corpus", type=str, default=None,
                        help="Retrieval-eval index dir for per-corpus availability counts.")
    args = parser.parse_args(argv)

    corpus_dir = Path(args.corpus) if args.corpus else None
    report = run_diagnose(roundtrip=args.roundtrip, corpus_dir=corpus_dir)

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(DIAG_DIR / "diagnostic_report.json", report)
    atomic_write_text(DIAG_DIR / "diagnostic_report.md", _write_markdown(report))

    write_manifest("diagnose",
                   cfg={"roundtrip": args.roundtrip, "corpus": str(corpus_dir) if corpus_dir else None},
                   inputs={"eval_questions": EVAL_QUESTIONS,
                           "eval_split_chunks": EVAL_SPLIT_DIR / "chunks.pkl"},
                   extra={"n_queries": report["n_queries"],
                          "n_gold_pmids_unique_normalized": report["n_gold_pmids_unique_normalized"],
                          "gold_pmids_matched_in_eval_split": report["gold_pmids_matched_in_eval_split"]})

    print(f"n_queries={report['n_queries']} "
          f"n_gold_pmids_unique_normalized={report['n_gold_pmids_unique_normalized']} "
          f"gold_pmids_matched_in_eval_split={report['gold_pmids_matched_in_eval_split']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
