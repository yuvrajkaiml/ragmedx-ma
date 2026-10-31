"""
Full retrieval evaluation for RAGMedX-MA — Dense (BGE-M3), Sparse (BM25), Hybrid (RRF).

Computes, from REAL index/dataset/gold outputs (no fabricated numbers):
  - Classification metrics: Accuracy, Precision, Recall (sensitivity), Specificity, F1
    over a documented (question, candidate-article) pair universe.
  - IR metrics: Precision@5, Recall@5, MRR, nDCG@5 (two aggregate views).

Retrievers evaluated:
  - Diagnostic index (data/indexes/bioasq_retrieval_eval/v1): Dense + Sparse + Hybrid.
  - Production index (data/faiss_index): Dense ONLY (no bm25.pkl exists), so production
    Sparse and production Hybrid are reported as NOT MEASURABLE with the exact reason.

This script is ADDITIVE and strictly READ-ONLY on the indexes:
  - It NEVER calls .save() / .add_chunks() / rebuilds / re-embeds anything.
  - It writes all NEW artifacts to results/full_retrieval_eval_<UTC>/.
  - It makes NO git commits, downloads no datasets, changes no dependencies.

Reuses IR math from evaluation.metrics and PMID/IO helpers from
evaluation.bioasq_eval.common (no reimplemented IR math).

Run from cwd = d:\\Ragmed_2\\backend with the CPU venv:

    cd d:\\Ragmed_2\\backend
    $env:HF_HUB_OFFLINE=1
    $env:TRANSFORMERS_OFFLINE=1
    d:\\Ragmed_2\\.venv\\Scripts\\python.exe scripts\\run_full_retrieval_eval.py

If BGE-M3 cannot load from the local HF cache, the script STOPS and reports the
blocker (exit 2). It never fabricates dense/hybrid results.
"""

from __future__ import annotations

# --- Force HF offline BEFORE importing app/transformers/BGE ------------------
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Path anchoring (never rely on cwd) --------------------------------------
BACKEND_DIR = Path(__file__).resolve().parents[1]   # d:\Ragmed_2\backend
REPO_ROOT = BACKEND_DIR.parent                      # d:\Ragmed_2

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

DIAG_INDEX_DIR = BACKEND_DIR / "data" / "indexes" / "bioasq_retrieval_eval" / "v1"
BM25_PATH = DIAG_INDEX_DIR / "bm25.pkl"
PROD_INDEX_DIR = BACKEND_DIR / "data" / "faiss_index"
EVAL_QUESTIONS = BACKEND_DIR / "data" / "datasets" / "bioasq_eval_split" / "eval_questions.json"

TOP_K = 5
SEED = 42

EMBEDDING_MODEL = "BAAI/bge-m3"
GENERATION_MODEL = "Qwen/Qwen3-8B (ollama: qwen3:8b)"
STALE_CONFIG_NOTE = (
    "backend/app/config.py (~line 132) declares embedding_model_name="
    "'BAAI/bge-small-en-v1.5', which is STALE and NOT what the indexes use. "
    "Both index manifests and BGEEmbedding.DEFAULT_MODEL use BAAI/bge-m3 "
    "(1024-dim). Reported as a limitation; NOT corrected (read-only scope)."
)
NOT_MEASURABLE_REASON = (
    "no sparse (bm25.pkl) index exists for the production corpus; building one "
    "would modify artifacts and is out of scope."
)
GENERATION_NOTE = (
    "Retrieval metrics are LLM-independent: the generation model "
    f"({GENERATION_MODEL}) is NOT invoked anywhere in this evaluation."
)
BINARY_THRESHOLD_NOTE = (
    "Gold labels are article-level binary (relevant=1, irrelevant=0). This "
    "dataset has NO partial (0.5) grades, so partial-grade sensitivity is N/A. "
    "nDCG uses binary gain (gain=1 if retrieved article PMID in gold else 0)."
)
TN_CAVEAT = (
    "TN is small by construction: the negative universe per question is "
    "(top-5 retrieved) union (available gold), so true negatives are bounded "
    "to that set. Specificity and Accuracy are therefore optimistic/limited; "
    "a corpus-universe Specificity is reported separately as a sensitivity "
    "analysis only."
)

HYBRID_CONFIG = {
    "rrf_k": 60,
    "query_expansion": True,
    "reranker": False,
    "dense_top_k": 10,
    "sparse_top_k": 10,
}

_RUN_UTC = datetime.now(timezone.utc)
_RUN_STAMP = _RUN_UTC.strftime("%Y%m%dT%H%M%SZ")
OUT_DIR = BACKEND_DIR / "results" / f"full_retrieval_eval_{_RUN_STAMP}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Output safety
# ---------------------------------------------------------------------------

def _assert_output_safety() -> None:
    """Guard: never write under the production FAISS index or inside the diag index."""
    out_resolved = OUT_DIR.resolve()

    prod = PROD_INDEX_DIR.resolve()
    assert out_resolved != prod and prod not in out_resolved.parents, (
        "output dir must not be under backend/data/faiss_index"
    )

    diag = DIAG_INDEX_DIR.resolve()
    assert out_resolved != diag and diag not in out_resolved.parents, (
        "output dir must not be inside the diagnostic index directory"
    )


# ---------------------------------------------------------------------------
# Dataset + gold loaders, PMID/coverage helpers
# ---------------------------------------------------------------------------

def load_questions() -> List[Dict[str, Any]]:
    with open(EVAL_QUESTIONS, "r", encoding="utf-8") as fh:
        return json.load(fh)


def gold_set(q: Dict[str, Any]) -> List[str]:
    """Normalized, sorted unique gold PMID list from a question's documents field."""
    from evaluation.bioasq_eval.common import normalize_pmid

    golds = {normalize_pmid(d) for d in (q.get("documents") or [])}
    return sorted(g for g in golds if g is not None)


def corpus_pmid_set(chunks) -> set:
    """Distinct canonical PMIDs present in a chunk collection."""
    from evaluation.bioasq_eval.common import canonical_pmid

    pmids = set()
    for c in chunks:
        pmid = canonical_pmid(c)
        if pmid is not None:
            pmids.add(pmid)
    return pmids


# ---------------------------------------------------------------------------
# Item 2 — retriever construction + per-question retrieval capture
# ---------------------------------------------------------------------------

def build_retrievers(index_dir: Path, with_sparse: bool, with_hybrid: bool,
                     embedder=None) -> Dict[str, Any]:
    """Construct the requested retrievers READ-ONLY. Shares one BGE-M3 embedder.

    Returns a dict possibly containing keys "dense", "sparse", "hybrid".
    The BGE-M3 embedder is loaded ONCE (either passed in or created here) and
    shared between dense and hybrid.
    """
    from app.rag.embeddings.bge_embedding import BGEEmbedding
    from app.rag.retrievers.dense_retriever import DenseRetriever
    from app.rag.retrievers.sparse_retriever import SparseRetriever
    from app.rag.retrievers.hybrid_retriever import HybridRetriever
    from app.rag.retrievers.query_expander import QueryExpander

    if embedder is None:
        embedder = BGEEmbedding()

    retrievers: Dict[str, Any] = {}

    dense = DenseRetriever.load(index_dir, embedder=embedder)
    retrievers["dense"] = dense

    sparse = None
    if with_sparse:
        sparse = SparseRetriever.load(BM25_PATH)
        retrievers["sparse"] = sparse

    if with_hybrid:
        if sparse is None:
            # Hybrid needs a sparse component; load it from the diagnostic bm25.
            sparse = SparseRetriever.load(BM25_PATH)
        retrievers["hybrid"] = HybridRetriever(
            dense_retriever=dense,
            sparse_retriever=sparse,
            query_expander=QueryExpander(),
            reranker=None,
            rrf_k=HYBRID_CONFIG["rrf_k"],
        )

    return retrievers


def retrieve_articles(retriever, body: str, top_k: int = TOP_K):
    """Run one retriever for one question.

    Returns (retrieved_articles, raw_rows, n_results, error). Exceptions are
    captured as strings and never crash the run (mirrors _capture).
    """
    from evaluation.bioasq_eval.common import canonical_pmid, dedup_articles, normalize_pmid

    body_stripped = (body or "").strip()
    if not body_stripped:
        return [], [], 0, "empty_query"

    is_hybrid = retriever.__class__.__name__ == "HybridRetriever"
    try:
        if is_hybrid:
            raw = retriever.retrieve(
                body_stripped,
                top_k=top_k,
                dense_top_k=HYBRID_CONFIG["dense_top_k"],
                sparse_top_k=HYBRID_CONFIG["sparse_top_k"],
            )
        else:
            raw = retriever.retrieve(body_stripped, top_k=top_k)
    except Exception as exc:  # record failures honestly, do not crash the run
        return [], [], 0, f"{type(exc).__name__}: {exc}"

    raw_rows = []
    for rank, (chunk, score) in enumerate(raw, start=1):
        raw_rows.append(
            {
                "rank": rank,
                "chunk_id": getattr(chunk, "id", None),
                "pmid": canonical_pmid(chunk),
                "score": float(score),
            }
        )

    retrieved_articles = dedup_articles((c for c, _ in raw), None)
    return retrieved_articles, raw_rows, len(raw), None


# ---------------------------------------------------------------------------
# Item 3 — IR metric computation + two aggregate views
# ---------------------------------------------------------------------------

def per_question_ir(retrieved_articles: List[str], gold: List[str]) -> Dict[str, float]:
    """Per-question P@5, R@5, nDCG@5, reciprocal_rank via evaluation.metrics."""
    from evaluation import metrics

    return {
        "precision_at_5": metrics.precision_at_k(retrieved_articles, gold, TOP_K),
        "recall_at_5": metrics.recall_at_k(retrieved_articles, gold, TOP_K),
        "ndcg_at_5": metrics.ndcg_at_k(retrieved_articles, gold, TOP_K),
        "reciprocal_rank": metrics.mrr([retrieved_articles], [gold]),
    }


def aggregate_ir(records: List[Dict[str, Any]], corpus_pmids: set) -> Dict[str, Any]:
    """Three IR views (mirrors bioasq_eval/run_retrieval_eval.py).

    records: list of {"retrieved_articles", "gold", "available", "covered", ir...}
    """
    from evaluation import metrics

    n_total = len(records)
    covered_recs = [r for r in records if r["covered"]]
    n_covered = len(covered_recs)

    def _macro(recs: List[Dict[str, Any]], field: str, denom: int) -> Optional[float]:
        if denom <= 0:
            return None
        return round(sum(r[field] for r in recs) / denom, 6)

    coverage_ge1 = sum(1 for r in records if r["covered"])
    coverage_all = sum(
        1 for r in records if r["gold"] and set(r["available"]) == set(r["gold"])
    )
    uncovered = sum(1 for r in records if r["gold"] and not r["covered"])
    partially_covered = sum(
        1
        for r in records
        if r["gold"] and r["covered"] and set(r["available"]) != set(r["gold"])
    )

    return {
        "corpus_coverage": {
            "n_queries": n_total,
            "coverage_at_least_1": round(coverage_ge1 / max(1, n_total), 6),
            "coverage_all": round(coverage_all / max(1, n_total), 6),
            "n_queries_with_at_least_one_available_gold": coverage_ge1,
            "n_queries_with_all_gold_available": coverage_all,
            "n_uncovered_gold_queries": uncovered,
            "n_partially_covered_queries": partially_covered,
            "corpus_distinct_pmids": len(corpus_pmids),
        },
        "end_to_end": {
            "denominator": n_total,
            "note": "All evaluated queries; uncovered-gold queries contribute 0.",
            "precision_at_5": _macro(records, "precision_at_5", n_total),
            "recall_at_5": _macro(records, "recall_at_5", n_total),
            "ndcg_at_5": _macro(records, "ndcg_at_5", n_total),
            "mrr": (
                metrics.mrr(
                    [r["retrieved_articles"] for r in records],
                    [r["gold"] for r in records],
                )
                if n_total
                else None
            ),
        },
        "retrieval_effectiveness": {
            "denominator": n_covered,
            "note": "Only covered queries (available gold non-empty); metric vs FULL gold.",
            "precision_at_5": _macro(covered_recs, "precision_at_5", n_covered),
            "recall_at_5": _macro(covered_recs, "recall_at_5", n_covered),
            "ndcg_at_5": _macro(covered_recs, "ndcg_at_5", n_covered),
            "mrr": (
                metrics.mrr(
                    [r["retrieved_articles"] for r in covered_recs],
                    [r["gold"] for r in covered_recs],
                )
                if n_covered
                else None
            ),
        },
    }


# ---------------------------------------------------------------------------
# Item 4 — classification metrics (documented pair universe)
# ---------------------------------------------------------------------------

def classification_counts(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """Micro-sum TP/FP/FN/TN over the (question, candidate-article) pair universe.

    Per question:
      universe = (top-5 retrieved article PMIDs) union (available gold PMIDs)
      gold_set = that question's available gold PMIDs
      retrieved = top-5 retrieved article PMIDs
      TP = article in retrieved AND in gold
      FP = article in retrieved AND not in gold
      FN = article in gold AND not in retrieved
      TN = article in universe AND not in retrieved AND not in gold
    """
    tp = fp = fn = tn = 0
    for r in records:
        retrieved = set(r["retrieved_articles"][:TOP_K])
        gold = set(r.get("available") or [])
        universe = retrieved | gold
        for article in universe:
            in_ret = article in retrieved
            in_gold = article in gold
            if in_ret and in_gold:
                tp += 1
            elif in_ret and not in_gold:
                fp += 1
            elif (not in_ret) and in_gold:
                fn += 1
            else:  # not in_ret and not in_gold
                tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def classification_metrics(counts: Dict[str, int]) -> Dict[str, Any]:
    """Accuracy/Precision/Recall/Specificity/F1 with undefined denominators as null."""
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]

    def _ratio(num: int, den: int) -> Optional[float]:
        if den <= 0:
            return None
        return round(num / den, 6)

    accuracy = _ratio(tp + tn, tp + tn + fp + fn)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)

    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = round(2 * precision * recall / (precision + recall), 6)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tn_caveat": TN_CAVEAT,
    }


def specificity_corpus_sensitivity(records: List[Dict[str, Any]], corpus_pmids: set) -> Dict[str, Any]:
    """Sensitivity analysis: TN over the full-corpus negative universe (not primary).

    Negative universe per question = all corpus PMIDs not in that question's gold.
    TN = corpus PMIDs not in gold and not in top-5 retrieved.
    FP = top-5 retrieved not in gold (same as primary).
    """
    n_corpus = len(corpus_pmids)
    tn = 0
    fp = 0
    for r in records:
        retrieved = set(r["retrieved_articles"][:TOP_K])
        gold = set(r.get("available") or [])
        # retrieved articles that are in the corpus (they always are) and not gold
        fp_set = {a for a in retrieved if a not in gold}
        fp += len(fp_set)
        # negatives = corpus \ gold ; TN = negatives not retrieved
        n_negatives = n_corpus - len(gold & corpus_pmids)
        tn += n_negatives - len(fp_set)

    if (tn + fp) <= 0:
        spec = None
    else:
        spec = round(tn / (tn + fp), 6)
    return {
        "label": "sensitivity analysis only (NOT the primary specificity)",
        "negative_universe": "all corpus PMIDs not in gold",
        "tn": tn,
        "fp": fp,
        "specificity_corpus_universe": spec,
    }


# ---------------------------------------------------------------------------
# Item 5 — full run orchestration per index
# ---------------------------------------------------------------------------

def _data_integrity(questions: List[Dict[str, Any]], corpus_pmids: set) -> Dict[str, Any]:
    """Dataset-level integrity/coverage stats (index-independent + coverage)."""
    seen_ids: Dict[str, int] = {}
    n_with_gold = 0
    n_zero_gold = 0
    n_empty_body = 0
    n_covered = 0
    n_fully_absent_gold = 0
    duplicate_ids: List[str] = []

    for q in questions:
        qid = q.get("id")
        if qid is not None:
            seen_ids[qid] = seen_ids.get(qid, 0) + 1
        body = (q.get("body") or "").strip()
        if not body:
            n_empty_body += 1
        gold = gold_set(q)
        if gold:
            n_with_gold += 1
            available = sorted(set(gold) & corpus_pmids)
            if available:
                n_covered += 1
            else:
                n_fully_absent_gold += 1
        else:
            n_zero_gold += 1

    duplicate_ids = sorted(qid for qid, c in seen_ids.items() if c > 1)

    return {
        "n_questions": len(questions),
        "n_with_valid_gold": n_with_gold,
        "n_zero_gold_excluded": n_zero_gold,
        "n_empty_body": n_empty_body,
        "n_covered_at_least_one_available_gold": n_covered,
        "n_fully_absent_gold": n_fully_absent_gold,
        "coverage_ratio": round(n_covered / max(1, n_with_gold), 6),
        "duplicate_question_ids": duplicate_ids,
        "exclusion_reasons": {
            "zero_gold": "question has no gold PMID labels; excluded from evaluation",
            "empty_body": "question body is empty; cannot be queried",
        },
    }


def run_index(label: str, index_dir: Path, retrievers: Dict[str, Any],
              questions: List[Dict[str, Any]], is_production: bool,
              not_measurable: List[str]) -> Dict[str, Any]:
    """Evaluate every retriever in `retrievers` over `questions` for one index.

    Returns a result object with per-retriever IR views + classification, plus
    per-question records (for JSONL dumps), coverage, and integrity stats.
    `not_measurable` lists retriever names to emit as explicit NOT MEASURABLE.
    """
    # Build corpus PMID set from the dense retriever's chunks (shared corpus).
    dense = retrievers["dense"]
    corpus_pmids = corpus_pmid_set(dense.vector_store.chunks)

    integrity = _data_integrity(questions, corpus_pmids)

    # Evaluable questions: non-empty body AND has valid gold labels.
    eval_questions: List[Dict[str, Any]] = []
    for q in questions:
        body = (q.get("body") or "").strip()
        gold = gold_set(q)
        if not body or not gold:
            continue
        available = sorted(set(gold) & corpus_pmids)
        eval_questions.append(
            {
                "id": q.get("id"),
                "body": q.get("body"),
                "type": q.get("type"),
                "gold": gold,
                "available": available,
                "covered": bool(available),
            }
        )

    retriever_keys = [k for k in ("dense", "sparse", "hybrid") if k in retrievers]

    # Per-retriever per-question records.
    per_retriever_records: Dict[str, List[Dict[str, Any]]] = {k: [] for k in retriever_keys}
    # Raw rows retained for JSONL dumps keyed by retriever then question id.
    per_retriever_jsonl: Dict[str, List[Dict[str, Any]]] = {k: [] for k in retriever_keys}

    for qd in eval_questions:
        for key in retriever_keys:
            retrieved_articles, raw_rows, n_results, error = retrieve_articles(
                retrievers[key], qd["body"], TOP_K
            )
            # IR against the FULL normalized gold set (matches run_retrieval_eval).
            ir = per_question_ir(retrieved_articles, qd["gold"])
            rec = {
                "question_id": qd["id"],
                "type": qd["type"],
                "gold": qd["gold"],
                "available": qd["available"],
                "covered": qd["covered"],
                "retrieved_articles": retrieved_articles,
                "n_results": n_results,
                "error": error,
                **ir,
            }
            per_retriever_records[key].append(rec)
            per_retriever_jsonl[key].append(
                {
                    "question_id": qd["id"],
                    "body": qd["body"],
                    "type": qd["type"],
                    "gold": qd["gold"],
                    "available_gold": qd["available"],
                    "covered": qd["covered"],
                    "retrieved_articles": retrieved_articles,
                    "n_results": n_results,
                    "retrieval_error": error,
                    "raw_top5": raw_rows,
                    "precision_at_5": ir["precision_at_5"],
                    "recall_at_5": ir["recall_at_5"],
                    "ndcg_at_5": ir["ndcg_at_5"],
                    "reciprocal_rank": ir["reciprocal_rank"],
                }
            )

    # Aggregate IR + classification per measurable retriever.
    retrievers_out: Dict[str, Any] = {}
    for key in retriever_keys:
        recs = per_retriever_records[key]
        ir_views = aggregate_ir(recs, corpus_pmids)
        counts = classification_counts(recs)
        cls = classification_metrics(counts)
        cls_sens = specificity_corpus_sensitivity(recs, corpus_pmids)
        failures = [
            {"question_id": r["question_id"], "error": r["error"]}
            for r in recs
            if r["error"]
        ]
        short = [
            {"question_id": r["question_id"], "n_results": r["n_results"]}
            for r in recs
            if r["error"] is None and r["n_results"] < TOP_K
        ]
        retrievers_out[key] = {
            "measurable": True,
            "ir_views": ir_views,
            "classification": cls,
            "classification_specificity_sensitivity": cls_sens,
            "n_evaluated": len(recs),
            "failures": failures,
            "short_result_questions": short,
        }

    # Explicit NOT MEASURABLE entries.
    for key in not_measurable:
        retrievers_out[key] = {
            "measurable": False,
            "not_measurable_reason": NOT_MEASURABLE_REASON,
        }

    if HYBRID_CONFIG and "hybrid" in retrievers_out and retrievers_out["hybrid"].get("measurable"):
        retrievers_out["hybrid"]["config"] = dict(HYBRID_CONFIG)

    return {
        "label": label,
        "is_production": is_production,
        "index_dir": str(index_dir),
        "corpus_distinct_pmids": len(corpus_pmids),
        "n_questions_total": len(questions),
        "n_evaluated": len(eval_questions),
        "data_integrity": integrity,
        "retrievers": retrievers_out,
        "_per_retriever_jsonl": per_retriever_jsonl,  # stripped before JSON dump
    }


# ---------------------------------------------------------------------------
# Item 6 — output writers
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None:
        return "NOT MEASURABLE"
    return f"{v:.4f}"


def _row_for_table_a(name: str, block: Dict[str, Any]) -> str:
    if not block.get("measurable", False):
        return f"| {name} | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE |"
    c = block["classification"]
    return (
        f"| {name} | {_fmt(c['accuracy'])} | {_fmt(c['precision'])} | "
        f"{_fmt(c['recall'])} | {_fmt(c['specificity'])} | {_fmt(c['f1'])} |"
    )


def _row_for_table_b(name: str, block: Dict[str, Any]) -> str:
    if not block.get("measurable", False):
        return f"| {name} | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE |"
    e = block["ir_views"]["end_to_end"]
    return (
        f"| {name} | {_fmt(e['precision_at_5'])} | {_fmt(e['recall_at_5'])} | "
        f"{_fmt(e['mrr'])} | {_fmt(e['ndcg_at_5'])} |"
    )


def write_outputs(diag_result: Dict[str, Any], prod_result: Dict[str, Any],
                  commands_text: str) -> Dict[str, Path]:
    from evaluation.bioasq_eval.common import atomic_write_json

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Strip the non-serializable helper key before dumping, keep a local copy.
    diag_jsonl = diag_result.pop("_per_retriever_jsonl")
    prod_jsonl = prod_result.pop("_per_retriever_jsonl")

    # ---- metrics.json ----
    metrics_obj = {
        "generated_utc": _utc_now_iso(),
        "seed": SEED,
        "top_k": TOP_K,
        "embedding_model": EMBEDDING_MODEL,
        "generation_model": GENERATION_MODEL,
        "generation_model_note": GENERATION_NOTE,
        "stale_config_limitation": STALE_CONFIG_NOTE,
        "binary_threshold_note": BINARY_THRESHOLD_NOTE,
        "classification_unit": (
            "a (question, candidate-article) pair; candidate universe per "
            "question = (top-5 retrieved article PMIDs) union (available gold "
            "PMIDs = gold PMIDs present in that index's corpus)."
        ),
        "tn_caveat": TN_CAVEAT,
        "hybrid_config": dict(HYBRID_CONFIG),
        "ndcg_formula": "binary gain (1 if in gold else 0); discount log2(rank+1); 0.0 when idcg==0.",
        "indexes": {
            "diagnostic": diag_result,
            "production": prod_result,
        },
    }
    metrics_path = OUT_DIR / "metrics.json"
    atomic_write_json(metrics_path, metrics_obj)

    # ---- per-question JSONL files ----
    jsonl_paths: List[Path] = []
    for key, rows in diag_jsonl.items():
        p = OUT_DIR / f"diagnostic_{key}.jsonl"
        with open(p, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, default=str) + "\n")
        jsonl_paths.append(p)
    for key, rows in prod_jsonl.items():
        p = OUT_DIR / f"production_{key}.jsonl"
        with open(p, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, default=str) + "\n")
        jsonl_paths.append(p)

    # ---- commands.txt ----
    commands_path = OUT_DIR / "commands.txt"
    with open(commands_path, "w", encoding="utf-8") as fh:
        fh.write(commands_text)

    # ---- report.md ----
    report_path = OUT_DIR / "report.md"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(_build_report(diag_result, prod_result))

    return {
        "metrics": metrics_path,
        "report": report_path,
        "commands": commands_path,
        "jsonl": jsonl_paths,  # type: ignore[dict-item]
    }


def _build_report(diag: Dict[str, Any], prod: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append("# Full Retrieval Evaluation — Dense (BGE-M3) vs Sparse (BM25) vs Hybrid (RRF)")
    L.append("")
    L.append(f"Generated (UTC): {_utc_now_iso()}")
    L.append(f"Seed: {SEED} | Top-k: {TOP_K}")
    L.append("")
    L.append("## Models")
    L.append("")
    L.append(f"- Embedding / retrieval model: `{EMBEDDING_MODEL}` (1024-dim, FAISS inner-product).")
    L.append(f"- Generation model (NOT used for retrieval): `{GENERATION_MODEL}`.")
    L.append(f"- {GENERATION_NOTE}")
    L.append(f"- Limitation: {STALE_CONFIG_NOTE}")
    L.append("")
    L.append("## Methodology")
    L.append("")
    L.append(f"- {BINARY_THRESHOLD_NOTE}")
    L.append(
        "- IR metrics (Precision@5, Recall@5, MRR, nDCG@5) reuse "
        "`evaluation/metrics.py`; article-level PMID matching reuses "
        "`evaluation/bioasq_eval/common.py` (`canonical_pmid`, `dedup_articles`)."
    )
    L.append(
        "- Classification unit = a (question, candidate-article) pair. Candidate "
        "universe per question = (top-5 retrieved article PMIDs) ∪ (available gold "
        "PMIDs present in that index's corpus). Micro-summed TP/FP/FN/TN, then "
        "Accuracy=(TP+TN)/(TP+TN+FP+FN), Precision=TP/(TP+FP), Recall=TP/(TP+FN), "
        "Specificity=TN/(TN+FP), F1=2PR/(P+R). Undefined denominators → NOT MEASURABLE."
    )
    L.append(f"- Caveat: {TN_CAVEAT}")
    L.append("- nDCG: binary gain (1 if retrieved article PMID ∈ gold else 0); discount log2(rank+1); 0.0 when idcg==0.")
    L.append(
        "- Headline IR table uses the end-to-end view (denominator = all evaluated "
        "questions; uncovered-gold questions score 0). The retrieval-effectiveness "
        "view (covered questions only) is also reported in metrics.json."
    )
    L.append("")

    # Per-index coverage summary.
    for res in (diag, prod):
        name = "Diagnostic" if not res["is_production"] else "Production"
        L.append(f"## {name} index")
        L.append("")
        L.append(f"- Index path (read-only): `{res['index_dir']}`")
        L.append(f"- Corpus distinct PMIDs: {res['corpus_distinct_pmids']}")
        L.append(f"- Questions evaluated: {res['n_evaluated']} of {res['n_questions_total']}")
        di = res["data_integrity"]
        L.append(
            f"- With valid gold: {di['n_with_valid_gold']}; "
            f"covered (≥1 available gold): {di['n_covered_at_least_one_available_gold']}; "
            f"coverage ratio: {di['coverage_ratio']}"
        )
        L.append(
            f"- Excluded — zero gold: {di['n_zero_gold_excluded']}; "
            f"empty body: {di['n_empty_body']}; "
            f"gold entirely absent from corpus: {di['n_fully_absent_gold']}; "
            f"duplicate ids: {len(di['duplicate_question_ids'])}"
        )
        L.append("")

    L.append("## Table A — Classification metrics")
    L.append("")
    L.append("| Retriever | Accuracy | Precision | Recall | Specificity | F1 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    dr = diag["retrievers"]
    pr = prod["retrievers"]
    L.append(_row_for_table_a("Dense (diagnostic)", dr["dense"]))
    L.append(_row_for_table_a("Sparse (diagnostic)", dr["sparse"]))
    L.append(_row_for_table_a("Hybrid-RRF (diagnostic)", dr["hybrid"]))
    L.append(_row_for_table_a("Dense (production)", pr["dense"]))
    L.append(_row_for_table_a("Sparse (production)", pr["sparse"]))
    L.append(_row_for_table_a("Hybrid (production)", pr["hybrid"]))
    L.append("")

    L.append("## Table B — Retrieval (IR) metrics — end-to-end view")
    L.append("")
    L.append("| Retriever | Precision@5 | Recall@5 | MRR | nDCG@5 |")
    L.append("|---|---:|---:|---:|---:|")
    L.append(_row_for_table_b("Dense (diagnostic)", dr["dense"]))
    L.append(_row_for_table_b("Sparse (diagnostic)", dr["sparse"]))
    L.append(_row_for_table_b("Hybrid-RRF (diagnostic)", dr["hybrid"]))
    L.append(_row_for_table_b("Dense (production)", pr["dense"]))
    L.append(_row_for_table_b("Sparse (production)", pr["sparse"]))
    L.append(_row_for_table_b("Hybrid (production)", pr["hybrid"]))
    L.append("")

    # Retrieval-effectiveness view (secondary) for measurable retrievers.
    L.append("## Table B2 — Retrieval (IR) metrics — retrieval-effectiveness view (covered only)")
    L.append("")
    L.append("| Retriever | Precision@5 | Recall@5 | MRR | nDCG@5 | n_covered |")
    L.append("|---|---:|---:|---:|---:|---:|")

    def _eff_row(nm: str, block: Dict[str, Any]) -> str:
        if not block.get("measurable", False):
            return f"| {nm} | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE | - |"
        eff = block["ir_views"]["retrieval_effectiveness"]
        return (
            f"| {nm} | {_fmt(eff['precision_at_5'])} | {_fmt(eff['recall_at_5'])} | "
            f"{_fmt(eff['mrr'])} | {_fmt(eff['ndcg_at_5'])} | {eff['denominator']} |"
        )

    L.append(_eff_row("Dense (diagnostic)", dr["dense"]))
    L.append(_eff_row("Sparse (diagnostic)", dr["sparse"]))
    L.append(_eff_row("Hybrid-RRF (diagnostic)", dr["hybrid"]))
    L.append(_eff_row("Dense (production)", pr["dense"]))
    L.append("")

    L.append("## NOT MEASURABLE rows")
    L.append("")
    L.append(f"- Sparse (production): NOT MEASURABLE — {NOT_MEASURABLE_REASON}")
    L.append(f"- Hybrid (production): NOT MEASURABLE — {NOT_MEASURABLE_REASON}")
    L.append("")

    L.append("## Hybrid configuration (diagnostic)")
    L.append("")
    L.append(
        f"- rrf_k={HYBRID_CONFIG['rrf_k']}, query_expansion="
        f"{HYBRID_CONFIG['query_expansion']}, reranker={HYBRID_CONFIG['reranker']}, "
        f"dense_top_k={HYBRID_CONFIG['dense_top_k']}, sparse_top_k={HYBRID_CONFIG['sparse_top_k']}."
    )
    L.append("")

    L.append("## Classification micro counts (TP/FP/FN/TN)")
    L.append("")
    L.append("| Retriever | TP | FP | FN | TN |")
    L.append("|---|---:|---:|---:|---:|")
    for nm, block in (
        ("Dense (diagnostic)", dr["dense"]),
        ("Sparse (diagnostic)", dr["sparse"]),
        ("Hybrid-RRF (diagnostic)", dr["hybrid"]),
        ("Dense (production)", pr["dense"]),
    ):
        c = block["classification"]
        L.append(f"| {nm} | {c['tp']} | {c['fp']} | {c['fn']} | {c['tn']} |")
    L.append("")

    L.append("## Limitations")
    L.append("")
    L.append(f"- {TN_CAVEAT}")
    L.append(
        "- Production Sparse/Hybrid are NOT MEASURABLE (no bm25.pkl for the "
        "production corpus; building one would modify artifacts, out of scope)."
    )
    L.append(f"- {STALE_CONFIG_NOTE}")
    L.append(
        "- Hybrid is evaluated with the cross-encoder reranker OFF (clean three-way "
        "comparison). A reranked variant is not included."
    )
    L.append(
        "- Production corpus is heterogeneous (BioASQ snippet chunks + a few MedQA "
        "records); only PMID-bearing chunks participate in article-level matching."
    )
    L.append("")
    L.append("## Output files")
    L.append("")
    L.append(f"All artifacts under: `{OUT_DIR}`")
    L.append("- metrics.json, report.md, commands.txt")
    L.append("- diagnostic_dense.jsonl, diagnostic_sparse.jsonl, diagnostic_hybrid.jsonl")
    L.append("- production_dense.jsonl")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

COMMANDS_TEXT = (
    "cd d:\\Ragmed_2\\backend\n"
    "$env:HF_HUB_OFFLINE=1\n"
    "$env:TRANSFORMERS_OFFLINE=1\n"
    "d:\\Ragmed_2\\.venv\\Scripts\\python.exe scripts\\run_full_retrieval_eval.py\n"
)


def main() -> int:
    _assert_output_safety()

    questions = load_questions()

    # --- Load BGE-M3 ONCE; blocker (exit 2) if it cannot load ---------------
    try:
        from app.rag.embeddings.bge_embedding import BGEEmbedding

        embedder = BGEEmbedding()
    except Exception as exc:
        print(
            f"[BLOCKER] BGE-M3 could not load from local HF cache: "
            f"{type(exc).__name__}: {exc}\n"
            f"          Dense/Hybrid cannot run. NOT fabricating results.",
            file=sys.stderr,
        )
        return 2

    # --- Diagnostic index: dense + sparse + hybrid --------------------------
    diag_retrievers = build_retrievers(
        DIAG_INDEX_DIR, with_sparse=True, with_hybrid=True, embedder=embedder
    )
    assert diag_retrievers["dense"].size == 2947, (
        f"diag dense size unexpected: {diag_retrievers['dense'].size}"
    )
    assert diag_retrievers["sparse"].size == 2947, (
        f"diag sparse size unexpected: {diag_retrievers['sparse'].size}"
    )
    diag_result = run_index(
        "diagnostic", DIAG_INDEX_DIR, diag_retrievers, questions,
        is_production=False, not_measurable=[],
    )

    # --- Production index: dense ONLY ---------------------------------------
    prod_retrievers = build_retrievers(
        PROD_INDEX_DIR, with_sparse=False, with_hybrid=False, embedder=embedder
    )
    prod_result = run_index(
        "production", PROD_INDEX_DIR, prod_retrievers, questions,
        is_production=True, not_measurable=["sparse", "hybrid"],
    )

    paths = write_outputs(diag_result, prod_result, COMMANDS_TEXT)

    # --- Console summary -----------------------------------------------------
    print("=" * 72)
    print("Full retrieval evaluation — summary")
    print("=" * 72)
    print(f"Output dir: {OUT_DIR}")
    for idx_label, res in (("DIAGNOSTIC", diag_result), ("PRODUCTION", prod_result)):
        print(f"\n[{idx_label}] index={res['index_dir']} "
              f"corpus_pmids={res['corpus_distinct_pmids']} "
              f"evaluated={res['n_evaluated']}")
        for key in ("dense", "sparse", "hybrid"):
            block = res["retrievers"].get(key)
            if block is None:
                continue
            if not block.get("measurable", False):
                print(f"  {key:7s}: NOT MEASURABLE ({block['not_measurable_reason']})")
                continue
            e = block["ir_views"]["end_to_end"]
            c = block["classification"]
            print(
                f"  {key:7s}: IR P@5={_fmt(e['precision_at_5'])} R@5={_fmt(e['recall_at_5'])} "
                f"MRR={_fmt(e['mrr'])} nDCG@5={_fmt(e['ndcg_at_5'])} | "
                f"CLS Acc={_fmt(c['accuracy'])} P={_fmt(c['precision'])} "
                f"R={_fmt(c['recall'])} Spec={_fmt(c['specificity'])} F1={_fmt(c['f1'])}"
            )
    print(f"\nmetrics.json: {paths['metrics']}")
    print(f"report.md:    {paths['report']}")
    print("Indexes were NOT modified. NO git commit was made.")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
