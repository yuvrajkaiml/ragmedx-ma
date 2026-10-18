"""
Full retrieval evaluation for RAGMedX-MA.

Runs a fresh, real retrieval evaluation of the three retrievers that exist in the
codebase and reports classification metrics (Accuracy, Precision, Recall/Sensitivity,
Specificity, F1) and IR metrics (Precision@5, Recall@5, MRR, nDCG@5):

  - Dense  (BGE-M3 + FAISS)   on the DIAGNOSTIC index
  - Sparse (Okapi BM25)        on the DIAGNOSTIC index
  - Hybrid (RRF, reranker off) on the DIAGNOSTIC index
  - Dense  (BGE-M3 + FAISS)    on the PRODUCTION index  (separately labeled)

Production SPARSE and production HYBRID are reported NOT MEASURABLE because the
production corpus has no BM25/sparse index (bm25.pkl); building one would modify
artifacts and is out of scope.

This script is ADDITIVE and strictly READ-ONLY on the indexes:
  - It loads retrievers via .load(); it NEVER calls .save()/.add_chunks()/rebuilds/re-embeds.
  - It writes all NEW artifacts to a NEW timestamped output dir under backend/results/
    and mirrors a copy under the task handoff dir.
  - It makes NO git commits and downloads nothing.

Run from cwd = d:\\Ragmed_2\\backend with the CPU venv:

    cd d:\\Ragmed_2\\backend
    $env:HF_HUB_OFFLINE=1
    $env:TRANSFORMERS_OFFLINE=1
    d:\\Ragmed_2\\.venv\\Scripts\\python.exe scripts\\run_full_retrieval_eval.py

The embedding/retrieval model is BAAI/bge-m3 (1024-dim). The generation/answer model
(Qwen/Qwen3-8B, ollama qwen3:8b) is SEPARATE and is NOT invoked here — retrieval
metrics never call the generation LLM.
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

# --- Index / dataset locations (read-only) -----------------------------------
DIAG_INDEX_DIR = BACKEND_DIR / "data" / "indexes" / "bioasq_retrieval_eval" / "v1"
BM25_PATH = DIAG_INDEX_DIR / "bm25.pkl"
PROD_INDEX_DIR = BACKEND_DIR / "data" / "faiss_index"
EVAL_QUESTIONS = BACKEND_DIR / "data" / "datasets" / "bioasq_eval_split" / "eval_questions.json"

TOP_K = 5
SEED = 42
DIAG_EXPECTED_SIZE = 2947

# Models (reported SEPARATELY; retrieval is LLM-independent).
EMBEDDING_MODEL = "BAAI/bge-m3"            # real embedding/retrieval model (1024-dim)
GENERATION_MODEL = "Qwen/Qwen3-8B"         # real generation/answer model (NOT used here)
GENERATION_MODEL_OLLAMA = "qwen3:8b"
STALE_CONFIG_EMBEDDING_ID = "BAAI/bge-small-en-v1.5"  # stale value in app/config.py

PROD_NOT_MEASURABLE_REASON = (
    "no BM25/sparse index (bm25.pkl) exists for the production corpus; "
    "building one would modify artifacts and is out of scope"
)

# Hybrid config (documented).
HYBRID_CONFIG = {
    "rrf_k": 60,
    "query_expansion": True,
    "reranker": False,
    "dense_top_k": 10,
    "sparse_top_k": 10,
    "dedup_jaccard_threshold": 0.90,
    "note": (
        "HybridRetriever default pool sizes: when dense_top_k/sparse_top_k are None "
        "they default to max(top_k, 10) = 10. Query expansion is ON (QueryExpander). "
        "Cross-encoder reranker is OFF (reranker=None) for a clean three-way comparison."
    ),
}

# Timestamped output dir (UTC); never overwrites earlier results.
RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
OUT_DIR = BACKEND_DIR / "results" / f"full_retrieval_eval_{RUN_TS}"
MIRROR_DIR = (
    REPO_ROOT / ".agents" / "tasks" / "ragmed2-retrieval-eval-2026-10-19"
    / f"full_retrieval_eval_{RUN_TS}"
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _assert_output_safety() -> None:
    """Guard: never write under the production FAISS index or inside the diagnostic index."""
    prod = PROD_INDEX_DIR.resolve()
    diag = DIAG_INDEX_DIR.resolve()
    for out in (OUT_DIR.resolve(), MIRROR_DIR.resolve()):
        assert out != prod and prod not in out.parents, (
            "output dir must not be under backend/data/faiss_index"
        )
        assert out != diag and diag not in out.parents, (
            "output dir must not be inside the diagnostic index directory"
        )


def load_questions() -> List[Dict[str, Any]]:
    with open(EVAL_QUESTIONS, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _fmt(v: Optional[float]) -> str:
    """Format a metric for Markdown tables; None -> 'NOT MEASURABLE'/'null'."""
    if v is None:
        return "null"
    return f"{v:.4f}"


# ---------------------------------------------------------------------------
# Retriever construction (read-only)
# ---------------------------------------------------------------------------

def build_retrievers(index_dir: Path, with_sparse: bool, with_hybrid: bool,
                     embedder: Any = None) -> Dict[str, Any]:
    """Construct the requested retrievers for ``index_dir`` (read-only loads).

    Returns a dict with keys 'dense', 'sparse', 'hybrid' (value None when not built).
    A single BGEEmbedding instance is shared across dense and hybrid.
    """
    from app.rag.embeddings.bge_embedding import BGEEmbedding
    from app.rag.retrievers.dense_retriever import DenseRetriever
    from app.rag.retrievers.sparse_retriever import SparseRetriever
    from app.rag.retrievers.hybrid_retriever import HybridRetriever
    from app.rag.retrievers.query_expander import QueryExpander

    if embedder is None:
        embedder = BGEEmbedding()

    dense = DenseRetriever.load(index_dir, embedder=embedder)

    sparse = None
    if with_sparse:
        sparse = SparseRetriever.load(BM25_PATH)

    hybrid = None
    if with_hybrid:
        if sparse is None:
            raise ValueError("hybrid requires a sparse retriever")
        hybrid = HybridRetriever(
            dense_retriever=dense,
            sparse_retriever=sparse,
            query_expander=QueryExpander(),
            reranker=None,
            rrf_k=HYBRID_CONFIG["rrf_k"],
        )

    return {"dense": dense, "sparse": sparse, "hybrid": hybrid}


def retrieve_articles(retriever: Any, body: str, dedup_articles, normalize_pmid,
                      ) -> Tuple[List[str], List[Dict[str, Any]], int, Optional[str]]:
    """Retrieve top-5 for one question and collapse chunks -> rank-ordered unique PMIDs.

    Returns (retrieved_articles, raw_rows, n_results, error). Exceptions are captured
    as strings (never crash the run); mirrors run_standalone_top5_eval.py::_capture.
    """
    body_stripped = (body or "").strip()
    if not body_stripped:
        return [], [], 0, "empty_query"
    try:
        raw = retriever.retrieve(body_stripped, top_k=TOP_K)
    except Exception as exc:  # record failures honestly
        return [], [], 0, f"{type(exc).__name__}: {exc}"

    rows: List[Dict[str, Any]] = []
    for rank, (chunk, score) in enumerate(raw, start=1):
        rows.append(
            {
                "rank": rank,
                "chunk_id": getattr(chunk, "id", None),
                "source_pmid": normalize_pmid((getattr(chunk, "metadata", None) or {}).get("pmid")),
                "score": float(score),
            }
        )
    fb: Dict[str, int] = {}
    retrieved_articles = dedup_articles((c for c, _ in raw), fb)
    return retrieved_articles, rows, len(rows), None


# ---------------------------------------------------------------------------
# IR metrics (reuse evaluation/metrics.py)
# ---------------------------------------------------------------------------

def per_question_ir(metrics, retrieved_articles: List[str], gold: List[str]) -> Dict[str, Any]:
    """Per-question P@5, R@5, nDCG@5, reciprocal_rank using the project's metric funcs."""
    return {
        "precision_at_5": metrics.precision_at_k(retrieved_articles, gold, TOP_K),
        "recall_at_5": metrics.recall_at_k(retrieved_articles, gold, TOP_K),
        "ndcg_at_5": metrics.ndcg_at_k(retrieved_articles, gold, TOP_K),
        "reciprocal_rank": metrics.mrr([retrieved_articles], [gold]),
    }


def aggregate_ir(metrics, records: List[Dict[str, Any]], corpus_pmids: set) -> Dict[str, Any]:
    """Build the coverage + end-to-end + retrieval-effectiveness IR views.

    Mirrors evaluation/bioasq_eval/run_retrieval_eval.py. Undefined denominators -> None.
    ``records`` are the evaluated questions (non-excluded), each carrying
    gold / available / covered / retrieved_articles / per-question IR fields.
    """
    ordered = records
    n_total = len(ordered)
    covered_recs = [r for r in ordered if r["covered"]]
    n_covered = len(covered_recs)

    def _macro(recs: List[Dict[str, Any]], field: str, denom: int) -> Optional[float]:
        if denom == 0:
            return None
        return round(sum(r[field] for r in recs) / denom, 6)

    coverage_ge1 = sum(1 for r in ordered if r["covered"])
    coverage_all = sum(1 for r in ordered
                       if r["gold"] and set(r["available"]) == set(r["gold"]))
    uncovered = sum(1 for r in ordered if r["gold"] and not r["covered"])
    partially_covered = sum(1 for r in ordered
                            if r["covered"] and set(r["available"]) != set(r["gold"]))

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
            "precision_at_5": _macro(ordered, "precision_at_5", n_total),
            "recall_at_5": _macro(ordered, "recall_at_5", n_total),
            "ndcg_at_5": _macro(ordered, "ndcg_at_5", n_total),
            "mrr": (metrics.mrr([r["retrieved_articles"] for r in ordered],
                                [r["gold"] for r in ordered]) if n_total else None),
        },
        "retrieval_effectiveness": {
            "denominator": n_covered,
            "note": "Only covered queries (available gold non-empty); metric vs FULL gold.",
            "precision_at_5": _macro(covered_recs, "precision_at_5", n_covered),
            "recall_at_5": _macro(covered_recs, "recall_at_5", n_covered),
            "ndcg_at_5": _macro(covered_recs, "ndcg_at_5", n_covered),
            "mrr": (metrics.mrr([r["retrieved_articles"] for r in covered_recs],
                                [r["gold"] for r in covered_recs]) if n_covered else None),
        },
    }


# ---------------------------------------------------------------------------
# Classification metrics (documented (question, candidate-article) pair universe)
# ---------------------------------------------------------------------------

def classification_counts(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Micro-sum TP/FP/FN/TN over (question, candidate-article) pairs.

    Evaluated unit  : a (question, candidate-article) pair.
    Candidate universe per question = (top-5 retrieved article PMIDs) UNION
                                      (available gold PMIDs = gold present in that index).
    Per pair, for a question's gold set G and top-5 retrieved set R:
        TP = article in R and in G
        FP = article in R and not in G
        FN = article in G and not in R
        TN = article in universe and not in R and not in G
    """
    TP = FP = FN = TN = 0
    for r in records:
        gold = set(r["available"])            # gold grounded in this index's corpus
        retrieved = set(r["retrieved_articles"][:TOP_K])
        universe = retrieved | gold
        for art in universe:
            in_r = art in retrieved
            in_g = art in gold
            if in_r and in_g:
                TP += 1
            elif in_r and not in_g:
                FP += 1
            elif in_g and not in_r:
                FN += 1
            else:
                TN += 1
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN}


def classification_metrics(counts: Dict[str, int]) -> Dict[str, Any]:
    """Accuracy/Precision/Recall/Specificity/F1 with undefined denominators -> None."""
    TP, FP, FN, TN = counts["TP"], counts["FP"], counts["FN"], counts["TN"]

    def _safe(num: float, den: float) -> Optional[float]:
        return round(num / den, 6) if den else None

    accuracy = _safe(TP + TN, TP + TN + FP + FN)
    precision = _safe(TP, TP + FP)
    recall = _safe(TP, TP + FN)
    specificity = _safe(TN, TN + FP)
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1: Optional[float] = round(2 * precision * recall / (precision + recall), 6)
    else:
        f1 = None
    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def specificity_corpus_sensitivity(records: List[Dict[str, Any]],
                                    corpus_pmids: set) -> Dict[str, Any]:
    """SEPARATE sensitivity analysis: negatives = all corpus PMIDs not gold.

    This is NOT the primary number; the primary table uses the top-5 ∪ gold universe.
    Here TN = |corpus_pmids - gold - retrieved|, FP = retrieved & not gold, micro-summed.
    """
    TN = 0
    FP = 0
    for r in records:
        gold = set(r["available"])
        retrieved = set(r["retrieved_articles"][:TOP_K])
        negatives = corpus_pmids - gold
        FP += len(retrieved & negatives)
        TN += len(negatives - retrieved)
    spec = round(TN / (TN + FP), 6) if (TN + FP) else None
    return {"TN": TN, "FP": FP, "specificity_corpus_universe": spec}


# ---------------------------------------------------------------------------
# Per-index orchestration
# ---------------------------------------------------------------------------

def corpus_pmid_set(chunks, normalize_pmid) -> set:
    s = {normalize_pmid((getattr(c, "metadata", None) or {}).get("pmid")) for c in chunks}
    s.discard(None)
    return s


def gold_set(q: Dict[str, Any], normalize_pmid) -> List[str]:
    golds = {normalize_pmid(d) for d in (q.get("documents") or [])}
    return sorted(g for g in golds if g is not None)


def data_integrity(questions: List[Dict[str, Any]], normalize_pmid) -> Dict[str, Any]:
    """Count missing/empty gold, duplicate ids, empty bodies, inconsistent doc->PMID maps."""
    seen_ids: Dict[str, int] = {}
    n_empty_body = 0
    n_zero_gold = 0
    n_unparseable_doc_urls = 0
    for q in questions:
        qid = q.get("id")
        seen_ids[qid] = seen_ids.get(qid, 0) + 1
        if not (q.get("body") or "").strip():
            n_empty_body += 1
        docs = q.get("documents") or []
        gold = gold_set(q, normalize_pmid)
        if not gold:
            n_zero_gold += 1
        for d in docs:
            if normalize_pmid(d) is None:
                n_unparseable_doc_urls += 1
    duplicate_ids = {k: v for k, v in seen_ids.items() if v > 1}
    return {
        "n_questions_total": len(questions),
        "n_distinct_question_ids": len(seen_ids),
        "n_duplicate_question_ids": len(duplicate_ids),
        "duplicate_question_ids": duplicate_ids,
        "n_empty_body": n_empty_body,
        "n_zero_gold_label": n_zero_gold,
        "n_unparseable_document_urls": n_unparseable_doc_urls,
    }


def run_index(label: str, is_production: bool, retrievers: Dict[str, Any],
              questions: List[Dict[str, Any]], corpus_pmids: set,
              metrics, dedup_articles, normalize_pmid) -> Dict[str, Any]:
    """Evaluate every available retriever in ``retrievers`` on the same questions/gold."""
    active = {k: v for k, v in retrievers.items() if v is not None}

    # Per-question base records (gold/available/covered) + per-retriever retrieval.
    base_records: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    # per-retriever list of IR-enriched records
    per_ret_records: Dict[str, List[Dict[str, Any]]] = {k: [] for k in active}
    # per-retriever per-question jsonl rows
    per_ret_jsonl: Dict[str, List[Dict[str, Any]]] = {k: [] for k in active}

    for q in questions:
        qid = q.get("id")
        body = q.get("body") or ""
        gold = gold_set(q, normalize_pmid)
        available = sorted(set(gold) & corpus_pmids)
        covered = len(available) > 0

        # Exclusions: empty body or no gold label at all (cannot score retrieval).
        if not body.strip():
            excluded.append({"question_id": qid, "reason": "empty_body"})
            continue
        if not gold:
            excluded.append({"question_id": qid, "reason": "no_gold_label"})
            continue

        for rk, retr in active.items():
            retrieved_articles, rows, n_results, err = retrieve_articles(
                retr, body, dedup_articles, normalize_pmid
            )
            ir = per_question_ir(metrics, retrieved_articles, gold)
            rec = {
                "question_id": qid,
                "type": q.get("type"),
                "gold": gold,
                "available": available,
                "covered": covered,
                "retrieved_articles": retrieved_articles,
                "n_results": n_results,
                "error": err,
                **ir,
            }
            per_ret_records[rk].append(rec)
            per_ret_jsonl[rk].append(
                {
                    "question_id": qid,
                    "type": q.get("type"),
                    "body": body,
                    "gold": gold,
                    "available_gold": available,
                    "covered": covered,
                    "retrieved_articles": retrieved_articles,
                    "n_results": n_results,
                    "retrieval_error": err,
                    "precision_at_5": ir["precision_at_5"],
                    "recall_at_5": ir["recall_at_5"],
                    "ndcg_at_5": ir["ndcg_at_5"],
                    "reciprocal_rank": ir["reciprocal_rank"],
                    "top5_rows": rows,
                }
            )

        base_records.append(
            {"question_id": qid, "gold": gold, "available": available, "covered": covered}
        )

    n_evaluated = len(base_records)
    n_with_gold = sum(1 for r in base_records if r["gold"])
    n_covered = sum(1 for r in base_records if r["covered"])
    n_uncovered = n_with_gold - n_covered
    coverage_ratio = round(n_covered / n_with_gold, 6) if n_with_gold else None
    fully_absent_gold = sum(
        1 for r in base_records if r["gold"] and not set(r["gold"]) & corpus_pmids
    )

    retrievers_out: Dict[str, Any] = {}
    for rk in active:
        recs = per_ret_records[rk]
        ir_views = aggregate_ir(metrics, recs, corpus_pmids)
        counts = classification_counts(recs)
        clf = classification_metrics(counts)
        clf_sensitivity = specificity_corpus_sensitivity(recs, corpus_pmids)
        n_errors = sum(1 for r in recs if r["error"])
        retrievers_out[rk] = {
            "ir_views": ir_views,
            "classification_primary": clf,
            "classification_specificity_sensitivity": clf_sensitivity,
            "n_questions": len(recs),
            "n_retrieval_errors": n_errors,
            "config": HYBRID_CONFIG if rk == "hybrid" else (
                {"k1": 1.5, "b": 0.75} if rk == "sparse" else
                {"model": EMBEDDING_MODEL, "dim": 1024, "metric": "FAISS inner-product"}
            ),
        }

    result: Dict[str, Any] = {
        "label": label,
        "evaluation_type": "production" if is_production else "diagnostic",
        "index_dir": str(PROD_INDEX_DIR if is_production else DIAG_INDEX_DIR),
        "embedding_model": EMBEDDING_MODEL,
        "generation_model": GENERATION_MODEL,
        "generation_model_ollama": GENERATION_MODEL_OLLAMA,
        "generation_model_note": "generation LLM is NOT invoked; retrieval metrics are LLM-independent",
        "coverage": {
            "corpus_distinct_pmids": len(corpus_pmids),
            "n_evaluated": n_evaluated,
            "n_excluded": len(excluded),
            "n_total_questions": len(questions),
            "consistency_check_evaluated_plus_excluded": n_evaluated + len(excluded),
            "n_with_gold": n_with_gold,
            "n_covered": n_covered,
            "n_uncovered": n_uncovered,
            "coverage_ratio_covered_over_with_gold": coverage_ratio,
            "n_fully_absent_gold": fully_absent_gold,
        },
        "excluded_questions": excluded,
        "retrievers": retrievers_out,
        "_per_ret_jsonl": per_ret_jsonl,  # stripped before writing metrics.json
    }

    # Production: mark sparse + hybrid NOT MEASURABLE with the exact reason.
    if is_production:
        result["not_measurable"] = {
            "sparse": PROD_NOT_MEASURABLE_REASON,
            "hybrid": PROD_NOT_MEASURABLE_REASON,
        }
    return result


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, default=str) + "\n")


def _ir_row(result: Dict[str, Any], rk: str, view: str) -> Tuple[Optional[float], ...]:
    v = result["retrievers"][rk]["ir_views"][view]
    return v["precision_at_5"], v["recall_at_5"], v["mrr"], v["ndcg_at_5"]


def _clf_row(result: Dict[str, Any], rk: str) -> Dict[str, Any]:
    return result["retrievers"][rk]["classification_primary"]


def write_report(path: Path, diag: Dict[str, Any], prod: Dict[str, Any],
                 integrity: Dict[str, Any], commands_rel: str) -> None:
    L: List[str] = []
    L.append("# RAGMedX-MA — Full Retrieval Evaluation")
    L.append("")
    L.append(f"Generated (UTC): {_utc_now_iso()}")
    L.append(f"Seed: {SEED} | Top-k: {TOP_K} | Device: CPU")
    L.append("")
    L.append("## Models (reported separately)")
    L.append("")
    L.append(f"- Embedding / retrieval model: `{EMBEDDING_MODEL}` (1024-dim, FAISS inner-product). "
             "Confirmed by both index manifests and `BGEEmbedding.DEFAULT_MODEL`.")
    L.append(f"- Generation / answer model: `{GENERATION_MODEL}` (ollama `{GENERATION_MODEL_OLLAMA}`). "
             "This is a SEPARATE model and is NOT invoked by this evaluation — retrieval metrics are "
             "LLM-independent.")
    L.append("")
    L.append("## Indexes (read-only)")
    L.append("")
    L.append(f"- Diagnostic: `{DIAG_INDEX_DIR}` (dense + sparse + hybrid; "
             f"{diag['coverage']['corpus_distinct_pmids']} distinct corpus PMIDs).")
    L.append(f"- Production: `{PROD_INDEX_DIR}` (dense only; "
             f"{prod['coverage']['corpus_distinct_pmids']} distinct corpus PMIDs). "
             "No bm25.pkl -> production sparse + hybrid NOT MEASURABLE.")
    L.append("")
    L.append("## Methodology")
    L.append("")
    L.append("Each eval question's gold documents are PubMed URLs mapped to PMIDs via "
             "`normalize_pmid`. Gold labels are ARTICLE-LEVEL BINARY: relevant=1, irrelevant=0. "
             "There are NO graded (0.5) relevance labels in this dataset, so partial-relevance "
             "handling / sensitivity analysis is N/A.")
    L.append("")
    L.append("For each retriever we retrieve the real top-5, collapse retrieved chunks to a "
             "rank-ordered unique article PMID list via `dedup_articles` (reused from "
             "`evaluation/bioasq_eval/common.py`), and compute metrics against the full normalized "
             "gold set. All retrievers are evaluated on the SAME questions with the SAME gold labels.")
    L.append("")
    L.append("### IR metric definitions (reused from `evaluation/metrics.py`)")
    L.append("")
    L.append("- `precision_at_k`: hits in top-5 / k (denominator is always k=5).")
    L.append("- `recall_at_k`: hits in top-5 / |gold| (denominator = number of gold PMIDs).")
    L.append("- `mrr`: mean of 1/rank of first relevant article; denominator = number of queries "
             "in the view.")
    L.append("- `ndcg_at_k`: binary gain (1 if retrieved article in gold else 0), discount "
             "log2(rank+1); IDCG places min(k, |gold|) relevant at the top; returns 0.0 when IDCG=0.")
    L.append("")
    L.append("Two IR aggregate views are reported for each retriever:")
    L.append("")
    L.append("- **end-to-end**: denominator = all evaluated questions; uncovered-gold questions "
             "contribute 0. This is the headline view.")
    L.append("- **retrieval-effectiveness**: denominator = covered questions only (>=1 gold PMID "
             "present in the index); `null` when there are zero covered questions.")
    L.append("")
    L.append("Undefined denominators are reported as `null`, never 0.")
    L.append("")
    L.append("### Classification metric definitions")
    L.append("")
    L.append("- Evaluated unit: a `(question, candidate-article)` pair.")
    L.append("- Candidate universe per question = (top-5 retrieved article PMIDs) UNION "
             "(available gold PMIDs = gold PMIDs present in that index's corpus). This bounds the "
             "negative set to a documented, finite, index-grounded universe (NOT an arbitrary one).")
    L.append("- TP = pair in top-5 AND in gold; FP = in top-5 AND not gold; FN = in gold AND not "
             "top-5; TN = in universe AND neither retrieved nor gold.")
    L.append("- Micro-sum TP/FP/FN/TN across all evaluated questions, then "
             "Accuracy=(TP+TN)/(TP+TN+FP+FN), Precision=TP/(TP+FP), Recall=TP/(TP+FN), "
             "Specificity=TN/(TN+FP), F1=2PR/(P+R). Any undefined denominator -> `null`.")
    L.append("- CAVEAT: TN is small by construction (universe = top5 ∪ available-gold), so "
             "Specificity and Accuracy are optimistic/limited. A separate corpus-universe "
             "Specificity sensitivity value (negatives = all corpus PMIDs not gold) is recorded in "
             "`metrics.json` under `classification_specificity_sensitivity`, labeled as a sensitivity "
             "analysis, NOT the primary number.")
    L.append("")
    L.append("### Retriever configurations")
    L.append("")
    L.append("- Dense: `DenseRetriever.load(index_dir, embedder=BGEEmbedding())`, BGE-M3 1024-dim, "
             "FAISS inner-product, top_k=5.")
    L.append("- Sparse: `SparseRetriever.load(bm25.pkl)`, Okapi BM25 (k1=1.5, b=0.75), top_k=5. "
             "BM25 only returns score>0 chunks, so it can return fewer than 5.")
    L.append(f"- Hybrid: `HybridRetriever(dense, sparse, query_expander=QueryExpander(), "
             f"reranker=None, rrf_k={HYBRID_CONFIG['rrf_k']})`; RRF fusion, query expansion ON, "
             f"cross-encoder reranker OFF, dense_top_k=sparse_top_k=10 (pool), top_k=5.")
    L.append("")
    L.append("## Data validation & coverage")
    L.append("")
    L.append(f"- Total questions in dataset: {integrity['n_questions_total']}")
    L.append(f"- Distinct question ids: {integrity['n_distinct_question_ids']} "
             f"(duplicates: {integrity['n_duplicate_question_ids']})")
    L.append(f"- Questions with empty body: {integrity['n_empty_body']}")
    L.append(f"- Questions with zero gold label: {integrity['n_zero_gold_label']}")
    L.append(f"- Unparseable document URLs (no trailing PMID): "
             f"{integrity['n_unparseable_document_urls']}")
    L.append("")
    for res in (diag, prod):
        c = res["coverage"]
        L.append(f"### {res['label']} ({res['evaluation_type']})")
        L.append("")
        L.append(f"- Index: `{res['index_dir']}`")
        L.append(f"- Corpus distinct PMIDs: {c['corpus_distinct_pmids']}")
        L.append(f"- Evaluated questions: {c['n_evaluated']} | Excluded: {c['n_excluded']} "
                 f"(evaluated+excluded = {c['consistency_check_evaluated_plus_excluded']} "
                 f"= total {c['n_total_questions']})")
        L.append(f"- Questions with >=1 gold available in corpus (covered): {c['n_covered']} / "
                 f"{c['n_with_gold']} with gold -> coverage = "
                 f"{c['coverage_ratio_covered_over_with_gold']}")
        L.append(f"- Uncovered (gold entirely absent from corpus): {c['n_uncovered']} "
                 f"(fully-absent-gold questions: {c['n_fully_absent_gold']})")
        if res["coverage"]["n_excluded"]:
            reasons: Dict[str, int] = {}
            for e in res["excluded_questions"]:
                reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
            L.append(f"- Exclusion reasons: {reasons}")
        L.append("")

    # Table A — classification
    L.append("## Table A — Classification metrics")
    L.append("")
    L.append("Primary (candidate universe = top-5 ∪ available-gold; relevant=1, irrelevant=0; "
             "no 0.5 grades).")
    L.append("")
    L.append("| Retriever | Accuracy | Precision | Recall | Specificity | F1 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for rk, name in (("dense", "Dense (diagnostic)"), ("sparse", "Sparse (diagnostic)"),
                     ("hybrid", "Hybrid-RRF (diagnostic)")):
        m = _clf_row(diag, rk)
        L.append(f"| {name} | {_fmt(m['accuracy'])} | {_fmt(m['precision'])} | {_fmt(m['recall'])} "
                 f"| {_fmt(m['specificity'])} | {_fmt(m['f1'])} |")
    pm = _clf_row(prod, "dense")
    L.append(f"| Dense (production) | {_fmt(pm['accuracy'])} | {_fmt(pm['precision'])} "
             f"| {_fmt(pm['recall'])} | {_fmt(pm['specificity'])} | {_fmt(pm['f1'])} |")
    L.append("| Sparse (production) | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE "
             "| NOT MEASURABLE | NOT MEASURABLE |")
    L.append("| Hybrid (production) | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE "
             "| NOT MEASURABLE | NOT MEASURABLE |")
    L.append("")
    L.append(f"Production sparse/hybrid reason: {PROD_NOT_MEASURABLE_REASON}.")
    L.append("")

    # Table B — IR (headline end-to-end view)
    L.append("## Table B — Retrieval metrics (end-to-end view)")
    L.append("")
    L.append("| Retriever | Precision@5 | Recall@5 | MRR | nDCG@5 |")
    L.append("|---|---:|---:|---:|---:|")
    for rk, name in (("dense", "Dense (diagnostic)"), ("sparse", "Sparse (diagnostic)"),
                     ("hybrid", "Hybrid-RRF (diagnostic)")):
        p, r, mr, nd = _ir_row(diag, rk, "end_to_end")
        L.append(f"| {name} | {_fmt(p)} | {_fmt(r)} | {_fmt(mr)} | {_fmt(nd)} |")
    p, r, mr, nd = _ir_row(prod, "dense", "end_to_end")
    L.append(f"| Dense (production) | {_fmt(p)} | {_fmt(r)} | {_fmt(mr)} | {_fmt(nd)} |")
    L.append("| Sparse (production) | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE "
             "| NOT MEASURABLE |")
    L.append("| Hybrid (production) | NOT MEASURABLE | NOT MEASURABLE | NOT MEASURABLE "
             "| NOT MEASURABLE |")
    L.append("")
    L.append("### Table B (retrieval-effectiveness view — covered questions only)")
    L.append("")
    L.append("| Retriever | Precision@5 | Recall@5 | MRR | nDCG@5 |")
    L.append("|---|---:|---:|---:|---:|")
    for rk, name in (("dense", "Dense (diagnostic)"), ("sparse", "Sparse (diagnostic)"),
                     ("hybrid", "Hybrid-RRF (diagnostic)")):
        p, r, mr, nd = _ir_row(diag, rk, "retrieval_effectiveness")
        L.append(f"| {name} | {_fmt(p)} | {_fmt(r)} | {_fmt(mr)} | {_fmt(nd)} |")
    p, r, mr, nd = _ir_row(prod, "dense", "retrieval_effectiveness")
    L.append(f"| Dense (production) | {_fmt(p)} | {_fmt(r)} | {_fmt(mr)} | {_fmt(nd)} |")
    L.append("")

    L.append("## Per-result summary")
    L.append("")
    for res in (diag, prod):
        c = res["coverage"]
        L.append(f"- **{res['label']}** ({res['evaluation_type']}): embedding=`{EMBEDDING_MODEL}`, "
                 f"generation=`{GENERATION_MODEL}` (not invoked), index=`{res['index_dir']}`, "
                 f"#evaluated={c['n_evaluated']}, coverage="
                 f"{c['coverage_ratio_covered_over_with_gold']} "
                 f"({c['n_covered']}/{c['n_with_gold']} with gold).")
    L.append("")
    L.append("## Limitations")
    L.append("")
    L.append(f"- `backend/app/config.py` has a STALE `embedding_model_name=\"{STALE_CONFIG_EMBEDDING_ID}\"` "
             f"that is NOT what the indexes use; the real retrieval model is `{EMBEDDING_MODEL}` per the "
             "index manifests. Reported, not corrected (read-only scope).")
    L.append("- Classification TN is small by construction (universe = top5 ∪ available-gold), so "
             "Specificity and Accuracy are optimistic; see the corpus-universe sensitivity value in "
             "`metrics.json`.")
    L.append("- Gold labels are binary (relevant=1 / irrelevant=0); there are no 0.5 grades, so "
             "partial-relevance handling is N/A.")
    L.append("- Production sparse + hybrid are NOT MEASURABLE: " + PROD_NOT_MEASURABLE_REASON + ".")
    L.append("- CPU-only run; the generation LLM (Qwen3-8B) was never invoked.")
    L.append("")
    L.append("## Reproduce")
    L.append("")
    L.append(f"See `{commands_rel}` for the exact PowerShell commands.")
    L.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def write_commands(path: Path) -> None:
    text = (
        "# Exact commands used to run this evaluation\n\n"
        "```powershell\n"
        "cd d:\\Ragmed_2\\backend\n"
        "$env:HF_HUB_OFFLINE=1\n"
        "$env:TRANSFORMERS_OFFLINE=1\n"
        "d:\\Ragmed_2\\.venv\\Scripts\\python.exe scripts\\run_full_retrieval_eval.py\n"
        "```\n\n"
        f"venv python: d:\\Ragmed_2\\.venv\\Scripts\\python.exe\n"
        f"cwd: d:\\Ragmed_2\\backend\n"
        f"env: HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1 (offline CPU)\n"
        f"Embedding model: {EMBEDDING_MODEL}; Generation model (not invoked): {GENERATION_MODEL}\n"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _strip_internal(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of a per-index result without the internal jsonl payload."""
    clean = dict(result)
    clean.pop("_per_ret_jsonl", None)
    return clean


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    _assert_output_safety()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MIRROR_DIR.mkdir(parents=True, exist_ok=True)

    from evaluation import metrics
    from evaluation.bioasq_eval.common import normalize_pmid, dedup_articles, atomic_write_json
    from app.rag.embeddings.bge_embedding import BGEEmbedding

    questions = load_questions()
    integrity = data_integrity(questions, normalize_pmid)

    # Shared BGE-M3 embedder (single load).
    embedder = BGEEmbedding()

    # ---- Diagnostic index: dense + sparse + hybrid -------------------------
    diag_retr = build_retrievers(DIAG_INDEX_DIR, with_sparse=True, with_hybrid=True,
                                 embedder=embedder)
    assert diag_retr["dense"].size == DIAG_EXPECTED_SIZE, (
        f"diag dense size {diag_retr['dense'].size} != {DIAG_EXPECTED_SIZE}")
    assert diag_retr["sparse"].size == DIAG_EXPECTED_SIZE, (
        f"diag sparse size {diag_retr['sparse'].size} != {DIAG_EXPECTED_SIZE}")
    assert diag_retr["hybrid"].size == DIAG_EXPECTED_SIZE, (
        f"diag hybrid size {diag_retr['hybrid'].size} != {DIAG_EXPECTED_SIZE}")
    diag_corpus = corpus_pmid_set(diag_retr["dense"].vector_store.chunks, normalize_pmid)
    diag = run_index("Diagnostic", False, diag_retr, questions, diag_corpus,
                     metrics, dedup_articles, normalize_pmid)

    # ---- Production index: dense only --------------------------------------
    prod_retr = build_retrievers(PROD_INDEX_DIR, with_sparse=False, with_hybrid=False,
                                 embedder=embedder)
    prod_corpus = corpus_pmid_set(prod_retr["dense"].vector_store.chunks, normalize_pmid)
    prod = run_index("Production", True, prod_retr, questions, prod_corpus,
                     metrics, dedup_articles, normalize_pmid)

    # ---- Write per-question JSONL ------------------------------------------
    for out_root in (OUT_DIR, MIRROR_DIR):
        for rk in diag["_per_ret_jsonl"]:
            _write_jsonl(out_root / f"per_question_{rk}.jsonl", diag["_per_ret_jsonl"][rk])
        _write_jsonl(out_root / "per_question_production_dense.jsonl",
                     prod["_per_ret_jsonl"]["dense"])

    # ---- Write metrics.json ------------------------------------------------
    metrics_obj = {
        "generated_utc": _utc_now_iso(),
        "seed": SEED,
        "top_k": TOP_K,
        "device": "cpu",
        "embedding_model": EMBEDDING_MODEL,
        "generation_model": GENERATION_MODEL,
        "generation_model_ollama": GENERATION_MODEL_OLLAMA,
        "generation_model_note": (
            "SEPARATE from retrieval; NOT invoked by this evaluation"),
        "stale_config_embedding_id": STALE_CONFIG_EMBEDDING_ID,
        "binary_relevance": {"relevant": 1, "irrelevant": 0,
                             "partial_grades": "none (N/A)"},
        "hybrid_config": HYBRID_CONFIG,
        "data_integrity": integrity,
        "production_not_measurable": {
            "sparse": PROD_NOT_MEASURABLE_REASON,
            "hybrid": PROD_NOT_MEASURABLE_REASON,
        },
        "results": {
            "diagnostic": _strip_internal(diag),
            "production": _strip_internal(prod),
        },
    }
    for out_root in (OUT_DIR, MIRROR_DIR):
        atomic_write_json(out_root / "metrics.json", metrics_obj)
        write_commands(out_root / "commands.md")
        write_report(out_root / "report.md", diag, prod, integrity, "commands.md")

    # ---- Console summary ---------------------------------------------------
    print("=" * 72)
    print("Full retrieval evaluation — summary")
    print("=" * 72)
    print(f"Output dir: {OUT_DIR}")
    print(f"Mirror dir: {MIRROR_DIR}")
    print(f"Embedding model: {EMBEDDING_MODEL} | Generation model (NOT invoked): {GENERATION_MODEL}")
    print(f"Diagnostic corpus PMIDs={len(diag_corpus)}  Production corpus PMIDs={len(prod_corpus)}")
    for res in (diag, prod):
        c = res["coverage"]
        print(f"\n[{res['label']} / {res['evaluation_type']}] "
              f"evaluated={c['n_evaluated']} excluded={c['n_excluded']} "
              f"coverage={c['coverage_ratio_covered_over_with_gold']} "
              f"({c['n_covered']}/{c['n_with_gold']})")
        for rk in res["retrievers"]:
            e2e = res["retrievers"][rk]["ir_views"]["end_to_end"]
            clf = res["retrievers"][rk]["classification_primary"]
            print(f"  {rk:8s} IR(e2e) P@5={_fmt(e2e['precision_at_5'])} "
                  f"R@5={_fmt(e2e['recall_at_5'])} MRR={_fmt(e2e['mrr'])} "
                  f"nDCG@5={_fmt(e2e['ndcg_at_5'])} | "
                  f"CLF Acc={_fmt(clf['accuracy'])} P={_fmt(clf['precision'])} "
                  f"R={_fmt(clf['recall'])} Spec={_fmt(clf['specificity'])} "
                  f"F1={_fmt(clf['f1'])} (TP={clf['TP']} FP={clf['FP']} "
                  f"FN={clf['FN']} TN={clf['TN']})")
    print("\nProduction sparse + hybrid: NOT MEASURABLE — " + PROD_NOT_MEASURABLE_REASON)
    print("Indexes were NOT modified. NO git commit was made. Generation LLM NOT invoked.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
