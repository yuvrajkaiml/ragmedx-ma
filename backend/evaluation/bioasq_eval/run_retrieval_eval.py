"""
Phase C — Retrieval evaluation against the isolated BioASQ retrieval-eval index.

Reuses ``evaluation/metrics.py`` (precision_at_k / recall_at_k / ndcg_at_k / mrr).
For each of the 250 eval questions, retrieves the actual top-5 from the v1 index,
maps retrieved chunks -> articles via rank-preserving dedup, and computes per-query
metrics against the full normalized gold set. Emits three views (design 7.3):

  - corpus coverage  (coverage@>=1, coverage@all over 250)
  - end-to-end       (denominator all 250; uncovered-gold queries score 0)
  - retrieval effectiveness (denominator = covered queries only)

Mere PMID presence is never a hit — only top-5 article-rank membership counts.

Run from ``backend/``::

    d:\\Ragmed_2\\.venv\\Scripts\\python.exe -m evaluation.bioasq_eval.run_retrieval_eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluation import metrics  # noqa: E402
from evaluation.bioasq_eval.common import (  # noqa: E402
    SEED, JsonlCheckpoint, atomic_write_json, dedup_articles, normalize_pmid,
    write_manifest,
)

EVAL_QUESTIONS = BACKEND_DIR / "data" / "datasets" / "bioasq_eval_split" / "eval_questions.json"
INDEX_DIR = BACKEND_DIR / "data" / "indexes" / "bioasq_retrieval_eval" / "v1"
RESULTS_DIR = BACKEND_DIR / "results" / "bioasq_eval_20261018"
RETRIEVAL_DIR = RESULTS_DIR / "retrieval"
TOP_K = 5


def _gold_set(q: Dict[str, Any]) -> List[str]:
    golds = {normalize_pmid(d) for d in (q.get("documents") or [])}
    return sorted(g for g in golds if g is not None)


def run(index_dir: Path = INDEX_DIR) -> Dict[str, Any]:
    with open(EVAL_QUESTIONS, "r", encoding="utf-8") as fh:
        questions = json.load(fh)

    from app.rag.embeddings.bge_embedding import BGEEmbedding
    from app.rag.retrievers.dense_retriever import DenseRetriever

    embedder = BGEEmbedding()  # single load
    retriever = DenseRetriever.load(index_dir, embedder=embedder)
    corpus_pmids = {normalize_pmid((c.metadata or {}).get("pmid"))
                    for c in retriever.vector_store.chunks}
    corpus_pmids.discard(None)

    RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = JsonlCheckpoint(RETRIEVAL_DIR / "per_query.jsonl", key_field="question_id")

    per_query: Dict[str, Dict[str, Any]] = {}
    # Load any completed records (resume).
    for rec in ckpt.read_all():
        per_query[rec["question_id"]] = rec

    for q in questions:
        qid = q.get("id")
        if ckpt.is_done(qid):
            continue
        gold = _gold_set(q)
        available = sorted(set(gold) & corpus_pmids)
        covered = len(available) > 0
        body = q.get("body") or ""
        fb: Dict[str, int] = {}
        if not body.strip():
            retrieved_articles: List[str] = []
        else:
            results = retriever.retrieve(body, top_k=TOP_K)
            retrieved_articles = dedup_articles((c for c, _ in results), fb)

        p = metrics.precision_at_k(retrieved_articles, gold, TOP_K)
        r = metrics.recall_at_k(retrieved_articles, gold, TOP_K)
        nd = metrics.ndcg_at_k(retrieved_articles, gold, TOP_K)
        # Reciprocal rank for this single query (via the batch helper).
        rr = metrics.mrr([retrieved_articles], [gold])

        rec = {
            "question_id": qid,
            "type": q.get("type"),
            "gold": gold,
            "available": available,
            "covered": covered,
            "retrieved_articles": retrieved_articles,
            "precision_at_5": p,
            "recall_at_5": r,
            "ndcg_at_5": nd,
            "reciprocal_rank": rr,
        }
        ckpt.append(rec)
        per_query[qid] = rec

    # Assemble the three views (one metric computation; denominator differs).
    ordered = [per_query[q["id"]] for q in questions if q["id"] in per_query]
    n_total = len(ordered)
    covered_recs = [rec for rec in ordered if rec["covered"]]
    n_covered = len(covered_recs)

    def _macro(recs: List[Dict[str, Any]], field: str, denom: int) -> Optional[float]:
        if denom == 0:
            return None
        return round(sum(rec[field] for rec in recs) / denom, 6)

    # coverage view
    coverage_ge1 = sum(1 for rec in ordered if rec["covered"])
    coverage_all = sum(1 for rec in ordered
                       if rec["gold"] and set(rec["available"]) == set(rec["gold"]))
    uncovered = sum(1 for rec in ordered if rec["gold"] and not rec["covered"])
    partially_covered = sum(1 for rec in ordered
                            if rec["covered"] and set(rec["available"]) != set(rec["gold"]))

    views = {
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
            "note": "All 250 queries; uncovered-gold queries contribute 0.",
            "precision_at_5": _macro(ordered, "precision_at_5", n_total),
            "recall_at_5": _macro(ordered, "recall_at_5", n_total),
            "ndcg_at_5": _macro(ordered, "ndcg_at_5", n_total),
            "mrr": metrics.mrr([rec["retrieved_articles"] for rec in ordered],
                               [rec["gold"] for rec in ordered]),
        },
        "retrieval_effectiveness": {
            "denominator": n_covered,
            "note": "Only covered queries (available gold non-empty); metric vs FULL gold.",
            "precision_at_5": _macro(covered_recs, "precision_at_5", n_covered),
            "recall_at_5": _macro(covered_recs, "recall_at_5", n_covered),
            "ndcg_at_5": _macro(covered_recs, "ndcg_at_5", n_covered),
            "mrr": (metrics.mrr([rec["retrieved_articles"] for rec in covered_recs],
                                [rec["gold"] for rec in covered_recs])
                    if n_covered else None),
        },
    }

    out = {
        "seed": SEED,
        "top_k": TOP_K,
        "index_dir": str(index_dir),
        "n_queries_evaluated": n_total,
        "views": views,
        "per_query": ordered,
    }
    atomic_write_json(RETRIEVAL_DIR / "retrieval_metrics.json", out)

    # Side file for the diagnostic's per-corpus availability counts.
    atomic_write_json(RETRIEVAL_DIR / "corpus_availability.json", {
        "corpus": str(index_dir),
        "corpus_distinct_pmids": len(corpus_pmids),
        "n_queries_with_at_least_one_available_gold": coverage_ge1,
        "n_queries_with_all_gold_available": coverage_all,
    })

    write_manifest("retrieval", cfg={"top_k": TOP_K, "index_dir": str(index_dir), "seed": SEED},
                   inputs={"eval_questions": EVAL_QUESTIONS,
                           "index": index_dir / "index.faiss"},
                   extra={"n_queries": n_total, "views": views})
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase C retrieval evaluation")
    parser.add_argument("--index-dir", type=str, default=str(INDEX_DIR))
    args = parser.parse_args(argv)
    out = run(Path(args.index_dir))
    e2e = out["views"]["end_to_end"]
    cov = out["views"]["corpus_coverage"]
    eff = out["views"]["retrieval_effectiveness"]
    print(f"n={out['n_queries_evaluated']} "
          f"coverage@>=1={cov['coverage_at_least_1']} "
          f"end_to_end P@5={e2e['precision_at_5']} R@5={e2e['recall_at_5']} "
          f"MRR={e2e['mrr']} nDCG@5={e2e['ndcg_at_5']} | "
          f"effectiveness(n={eff['denominator']}) P@5={eff['precision_at_5']} "
          f"MRR={eff['mrr']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
