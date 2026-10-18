"""
Standalone top-5 retrieval evaluation (dense-only AND sparse-only, SEPARATELY)
over the existing BioASQ retrieval-eval indexes.

This script is ADDITIVE and READ-ONLY on the indexes:
  - It loads the dense (BGE-M3 + FAISS) and sparse (Okapi BM25) retrievers.
  - It NEVER calls .save() / .add_chunks() / rebuilds / re-embeds anything.
  - It writes all NEW artifacts to results/standalone_top5_eval_20261018/.
  - It makes NO git commits.

Run from cwd = d:\\Ragmed_2\\backend with the CPU venv:

    cd d:\\Ragmed_2\\backend
    $env:HF_HUB_OFFLINE=1
    $env:TRANSFORMERS_OFFLINE=1
    d:\\Ragmed_2\\.venv\\Scripts\\python.exe scripts\\run_standalone_top5_eval.py

If BGE-M3 cannot load from the local HF cache, the script STOPS the dense
portion, reports the blocker, and still runs + reports the sparse portion
(sparse needs no model). It never fabricates dense results.
"""

from __future__ import annotations

# --- Force HF offline BEFORE importing app/transformers/BGE ------------------
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Path anchoring (never rely on cwd) --------------------------------------
BACKEND_DIR = Path(__file__).resolve().parents[1]   # d:\Ragmed_2\backend
REPO_ROOT = BACKEND_DIR.parent                      # d:\Ragmed_2

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

INDEX_DIR = BACKEND_DIR / "data" / "indexes" / "bioasq_retrieval_eval" / "v1"
BM25_PATH = INDEX_DIR / "bm25.pkl"
EVAL_QUESTIONS = BACKEND_DIR / "data" / "datasets" / "bioasq_eval_split" / "eval_questions.json"
OUT_DIR = BACKEND_DIR / "results" / "standalone_top5_eval_20261018"
HANDOFF_DIR = REPO_ROOT / ".agents" / "tasks" / "ragmed2-yuvraj-llm-2026-10-17" / "handoff"

TOP_K = 5
N_QUESTIONS = 18
SEED = 42
TRUNC = 200

RELEVANCE_BASIS = "article-level (BioASQ gold PMIDs; dataset ground truth)"
PASSAGE_NOTE = (
    "Passage-level relevance labels are PENDING Sana's independent review and "
    "were NOT fabricated here (relevance_label is null/blank in every per-result row)."
)

# Exact template column order (verified from
# .agents/tasks/ragmed2-yuvraj-llm-2026-10-17/handoff/sana_evaluation_sheet_TEMPLATE.csv)
SANA_COLUMNS = [
    "question_id",
    "query",
    "expected_evidence_pmids",
    "expected_passage_notes",
    "retriever",
    "retrieved_chunk_id",
    "source_pmid",
    "rank",
    "score",
    "relevance_label",
    "notes",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _assert_output_safety() -> None:
    """Guard: never write under the production FAISS index or inside the eval index."""
    forbidden = (BACKEND_DIR / "data" / "faiss_index").resolve()
    out_resolved = OUT_DIR.resolve()
    assert out_resolved != forbidden and forbidden not in out_resolved.parents, (
        "output dir must not be under backend/data/faiss_index"
    )
    index_resolved = INDEX_DIR.resolve()
    assert out_resolved != index_resolved and index_resolved not in out_resolved.parents, (
        "output dir must not be inside the eval index directory"
    )


def _gold_set(q: Dict[str, Any], normalize_pmid) -> List[str]:
    golds = {normalize_pmid(d) for d in (q.get("documents") or [])}
    return sorted(g for g in golds if g is not None)


def _select_questions(questions: List[Dict[str, Any]], corpus_pmids: set, normalize_pmid):
    """Select FIRST N_QUESTIONS covered questions in file order (deterministic)."""
    selected: List[Dict[str, Any]] = []
    for q in questions:
        body = (q.get("body") or "").strip()
        if not body:
            continue
        gold = _gold_set(q, normalize_pmid)
        available = sorted(set(gold) & corpus_pmids)
        if not available:
            continue
        gold_snippets = [s.get("text", "") for s in (q.get("snippets") or []) if s.get("text")]
        selected.append(
            {
                "id": q.get("id"),
                "body": q.get("body"),
                "type": q.get("type"),
                "gold": gold,
                "available": available,
                "gold_snippets": gold_snippets,
            }
        )
        if len(selected) >= N_QUESTIONS:
            break
    return selected


def _capture(retriever_name: str, retriever, body: str, normalize_pmid):
    """Run one retriever for one question. Returns (rows, raw_results, n_results, error)."""
    body_stripped = (body or "").strip()
    if not body_stripped:
        return [], [], 0, "empty_query"
    try:
        raw = retriever.retrieve(body_stripped, top_k=TOP_K)
    except Exception as exc:  # record failures honestly, do not crash the run
        return [], [], 0, f"{type(exc).__name__}: {exc}"

    rows = []
    for rank, (chunk, score) in enumerate(raw, start=1):
        content = chunk.content or ""
        rows.append(
            {
                "rank": rank,
                "chunk_id": chunk.id,
                "source_pmid": normalize_pmid((chunk.metadata or {}).get("pmid")),
                "score": float(score),
                "text": content,
                "text_truncated_200": content[:TRUNC],
                "relevance_label": None,
            }
        )
    return rows, raw, len(rows), None


def _per_retriever_metrics(questions_data, retriever_key, metrics, dedup_articles):
    """Compute per-question + aggregate article-level metrics for one retriever."""
    per_question = []
    all_retrieved: List[List[str]] = []
    all_gold: List[List[str]] = []
    failures = []
    short_results = []

    for qd in questions_data:
        gold = qd["gold"]
        block = qd[retriever_key]
        fb: Dict[str, int] = {}
        retrieved_articles = dedup_articles((c for c, _ in block["raw"]), fb)

        p = metrics.precision_at_k(retrieved_articles, gold, TOP_K)
        r = metrics.recall_at_k(retrieved_articles, gold, TOP_K)
        nd = metrics.ndcg_at_k(retrieved_articles, gold, TOP_K)
        rr = metrics.mrr([retrieved_articles], [gold])

        per_question.append(
            {
                "question_id": qd["id"],
                "type": qd["type"],
                "precision_at_5": p,
                "recall_at_5": r,
                "ndcg_at_5": nd,
                "reciprocal_rank": rr,
                "retrieved_articles": retrieved_articles,
                "gold": gold,
                "n_results": block["n_results"],
            }
        )
        all_retrieved.append(retrieved_articles)
        all_gold.append(gold)

        if block["error"]:
            failures.append({"question_id": qd["id"], "error": block["error"]})
        if block["error"] is None and block["n_results"] < TOP_K:
            short_results.append(
                {"question_id": qd["id"], "n_results": block["n_results"]}
            )

    n = len(per_question)
    macro = {
        "precision_at_5": round(sum(x["precision_at_5"] for x in per_question) / n, 6) if n else None,
        "recall_at_5": round(sum(x["recall_at_5"] for x in per_question) / n, 6) if n else None,
        "ndcg_at_5": round(sum(x["ndcg_at_5"] for x in per_question) / n, 6) if n else None,
    }
    overall_mrr = metrics.mrr(all_retrieved, all_gold) if n else None

    return {
        "relevance_basis": RELEVANCE_BASIS,
        "passage_level_note": PASSAGE_NOTE,
        "macro_precision_at_5": macro["precision_at_5"],
        "macro_recall_at_5": macro["recall_at_5"],
        "macro_ndcg_at_5": macro["ndcg_at_5"],
        "overall_mrr": overall_mrr,
        "coverage": {
            "n_questions": n,
            "n_with_full_5_results": sum(
                1 for qd in questions_data if qd[retriever_key]["n_results"] == TOP_K
            ),
            "n_with_fewer_than_5": len(short_results),
            "n_with_zero_results": sum(
                1 for qd in questions_data if qd[retriever_key]["n_results"] == 0
            ),
        },
        "failures": failures,
        "short_result_questions": short_results,
        "per_question": per_question,
    }


def _write_jsonl(path: Path, questions_data, retriever_key) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for qd in questions_data:
            rec = {
                "question_id": qd["id"],
                "type": qd["type"],
                "body": qd["body"],
                "gold_pmids": qd["gold"],
                "available_gold_pmids": qd["available"],
                "n_results": qd[retriever_key]["n_results"],
                "retrieval_error": qd[retriever_key]["error"],
                "results": qd[retriever_key]["rows"],
            }
            fh.write(json.dumps(rec, default=str) + "\n")


def _write_questions_json(path: Path, selected) -> None:
    obj = {
        "provisional": True,
        "source": (
            "auto-selected first 18 covered questions from eval_questions.json; "
            "pending Sana confirmation"
        ),
        "selection_rule": (
            "first N covered questions in file order; covered = non-empty body AND "
            "at least one gold PMID present in the corpus"
        ),
        "seed": SEED,
        "n_selected": len(selected),
        "questions": [
            {
                "id": q["id"],
                "body": q["body"],
                "type": q["type"],
                "gold_pmids": q["gold"],
                "available_gold_pmids": q["available"],
                "expected_evidence_snippets": q["gold_snippets"],
                "provisional": True,
            }
            for q in selected
        ],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def _write_sana_csv(path: Path, questions_data) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(SANA_COLUMNS)
        for qd in questions_data:
            expected = ";".join(qd["gold"])
            for retriever_key, retriever_name in (("dense", "dense"), ("sparse", "sparse")):
                block = qd[retriever_key]
                if not block["rows"]:
                    note = block["error"] or "no results"
                    writer.writerow(
                        [
                            qd["id"], qd["body"], expected, "", retriever_name,
                            "", "", "", "", "", note,
                        ]
                    )
                    continue
                for row in block["rows"]:
                    writer.writerow(
                        [
                            qd["id"],
                            qd["body"],
                            expected,
                            "",  # expected_passage_notes (Sana fills)
                            retriever_name,
                            row["chunk_id"],
                            row["source_pmid"] or "",
                            row["rank"],
                            row["score"],
                            "",  # relevance_label (Sana fills)
                            "",  # notes
                        ]
                    )


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def _write_report(path: Path, selected, questions_data, dense_metrics,
                  sparse_metrics, dense_available: bool, dense_blocker,
                  corpus_distinct_pmids: int) -> None:
    types = sorted({q["type"] for q in selected})
    lines: List[str] = []
    lines.append("# Standalone Top-5 Retrieval Evaluation (dense vs sparse)")
    lines.append("")
    lines.append(f"Generated (UTC): {_utc_now_iso()}")
    lines.append(f"Index (read-only): `{INDEX_DIR}`")
    lines.append(f"Embedding model: BAAI/bge-m3 (1024-dim, FAISS inner-product)")
    lines.append(f"Top-k: {TOP_K} | Questions: {len(selected)} | Seed: {SEED}")
    lines.append("")
    lines.append("## Provisional question set")
    lines.append("")
    lines.append(
        "The 18 questions were AUTO-SELECTED as the first 18 covered questions "
        "(non-empty body AND at least one gold PMID present in the corpus) in "
        "`eval_questions.json` file order. This set is PROVISIONAL and is pending "
        "Sana's confirmation. Sana's hand-authored question set is not yet ready; "
        "only the template and task checklist were present in the handoff dir."
    )
    lines.append("")
    lines.append(f"Question types spanned: {', '.join(types)}")
    lines.append("")
    lines.append("## Article-level metrics (dense vs sparse)")
    lines.append("")
    lines.append(f"Relevance basis: {RELEVANCE_BASIS}.")
    lines.append("")
    lines.append("| Metric | Dense | Sparse |")
    lines.append("|---|---|---|")
    lines.append(
        f"| macro Precision@5 | {_fmt(dense_metrics.get('macro_precision_at_5') if dense_available else None)} "
        f"| {_fmt(sparse_metrics.get('macro_precision_at_5'))} |"
    )
    lines.append(
        f"| macro Recall@5 | {_fmt(dense_metrics.get('macro_recall_at_5') if dense_available else None)} "
        f"| {_fmt(sparse_metrics.get('macro_recall_at_5'))} |"
    )
    lines.append(
        f"| macro nDCG@5 | {_fmt(dense_metrics.get('macro_ndcg_at_5') if dense_available else None)} "
        f"| {_fmt(sparse_metrics.get('macro_ndcg_at_5'))} |"
    )
    lines.append(
        f"| overall MRR | {_fmt(dense_metrics.get('overall_mrr') if dense_available else None)} "
        f"| {_fmt(sparse_metrics.get('overall_mrr'))} |"
    )
    lines.append("")
    lines.append(
        "These are article-level metrics computed against the real BioASQ gold "
        "PMIDs in each question's `documents` field (dataset ground truth), matching "
        "the convention in `evaluation/bioasq_eval/run_retrieval_eval.py` (gold = full "
        "normalized gold set). They are NOT invented."
    )
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Corpus distinct PMIDs: {corpus_distinct_pmids}")
    lines.append(f"- Questions evaluated: {len(selected)} (all covered)")
    if dense_available:
        dc = dense_metrics["coverage"]
        lines.append(
            f"- Dense: {dc['n_with_full_5_results']} with 5 results, "
            f"{dc['n_with_fewer_than_5']} with <5, {dc['n_with_zero_results']} with 0."
        )
    else:
        lines.append("- Dense: NOT RUN (BGE-M3 could not load — see blocker below).")
    sc = sparse_metrics["coverage"]
    lines.append(
        f"- Sparse: {sc['n_with_full_5_results']} with 5 results, "
        f"{sc['n_with_fewer_than_5']} with <5, {sc['n_with_zero_results']} with 0."
    )
    lines.append("")
    lines.append("## Failures & limitations (honest)")
    lines.append("")
    if dense_blocker:
        lines.append(f"- BLOCKER (dense): {dense_blocker}")
    if dense_available and dense_metrics["failures"]:
        lines.append(f"- Dense retrieval errors: {dense_metrics['failures']}")
    if sparse_metrics["failures"]:
        lines.append(f"- Sparse retrieval errors: {sparse_metrics['failures']}")
    if dense_available and dense_metrics["short_result_questions"]:
        lines.append(
            f"- Dense questions with <5 results: {dense_metrics['short_result_questions']}"
        )
    if sparse_metrics["short_result_questions"]:
        lines.append(
            f"- Sparse questions with <5 results (BM25 returns only score>0 chunks): "
            f"{sparse_metrics['short_result_questions']}"
        )
    lines.append(
        "- Question set is PROVISIONAL (auto-selected); Sana has NOT reviewed or "
        "confirmed it."
    )
    lines.append(
        "- NO passage-level relevance labels were fabricated. `relevance_label` is "
        "null (JSONL) / blank (CSV) in every per-result row for Sana's independent review."
    )
    lines.append(
        "- Metrics are article-level (PMID) only. Passage-level relevance is a separate, "
        "pending human judgment."
    )
    lines.append("- Hybrid retrieval was NOT run; it is available for a later run.")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```powershell")
    lines.append("cd d:\\Ragmed_2\\backend")
    lines.append("$env:HF_HUB_OFFLINE=1")
    lines.append("$env:TRANSFORMERS_OFFLINE=1")
    lines.append("d:\\Ragmed_2\\.venv\\Scripts\\python.exe scripts\\run_standalone_top5_eval.py")
    lines.append("```")
    lines.append("")
    lines.append(f"Outputs under: `{OUT_DIR}`")
    lines.append(
        "- questions.json, dense_top5.jsonl, sparse_top5.jsonl, metrics.json, "
        "sana_review_sheet.csv, report.md"
    )
    lines.append(
        "Provisional handoff copies: "
        "`sana_questions_PROVISIONAL.json`, `sana_review_sheet_PROVISIONAL.csv`."
    )
    lines.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> int:
    _assert_output_safety()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from evaluation import metrics
    from evaluation.bioasq_eval.common import normalize_pmid, dedup_articles
    from app.rag.retrievers.sparse_retriever import SparseRetriever

    with open(EVAL_QUESTIONS, "r", encoding="utf-8") as fh:
        questions = json.load(fh)

    # --- Sparse loads with no model (always available) ----------------------
    sparse = SparseRetriever.load(BM25_PATH)
    assert sparse.size == 2947, f"sparse size unexpected: {sparse.size}"

    # --- Dense: try to load; blocker if BGE-M3 cannot load ------------------
    dense = None
    dense_blocker: Optional[str] = None
    try:
        from app.rag.embeddings.bge_embedding import BGEEmbedding
        from app.rag.retrievers.dense_retriever import DenseRetriever

        embedder = BGEEmbedding()
        dense = DenseRetriever.load(INDEX_DIR, embedder=embedder)
        assert dense.size == 2947, f"dense size unexpected: {dense.size}"
    except Exception as exc:
        dense_blocker = f"{type(exc).__name__}: {exc}"
        print(
            f"[BLOCKER] Dense (BGE-M3) could not load from cache: {dense_blocker}\n"
            f"          Continuing with SPARSE only (no model needed).",
            file=sys.stderr,
        )

    # --- Build corpus PMID set (prefer dense chunks; sparse is identical) ----
    chunk_source = dense.vector_store.chunks if dense is not None else sparse.chunks
    corpus_pmids = {normalize_pmid((c.metadata or {}).get("pmid")) for c in chunk_source}
    corpus_pmids.discard(None)
    corpus_distinct_pmids = len(corpus_pmids)

    # --- Deterministic provisional selection --------------------------------
    selected = _select_questions(questions, corpus_pmids, normalize_pmid)
    assert selected, "no covered questions selected"

    # --- Per-question dense + sparse capture --------------------------------
    questions_data = []
    for q in selected:
        qd = {
            "id": q["id"],
            "body": q["body"],
            "type": q["type"],
            "gold": q["gold"],
            "available": q["available"],
            "gold_snippets": q["gold_snippets"],
        }
        if dense is not None:
            d_rows, d_raw, d_n, d_err = _capture("dense", dense, q["body"], normalize_pmid)
        else:
            d_rows, d_raw, d_n, d_err = [], [], 0, "dense_not_loaded (BGE-M3 blocker)"
        s_rows, s_raw, s_n, s_err = _capture("sparse", sparse, q["body"], normalize_pmid)
        qd["dense"] = {"rows": d_rows, "raw": d_raw, "n_results": d_n, "error": d_err}
        qd["sparse"] = {"rows": s_rows, "raw": s_raw, "n_results": s_n, "error": s_err}
        questions_data.append(qd)

    dense_available = dense is not None
    dense_metrics = (
        _per_retriever_metrics(questions_data, "dense", metrics, dedup_articles)
        if dense_available
        else {
            "relevance_basis": RELEVANCE_BASIS,
            "passage_level_note": PASSAGE_NOTE,
            "blocker": dense_blocker,
            "note": "Dense not run — BGE-M3 could not load from cache.",
        }
    )
    sparse_metrics = _per_retriever_metrics(questions_data, "sparse", metrics, dedup_articles)

    # --- Write outputs -------------------------------------------------------
    questions_path = OUT_DIR / "questions.json"
    dense_jsonl = OUT_DIR / "dense_top5.jsonl"
    sparse_jsonl = OUT_DIR / "sparse_top5.jsonl"
    metrics_path = OUT_DIR / "metrics.json"
    sana_csv = OUT_DIR / "sana_review_sheet.csv"
    report_path = OUT_DIR / "report.md"

    _write_questions_json(questions_path, selected)
    if dense_available:
        _write_jsonl(dense_jsonl, questions_data, "dense")
    else:
        # Still emit the file so the artifact set is complete; empty results, honest.
        _write_jsonl(dense_jsonl, questions_data, "dense")
    _write_jsonl(sparse_jsonl, questions_data, "sparse")

    metrics_obj = {
        "top_k": TOP_K,
        "n_questions": len(selected),
        "seed": SEED,
        "index_dir": str(INDEX_DIR),
        "embedding_model": "BAAI/bge-m3",
        "provisional": True,
        "corpus_distinct_pmids": corpus_distinct_pmids,
        "relevance_basis": RELEVANCE_BASIS,
        "metrics_basis_note": (
            "Article-level PMID relevance from eval_questions.json 'documents' "
            "(dataset ground truth); gold = full normalized gold set, matching "
            "run_retrieval_eval.py. " + PASSAGE_NOTE
        ),
        "dense_available": dense_available,
        "dense_blocker": dense_blocker,
        "retrievers": {
            "dense": dense_metrics,
            "sparse": sparse_metrics,
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_obj, fh, indent=2, default=str)

    _write_sana_csv(sana_csv, questions_data)
    _write_report(report_path, selected, questions_data, dense_metrics,
                  sparse_metrics, dense_available, dense_blocker,
                  corpus_distinct_pmids)

    # --- Provisional handoff copies (never overwrite template) --------------
    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
    prov_q = HANDOFF_DIR / "sana_questions_PROVISIONAL.json"
    prov_csv = HANDOFF_DIR / "sana_review_sheet_PROVISIONAL.csv"
    template = HANDOFF_DIR / "sana_evaluation_sheet_TEMPLATE.csv"
    assert prov_q.resolve() != template.resolve()
    assert prov_csv.resolve() != template.resolve()
    shutil.copyfile(questions_path, prov_q)
    shutil.copyfile(sana_csv, prov_csv)

    # --- Console summary -----------------------------------------------------
    print("=" * 72)
    print("Standalone top-5 retrieval evaluation — summary")
    print("=" * 72)
    print(f"Indexes loaded READ-ONLY (no .save, no rebuild). Dense available: {dense_available}")
    if dense_blocker:
        print(f"Dense blocker: {dense_blocker}")
    print(f"sparse.size={sparse.size}" + (f"  dense.size={dense.size}" if dense else ""))
    print(f"corpus_distinct_pmids={corpus_distinct_pmids}  n_questions={len(selected)}")
    if dense_available:
        print(
            f"DENSE : P@5={_fmt(dense_metrics['macro_precision_at_5'])} "
            f"R@5={_fmt(dense_metrics['macro_recall_at_5'])} "
            f"MRR={_fmt(dense_metrics['overall_mrr'])} "
            f"nDCG@5={_fmt(dense_metrics['macro_ndcg_at_5'])}"
        )
    else:
        print("DENSE : NOT RUN (BGE-M3 blocker)")
    print(
        f"SPARSE: P@5={_fmt(sparse_metrics['macro_precision_at_5'])} "
        f"R@5={_fmt(sparse_metrics['macro_recall_at_5'])} "
        f"MRR={_fmt(sparse_metrics['overall_mrr'])} "
        f"nDCG@5={_fmt(sparse_metrics['macro_ndcg_at_5'])}"
    )
    print(f"Output dir: {OUT_DIR}")
    print("Limitations: provisional auto-selected questions (Sana not confirmed); "
          "no passage-level labels yet (pending Sana); sparse may return <5; "
          "metrics are article-level PMID only; hybrid not run.")
    print("Indexes were NOT modified. NO git commit was made.")
    print("=" * 72)

    return 2 if (dense_blocker is not None) else 0


if __name__ == "__main__":
    raise SystemExit(main())
