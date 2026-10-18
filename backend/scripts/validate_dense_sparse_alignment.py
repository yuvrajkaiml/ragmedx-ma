"""
Deterministic validation of dense/sparse retrieval-index alignment.

Loads BOTH the dense FAISS index and the BM25 sparse index for the BioASQ
retrieval evaluation independently, in-process, and runs a fixed battery of
checks.  Prints PASS/FAIL + exact counts per check, writes a machine-readable
JSON report and a human-readable Markdown report, and EXITS NON-ZERO if any
hard check fails.

Checks (see instructions):
  1. Dense loads (size, FAISS index.d).
  2. Sparse loads (size).
  3. Chunk-id sets identical between dense and sparse.
  4. Chunk ids unique within the corpus.
  5. Every dense FAISS vector id resolves to a valid chunk record.
  6. Every sparse document id (list position) resolves to a valid chunk with
     non-empty content and a metadata['pmid'].
  7. PMID provenance: metadata['pmid'] normalizes equal to trailing-int of
     source (normalize_pmid).
  8. Dimension/metric: FAISS metric is inner-product, sampled reconstructed
     vectors are finite and (normalize=True) L2-norm ~ 1.0.
  9. Smoke queries run on BOTH retrievers without error, each returning >= 1
     result; print top-1 chunk_id + source_pmid + score.
 10. Consistency: smoke-query result ids exist in the canonical set and the
     returned passage text equals the stored content for that id.

Run from backend with the CPU venv::

    cd d:\\Ragmed_2\\backend
    d:\\Ragmed_2\\.venv\\Scripts\\python.exe scripts/validate_dense_sparse_alignment.py

BGE-M3 is loaded (offline from cache if available) for dense retrieval. If the
embedding model cannot load, the dense-dependent checks are marked BLOCKED and
the script still runs every network-free check from v1/chunks.pkl + bm25.pkl.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Prefer offline model loading when a cached copy exists; avoids silent
# re-download attempts turning a cached run into a network run.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BACKEND_DIR = Path(__file__).resolve().parents[1]  # d:\Ragmed_2\backend
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.rag.vectorstore.faiss_store import FAISSVectorStore  # noqa: E402
from app.rag.retrievers.sparse_retriever import SparseRetriever  # noqa: E402
from evaluation.bioasq_eval.common import normalize_pmid  # noqa: E402

import faiss  # noqa: E402

V1_DIR = BACKEND_DIR / "data" / "indexes" / "bioasq_retrieval_eval" / "v1"
SPARSE_PATH = V1_DIR / "bm25.pkl"
RESULTS_DIR = BACKEND_DIR / "results" / "bioasq_eval_20261018"
JSON_OUT = RESULTS_DIR / "dense_sparse_validation.json"
MD_OUT = RESULTS_DIR / "dense_sparse_validation_report.md"

EXPECTED_N = 2947
EXPECTED_DIM = 1024

SMOKE_QUERIES = [
    "What is the mechanism of action of metformin?",
    "first-line treatment for hypertension",
    "BRCA1 breast cancer risk",
    "symptoms of hypothyroidism",
    "CTLA-4 immune checkpoint",
]


class Checks:
    """Collects check results; tracks whether any HARD check failed."""

    def __init__(self) -> None:
        self.results: list[dict] = []
        self.any_hard_fail = False

    def record(self, name, status, counts=None, hard=True, detail=None):
        # status: 'PASS' | 'FAIL' | 'BLOCKED'
        entry = {
            "name": name,
            "status": status,
            "hard": hard,
            "counts": counts or {},
        }
        if detail:
            entry["detail"] = detail
        self.results.append(entry)
        if hard and status == "FAIL":
            self.any_hard_fail = True
        tag = {"PASS": "PASS", "FAIL": "FAIL", "BLOCKED": "BLOCKED"}[status]
        print(f"[{tag}] {name}  counts={entry['counts']}"
              + (f"  detail={detail}" if detail else ""))
        return entry


def _source_pmid(chunk) -> str | None:
    return normalize_pmid(getattr(chunk, "source", None))


def main() -> int:
    checks = Checks()
    print(f"[info] dense dir : {V1_DIR}")
    print(f"[info] sparse file: {SPARSE_PATH}")

    # ------------------------------------------------------------------
    # Load the FAISS store (no embedder needed) — basis for most checks.
    # ------------------------------------------------------------------
    try:
        store = FAISSVectorStore.load(V1_DIR)
        dense_chunks = store.chunks
    except Exception as exc:  # pragma: no cover - fatal
        print(f"[fatal] cannot load dense FAISS store: {exc}")
        traceback.print_exc()
        checks.record("1_dense_store_load", "FAIL",
                      detail=str(exc))
        _write_reports(checks, blocked_reason="dense store load failed")
        return 1

    # ------------------------------------------------------------------
    # Load sparse retriever.
    # ------------------------------------------------------------------
    try:
        sparse = SparseRetriever.load(SPARSE_PATH)
        sparse_chunks = sparse.chunks
    except Exception as exc:
        print(f"[fatal] cannot load sparse retriever: {exc}")
        traceback.print_exc()
        checks.record("2_sparse_load", "FAIL", detail=str(exc))
        _write_reports(checks, blocked_reason="sparse load failed")
        return 1

    # ------------------------------------------------------------------
    # Try to construct the dense retriever (needs the embedding model).
    # Failure here only BLOCKS dense-retrieval smoke checks (9/10 dense).
    # ------------------------------------------------------------------
    dense_retriever = None
    dense_model_error = None
    try:
        from app.rag.embeddings.bge_embedding import BGEEmbedding
        from app.rag.retrievers.dense_retriever import DenseRetriever
        dense_retriever = DenseRetriever.load(V1_DIR, embedder=BGEEmbedding())
    except Exception as exc:
        dense_model_error = f"{type(exc).__name__}: {exc}"
        print(f"[warn] BGE-M3 / dense retriever unavailable: {dense_model_error}")

    # ================================================================
    # Check 1: Dense loads (size + FAISS index.d).
    # ================================================================
    d = int(store.index.d)
    n_dense = len(dense_chunks)
    ok1 = (n_dense == EXPECTED_N) and (d == EXPECTED_DIM) \
        and (int(store.index.ntotal) == n_dense)
    checks.record(
        "1_dense_loads",
        "PASS" if ok1 else "FAIL",
        counts={"dense_chunks": n_dense, "ntotal": int(store.index.ntotal),
                "faiss_d": d, "expected_n": EXPECTED_N,
                "expected_dim": EXPECTED_DIM},
    )

    # ================================================================
    # Check 2: Sparse loads (size).
    # ================================================================
    n_sparse = sparse.size
    ok2 = (n_sparse == EXPECTED_N) and (len(sparse_chunks) == n_sparse)
    checks.record(
        "2_sparse_loads",
        "PASS" if ok2 else "FAIL",
        counts={"sparse_size": n_sparse, "sparse_chunks": len(sparse_chunks),
                "vocabulary_size": sparse.vocabulary_size,
                "expected_n": EXPECTED_N},
    )

    # ================================================================
    # Check 3: Chunk-id sets identical.
    # ================================================================
    dense_ids = [c.id for c in dense_chunks]
    sparse_ids = [c.id for c in sparse_chunks]
    dense_set = set(dense_ids)
    sparse_set = set(sparse_ids)
    only_dense = sorted(dense_set - sparse_set)
    only_sparse = sorted(sparse_set - dense_set)
    ok3 = (not only_dense) and (not only_sparse)
    checks.record(
        "3_chunk_id_sets_identical",
        "PASS" if ok3 else "FAIL",
        counts={"dense_ids": len(dense_set), "sparse_ids": len(sparse_set),
                "only_in_dense": len(only_dense),
                "only_in_sparse": len(only_sparse)},
        detail=None if ok3 else {
            "only_in_dense_sample": only_dense[:10],
            "only_in_sparse_sample": only_sparse[:10],
        },
    )

    # ================================================================
    # Check 4: Chunk ids unique within corpus (both lists).
    # ================================================================
    def _dups(ids):
        seen, dups = set(), []
        for i in ids:
            if i in seen:
                dups.append(i)
            seen.add(i)
        return dups

    dense_dups = _dups(dense_ids)
    sparse_dups = _dups(sparse_ids)
    ok4 = (not dense_dups) and (not sparse_dups)
    checks.record(
        "4_chunk_ids_unique",
        "PASS" if ok4 else "FAIL",
        counts={"dense_duplicates": len(dense_dups),
                "sparse_duplicates": len(sparse_dups)},
        detail=None if ok4 else {"dense_dup_sample": dense_dups[:10],
                                 "sparse_dup_sample": sparse_dups[:10]},
    )

    # ================================================================
    # Check 5: Every dense FAISS vector id resolves to a valid chunk.
    # ================================================================
    ntotal = int(store.index.ntotal)
    # IndexIDMap2 exposes stored ids; map each via _id_to_position.
    unresolved = 0
    try:
        id_map = faiss.vector_to_array(store.index.id_map)  # type: ignore[attr-defined]
        stored_ids = [int(x) for x in id_map]
    except Exception:
        stored_ids = list(store._id_to_position.keys())
    for fid in stored_ids:
        pos = store._id_to_position.get(int(fid))
        if pos is None or pos < 0 or pos >= len(dense_chunks):
            unresolved += 1
    ok5 = (ntotal == len(dense_chunks)) and (unresolved == 0) \
        and (len(stored_ids) == ntotal)
    checks.record(
        "5_dense_vector_ids_resolve",
        "PASS" if ok5 else "FAIL",
        counts={"ntotal": ntotal, "mapped_ids": len(stored_ids),
                "unresolved": unresolved, "chunks": len(dense_chunks)},
    )

    # ================================================================
    # Check 6: Every sparse doc id (position) resolves to a valid chunk
    #          with non-empty content and metadata['pmid'].
    # ================================================================
    empty_content = 0
    missing_pmid = 0
    for c in sparse_chunks:
        if not (c.content and c.content.strip()):
            empty_content += 1
        pmid = (c.metadata or {}).get("pmid")
        if normalize_pmid(pmid) is None:
            missing_pmid += 1
    ok6 = (empty_content == 0) and (missing_pmid == 0) \
        and (len(sparse_chunks) == n_sparse)
    checks.record(
        "6_sparse_doc_ids_resolve",
        "PASS" if ok6 else "FAIL",
        counts={"sparse_chunks": len(sparse_chunks),
                "empty_content": empty_content,
                "missing_pmid": missing_pmid},
    )

    # ================================================================
    # Check 7: PMID provenance — metadata['pmid'] == trailing-int(source).
    # ================================================================
    pmid_mismatch = 0
    pmid_mismatch_samples = []
    for c in dense_chunks:
        meta_pmid = normalize_pmid((c.metadata or {}).get("pmid"))
        src_pmid = _source_pmid(c)
        if meta_pmid != src_pmid:
            pmid_mismatch += 1
            if len(pmid_mismatch_samples) < 10:
                pmid_mismatch_samples.append({
                    "chunk_id": c.id, "meta_pmid": meta_pmid,
                    "source": getattr(c, "source", None),
                    "source_pmid": src_pmid,
                })
    ok7 = pmid_mismatch == 0
    checks.record(
        "7_pmid_provenance",
        "PASS" if ok7 else "FAIL",
        counts={"checked": len(dense_chunks), "mismatches": pmid_mismatch},
        detail=None if ok7 else {"samples": pmid_mismatch_samples},
    )

    # ================================================================
    # Check 8: Dimension/metric — inner product + finite + unit-norm sample.
    # ================================================================
    try:
        base = faiss.downcast_index(store.index.index) \
            if hasattr(store.index, "index") else store.index
        metric = int(base.metric_type)
    except Exception:
        metric = int(getattr(store.index, "metric_type", -1))
    is_ip = (metric == int(faiss.METRIC_INNER_PRODUCT))

    sample_ids = stored_ids[:min(50, len(stored_ids))]
    all_finite = True
    norms_ok = True
    norm_samples = []
    for fid in sample_ids:
        try:
            vec = store.index.reconstruct(int(fid))
        except Exception as exc:
            all_finite = False
            norm_samples.append({"id": fid, "error": str(exc)})
            continue
        v = np.asarray(vec, dtype=np.float64)
        if not np.isfinite(v).all():
            all_finite = False
        nrm = float(np.linalg.norm(v))
        if len(norm_samples) < 5:
            norm_samples.append({"id": int(fid), "norm": nrm})
        if abs(nrm - 1.0) > 1e-3:
            norms_ok = False
    ok8 = is_ip and all_finite and norms_ok
    checks.record(
        "8_dimension_metric_finite_norm",
        "PASS" if ok8 else "FAIL",
        counts={"metric_is_inner_product": is_ip,
                "sampled": len(sample_ids),
                "all_finite": all_finite,
                "all_unit_norm": norms_ok},
        detail={"norm_samples": norm_samples},
    )

    # ================================================================
    # Check 9 + 10: smoke queries on both retrievers.
    # ================================================================
    id_to_chunk = {c.id: c for c in dense_chunks}

    smoke_rows = []
    sparse_smoke_ok = True
    dense_smoke_ok = True
    consistency_ok = True
    consistency_failures = []

    for q in SMOKE_QUERIES:
        row = {"query": q}
        # --- sparse ---
        try:
            s_res = sparse.retrieve(q, top_k=5)
            if not s_res:
                sparse_smoke_ok = False
                row["sparse"] = {"error": "no results"}
            else:
                top_chunk, top_score = s_res[0]
                row["sparse"] = {
                    "top1_chunk_id": top_chunk.id,
                    "source_pmid": _source_pmid(top_chunk),
                    "score": round(float(top_score), 6),
                    "n_results": len(s_res),
                }
                for ch, _ in s_res:
                    if ch.id not in id_to_chunk:
                        consistency_ok = False
                        consistency_failures.append(
                            {"query": q, "retriever": "sparse",
                             "chunk_id": ch.id, "reason": "id not in canonical set"})
                    elif ch.content != id_to_chunk[ch.id].content:
                        consistency_ok = False
                        consistency_failures.append(
                            {"query": q, "retriever": "sparse",
                             "chunk_id": ch.id, "reason": "content mismatch"})
        except Exception as exc:
            sparse_smoke_ok = False
            row["sparse"] = {"error": f"{type(exc).__name__}: {exc}"}

        # --- dense ---
        if dense_retriever is not None:
            try:
                d_res = dense_retriever.retrieve(q, top_k=5)
                if not d_res:
                    dense_smoke_ok = False
                    row["dense"] = {"error": "no results"}
                else:
                    top_chunk, top_score = d_res[0]
                    row["dense"] = {
                        "top1_chunk_id": top_chunk.id,
                        "source_pmid": _source_pmid(top_chunk),
                        "score": round(float(top_score), 6),
                        "n_results": len(d_res),
                    }
                    for ch, _ in d_res:
                        if ch.id not in id_to_chunk:
                            consistency_ok = False
                            consistency_failures.append(
                                {"query": q, "retriever": "dense",
                                 "chunk_id": ch.id, "reason": "id not in canonical set"})
                        elif ch.content != id_to_chunk[ch.id].content:
                            consistency_ok = False
                            consistency_failures.append(
                                {"query": q, "retriever": "dense",
                                 "chunk_id": ch.id, "reason": "content mismatch"})
            except Exception as exc:
                dense_smoke_ok = False
                row["dense"] = {"error": f"{type(exc).__name__}: {exc}"}
        else:
            row["dense"] = {"blocked": dense_model_error}
        smoke_rows.append(row)
        print(f"[smoke] {q}\n        sparse={row.get('sparse')}\n        dense={row.get('dense')}")

    # Check 9
    if dense_retriever is None:
        checks.record(
            "9_smoke_queries",
            "BLOCKED" if sparse_smoke_ok else "FAIL",
            counts={"n_queries": len(SMOKE_QUERIES),
                    "sparse_ok": sparse_smoke_ok,
                    "dense_ok": "BLOCKED"},
            hard=True,
            detail={"dense_blocked_reason": dense_model_error},
        )
        # sparse part is hard; if sparse failed, that is a hard fail.
        if not sparse_smoke_ok:
            checks.any_hard_fail = True
    else:
        ok9 = sparse_smoke_ok and dense_smoke_ok
        checks.record(
            "9_smoke_queries",
            "PASS" if ok9 else "FAIL",
            counts={"n_queries": len(SMOKE_QUERIES),
                    "sparse_ok": sparse_smoke_ok,
                    "dense_ok": dense_smoke_ok},
        )

    # Check 10 (consistency) — hard for whichever retrievers actually ran.
    checks.record(
        "10_result_consistency",
        "PASS" if consistency_ok else "FAIL",
        counts={"failures": len(consistency_failures)},
        detail=None if consistency_ok else {"samples": consistency_failures[:10]},
    )

    # ------------------------------------------------------------------
    # Write reports.
    # ------------------------------------------------------------------
    _write_reports(
        checks,
        smoke_rows=smoke_rows,
        dense_model_error=dense_model_error,
        corpus={"n_chunks": len(dense_chunks),
                "faiss_dim": d,
                "vocabulary_size": sparse.vocabulary_size,
                "n_distinct_source_pmids": len({_source_pmid(c) for c in dense_chunks})},
    )

    overall = "FAIL" if checks.any_hard_fail else "PASS"
    print(f"\n==== OVERALL: {overall} ====")
    return 1 if checks.any_hard_fail else 0


def _write_reports(checks: "Checks", smoke_rows=None, dense_model_error=None,
                   corpus=None, blocked_reason=None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dense_index_path": str(V1_DIR.resolve()),
        "sparse_index_path": str(SPARSE_PATH.resolve()),
        "overall": "FAIL" if checks.any_hard_fail else "PASS",
        "dense_model_error": dense_model_error,
        "blocked_reason": blocked_reason,
        "corpus": corpus or {},
        "checks": checks.results,
        "smoke_queries": smoke_rows or [],
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2, default=str),
                        encoding="utf-8")

    lines = []
    lines.append("# Dense/Sparse Index Alignment Validation")
    lines.append("")
    lines.append(f"- Generated (UTC): {payload['generated_utc']}")
    lines.append(f"- Dense index: `{payload['dense_index_path']}`")
    lines.append(f"- Sparse index: `{payload['sparse_index_path']}`")
    lines.append(f"- **Overall: {payload['overall']}**")
    if dense_model_error:
        lines.append(f"- Dense model note: `{dense_model_error}`")
    if blocked_reason:
        lines.append(f"- Blocked: {blocked_reason}")
    if corpus:
        lines.append("")
        lines.append("## Corpus")
        for k, v in corpus.items():
            lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Status | Hard | Counts |")
    lines.append("|---|---|---|---|")
    for c in checks.results:
        lines.append(
            f"| {c['name']} | {c['status']} | {c['hard']} | "
            f"`{json.dumps(c['counts'], default=str)}` |"
        )
    # discrepancies
    disc = [c for c in checks.results if c.get("detail") and c["status"] != "PASS"]
    if disc:
        lines.append("")
        lines.append("## Discrepancies")
        for c in disc:
            lines.append(f"- **{c['name']}**: `{json.dumps(c['detail'], default=str)}`")
    if smoke_rows:
        lines.append("")
        lines.append("## Smoke queries (top-1)")
        lines.append("")
        lines.append("| Query | Dense top1 (pmid, score) | Sparse top1 (pmid, score) |")
        lines.append("|---|---|---|")
        for r in smoke_rows:
            dpart = r.get("dense", {})
            spart = r.get("sparse", {})
            lines.append(
                f"| {r['query']} | {dpart} | {spart} |"
            )
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[write] {JSON_OUT}")
    print(f"[write] {MD_OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
