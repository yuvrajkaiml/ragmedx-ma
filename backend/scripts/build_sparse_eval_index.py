"""
Build the aligned BM25 sparse index for BioASQ retrieval evaluation.

This script builds a sparse (Okapi BM25) retrieval index from the EXACT
canonical chunk set that backs the dense evaluation index at
``data/indexes/bioasq_retrieval_eval/v1``.  Dense and sparse MUST share
identical ``MedicalChunk.id`` values so that HybridRetriever's RRF fusion
(keyed on ``chunk.id``) is well defined.

Design decisions (all derived from the in-repo implementations, not assumed):
- The canonical chunk list is the ``.chunks`` attribute of the FAISS store
  loaded from the v1 dense index directory (``FAISSVectorStore.load``).
  We do NOT re-derive chunks from the raw dataset; we reuse the identical
  objects / ids used by the dense index.
- The sparse index is the in-repo ``SparseRetriever`` (hand-rolled Okapi
  BM25, k1=1.5, b=0.75, lowercasing ``[a-z0-9]+`` tokenizer, smoothed IDF).
- The sparse artifact is a single pickle written beside the dense files as
  ``bm25.pkl``.  The existing dense files are never touched.

Run from the backend directory with the CPU venv::

    cd d:\\Ragmed_2\\backend
    d:\\Ragmed_2\\.venv\\Scripts\\python.exe scripts/build_sparse_eval_index.py

No embedding model is loaded or downloaded; this build is pure-CPU and
network-free.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ``app`` must be importable; this script is intended to run with cwd=backend,
# but we also anchor sys.path on the backend dir defensively.
BACKEND_DIR = Path(__file__).resolve().parents[1]  # d:\Ragmed_2\backend
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.rag.retrievers.sparse_retriever import SparseRetriever  # noqa: E402
from app.rag.vectorstore.faiss_store import FAISSVectorStore  # noqa: E402

# Canonical locations (relative to backend dir).
V1_DIR = BACKEND_DIR / "data" / "indexes" / "bioasq_retrieval_eval" / "v1"
DENSE_MANIFEST = V1_DIR / "manifest.json"
SPARSE_INDEX_PATH = V1_DIR / "bm25.pkl"
SPARSE_MANIFEST_PATH = V1_DIR / "sparse_manifest.json"

# Protected production index; the sparse artifact must never land here.
PRODUCTION_INDEX = (BACKEND_DIR / "data" / "faiss_index").resolve()

EXPECTED_N_CHUNKS = 2947


def _assert_safe_output(path: Path) -> None:
    """Fail loudly unless ``path`` is outside the protected production index."""
    resolved = path.resolve()
    print(f"[path] resolved output path: {resolved}")
    print(f"[path] protected production index: {PRODUCTION_INDEX}")

    if resolved == PRODUCTION_INDEX:
        raise SystemExit(
            f"REFUSING: output path is the production index: {resolved}"
        )
    # Is resolved a descendant of the production index?
    if PRODUCTION_INDEX in resolved.parents:
        raise SystemExit(
            f"REFUSING: output path is under the production index: {resolved}"
        )
    print("[path] OK: output is not the production index nor a descendant.")


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("faiss", "numpy", "torch", "transformers"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = "not-installed"
    return versions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing bm25.pkl if present.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Safety: resolve + assert the output path is not production.
    # ------------------------------------------------------------------
    _assert_safe_output(SPARSE_INDEX_PATH)
    _assert_safe_output(SPARSE_MANIFEST_PATH)

    # ------------------------------------------------------------------
    # Load the canonical chunk set from the DENSE index (no embedder needed).
    # ------------------------------------------------------------------
    print(f"[load] loading canonical chunks from {V1_DIR}")
    store = FAISSVectorStore.load(V1_DIR)
    chunks = store.chunks
    print(
        f"[load] loaded {len(chunks)} chunks "
        f"(index.ntotal={store.index.ntotal}, dim={store.index.d})"
    )

    # ------------------------------------------------------------------
    # Hard assertions on the canonical corpus.
    # ------------------------------------------------------------------
    if len(chunks) != EXPECTED_N_CHUNKS:
        raise SystemExit(
            f"REFUSING: expected {EXPECTED_N_CHUNKS} chunks, "
            f"got {len(chunks)}."
        )

    ids = [c.id for c in chunks]
    unique_ids = set(ids)
    if len(unique_ids) != len(ids):
        # Report the duplicates explicitly before failing.
        seen: set[str] = set()
        dups: list[str] = []
        for cid in ids:
            if cid in seen:
                dups.append(cid)
            seen.add(cid)
        raise SystemExit(
            f"REFUSING: chunk ids are not unique: "
            f"{len(ids) - len(unique_ids)} duplicate(s), "
            f"examples: {dups[:5]}"
        )
    print(f"[check] all {len(ids)} chunk ids are unique.")

    # ------------------------------------------------------------------
    # Refuse to clobber an existing bm25.pkl unless --force.
    # ------------------------------------------------------------------
    if SPARSE_INDEX_PATH.exists() and not args.force:
        print(
            f"[skip] {SPARSE_INDEX_PATH} already exists and --force not given; "
            f"not overwriting. Re-run with --force to rebuild."
        )
        return 0

    action = "overwriting (--force)" if SPARSE_INDEX_PATH.exists() else "writing new"
    print(f"[build] building SparseRetriever (Okapi BM25) over {len(chunks)} chunks...")

    retriever = SparseRetriever()  # k1=1.5, b=0.75 defaults
    retriever.add_chunks(chunks)

    if retriever.size != len(chunks):
        raise SystemExit(
            f"REFUSING: sparse index size {retriever.size} != "
            f"canonical chunk count {len(chunks)}."
        )

    print(f"[build] {action}: {SPARSE_INDEX_PATH}")
    retriever.save(SPARSE_INDEX_PATH)

    # ------------------------------------------------------------------
    # Linking manifest: ties the sparse index to the dense index + corpus.
    # ------------------------------------------------------------------
    dense_manifest = {}
    dense_config_hash = None
    if DENSE_MANIFEST.exists():
        dense_manifest = json.loads(DENSE_MANIFEST.read_text(encoding="utf-8"))
        dense_config_hash = dense_manifest.get("config_hash")

    manifest = {
        "sparse_algorithm": "Okapi BM25 (in-repo SparseRetriever)",
        "k1": retriever.k1,
        "b": retriever.b,
        "tokenizer": (
            "lowercases text then extracts [a-z0-9]+ runs via regex "
            "(punctuation removed; hyphenated terms split, e.g. CTLA-4 -> "
            "['ctla','4']); smoothed BM25 IDF = ln(1 + (N - df + 0.5)/(df + 0.5))"
        ),
        "n_chunks": retriever.size,
        "vocabulary_size": retriever.vocabulary_size,
        "average_document_length": retriever.average_document_length,
        "source_dense_manifest_config_hash": dense_config_hash,
        "dense_index_path": str(V1_DIR.resolve()),
        "sparse_index_path": str(SPARSE_INDEX_PATH.resolve()),
        "bge_model_note": (
            "Dense index uses BAAI/bge-m3 (1024-dim, normalize=True). The "
            "sparse index requires NO embedding model; it indexes chunk text "
            "lexically only."
        ),
        "alignment_note": (
            "The sparse index was built from the identical canonical chunk set "
            "as the dense index (same MedicalChunk objects, same chunk.ids), "
            "loaded via FAISSVectorStore.load(v1).chunks. No re-derivation from "
            "the raw dataset was performed; no eval queries/gold answers/labels "
            "were injected into the index."
        ),
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "package_versions": _package_versions(),
    }
    SPARSE_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[build] wrote linking manifest: {SPARSE_MANIFEST_PATH}")

    print(
        f"SUMMARY sparse_size={retriever.size} "
        f"vocab={retriever.vocabulary_size} "
        f"path={SPARSE_INDEX_PATH.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
