"""
Phase B — Build the isolated, versioned BioASQ retrieval-eval index.

Builds ``backend/data/indexes/bioasq_retrieval_eval/v1/{index.faiss, chunks.pkl,
manifest.json}`` from the ``eval_questions.json`` SNIPPETS ONLY (local). Preserves
source PMIDs and provenance through chunking, embedding, and indexing. Dedupes raw
snippets by ``(pmid, normalized_text)``. Reuses the project's BGE-M3 embedding model.

NEVER touches the production index or eval_split. ``--force-rebuild`` writes ``v2/``
rather than overwriting ``v1/``.

Run from ``backend/``::

    d:\\Ragmed_2\\.venv\\Scripts\\python.exe -m evaluation.bioasq_eval.build_eval_index
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluation.bioasq_eval.common import (  # noqa: E402
    SEED, atomic_write_json, config_hash, file_sha256, normalize_pmid,
    utc_now_iso, write_manifest,
)
from evaluation.bioasq_eval.integrity import check_pre_run  # noqa: E402

EVAL_QUESTIONS = BACKEND_DIR / "data" / "datasets" / "bioasq_eval_split" / "eval_questions.json"
INDEX_ROOT = BACKEND_DIR / "data" / "indexes" / "bioasq_retrieval_eval"
RESULTS_DIR = BACKEND_DIR / "results" / "bioasq_eval_20261018"
TRAINING13B = BACKEND_DIR / "data" / "datasets" / "bioasq" / "training13b.json"

BATCH_SIZE = 16  # overrides the embedder default of 8; recorded in manifest.
EMBED_DIM = 1024

_WS_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    return _WS_RE.sub(" ", str(text or "").strip()).lower()


def _build_chunks(questions: List[Dict[str, Any]]) -> Tuple[List[Any], Dict[str, Any]]:
    """Build deduped MedicalChunks from eval question snippets, with provenance."""
    from app.rag.chunking.models import MedicalChunk

    seen: set = set()  # (pmid, normalized_text)
    chunks: List[MedicalChunk] = []
    raw_snippets = 0
    skipped_no_pmid = 0
    skipped_dup = 0
    distinct_pmids: set = set()

    for q in questions:
        qid = q.get("id")
        for si, snip in enumerate(q.get("snippets") or []):
            raw_snippets += 1
            doc_url = snip.get("document")
            pmid = normalize_pmid(doc_url)
            if pmid is None:
                skipped_no_pmid += 1
                continue
            text = snip.get("text") or ""
            norm = _normalize_text(text)
            key = (pmid, norm)
            if key in seen:
                skipped_dup += 1
                continue
            seen.add(key)
            begin_section = snip.get("beginSection")
            end_section = snip.get("endSection")
            # Section-prefix ON using beginSection only (endSection ignored) — matches
            # the eval_split/phase5 convention; recorded verbatim in the manifest.
            prefix = f"[{begin_section}] " if begin_section else ""
            content = prefix + str(text)
            pmid_url = f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}"
            chunk = MedicalChunk(
                id=f"bioasq_{qid}_s{si}_chunk_0",
                document_id=f"bioasq_{qid}_s{si}",
                content=content,
                title=None,
                section=begin_section,
                source=pmid_url,
                metadata={
                    "pmid": pmid,
                    "pmid_url": pmid_url,
                    "begin_section": begin_section,
                    "end_section": end_section,
                    "offset_begin": snip.get("offsetInBeginSection"),
                    "offset_end": snip.get("offsetInEndSection"),
                    "question_id": qid,
                },
            )
            chunks.append(chunk)
            distinct_pmids.add(pmid)

    stats = {
        "raw_snippets": raw_snippets,
        "n_chunks": len(chunks),
        "n_distinct_pmids": len(distinct_pmids),
        "skipped_no_pmid": skipped_no_pmid,
        "skipped_duplicate": skipped_dup,
    }
    return chunks, stats


def _verify_index(store: Any, chunks: List[Any], embedder: Any,
                  questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Round-trip + ID-mapping verification. Fatal on failure (raises ValueError)."""
    report: Dict[str, Any] = {}
    report["ntotal"] = int(store.index.ntotal)
    report["len_chunks"] = len(store.chunks)
    report["ntotal_equals_len_chunks"] = store.index.ntotal == len(store.chunks)
    report["d"] = int(store.index.d)
    report["d_equals_1024"] = store.index.d == EMBED_DIM
    if not report["ntotal_equals_len_chunks"]:
        raise ValueError(f"index.ntotal={store.index.ntotal} != len(chunks)={len(store.chunks)}")
    if not report["d_equals_1024"]:
        raise ValueError(f"index.d={store.index.d} != {EMBED_DIM}")

    # Sample 10 chunks: metadata['pmid'] must match source-parsed PMID.
    import random
    rng = random.Random(SEED)
    sample = rng.sample(range(len(store.chunks)), min(10, len(store.chunks)))
    meta_source_ok = 0
    for i in sample:
        ch = store.chunks[i]
        meta_pmid = normalize_pmid((ch.metadata or {}).get("pmid"))
        src_pmid = normalize_pmid(ch.source)
        if meta_pmid is not None and meta_pmid == src_pmid:
            meta_source_ok += 1
    report["metadata_source_pmid_match"] = f"{meta_source_ok}/{len(sample)}"
    report["metadata_source_pmid_all_ok"] = meta_source_ok == len(sample)
    if not report["metadata_source_pmid_all_ok"]:
        raise ValueError("metadata['pmid'] does not match source-parsed PMID for sampled chunks")

    # 10-query gold round-trip: a query's own gold snippet PMID should appear in top-5.
    from app.rag.retrievers.dense_retriever import DenseRetriever
    retriever = DenseRetriever(embedder=embedder, vector_store=store)
    corpus_pmids = {normalize_pmid((c.metadata or {}).get("pmid")) for c in store.chunks}
    # Pick 10 questions whose gold is in the corpus.
    candidates = []
    for q in questions:
        golds = {normalize_pmid(d) for d in (q.get("documents") or [])}
        golds = {g for g in golds if g is not None}
        if golds & corpus_pmids and q.get("body"):
            candidates.append((q, golds & corpus_pmids))
        if len(candidates) >= 10:
            break
    roundtrip_hits = 0
    for q, avail in candidates:
        results = retriever.retrieve(q["body"], top_k=5)
        got = {normalize_pmid((c.metadata or {}).get("pmid")) for c, _ in results}
        if got & avail:
            roundtrip_hits += 1
    report["gold_roundtrip"] = f"{roundtrip_hits}/{len(candidates)}"
    report["gold_roundtrip_queries"] = len(candidates)
    report["gold_roundtrip_hits"] = roundtrip_hits
    return report


def build(force_rebuild: bool = False) -> Dict[str, Any]:
    # Integrity pre-check must pass before building.
    pre = check_pre_run()
    if not pre["ok"]:
        raise RuntimeError("Integrity pre-check failed; refusing to build. Run is INVALID.")

    with open(EVAL_QUESTIONS, "r", encoding="utf-8") as fh:
        questions = json.load(fh)

    version = "v2" if force_rebuild else "v1"
    out_dir = INDEX_ROOT / version

    cfg = {
        "corpus": "eval_questions.json snippets only",
        "embedding_model": "BAAI/bge-m3",
        "normalize": True,
        "max_length": 512,
        "batch_size": BATCH_SIZE,
        "dimension": EMBED_DIM,
        "section_prefix_rule": "[{beginSection}] prepended; endSection ignored",
        "dense_only": True,
        "dedup_key": "(pmid, normalized_text)",
        "seed": SEED,
    }
    chash = config_hash(cfg)

    # Idempotence: skip if v1 manifest has a matching config_hash.
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not force_rebuild:
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("config_hash") == chash and (out_dir / "index.faiss").exists():
                print(f"[build] v1 up-to-date (config_hash match); skipping rebuild.")
                return {"skipped": True, "config_hash": chash, "index_dir": str(out_dir)}
        except (json.JSONDecodeError, OSError):
            pass

    chunks, stats = _build_chunks(questions)
    if not chunks:
        raise ValueError("No chunks built from snippets; cannot build index.")

    from app.rag.embeddings.bge_embedding import BGEEmbedding
    from app.rag.vectorstore.faiss_store import FAISSVectorStore

    embedder = BGEEmbedding()  # single load; BAAI/bge-m3, normalize=True
    texts = [c.content for c in chunks]
    embeddings = embedder.embed(texts, batch_size=BATCH_SIZE)

    store = FAISSVectorStore(dimension=EMBED_DIM)
    store.add(embeddings=embeddings, chunks=chunks)
    store.save(out_dir)  # writes index.faiss + chunks.pkl atomically within dir

    verify = _verify_index(store, chunks, embedder, questions)

    manifest = {
        "version": version,
        "config": cfg,
        "config_hash": chash,
        "seed": SEED,
        "source_file": str(EVAL_QUESTIONS),
        "source_file_sha256": file_sha256(EVAL_QUESTIONS),
        "n_chunks": stats["n_chunks"],
        "n_distinct_pmids": stats["n_distinct_pmids"],
        "build_stats": stats,
        "forced_bge_m3_note": (
            "Embedding model is BAAI/bge-m3 (1024-dim), instantiated directly, NOT the "
            "stale bge-small-en-v1.5 id in app/config.py — required for comparability with "
            "the 1024-dim production/eval_split indexes."
        ),
        "batch_size_note": f"batch_size={BATCH_SIZE} overrides the embedder default of 8.",
        "section_prefix_note": "[{beginSection}] prepended to content; endSection ignored.",
        "build_timestamp_utc": utc_now_iso(),
    }
    atomic_write_json(manifest_path, manifest)

    build_report = {
        "index_dir": str(out_dir),
        "version": version,
        "config_hash": chash,
        **stats,
        **verify,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(RESULTS_DIR / "build_report.json", build_report)

    write_manifest("build_index", cfg=cfg,
                   inputs={"eval_questions": EVAL_QUESTIONS},
                   extra={"index_dir": str(out_dir), "build_report": build_report})

    return build_report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase B build retrieval-eval index")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Build v2/ instead of v1/ (never overwrites v1).")
    parser.add_argument("--add-training-distractors", type=int, default=0,
                        help="(default OFF) add N distractor questions from training13b.json")
    parser.add_argument("--fetch-missing", action="store_true",
                        help="(default OFF) fetch missing gold PMIDs via NCBI E-utilities")
    parser.add_argument("--reranker", action="store_true",
                        help="(default OFF) documented extension; not used in dense-only build")
    args = parser.parse_args(argv)

    if args.add_training_distractors and not TRAINING13B.exists():
        print(f"[build] training13b.json not found at {TRAINING13B}; distractors unavailable.",
              file=sys.stderr)
    if args.fetch_missing:
        print("[build] --fetch-missing requested but default build is snippet-only; "
              "fetch path is a documented extension and is not auto-enabled here.",
              file=sys.stderr)

    report = build(force_rebuild=args.force_rebuild)
    if report.get("skipped"):
        print(f"skipped=true config_hash={report['config_hash']}")
        return 0
    print(f"ntotal={report['ntotal']} len_chunks={report['len_chunks']} "
          f"d={report['d']} distinct_pmids={report['n_distinct_pmids']} "
          f"gold_roundtrip={report['gold_roundtrip']}")

    # Post-build: confirm protected files unchanged.
    post = check_pre_run()
    if not post["ok"]:
        raise RuntimeError("Protected file changed during build. Run is INVALID.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
