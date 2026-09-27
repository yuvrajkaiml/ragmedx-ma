"""
Production FAISS index build script for RAGMedX-MA.

Pipeline:
  DocumentLoader.load_all()        — discovers bioasq/ (and medqa/) under
                                     backend/data/datasets/
      ↓ deduplicate by exact text
  MedicalChunker.from_settings()   — splits long passages; short BioASQ
      ↓ .chunk_documents()           snippets pass through as single chunks
  BGEEmbedding(bge-m3)             — 1024-dim L2-normalised dense vectors
      ↓ .embed_documents()
  FAISSVectorStore(dim=1024)       — IndexFlatIP (≡ cosine for unit vecs)
      ↓ .add() then .save()
  settings.faiss_index_dir/        — index.faiss + chunks.pkl

Usage (from backend/):
    python -m scripts.build_index                   # default settings
    RAGMEDX_FAISS_INDEX_DIR=data/faiss_index \\
        python -m scripts.build_index               # explicit output dir

Env vars read via Settings:
    RAGMEDX_FAISS_INDEX_DIR   — output directory (default: data/faiss_index)
    RAGMEDX_LLM_MODE          — informational only; index build always runs
    RETRIEVAL_CHUNK_SIZE      — MedicalChunker.chunk_size   (default 1000)
    RETRIEVAL_CHUNK_OVERLAP   — MedicalChunker.chunk_overlap (default 150)
    RETRIEVAL_CHUNK_MIN_CHARS — MedicalChunker.min_chunk_size (default 50)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure the backend/ app package is on the path whether the script is run
# via `python scripts/build_index.py` or `python -m scripts.build_index`.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.config import DATASETS_DIR, get_settings
from app.rag.chunking.semantic_chunker import MedicalChunker
from app.rag.embeddings.bge_embedding import BGEEmbedding
from app.rag.vectorstore.faiss_store import FAISSVectorStore
from app.retrieval.document_loader import DocumentLoader
from app.utils.logger import logger

# BGE-M3 produces 1024-dim embeddings.  FAISSVectorStore must be initialised
# with the same dimension.  Do not use Settings.embedding_model_name here —
# that field defaults to bge-small-en-v1.5 (384-dim) and is inconsistent with
# the store's dimension; bge-m3 is the production embedding model for this index.
_EMBED_MODEL = BGEEmbedding.DEFAULT_MODEL   # "BAAI/bge-m3"
_EMBED_DIM   = 1024
_EMBED_BATCH = 64   # Reduced from 128; BGE-M3 hidden states are large.


def _deduplicate(documents):
    """Return a new list with exact-text duplicates removed (first-seen wins)."""
    seen: set[str] = set()
    unique = []
    skipped = 0
    for doc in documents:
        if doc.content in seen:
            skipped += 1
        else:
            seen.add(doc.content)
            unique.append(doc)
    return unique, skipped


def main() -> None:
    t_total = time.perf_counter()
    settings = get_settings()

    output_dir: Path = settings.faiss_index_dir
    # Resolve relative paths against the backend/ directory so the script
    # works regardless of the current working directory.
    if not output_dir.is_absolute():
        output_dir = _BACKEND_DIR / output_dir

    logger.info("=" * 60)
    logger.info("RAGMedX-MA — production index build")
    logger.info(f"  datasets_dir : {DATASETS_DIR}")
    logger.info(f"  output_dir   : {output_dir}")
    logger.info(f"  chunk_size   : {settings.retrieval_chunk_size}")
    logger.info(f"  chunk_overlap: {settings.retrieval_chunk_overlap}")
    logger.info(f"  min_chars    : {settings.retrieval_chunk_min_chars}")
    logger.info(f"  embed_model  : {_EMBED_MODEL}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Load documents
    # ------------------------------------------------------------------ #
    logger.info("Step 1/4 — loading documents …")
    t0 = time.perf_counter()
    loader = DocumentLoader(DATASETS_DIR)
    raw_docs = loader.load_all()
    logger.info(
        f"  Loaded {len(raw_docs)} raw documents "
        f"({time.perf_counter() - t0:.1f}s)"
    )

    if not raw_docs:
        logger.error(
            "No documents loaded — check that backend/data/datasets/bioasq/ "
            "exists and contains training13b.json (or any *.json file)."
        )
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 2. Deduplicate by exact snippet text
    # ------------------------------------------------------------------ #
    logger.info("Step 2/4 — deduplicating …")
    unique_docs, n_dupes = _deduplicate(raw_docs)
    logger.info(
        f"  {len(unique_docs)} unique documents kept, "
        f"{n_dupes} exact-text duplicates skipped"
    )

    # ------------------------------------------------------------------ #
    # 3. Chunk
    # ------------------------------------------------------------------ #
    logger.info("Step 3/4 — chunking …")
    t0 = time.perf_counter()
    chunker = MedicalChunker.from_settings()
    chunks = chunker.chunk_documents(unique_docs)
    logger.info(
        f"  {len(chunks)} chunks from {len(unique_docs)} documents "
        f"({time.perf_counter() - t0:.1f}s)"
    )

    if not chunks:
        logger.error("Chunker produced zero chunks — nothing to embed.")
        sys.exit(1)

    # Sanity check: near-1:1 ratio is expected for BioASQ (most snippets
    # are shorter than chunk_size so each becomes a single chunk).
    ratio = len(chunks) / len(unique_docs)
    if ratio < 0.5:
        logger.warning(
            f"Chunk/doc ratio is unusually low ({ratio:.2f}). "
            "Many documents may have been discarded by min_chunk_size. "
            "Consider lowering RETRIEVAL_CHUNK_MIN_CHARS."
        )
    elif ratio > 5.0:
        logger.warning(
            f"Chunk/doc ratio is unusually high ({ratio:.2f}). "
            "Documents may be much longer than expected."
        )

    # ------------------------------------------------------------------ #
    # 4. Embed + store
    # ------------------------------------------------------------------ #
    logger.info(f"Step 4/4 — embedding {len(chunks)} chunks with {_EMBED_MODEL} …")
    t0 = time.perf_counter()
    embedder = BGEEmbedding(model_name=_EMBED_MODEL)

    # Embed in reporting blocks so VRAM and throughput are visible.
    _LOG_EVERY = 2000   # log a progress line every this many chunks
    texts = [c.content for c in chunks]
    all_embeddings = []
    n = len(texts)

    import torch as _torch
    def _vram_str() -> str:
        if not _torch.cuda.is_available():
            return "cpu"
        alloc = _torch.cuda.memory_allocated() / 1_048_576
        reserv = _torch.cuda.memory_reserved() / 1_048_576
        total = _torch.cuda.get_device_properties(0).total_memory / 1_048_576
        return f"VRAM {alloc:.0f}/{total:.0f} MiB alloc, {reserv:.0f} MiB reserved"

    for start in range(0, n, _LOG_EVERY):
        end = min(start + _LOG_EVERY, n)
        block = texts[start:end]
        block_emb = embedder.embed_documents(block, batch_size=_EMBED_BATCH)
        all_embeddings.append(block_emb)
        elapsed = time.perf_counter() - t0
        rate = end / elapsed if elapsed > 0 else 0
        eta  = (n - end) / rate if rate > 0 else 0
        logger.info(
            f"  embedded {end}/{n} chunks "
            f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining) | {_vram_str()}"
        )

    import numpy as np
    embeddings = np.vstack(all_embeddings)
    logger.info(
        f"  Embedding complete — shape {embeddings.shape} "
        f"({time.perf_counter() - t0:.1f}s total)"
    )

    store = FAISSVectorStore(dimension=_EMBED_DIM)
    store.add(embeddings, chunks)

    output_dir.mkdir(parents=True, exist_ok=True)
    store.save(output_dir)

    logger.success(
        f"Index saved to {output_dir}  "
        f"({store.size} vectors, "
        f"{(output_dir / 'index.faiss').stat().st_size / 1_048_576:.1f} MB index, "
        f"{(output_dir / 'chunks.pkl').stat().st_size / 1_048_576:.1f} MB metadata)"
    )
    logger.info(f"Total build time: {time.perf_counter() - t_total:.1f}s")


if __name__ == "__main__":
    main()
