
"""Build dense FAISS and sparse BM25 indexes from chunk JSONL files.

Run from the backend directory:
    python scripts/build_index.py
    python scripts/build_index.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.chunking.models import MedicalChunk
from app.rag.embeddings.bge_embedding import BGEEmbedding
from app.rag.retrievers.dense_retriever import DenseRetriever
from app.rag.retrievers.sparse_retriever import SparseRetriever
from app.rag.vectorstore.faiss_store import FAISSVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FAISS and BM25 indexes from chunk JSONL files."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[
            PROJECT_ROOT / "data" / "cache" / "pubmed_chunks.jsonl",
            PROJECT_ROOT / "data" / "cache" / "pmc_chunks.jsonl",
        ],
        help="One or more JSONL chunk files.",
    )
    parser.add_argument(
        "--dense-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "dense",
        help="Directory for the FAISS index.",
    )
    parser.add_argument(
        "--sparse-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "bm25.pkl",
        help="File path for the BM25 index.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Embedding device. Default selects CUDA when available.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of chunks embedded per batch.",
    )
    return parser.parse_args()


def read_chunks(paths: list[Path]) -> list[MedicalChunk]:
    """Read chunk JSONL files and keep the first occurrence of each ID."""
    chunks_by_id: dict[str, MedicalChunk] = {}

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Chunk file not found: {path}")

        print(f"Reading {path}...")
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    chunk = MedicalChunk.model_validate_json(line)
                except Exception as exc:
                    raise ValueError(
                        f"Invalid chunk in {path}, line {line_number}: {exc}"
                    ) from exc

                chunks_by_id.setdefault(chunk.id, chunk)

    return list(chunks_by_id.values())


def main() -> int:
    args = parse_args()

    if args.batch_size < 1:
        print("ERROR: batch-size must be at least 1.")
        return 1

    try:
        chunks = read_chunks(args.inputs)
    except (OSError, ValueError) as exc:
        print(f"ERROR: Could not read chunks: {exc}")
        return 1

    if not chunks:
        print("ERROR: No chunks found; no indexes were built.")
        return 1

    print(f"Unique chunks to index: {len(chunks)}")
    print("Loading BGE-M3. The first run may download model files...")

    try:
        device = None if args.device == "auto" else args.device
        embedder = BGEEmbedding(
            device=device,
            batch_size=args.batch_size,
        )

        if embedder.embedding_dimension != 1024:
            raise ValueError(
                "BGE-M3 must produce 1024-dimensional vectors; got "
                f"{embedder.embedding_dimension}."
            )

        vector_store = FAISSVectorStore(dimension=1024)
        dense = DenseRetriever(embedder, vector_store)
        sparse = SparseRetriever()

        print("Building dense FAISS index...")
        dense.add_chunks(chunks)

        print("Building sparse BM25 index...")
        sparse.add_chunks(chunks)

        # Save to the configured paths.
        dense.save(args.dense_dir)
        sparse.save(args.sparse_file)

    except Exception as exc:
        print(f"ERROR: Index build failed: {exc}")
        return 1

    print("Index build completed.")
    print(f"Dense chunks: {dense.size}")
    print(f"Sparse chunks: {sparse.size}")
    print(f"FAISS directory: {args.dense_dir}")
    print(f"BM25 file: {args.sparse_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
