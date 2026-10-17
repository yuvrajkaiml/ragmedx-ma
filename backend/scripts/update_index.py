
"""Incrementally add new JSONL chunks to existing FAISS and BM25 indexes."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add new chunks to existing FAISS and BM25 indexes."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="JSONL files containing chunks to add.",
    )
    parser.add_argument(
        "--dense-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "dense",
        help="Directory containing the existing FAISS index.",
    )
    parser.add_argument(
        "--sparse-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "bm25.pkl",
        help="Path to the existing BM25 index.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def read_new_chunks(paths: list[Path]) -> list[MedicalChunk]:
    """Read JSONL and deduplicate chunks across the supplied input files."""
    chunks_by_id: dict[str, MedicalChunk] = {}

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Chunk file not found: {path}")

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

    if not args.dense_dir.is_dir():
        print(f"ERROR: Existing FAISS directory not found: {args.dense_dir}")
        print("Build the initial indexes first using scripts/build_index.py.")
        return 1

    if not args.sparse_file.is_file():
        print(f"ERROR: Existing BM25 index not found: {args.sparse_file}")
        print("Build the initial indexes first using scripts/build_index.py.")
        return 1

    try:
        incoming_chunks = read_new_chunks(args.inputs)
        if not incoming_chunks:
            print("No input chunks found; indexes were not changed.")
            return 0

        # Load the saved FAISS index first, without loading the embedding model.
        from app.rag.vectorstore.faiss_store import FAISSVectorStore

        vector_store = FAISSVectorStore.load(args.dense_dir)
        sparse = SparseRetriever.load(args.sparse_file)

        dense_ids = {
            str(chunk.id) for chunk in vector_store.chunks
        }
        sparse_ids = {
            str(chunk.id) for chunk in sparse.chunks
        }

        # Both indexes must contain the same chunk IDs before updating.
        if dense_ids != sparse_ids:
            missing_from_sparse = dense_ids - sparse_ids
            missing_from_dense = sparse_ids - dense_ids

            raise ValueError(
                "FAISS and BM25 indexes are inconsistent. "
                f"Missing from BM25: {len(missing_from_sparse)} chunks; "
                f"missing from FAISS: {len(missing_from_dense)} chunks. "
                "Rebuild both indexes before updating."
            )

        existing_ids = dense_ids
        new_chunks = [
            chunk for chunk in incoming_chunks
            if str(chunk.id) not in existing_ids
        ]

        print(f"Incoming unique chunks: {len(incoming_chunks)}")
        print(f"Already indexed: {len(incoming_chunks) - len(new_chunks)}")
        print(f"New chunks to add: {len(new_chunks)}")

        if not new_chunks:
            print("No new chunks to add; indexes were not changed.")
            return 0

        device = None if args.device == "auto" else args.device
        embedder = BGEEmbedding(
            device=device,
            batch_size=args.batch_size,
        )

        dense = DenseRetriever(embedder, vector_store)

        # Update both stores with the same filtered chunk set.
        dense.add_chunks(new_chunks)
        sparse.add_chunks(new_chunks)

        dense.save(args.dense_dir)
        sparse.save(args.sparse_file)

        print("Index update completed.")
        print(f"Dense chunks: {dense.size}")
        print(f"Sparse chunks: {sparse.size}")
        return 0

    except Exception as exc:
        print(f"ERROR: Index update failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
