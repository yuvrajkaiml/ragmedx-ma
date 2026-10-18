"""Build resumable dense FAISS and sparse BM25 indexes from chunk JSONL files.

This script reads one or more JSONL chunk files, deduplicates by chunk ID,
embeds with BGE-M3, and builds dense (FAISS) and sparse (BM25) indexes with
resumable checkpointing. It is the counterpart of ``scripts/update_index.py``.

Note: the earlier datasets-directory build flow (``DocumentLoader.load_all``
+ ``MedicalChunker.from_settings`` + ``--corpus-file``) still lives in its own
modules — ``app.retrieval.document_loader`` and
``app.rag.chunking.semantic_chunker`` — and is exercised by their own tests.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
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
        description="Build resumable FAISS and BM25 indexes from chunk JSONL files."
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
        help="Final FAISS index directory.",
    )
    parser.add_argument(
        "--sparse-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "bm25.pkl",
        help="Final BM25 index file.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "build_checkpoint",
        help="Directory used to resume an interrupted build.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=512,
        help="Number of chunks processed between checkpoints.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Embedding device.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of chunks embedded per batch.",
    )
    return parser.parse_args()


def read_chunks(paths: list[Path]) -> list[MedicalChunk]:
    """Read JSONL chunks, keeping the first occurrence of each ID."""
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


def save_json_atomic(path: Path, data: dict) -> None:
    """Write JSON via a temporary file, then atomically replace the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(
    checkpoint_dir: Path,
    dense: DenseRetriever,
    sparse: SparseRetriever,
    processed_ids: set[str],
) -> None:
    """Publish a checkpoint only after both indexes and metadata are saved."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    generation = f"generation_{uuid.uuid4().hex}"
    generation_dir = checkpoint_dir / generation
    generation_dir.mkdir()

    try:
        dense.save(generation_dir / "dense")
        sparse.save(generation_dir / "bm25.pkl")

        save_json_atomic(
            generation_dir / "metadata.json",
            {
                "processed_ids": sorted(processed_ids),
                "dense_size": dense.size,
                "sparse_size": sparse.size,
            },
        )

        # This manifest is updated last. It identifies a complete generation.
        save_json_atomic(
            checkpoint_dir / "current.json",
            {"generation": generation},
        )
    except Exception:
        shutil.rmtree(generation_dir, ignore_errors=True)
        raise

    # Keep the current generation and remove older complete generations.
    for old_dir in checkpoint_dir.glob("generation_*"):
        if old_dir != generation_dir and old_dir.is_dir():
            shutil.rmtree(old_dir, ignore_errors=True)


def load_checkpoint(
    checkpoint_dir: Path,
    embedder: BGEEmbedding,
) -> tuple[DenseRetriever, SparseRetriever, set[str]] | None:
    manifest_path = checkpoint_dir / "current.json"
    if not manifest_path.is_file():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generation = manifest.get("generation")

    # Do not allow a manifest to refer outside the checkpoint directory.
    if not isinstance(generation, str) or Path(generation).name != generation:
        raise ValueError("Invalid checkpoint generation manifest.")

    generation_dir = checkpoint_dir / generation
    metadata_path = generation_dir / "metadata.json"
    dense_dir = generation_dir / "dense"
    sparse_file = generation_dir / "bm25.pkl"

    if not metadata_path.is_file() or not dense_dir.is_dir() or not sparse_file.is_file():
        raise ValueError("Checkpoint is incomplete; do not delete it manually.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    processed_ids = set(metadata["processed_ids"])

    dense = DenseRetriever.load(dense_dir, embedder)
    sparse = SparseRetriever.load(sparse_file)

    if dense.size != sparse.size or dense.size != len(processed_ids):
        raise ValueError(
            "Checkpoint consistency check failed: dense, sparse, and processed "
            "chunk counts do not match."
        )

    if dense.size != metadata["dense_size"] or sparse.size != metadata["sparse_size"]:
        raise ValueError("Checkpoint sizes do not match saved metadata.")

    return dense, sparse, processed_ids


def main() -> int:
    args = parse_args()

    if args.batch_size < 1 or args.checkpoint_every < 1:
        print("ERROR: batch-size and checkpoint-every must be at least 1.")
        return 1

    try:
        chunks = read_chunks(args.inputs)
    except (OSError, ValueError) as exc:
        print(f"ERROR: Could not read chunks: {exc}")
        return 1

    if not chunks:
        print("ERROR: No chunks found; no indexes were built.")
        return 1

    print(f"Unique chunks available: {len(chunks)}")
    print("Loading BGE-M3. The first run may download model files...")

    try:
        device = None if args.device == "auto" else args.device
        embedder = BGEEmbedding(device=device, batch_size=args.batch_size)

        if embedder.embedding_dimension != 1024:
            raise ValueError(
                f"BGE-M3 must produce 1024-dimensional vectors; "
                f"got {embedder.embedding_dimension}."
            )

        checkpoint = load_checkpoint(args.checkpoint_dir, embedder)

        if checkpoint is None:
            print("No checkpoint found; starting a new build.")
            dense = DenseRetriever(
                embedder, FAISSVectorStore(dimension=1024)
            )
            sparse = SparseRetriever()
            processed_ids: set[str] = set()
        else:
            dense, sparse, processed_ids = checkpoint
            print(f"Resuming from checkpoint: {len(processed_ids)} chunks saved.")

        pending = [chunk for chunk in chunks if chunk.id not in processed_ids]
        print(f"Chunks remaining: {len(pending)}")

        for start in range(0, len(pending), args.checkpoint_every):
            batch = pending[start : start + args.checkpoint_every]

            print(
                f"Indexing chunks {len(processed_ids) + 1}–"
                f"{len(processed_ids) + len(batch)} of {len(chunks)}..."
            )

            dense.add_chunks(batch, batch_size=args.batch_size)
            sparse.add_chunks(batch)
            processed_ids.update(chunk.id for chunk in batch)

            if dense.size != sparse.size or dense.size != len(processed_ids):
                raise ValueError("Dense/sparse index counts diverged.")

            save_checkpoint(
                args.checkpoint_dir, dense, sparse, processed_ids
            )
            print(f"Checkpoint saved: {len(processed_ids)} chunks.")

        if len(processed_ids) != len(chunks):
            raise ValueError(
                "Not all chunks were indexed. Check input files and checkpoint."
            )

        # Publish only after the complete, consistent build is checkpointed.
        dense.save(args.dense_dir)
        sparse.save(args.sparse_file)

    except Exception as exc:
        print(f"ERROR: Index build failed: {exc}")
        print("If a checkpoint was saved, rerun the same command to resume.")
        return 1

    print("Index build completed successfully.")
    print(f"Dense chunks: {dense.size}")
    print(f"Sparse chunks: {sparse.size}")
    print(f"FAISS directory: {args.dense_dir}")
    print(f"BM25 file: {args.sparse_file}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
