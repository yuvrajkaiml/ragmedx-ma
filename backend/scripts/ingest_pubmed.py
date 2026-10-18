
"""Ingest PubMed XML.GZ files and save retrieval-ready chunks.

Usage:
    python scripts/ingest_pubmed.py
    python scripts/ingest_pubmed.py --input-dir data/datasets/pubmed
    python scripts/ingest_pubmed.py --input-dir data/datasets/pubmed --output data/cache/pubmed_chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running this file directly from the backend directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.chunking.semantic_chunker import MedicalChunker
from app.rag.loaders.pubmed_loader import PubMedLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load PubMed XML.GZ files and create medical text chunks."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "datasets" / "pubmed",
        help="Directory containing PubMed .xml.gz files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "cache" / "pubmed_chunks.jsonl",
        help="Output JSONL file for generated chunks.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Maximum target chunk size.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=400,
        help="Overlap between consecutive chunks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input_dir.is_dir():
        print(f"ERROR: Input directory does not exist: {args.input_dir}")
        return 1

    if args.chunk_size <= 0 or args.overlap < 0:
        print("ERROR: chunk-size must be positive and overlap cannot be negative.")
        return 1

    if args.overlap >= args.chunk_size:
        print("ERROR: overlap must be smaller than chunk-size.")
        return 1

    input_files = list(args.input_dir.glob("*.xml.gz"))
    if not input_files:
        print(f"No .xml.gz files found in {args.input_dir}")
        return 1

    print(f"Found {len(input_files)} PubMed XML.GZ file(s).")
    print("Loading PubMed articles...")

    try:
        documents = PubMedLoader(args.input_dir).load()
    except Exception as exc:
        print(f"ERROR: PubMed loading failed: {exc}")
        return 1

    if not documents:
        print("No PubMed documents were loaded.")
        return 1

    print(f"Loaded {len(documents)} document(s).")
    chunker = MedicalChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
    )
    chunks = chunker.chunk_documents(documents)

    if not chunks:
        print("No chunks were generated.")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary file first to avoid leaving a partial output.
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        with temporary_output.open("w", encoding="utf-8") as file:
            for chunk in chunks:
                file.write(
                    json.dumps(
                        chunk.model_dump(mode="json"),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        temporary_output.replace(args.output)
    except Exception as exc:
        temporary_output.unlink(missing_ok=True)
        print(f"ERROR: Could not write chunks: {exc}")
        return 1

    print(f"Generated {len(chunks)} chunk(s).")
    print(f"Saved chunks to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
