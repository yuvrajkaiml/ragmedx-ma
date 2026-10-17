
"""Ingest PMC BioC XML files and save retrieval-ready chunks.

Usage:
    python scripts/ingest_pmc.py
    python scripts/ingest_pmc.py --input-dir data/datasets/pmc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.chunking.semantic_chunker import MedicalChunker
from app.rag.loaders.pmc_loader import PMCLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load PMC BioC XML files and create medical text chunks."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "datasets" / "pmc",
        help="Directory containing PMC .xml files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "cache" / "pmc_chunks.jsonl",
        help="Output JSONL file for generated chunks.",
    )
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--overlap", type=int, default=400)
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

    xml_files = list(args.input_dir.glob("*.xml"))
    if not xml_files:
        print(f"No .xml files found in {args.input_dir}")
        return 1

    print(f"Found {len(xml_files)} PMC XML file(s).")
    print("Loading PMC articles...")

    try:
        documents = PMCLoader(args.input_dir).load()
    except Exception as exc:
        print(f"ERROR: PMC loading failed: {exc}")
        return 1

    if not documents:
        print("No PMC documents were loaded.")
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
