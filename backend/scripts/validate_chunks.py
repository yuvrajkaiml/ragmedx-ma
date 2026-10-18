
from pathlib import Path

from app.rag.chunking.models import MedicalChunk

files = [
    Path("data/cache/pubmed_chunks.jsonl"),
    Path("data/cache/pmc_chunks.jsonl"),
]

total = 0

for file_path in files:
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        raise SystemExit(1)

    count = 0

    with file_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                MedicalChunk.model_validate_json(line)
            except Exception as exc:
                print(
                    f"INVALID: {file_path}, line {line_number}: {exc}"
                )
                raise SystemExit(1)

            count += 1

    print(f"VALID: {file_path} — {count} chunks")
    total += count

print(f"\nSUCCESS: All {total} chunks are valid.")
