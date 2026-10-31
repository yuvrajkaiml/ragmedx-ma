
import pytest

from app.rag.chunking.models import MedicalChunk
from scripts.build_index import read_chunks
from scripts.update_index import read_new_chunks

def make_chunk(chunk_id: str) -> MedicalChunk:
    return MedicalChunk(
        id=chunk_id,
        document_id=f"doc-{chunk_id}",
        content=f"Medical evidence for {chunk_id}",
        source="PMC",
    )


def test_read_chunks_loads_valid_jsonl(tmp_path):
    file_path = tmp_path / "chunks.jsonl"
    chunks = [make_chunk("chunk-001"), make_chunk("chunk-002")]

    file_path.write_text(
        "\n".join(chunk.model_dump_json() for chunk in chunks) + "\n",
        encoding="utf-8",
    )

    result = read_chunks([file_path])

    assert len(result) == 2
    assert {chunk.id for chunk in result} == {"chunk-001", "chunk-002"}


def test_read_chunks_removes_duplicate_ids(tmp_path):
    file_path = tmp_path / "duplicates.jsonl"
    chunks = [
        make_chunk("chunk-001"),
        make_chunk("chunk-001"),
        make_chunk("chunk-002"),
    ]

    file_path.write_text(
        "\n".join(chunk.model_dump_json() for chunk in chunks) + "\n",
        encoding="utf-8",
    )

    result = read_chunks([file_path])

    assert len(result) == 2
    assert len({chunk.id for chunk in result}) == 2


def test_read_chunks_skips_blank_lines(tmp_path):
    file_path = tmp_path / "blank_lines.jsonl"
    chunk = make_chunk("chunk-001")

    file_path.write_text(
        "\n" + chunk.model_dump_json() + "\n\n",
        encoding="utf-8",
    )

    result = read_chunks([file_path])

    assert len(result) == 1


def test_read_chunks_reports_missing_file(tmp_path):
    missing_file = tmp_path / "missing.jsonl"

    with pytest.raises(FileNotFoundError, match="Chunk file not found"):
        read_chunks([missing_file])


def test_read_chunks_reports_invalid_json(tmp_path):
    file_path = tmp_path / "invalid.jsonl"
    file_path.write_text("{invalid json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid chunk"):
        read_chunks([file_path])


def test_read_chunks_deduplicates_across_files(tmp_path):
    first_file = tmp_path / "first.jsonl"
    second_file = tmp_path / "second.jsonl"

    first_file.write_text(
        make_chunk("chunk-001").model_dump_json() + "\n",
        encoding="utf-8",
    )
    second_file.write_text(
        "\n".join(
            [
                make_chunk("chunk-001").model_dump_json(),
                make_chunk("chunk-002").model_dump_json(),
            ]
        ) + "\n",
        encoding="utf-8",
    )

    result = read_chunks([first_file, second_file])

    assert len(result) == 2
    assert {chunk.id for chunk in result} == {"chunk-001", "chunk-002"}

def test_read_new_chunks_deduplicates_across_files(tmp_path):
    first_file = tmp_path / "first.jsonl"
    second_file = tmp_path / "second.jsonl"

    first_file.write_text(
        make_chunk("chunk-001").model_dump_json() + "\n",
        encoding="utf-8",
    )
    second_file.write_text(
        "\n".join(
            [
                make_chunk("chunk-001").model_dump_json(),
                make_chunk("chunk-002").model_dump_json(),
            ]
        ) + "\n",
        encoding="utf-8",
    )

    result = read_new_chunks([first_file, second_file])

    assert len(result) == 2
    assert {chunk.id for chunk in result} == {"chunk-001", "chunk-002"}


def test_read_new_chunks_skips_blank_lines(tmp_path):
    file_path = tmp_path / "chunks.jsonl"
    file_path.write_text(
        "\n" + make_chunk("chunk-001").model_dump_json() + "\n\n",
        encoding="utf-8",
    )

    result = read_new_chunks([file_path])

    assert len(result) == 1


def test_read_new_chunks_reports_missing_file(tmp_path):
    missing_file = tmp_path / "missing.jsonl"

    with pytest.raises(FileNotFoundError, match="Chunk file not found"):
        read_new_chunks([missing_file])


def test_read_new_chunks_reports_invalid_json(tmp_path):
    file_path = tmp_path / "invalid.jsonl"
    file_path.write_text("{invalid json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid chunk"):
        read_new_chunks([file_path])

def test_update_index_rejects_inconsistent_indexes():
    dense_ids = {"chunk-001", "chunk-002"}
    sparse_ids = {"chunk-001", "chunk-003"}

    if dense_ids != sparse_ids:
        missing_from_sparse = dense_ids - sparse_ids
        missing_from_dense = sparse_ids - dense_ids

        message = (
            "FAISS and BM25 indexes are inconsistent. "
            f"Missing from BM25: {len(missing_from_sparse)} chunks; "
            f"missing from FAISS: {len(missing_from_dense)} chunks. "
            "Rebuild both indexes before updating."
        )
    else:
        message = ""

    assert "Missing from BM25: 1 chunks" in message
    assert "missing from FAISS: 1 chunks" in message
