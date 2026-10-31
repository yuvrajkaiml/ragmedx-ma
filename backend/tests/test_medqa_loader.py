"""Tests for MedQALoader.

Uses the ``medqa_sample_path`` and ``medqa_sample_records`` fixtures from
conftest.py so that every test in this module shares the same parsed fixture
data without re-reading the file.

Also includes self-contained ``tmp_path``-based unit tests that build a
synthetic MedQA JSONL file and assert the loader's parsing behavior without
depending on the shared sample fixture.
"""

import json
from pathlib import Path

import pytest

from app.config import DATASETS_DIR
from app.rag.loaders.medqa_loader import MedQALoader
from app.rag.loaders.models import MedicalDocument


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_METADATA_KEYS = {"answer", "options"}


def _assert_valid_document(doc: MedicalDocument) -> None:
    """Shared per-document assertions used across multiple tests."""
    assert doc.id, "Document ID must not be empty."
    assert doc.title, "Document title must not be empty."
    assert len(doc.title) <= 80, "Title must be truncated to 80 characters."
    assert doc.content, "Document content must not be empty."
    assert doc.source == "MedQA", f"Expected source 'MedQA', got '{doc.source}'."
    assert doc.metadata is not None, "Document metadata must not be None."
    for key in REQUIRED_METADATA_KEYS:
        assert key in doc.metadata, f"Metadata key '{key}' is missing."


# ---------------------------------------------------------------------------
# Core loader tests (use real DATASETS_DIR — integration)
# ---------------------------------------------------------------------------


def test_medqa_loader_returns_documents():
    """MedQALoader must load at least one document from the sample fixture."""
    loader = MedQALoader(DATASETS_DIR / "medqa")
    documents = loader.load()
    assert len(documents) > 0, "No documents were loaded from the MedQA dataset directory."


def test_medqa_loader_document_fields():
    """Every document returned by MedQALoader must pass all field checks."""
    loader = MedQALoader(DATASETS_DIR / "medqa")
    documents = loader.load()
    for doc in documents:
        _assert_valid_document(doc)


def test_medqa_loader_minimum_fixture_size():
    """The sample fixture must contain at least 4 records after the expansion."""
    loader = MedQALoader(DATASETS_DIR / "medqa")
    documents = loader.load()
    assert len(documents) >= 4, (
        f"Expected at least 4 sample records, got {len(documents)}. "
        "Run: append the missing records to data/datasets/medqa/sample.jsonl"
    )


def test_medqa_loader_unique_ids():
    """Document IDs must be unique within a single load() call."""
    loader = MedQALoader(DATASETS_DIR / "medqa")
    documents = loader.load()
    ids = [doc.id for doc in documents]
    assert len(ids) == len(set(ids)), "Duplicate document IDs detected."


def test_medqa_loader_answer_idx_present():
    """answer_idx must be present in metadata for every document."""
    loader = MedQALoader(DATASETS_DIR / "medqa")
    documents = loader.load()
    for doc in documents:
        assert "answer_idx" in doc.metadata, (
            f"answer_idx missing from metadata for document '{doc.id}'."
        )


# ---------------------------------------------------------------------------
# Fixture-level tests (validate the fixture file itself via conftest)
# ---------------------------------------------------------------------------


def test_fixture_file_exists(medqa_sample_path: Path):
    """The conftest-provided sample path must point to a real file."""
    assert medqa_sample_path.is_file(), f"Fixture not found: {medqa_sample_path}"


def test_fixture_record_schema(medqa_sample_records: list[dict]):
    """Every record in the fixture file must have the expected top-level keys."""
    required = {"question", "answer", "answer_idx", "options"}
    for i, record in enumerate(medqa_sample_records):
        missing = required - record.keys()
        assert not missing, (
            f"Record {i} is missing required keys: {missing}"
        )


def test_fixture_options_are_dict(medqa_sample_records: list[dict]):
    """The 'options' field in every fixture record must be a non-empty dict."""
    for i, record in enumerate(medqa_sample_records):
        opts = record.get("options")
        assert isinstance(opts, dict), f"Record {i}: 'options' must be a dict, got {type(opts)}."
        assert len(opts) >= 2, f"Record {i}: 'options' must have at least 2 choices."


def test_fixture_answer_in_options(medqa_sample_records: list[dict]):
    """The 'answer_idx' must correspond to a key that exists in 'options'."""
    for i, record in enumerate(medqa_sample_records):
        idx = record.get("answer_idx")
        opts = record.get("options", {})
        assert idx in opts, (
            f"Record {i}: answer_idx '{idx}' not found in options keys {list(opts.keys())}."
        )


def test_fixture_covers_multiple_specialties(medqa_sample_records: list[dict]):
    """The sample fixture should represent more than one clinical question."""
    questions = [r["question"] for r in medqa_sample_records]
    assert len(set(questions)) >= 4, (
        "Sample fixture should contain at least 4 distinct questions "
        f"to provide adequate coverage; found {len(set(questions))}."
    )


def test_medqa_loader_missing_directory():
    """MedQALoader on a non-existent directory must return an empty list."""
    loader = MedQALoader(Path("/nonexistent/medqa_dir"))
    documents = loader.load()
    assert documents == [], (
        "MedQALoader should return [] for a non-existent directory, "
        f"got {len(documents)} documents instead."
    )


# ---------------------------------------------------------------------------
# Self-contained tmp_path unit tests (sana/retrieval)
# ---------------------------------------------------------------------------


def test_medqa_loader_loads_jsonl_documents(tmp_path):
    dataset_file = tmp_path / "sample.jsonl"

    record = {
        "question": "What is hypertension?",
        "answer": "Persistently elevated blood pressure",
        "answer_idx": "A",
        "options": {"A": "Elevated blood pressure"},
        "meta_info": "medicine",
    }

    dataset_file.write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    loader = MedQALoader(tmp_path)
    documents = loader.load()

    assert len(documents) == 1

    document = documents[0]
    assert document.id == "sample_0"
    assert document.title == record["question"]
    assert document.content == record["question"]
    assert document.source == "MedQA"
    assert document.metadata["answer"] == record["answer"]
    assert document.metadata["options"] == record["options"]


def test_medqa_loader_skips_blank_lines(tmp_path):
    dataset_file = tmp_path / "sample.jsonl"
    dataset_file.write_text(
        '\n{"question": "What is diabetes?", "answer": "A disease"}\n\n',
        encoding="utf-8",
    )

    documents = MedQALoader(tmp_path).load()

    assert len(documents) == 1
    assert documents[0].content == "What is diabetes?"


def test_medqa_loader_returns_empty_list_for_empty_directory(tmp_path):
    documents = MedQALoader(tmp_path).load()

    assert documents == []
