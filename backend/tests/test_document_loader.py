"""Tests for DocumentLoader (app.retrieval.document_loader).

DocumentLoader is the unified ingestion entry-point for the retrieval pipeline.
It delegates to the specialised loaders in app.rag.loaders and returns a flat
list of MedicalDocument objects.
"""

import tempfile
from pathlib import Path

import pytest

from app.config import DATASETS_DIR
from app.rag.loaders.models import MedicalDocument
from app.retrieval.document_loader import DocumentLoader


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_document_loader_instantiation():
    """DocumentLoader must instantiate with a valid datasets_dir."""
    loader = DocumentLoader(DATASETS_DIR)
    assert loader is not None
    assert loader.datasets_dir == DATASETS_DIR


def test_document_loader_accepts_string_path():
    """DocumentLoader must accept a plain string path and convert it to Path."""
    loader = DocumentLoader(str(DATASETS_DIR))
    assert isinstance(loader.datasets_dir, Path)


# ---------------------------------------------------------------------------
# load_all()
# ---------------------------------------------------------------------------


def test_load_all_returns_list():
    """load_all() must return a list (not None, not a generator)."""
    loader = DocumentLoader(DATASETS_DIR)
    result = loader.load_all()
    assert isinstance(result, list)


def test_load_all_returns_medical_documents():
    """Every item returned by load_all() must be a MedicalDocument."""
    loader = DocumentLoader(DATASETS_DIR)
    docs = loader.load_all()
    assert len(docs) > 0, "Expected at least one document from the sample fixture."
    for doc in docs:
        assert isinstance(doc, MedicalDocument), (
            f"Expected MedicalDocument, got {type(doc)}"
        )


def test_load_all_document_fields():
    """All documents from load_all() must have non-empty id, title, content, source."""
    loader = DocumentLoader(DATASETS_DIR)
    docs = loader.load_all()
    for doc in docs:
        assert doc.id, f"Document missing id: {doc!r}"
        assert doc.title, f"Document missing title: {doc!r}"
        assert doc.content, f"Document missing content: {doc!r}"
        assert doc.source, f"Document missing source: {doc!r}"


def test_load_all_medqa_source_tag():
    """Documents loaded from medqa/ sub-directory must carry source='MedQA'."""
    loader = DocumentLoader(DATASETS_DIR)
    docs = loader.load_all()
    medqa_docs = [d for d in docs if d.source == "MedQA"]
    assert len(medqa_docs) > 0, "No MedQA-sourced documents found in load_all() output."


def test_load_all_skips_missing_subdirs():
    """load_all() must not raise when a known sub-directory is absent."""
    with tempfile.TemporaryDirectory() as tmp:
        loader = DocumentLoader(Path(tmp))  # empty dir — no medqa/ subdir
        docs = loader.load_all()
        assert docs == [], f"Expected empty list for empty datasets_dir, got {docs}"


# ---------------------------------------------------------------------------
# load_file() — plain-text files
# ---------------------------------------------------------------------------


def test_load_file_text(tmp_path: Path):
    """load_file() must load a .txt file and return a single MedicalDocument."""
    txt_file = tmp_path / "note.txt"
    txt_file.write_text("Patient presents with chest pain.", encoding="utf-8")

    loader = DocumentLoader(DATASETS_DIR)
    docs = loader.load_file(txt_file)

    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, MedicalDocument)
    assert doc.id == "note"
    assert doc.title == "note.txt"
    assert "chest pain" in doc.content
    assert doc.source == "note.txt"


def test_load_file_text_accepts_string_path(tmp_path: Path):
    """load_file() must accept a plain string path for .txt files."""
    txt_file = tmp_path / "report.txt"
    txt_file.write_text("Blood pressure: 120/80 mmHg.", encoding="utf-8")

    loader = DocumentLoader(DATASETS_DIR)
    docs = loader.load_file(str(txt_file))
    assert len(docs) == 1
    assert "Blood pressure" in docs[0].content


# ---------------------------------------------------------------------------
# load_file() — error handling
# ---------------------------------------------------------------------------


def test_load_file_missing_raises():
    """load_file() must raise FileNotFoundError for a non-existent path."""
    loader = DocumentLoader(DATASETS_DIR)
    with pytest.raises(FileNotFoundError):
        loader.load_file(Path("/nonexistent/file.txt"))


def test_load_file_unsupported_extension_raises(tmp_path: Path):
    """load_file() must raise ValueError for an unsupported file extension."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("col1,col2\n1,2\n", encoding="utf-8")

    loader = DocumentLoader(DATASETS_DIR)
    with pytest.raises(ValueError, match="Unsupported file type"):
        loader.load_file(csv_file)
