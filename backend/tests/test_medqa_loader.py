import pytest
from app.config import DATASETS_DIR
from app.rag.loaders.medqa_loader import MedQALoader


def test_medqa_loader():
    """Verify MedQA dataset loader can parse dataset files."""
    loader = MedQALoader(DATASETS_DIR / "medqa")
    documents = loader.load()

    assert len(documents) > 0, "No documents were loaded from MedQA dataset directory."

    for document in documents:
        assert document.id, "Document ID is missing."
        assert document.title, "Document title is missing."
        assert document.content, "Document content is missing."
        assert document.source == "MedQA", "Incorrect document source."
        assert document.metadata is not None, "Document metadata is missing."
        assert "answer" in document.metadata, "Answer missing from metadata."
        assert "options" in document.metadata, "Options missing from metadata."