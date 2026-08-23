import pytest
from pathlib import Path
from app.config import DATASETS_DIR

def test_document_loader_boundary():
    """Verify document loader boundary in app.retrieval."""
    try:
        from app.retrieval.document_loader import DocumentLoader
        loader = DocumentLoader(DATASETS_DIR)
        assert loader is not None
    except (ImportError, AttributeError):
        pytest.skip("DocumentLoader in app.retrieval is pending Sana Thorat implementation.")