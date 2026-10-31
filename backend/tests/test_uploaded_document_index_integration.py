"""Opt-in, slow end-to-end smoke test for the uploaded-document pipeline using
the REAL cached BGE-M3 embedder.

Run explicitly with ``-m integration``. It is excluded from the default suite
(which runs ``-m "not integration"``) and skips automatically when the BGE-M3
model is not cached locally, so CI without the ~2 GB model stays green.

    cd d:\\Ragmed_2\\backend
    & d:\\Ragmed_2\\.venv\\Scripts\\python.exe -m pytest \
        tests/test_uploaded_document_index_integration.py -m integration -q
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest

from app.services.uploaded_document_index import UploadedDocumentIndexService

pytestmark = pytest.mark.integration

_MODEL_CACHE = (
    Path(os.path.expanduser("~"))
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--BAAI--bge-m3"
)
_SPARE_PDF = Path(r"d:\Ragmed_2\docs\Visual 1.pdf")


@pytest.mark.skipif(not _MODEL_CACHE.exists(), reason="BGE-M3 model not cached locally")
@pytest.mark.skipif(not _SPARE_PDF.exists(), reason="Spare PDF fixture not available")
async def test_real_build_index_and_retrieve(tmp_path):
    service = UploadedDocumentIndexService(uploads_dir=tmp_path)
    document_id = uuid4()
    service.pdf_path(document_id).write_bytes(_SPARE_PDF.read_bytes())

    chunk_count = service.build_index(document_id, filename="Visual 1.pdf")

    assert chunk_count > 0
    assert service.is_indexed(document_id)

    provider = service.evidence_provider(document_id)
    evidence, iterations = await provider(query="What is this document about?")

    assert iterations[0]["status"] == "completed"
    for item in evidence:
        assert item["source_type"] == "uploaded_document"
        assert 0.0 <= item["relevance_score"] <= 1.0
