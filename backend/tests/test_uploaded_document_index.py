"""Service-pipeline tests for the user-uploaded-document feature.

These exercise :class:`UploadedDocumentIndexService` end to end — parse → chunk
→ embed → persist an isolated FAISS index, then retrieve from it — without
loading the real ~2 GB BGE-M3 model. A tiny deterministic fake embedder is
injected via the constructor (the service already supports this), so the whole
pipeline runs fast and offline while still proving the real FAISS/chunking code
paths and the agent-contract evidence schema.

asyncio_mode = auto (pytest.ini) runs ``async def test_*`` without a decorator.
This file does not import ``app.main``, so no DEBUG env is required.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from app.services.uploaded_document_index import UploadedDocumentIndexService

_DIM = 1024


class _FakeEmbedder:
    """Deterministic 1024-dim L2-normalised embedder (no model, no network).

    Mirrors the slice of the :class:`BGEEmbedding` interface the index service
    uses: ``embed_documents(list[str]) -> (n, 1024) float32`` and
    ``embed_query(str) -> (1024,) float32``. Vectors are seeded from a hash of
    the text so identical text always maps to the same unit vector, which makes
    retrieval ranking deterministic.
    """

    def _vector(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(_DIM).astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def embed_documents(self, texts, batch_size=None) -> np.ndarray:
        if not texts:
            return np.empty((0, _DIM), dtype=np.float32)
        return np.stack([self._vector(text) for text in texts]).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self._vector(query)


def _write_pdf(path: Path, sentences: list[str]) -> None:
    """Write a one-page PDF with extractable text using the already-installed
    ``reportlab``. Falls back to copying the repo's spare PDF if reportlab or
    the draw step is unavailable so the test stays runnable either way."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        pdf = canvas.Canvas(str(path), pagesize=letter)
        text_object = pdf.beginText(72, 720)
        for sentence in sentences:
            text_object.textLine(sentence)
        pdf.drawText(text_object)
        pdf.showPage()
        pdf.save()
    except Exception:  # pragma: no cover - environment fallback
        spare = Path(r"d:\Ragmed_2\docs\Visual 1.pdf")
        if not spare.exists():
            pytest.skip("No PDF writer available and no spare PDF to copy.")
        path.write_bytes(spare.read_bytes())


@pytest.fixture
def service(tmp_path) -> UploadedDocumentIndexService:
    """Service scoped to a throwaway uploads dir with the fake embedder."""
    return UploadedDocumentIndexService(uploads_dir=tmp_path, embedder=_FakeEmbedder())


def _build_sample(service: UploadedDocumentIndexService):
    document_id = uuid4()
    _write_pdf(
        service.pdf_path(document_id),
        [
            "Metformin is a first-line treatment for type 2 diabetes mellitus.",
            "It lowers hepatic glucose production and improves insulin sensitivity.",
            "Common adverse effects include gastrointestinal upset and nausea.",
        ],
    )
    return document_id


def test_build_index_persists_isolated_faiss_store(service):
    document_id = _build_sample(service)

    chunk_count = service.build_index(document_id, filename="lab_report.pdf")

    assert chunk_count >= 1
    assert service.is_indexed(document_id)
    directory = service.index_dir(document_id)
    assert (directory / "index.faiss").is_file()
    assert (directory / "chunks.pkl").is_file()
    # Isolation: the index lives under this document's own directory.
    assert directory.parent == service.uploads_dir


def test_build_index_missing_pdf_raises(service):
    with pytest.raises(FileNotFoundError):
        service.build_index(uuid4())


def test_retrieve_on_unindexed_document_returns_empty(service):
    assert service.retrieve(uuid4(), "any question") == []


def test_retrieve_returns_ranked_chunks(service):
    document_id = _build_sample(service)
    service.build_index(document_id)

    results = service.retrieve(document_id, "type 2 diabetes treatment", top_k=3)

    assert results, "expected at least one retrieved chunk"
    chunk, score = results[0]
    assert chunk.content
    assert isinstance(score, float)


async def test_evidence_provider_returns_agent_contract_evidence(service):
    document_id = _build_sample(service)
    service.build_index(document_id)

    provider = service.evidence_provider(document_id)
    assert callable(provider)

    evidence, iterations = await provider(
        query="How is type 2 diabetes treated?",
        optimized_query=None,
        max_iterations=3,
    )

    assert evidence, "expected evidence from an indexed document"
    for item in evidence:
        assert item["source_type"] == "uploaded_document"
        assert item["document_id"]
        assert item["chunk_id"]
        assert item["cited_text"]
        assert 0.0 <= item["relevance_score"] <= 1.0
        assert item["citation_marker"].startswith("[Doc ")
    assert iterations[0]["status"] == "completed"
    assert iterations[0]["scope"] == "uploaded_document"


async def test_evidence_provider_on_unindexed_document_reports_status(service):
    provider = service.evidence_provider(uuid4())

    evidence, iterations = await provider(query="anything")

    assert evidence == []
    assert iterations[0]["status"] == "document_not_indexed"


async def test_evidence_provider_empty_query_short_circuits(service):
    document_id = _build_sample(service)
    service.build_index(document_id)

    provider = service.evidence_provider(document_id)
    evidence, iterations = await provider(query="   ", optimized_query=None)

    assert evidence == []
    assert iterations[0]["status"] == "empty_query"


def test_delete_index_removes_directory(service):
    document_id = _build_sample(service)
    service.build_index(document_id)
    assert service.is_indexed(document_id)

    service.delete_index(document_id)

    assert not service.is_indexed(document_id)
    assert not service.index_dir(document_id).exists()
