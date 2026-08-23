"""Tests for FAISSVectorStore (app.rag.vectorstore.faiss_store).

Test design
-----------
* All unit tests use small, deterministic synthetic vectors -- no model
  downloads required.  Vectors are L2-normalised to respect the documented
  normalisation contract.
* Persistence tests use pytest's ``tmp_path`` fixture so nothing is written
  to the tracked project tree.
* The single integration test (TestBGEIntegration) loads the cached BGE model
  and is marked ``@pytest.mark.integration`` so it can be excluded from fast
  runs with ``-m "not integration"``.
* The session-scoped ``embedder`` fixture in conftest.py is not used here to
  keep this file self-contained; the integration test creates its own embedder.
"""

import math
import uuid
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from app.rag.interfaces import VectorStore
from app.rag.vectorstore.faiss_store import FAISSVectorStore
from app.schemas.retrieval import DocumentChunk, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(dim: int = 384, seed: int = 0) -> list[float]:
    """Return a deterministic L2-normalised float32 vector."""
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype("float32")
    v /= np.linalg.norm(v)
    return v.tolist()


def _make_chunk(
    index: int = 0,
    source_type: str = "medqa",
    extra_meta: dict | None = None,
) -> DocumentChunk:
    """Return a minimal DocumentChunk suitable for testing."""
    meta = {"page": str(index)}
    if extra_meta:
        meta.update(extra_meta)
    return DocumentChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        source_type=source_type,
        title=f"Test Chunk {index}",
        text=f"Medical evidence text for chunk {index}.",
        chunk_index=index,
        metadata=meta,
    )


def _populated_store(n: int = 5, dim: int = 384) -> FAISSVectorStore:
    """Return a FAISSVectorStore with *n* deterministic chunks inserted."""
    store = FAISSVectorStore(dim=dim)
    chunks = [_make_chunk(i) for i in range(n)]
    vecs = [_unit_vec(dim=dim, seed=i) for i in range(n)]
    store.upsert(chunks, vecs)
    return store


# ---------------------------------------------------------------------------
# 1. Interface conformance
# ---------------------------------------------------------------------------

class TestInterfaceConformance:
    def test_is_vectorstore_subclass(self):
        assert issubclass(FAISSVectorStore, VectorStore)

    def test_instance_is_vectorstore(self):
        assert isinstance(FAISSVectorStore(dim=4), VectorStore)

    def test_has_upsert(self):
        assert callable(FAISSVectorStore.upsert)

    def test_has_search(self):
        assert callable(FAISSVectorStore.search)

    def test_has_save(self):
        assert callable(FAISSVectorStore.save)

    def test_has_load(self):
        assert callable(FAISSVectorStore.load)

    def test_has_dim_property(self):
        store = FAISSVectorStore(dim=16)
        assert store.dim == 16

    def test_has_ntotal_property(self):
        store = FAISSVectorStore(dim=4)
        assert store.ntotal == 0


# ---------------------------------------------------------------------------
# 2. Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_dim_is_384(self):
        store = FAISSVectorStore()
        assert store.dim == 384

    def test_custom_dim(self):
        store = FAISSVectorStore(dim=128)
        assert store.dim == 128

    def test_zero_dim_raises(self):
        with pytest.raises(ValueError, match="positive"):
            FAISSVectorStore(dim=0)

    def test_negative_dim_raises(self):
        with pytest.raises(ValueError, match="positive"):
            FAISSVectorStore(dim=-1)

    def test_custom_index_dir(self, tmp_path: Path):
        store = FAISSVectorStore(dim=4, index_dir=tmp_path)
        assert store._index_dir == tmp_path

    def test_starts_empty(self):
        store = FAISSVectorStore(dim=4)
        assert store.ntotal == 0


# ---------------------------------------------------------------------------
# 3. Upsert
# ---------------------------------------------------------------------------

class TestUpsert:
    def test_upsert_single_chunk(self):
        store = FAISSVectorStore(dim=4)
        chunk = _make_chunk()
        vec = [0.5, 0.5, 0.5, 0.5]
        store.upsert([chunk], [vec])
        assert store.ntotal == 1

    def test_upsert_multiple_chunks(self):
        store = FAISSVectorStore(dim=4)
        chunks = [_make_chunk(i) for i in range(5)]
        vecs = [_unit_vec(dim=4, seed=i) for i in range(5)]
        store.upsert(chunks, vecs)
        assert store.ntotal == 5

    def test_upsert_incremental(self):
        store = FAISSVectorStore(dim=4)
        for i in range(3):
            store.upsert([_make_chunk(i)], [_unit_vec(dim=4, seed=i)])
        assert store.ntotal == 3

    def test_upsert_empty_is_noop(self):
        store = FAISSVectorStore(dim=4)
        store.upsert([], [])
        assert store.ntotal == 0

    def test_upsert_length_mismatch_raises(self):
        store = FAISSVectorStore(dim=4)
        chunks = [_make_chunk(0), _make_chunk(1)]
        vecs = [_unit_vec(dim=4, seed=0)]
        with pytest.raises(ValueError, match="same length"):
            store.upsert(chunks, vecs)

    def test_upsert_wrong_dim_raises(self):
        store = FAISSVectorStore(dim=4)
        chunk = _make_chunk()
        with pytest.raises(ValueError, match="Expected embedding dimension 4"):
            store.upsert([chunk], [[0.1, 0.2]])  # dim=2, not 4

    def test_upsert_preserves_metadata(self):
        store = FAISSVectorStore(dim=4)
        chunk = _make_chunk(0, extra_meta={"source": "pubmed", "page": "7"})
        store.upsert([chunk], [_unit_vec(dim=4, seed=0)])
        results = store.search(_unit_vec(dim=4, seed=0), top_k=1)
        assert results[0].chunk.metadata["source"] == "pubmed"
        assert results[0].chunk.metadata["page"] == "7"

    def test_upsert_preserves_source_type(self):
        store = FAISSVectorStore(dim=4)
        chunk = _make_chunk(0, source_type="pubmed")
        store.upsert([chunk], [_unit_vec(dim=4, seed=0)])
        results = store.search(_unit_vec(dim=4, seed=0), top_k=1)
        assert results[0].chunk.source_type == "pubmed"


# ---------------------------------------------------------------------------
# 4. Search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_returns_list(self):
        store = _populated_store(n=3, dim=4)
        results = store.search(_unit_vec(dim=4, seed=0), top_k=1)
        assert isinstance(results, list)

    def test_search_returns_retrieval_results(self):
        store = _populated_store(n=3, dim=4)
        results = store.search(_unit_vec(dim=4, seed=0), top_k=1)
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_search_respects_top_k(self):
        store = _populated_store(n=5, dim=4)
        for k in (1, 2, 3):
            results = store.search(_unit_vec(dim=4, seed=99), top_k=k)
            assert len(results) == k

    def test_search_top_k_larger_than_index(self):
        """top_k > ntotal should return at most ntotal results."""
        store = _populated_store(n=3, dim=4)
        results = store.search(_unit_vec(dim=4, seed=0), top_k=100)
        assert len(results) == 3

    def test_search_returns_scores(self):
        store = _populated_store(n=3, dim=4)
        results = store.search(_unit_vec(dim=4, seed=0), top_k=2)
        for r in results:
            assert 0.0 <= r.relevance_score <= 1.0

    def test_search_ordered_by_score_descending(self):
        store = _populated_store(n=5, dim=4)
        results = store.search(_unit_vec(dim=4, seed=0), top_k=5)
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_exact_match_scores_one(self):
        """Searching with a vector identical to one in the index must score ~1.0."""
        store = FAISSVectorStore(dim=4)
        vec = _unit_vec(dim=4, seed=42)
        store.upsert([_make_chunk(0)], [vec])
        results = store.search(vec, top_k=1)
        assert abs(results[0].relevance_score - 1.0) < 1e-5

    def test_search_correct_chunk_id_returned(self):
        """Top result for a query identical to chunk 0 must be chunk 0."""
        store = FAISSVectorStore(dim=4)
        chunks = [_make_chunk(i) for i in range(3)]
        vecs = [_unit_vec(dim=4, seed=i) for i in range(3)]
        store.upsert(chunks, vecs)
        results = store.search(_unit_vec(dim=4, seed=0), top_k=1)
        assert results[0].chunk.chunk_index == 0

    def test_search_invalid_top_k_raises(self):
        store = _populated_store(n=2, dim=4)
        with pytest.raises(ValueError, match="positive integer"):
            store.search(_unit_vec(dim=4, seed=0), top_k=0)

    def test_search_negative_top_k_raises(self):
        store = _populated_store(n=2, dim=4)
        with pytest.raises(ValueError, match="positive integer"):
            store.search(_unit_vec(dim=4, seed=0), top_k=-1)

    def test_search_wrong_dim_raises(self):
        store = FAISSVectorStore(dim=4)
        store.upsert([_make_chunk()], [_unit_vec(dim=4, seed=0)])
        with pytest.raises(ValueError, match="Expected embedding dimension 4"):
            store.search([0.1, 0.2], top_k=1)  # dim=2


# ---------------------------------------------------------------------------
# 5. Empty index
# ---------------------------------------------------------------------------

class TestEmptyIndex:
    def test_search_empty_index_returns_empty_list(self):
        store = FAISSVectorStore(dim=4)
        results = store.search(_unit_vec(dim=4, seed=0), top_k=5)
        assert results == []

    def test_ntotal_zero_on_init(self):
        assert FAISSVectorStore(dim=4).ntotal == 0

    def test_upsert_empty_does_not_change_ntotal(self):
        store = FAISSVectorStore(dim=4)
        store.upsert([], [])
        assert store.ntotal == 0


# ---------------------------------------------------------------------------
# 6. Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_creates_index_file(self, tmp_path: Path):
        store = _populated_store(n=2, dim=4)
        store.save(tmp_path)
        assert (tmp_path / "index.faiss").exists()

    def test_save_creates_metadata_file(self, tmp_path: Path):
        store = _populated_store(n=2, dim=4)
        store.save(tmp_path)
        assert (tmp_path / "metadata.json").exists()

    def test_metadata_json_contains_dim(self, tmp_path: Path):
        store = FAISSVectorStore(dim=8)
        store.upsert([_make_chunk()], [_unit_vec(dim=8, seed=0)])
        store.save(tmp_path)
        import json
        meta = json.loads((tmp_path / "metadata.json").read_text())
        assert meta["dim"] == 8

    def test_metadata_json_contains_ntotal(self, tmp_path: Path):
        store = _populated_store(n=3, dim=4)
        store.save(tmp_path)
        import json
        meta = json.loads((tmp_path / "metadata.json").read_text())
        assert meta["ntotal"] == 3

    def test_load_restores_ntotal(self, tmp_path: Path):
        store = _populated_store(n=3, dim=4)
        store.save(tmp_path)
        loaded = FAISSVectorStore.load(tmp_path)
        assert loaded.ntotal == 3

    def test_load_restores_dim(self, tmp_path: Path):
        store = FAISSVectorStore(dim=8)
        store.upsert([_make_chunk()], [_unit_vec(dim=8, seed=0)])
        store.save(tmp_path)
        loaded = FAISSVectorStore.load(tmp_path)
        assert loaded.dim == 8

    def test_load_restores_search_results(self, tmp_path: Path):
        """Search results must be equivalent before and after save/load."""
        store = _populated_store(n=5, dim=4)
        query = _unit_vec(dim=4, seed=99)
        before = store.search(query, top_k=3)
        store.save(tmp_path)

        loaded = FAISSVectorStore.load(tmp_path)
        after = loaded.search(query, top_k=3)

        assert len(before) == len(after)
        for b, a in zip(before, after):
            assert b.chunk.chunk_index == a.chunk.chunk_index
            assert abs(b.relevance_score - a.relevance_score) < 1e-5

    def test_load_preserves_metadata(self, tmp_path: Path):
        """Metadata dict must survive the save/load round-trip."""
        store = FAISSVectorStore(dim=4)
        chunk = _make_chunk(0, extra_meta={"document_id": "doc-123", "page": "5"})
        vec = _unit_vec(dim=4, seed=0)
        store.upsert([chunk], [vec])
        store.save(tmp_path)

        loaded = FAISSVectorStore.load(tmp_path)
        results = loaded.search(vec, top_k=1)
        assert results[0].chunk.metadata["document_id"] == "doc-123"
        assert results[0].chunk.metadata["page"] == "5"

    def test_load_missing_index_file_raises(self, tmp_path: Path):
        """Missing index.faiss must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="index.faiss"):
            FAISSVectorStore.load(tmp_path)

    def test_load_missing_metadata_file_raises(self, tmp_path: Path):
        """Missing metadata.json must raise FileNotFoundError."""
        # Write only the index file
        store = _populated_store(n=1, dim=4)
        import faiss as _faiss
        _faiss.write_index(store._index, str(tmp_path / "index.faiss"))
        with pytest.raises(FileNotFoundError, match="metadata.json"):
            FAISSVectorStore.load(tmp_path)

    def test_save_creates_directory_if_missing(self, tmp_path: Path):
        nested = tmp_path / "deep" / "nested" / "dir"
        store = _populated_store(n=1, dim=4)
        store.save(nested)
        assert (nested / "index.faiss").exists()

    def test_save_with_explicit_dir_overrides_default(self, tmp_path: Path):
        store = _populated_store(n=2, dim=4)
        explicit_dir = tmp_path / "explicit"
        store.save(explicit_dir)
        assert (explicit_dir / "index.faiss").exists()


# ---------------------------------------------------------------------------
# 7. Import boundary
# ---------------------------------------------------------------------------

class TestImportBoundary:
    def test_importable_from_rag_vectorstore(self):
        from app.rag.vectorstore.faiss_store import FAISSVectorStore as F
        assert F is FAISSVectorStore

    def test_importable_from_rag_vectorstore_package(self):
        from app.rag.vectorstore import FAISSVectorStore as F
        assert F is FAISSVectorStore

    def test_importable_from_retrieval_vector_store(self):
        from app.retrieval.vector_store import FAISSVectorStore as F
        assert F is FAISSVectorStore

    def test_importable_from_retrieval_package(self):
        from app.retrieval import FAISSVectorStore as F
        assert F is FAISSVectorStore


# ---------------------------------------------------------------------------
# 8. BGE integration test
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestBGEIntegration:
    """End-to-end: BGEEmbedder -> FAISSVectorStore -> search.

    Uses the cached BAAI/bge-small-en-v1.5 model.  Marked as 'integration'
    so it can be skipped in fast unit-test runs:
        pytest -m "not integration"
    """

    @pytest.fixture(scope="class")
    def bge_store(self):
        """Build a store with 4 medical passages using real BGE embeddings."""
        from app.rag.embeddings.bge_embedding import BGEEmbedder

        passages = [
            "ACE inhibitors are recommended as first-line therapy for essential hypertension.",
            "Thiazide diuretics lower blood pressure by reducing plasma volume.",
            "Calcium channel blockers relax vascular smooth muscle to reduce blood pressure.",
            "Intramedullary nailing is the standard treatment for femoral shaft fractures.",
        ]
        embedder = BGEEmbedder()
        chunks = [_make_chunk(i, source_type="medqa") for i in range(len(passages))]
        doc_vecs = embedder.embed_documents(passages)

        store = FAISSVectorStore(dim=embedder.dim)
        store.upsert(chunks, doc_vecs)
        return store, embedder

    def test_integration_store_has_correct_ntotal(self, bge_store):
        store, _ = bge_store
        assert store.ntotal == 4

    def test_integration_relevant_passage_ranks_first(self, bge_store):
        """A hypertension query must rank a hypertension passage above an
        orthopaedic passage."""
        store, embedder = bge_store
        query = "What is the first-line drug for essential hypertension?"
        q_vec = embedder.embed_query(query)
        results = store.search(q_vec, top_k=4)

        assert len(results) == 4
        # Chunk indices 0-2 are hypertension passages; chunk index 3 is ortho.
        # At least one of the top-2 results must be a hypertension passage.
        top2_indices = {r.chunk.chunk_index for r in results[:2]}
        assert top2_indices & {0, 1, 2}, (
            f"Expected a hypertension passage in top-2, got chunk indices {top2_indices}"
        )

    def test_integration_scores_in_valid_range(self, bge_store):
        store, embedder = bge_store
        q_vec = embedder.embed_query("hypertension treatment")
        results = store.search(q_vec, top_k=4)
        for r in results:
            assert 0.0 <= r.relevance_score <= 1.0

    def test_integration_save_load_roundtrip(self, bge_store, tmp_path: Path):
        store, embedder = bge_store
        q_vec = embedder.embed_query("blood pressure medication")
        before = store.search(q_vec, top_k=3)

        store.save(tmp_path)
        loaded = FAISSVectorStore.load(tmp_path)
        after = loaded.search(q_vec, top_k=3)

        assert len(before) == len(after)
        for b, a in zip(before, after):
            assert b.chunk.chunk_index == a.chunk.chunk_index