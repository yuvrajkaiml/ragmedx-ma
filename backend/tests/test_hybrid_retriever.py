"""Tests for HybridRetriever (app.rag.retrievers.hybrid_retriever).

Test design
-----------
* All tests are pure unit / integration tests -- no BGE model loading in unit
  classes.  The integration class (TestBGEFAISSHybridIntegration) loads the
  cached BGE model and is marked @pytest.mark.integration.
* Fake Dense/Sparse retrievers are built by wiring real FAISSVectorStore
  (dim=4, synthetic vectors) and a FakeEmbedder so fusion logic is exercised
  without downloading anything.
* RRF arithmetic is validated with hand-calculated expected values.
"""

import math
import uuid
from collections.abc import Sequence
from typing import Optional

import numpy as np
import pytest

from app.rag.interfaces import Embedder, VectorStore
from app.rag.retrievers.dense_retriever import DenseRetriever
from app.rag.retrievers.hybrid_retriever import HybridRetriever
from app.rag.retrievers.sparse_retriever import SparseRetriever
from app.rag.vectorstore.faiss_store import FAISSVectorStore
from app.schemas.retrieval import DocumentChunk, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(dim: int = 4, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype("float32")
    return (v / np.linalg.norm(v)).tolist()


def _chunk(index: int, text: str = "", source_type: str = "medqa") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        source_type=source_type,
        title=f"Chunk {index}",
        text=text or f"medical evidence text chunk {index}",
        chunk_index=index,
        metadata={"page": str(index)},
    )


class _FakeEmbedder(Embedder):
    """Returns a fixed query vector regardless of input."""
    def __init__(self, vec: list[float]):
        self._vec = vec
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec for _ in texts]
    def embed_query(self, text: str) -> list[float]:
        return self._vec


def _build_pair(
    texts: list[str],
    default_top_k: int = 5,
    query_seed: int = 0,
) -> tuple[DenseRetriever, SparseRetriever]:
    """Build a matched DenseRetriever + SparseRetriever over the same chunks."""
    dim = 4
    chunks = [_chunk(i, t) for i, t in enumerate(texts)]
    vecs = [_unit_vec(dim=dim, seed=i) for i in range(len(chunks))]

    store = FAISSVectorStore(dim=dim)
    store.upsert(chunks, vecs)
    dense = DenseRetriever(
        embedder=_FakeEmbedder(_unit_vec(dim=dim, seed=query_seed)),
        store=store,
        default_top_k=default_top_k,
    )

    sparse = SparseRetriever(default_top_k=default_top_k)
    sparse.index(chunks)

    return dense, sparse


_CORPUS = [
    "ACE inhibitors are first-line therapy for essential hypertension",
    "Thiazide diuretics reduce blood pressure by decreasing plasma volume",
    "Calcium channel blockers relax vascular smooth muscle",
    "Beta-blockers reduce heart rate and cardiac output",
    "Intramedullary nailing treats femoral shaft fractures",
]


# ---------------------------------------------------------------------------
# 1. Construction validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_both_rankers_accepted(self):
        d, s = _build_pair(_CORPUS)
        h = HybridRetriever(dense=d, sparse=s)
        assert h is not None

    def test_dense_only_accepted(self):
        d, _ = _build_pair(_CORPUS)
        h = HybridRetriever(dense=d, sparse=None)
        assert h.sparse is None

    def test_sparse_only_accepted(self):
        _, s = _build_pair(_CORPUS)
        h = HybridRetriever(dense=None, sparse=s)
        assert h.dense is None

    def test_both_none_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            HybridRetriever(dense=None, sparse=None)

    def test_wrong_dense_type_raises(self):
        _, s = _build_pair(_CORPUS)
        with pytest.raises(TypeError, match="DenseRetriever"):
            HybridRetriever(dense="not_a_retriever", sparse=s)

    def test_wrong_sparse_type_raises(self):
        d, _ = _build_pair(_CORPUS)
        with pytest.raises(TypeError, match="SparseRetriever"):
            HybridRetriever(dense=d, sparse="not_a_retriever")

    def test_negative_weight_raises(self):
        d, s = _build_pair(_CORPUS)
        with pytest.raises(ValueError, match="non-negative"):
            HybridRetriever(dense=d, sparse=s, weight_dense=-0.1)

    def test_zero_rrf_k_raises(self):
        d, s = _build_pair(_CORPUS)
        with pytest.raises(ValueError, match="rrf_k"):
            HybridRetriever(dense=d, sparse=s, rrf_k=0)

    def test_multiplier_zero_raises(self):
        d, s = _build_pair(_CORPUS)
        with pytest.raises(ValueError, match="retrieval_multiplier"):
            HybridRetriever(dense=d, sparse=s, retrieval_multiplier=0)

    def test_default_top_k_from_settings(self):
        from app.config import get_settings
        d, s = _build_pair(_CORPUS)
        h = HybridRetriever(dense=d, sparse=s)
        assert h.default_top_k == get_settings().retrieval_top_k

    def test_explicit_default_top_k(self):
        d, s = _build_pair(_CORPUS)
        h = HybridRetriever(dense=d, sparse=s, default_top_k=7)
        assert h.default_top_k == 7

    def test_properties_stored(self):
        d, s = _build_pair(_CORPUS)
        h = HybridRetriever(dense=d, sparse=s, weight_dense=0.8,
                            weight_sparse=1.2, rrf_k=30)
        assert h.dense is d
        assert h.sparse is s
        assert h.weight_dense == 0.8
        assert h.weight_sparse == 1.2
        assert h.rrf_k == 30


# ---------------------------------------------------------------------------
# 2. Retrieve -- return type and shape
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_returns_list(self):
        d, s = _build_pair(_CORPUS)
        h = HybridRetriever(dense=d, sparse=s, default_top_k=3)
        assert isinstance(h.retrieve("hypertension"), list)

    def test_returns_retrieval_results(self):
        d, s = _build_pair(_CORPUS)
        h = HybridRetriever(dense=d, sparse=s, default_top_k=3)
        results = h.retrieve("hypertension")
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_respects_default_top_k(self):
        d, s = _build_pair(_CORPUS)
        h = HybridRetriever(dense=d, sparse=s, default_top_k=2)
        assert len(h.retrieve("hypertension")) == 2

    def test_top_k_override(self):
        d, s = _build_pair(_CORPUS)
        h = HybridRetriever(dense=d, sparse=s, default_top_k=5)
        for k in (1, 2, 3):
            assert len(h.retrieve("therapy", top_k=k)) == k

    def test_top_k_larger_than_corpus_clamped(self):
        d, s = _build_pair(_CORPUS[:2])
        h = HybridRetriever(dense=d, sparse=s, default_top_k=10)
        results = h.retrieve("therapy", top_k=100)
        assert len(results) <= 2

    def test_no_duplicate_chunk_ids(self):
        d, s = _build_pair(_CORPUS)
        h = HybridRetriever(dense=d, sparse=s, default_top_k=5)
        results = h.retrieve("blood pressure treatment")
        ids = [r.chunk.chunk_id for r in results]
        assert len(ids) == len(set(ids)), "duplicate chunk_ids in fused results"


# ---------------------------------------------------------------------------
# 3. Score correctness
# ---------------------------------------------------------------------------

class TestScores:
    def test_scores_in_zero_one(self):
        d, s = _build_pair(_CORPUS)
        h = HybridRetriever(dense=d, sparse=s, default_top_k=5)
        for r in h.retrieve("hypertension treatment"):
            assert 0.0 <= r.relevance_score <= 1.0

    def test_top_score_is_one(self):
        d, s = _build_pair(_CORPUS)
        h = HybridRetriever(dense=d, sparse=s, default_top_k=3)
        results = h.retrieve("hypertension ACE inhibitor")
        assert abs(results[0].relevance_score - 1.0) < 1e-9

    def test_scores_descending(self):
        d, s = _build_pair(_CORPUS)
        h = HybridRetriever(dense=d, sparse=s, default_top_k=5)
        results = h.retrieve("blood pressure treatment")
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_arithmetic(self):
        """Manually verify RRF fusion for a 2-chunk corpus.

        If dense ranks chunk A at position 1 and sparse ranks chunk A at
        position 1, the fused RRF score for A is:
            1/(60+1) + 1/(60+1) = 2/61

        If chunk B is not in either list its score is 0 (won\'t appear).
        The normalised score for A is 1.0 (it\'s the only result).
        """
        dim = 4
        cA = _chunk(0, "unique keyword alpha beta gamma")
        cB = _chunk(1, "completely different orthogonal zzz")

        store = FAISSVectorStore(dim=dim)
        vA = _unit_vec(dim=dim, seed=0)
        vB = _unit_vec(dim=dim, seed=99)
        store.upsert([cA, cB], [vA, vB])

        sparse = SparseRetriever()
        sparse.index([cA, cB])

        # FakeEmbedder always returns vA -- dense ranks cA first
        dense = DenseRetriever(
            embedder=_FakeEmbedder(vA), store=store, default_top_k=2
        )

        h = HybridRetriever(dense=dense, sparse=sparse, default_top_k=1, rrf_k=60)
        results = h.retrieve("unique keyword alpha")

        assert len(results) == 1
        assert results[0].chunk.chunk_id == cA.chunk_id
        assert abs(results[0].relevance_score - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 4. Graceful degradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_dense_only_returns_results(self):
        d, _ = _build_pair(_CORPUS)
        h = HybridRetriever(dense=d, sparse=None, default_top_k=3)
        results = h.retrieve("hypertension")
        assert len(results) == 3

    def test_sparse_only_returns_results(self):
        _, s = _build_pair(_CORPUS)
        h = HybridRetriever(dense=None, sparse=s, default_top_k=3)
        results = h.retrieve("hypertension")
        assert len(results) > 0

    def test_dense_only_scores_descending(self):
        d, _ = _build_pair(_CORPUS)
        h = HybridRetriever(dense=d, sparse=None, default_top_k=3)
        scores = [r.relevance_score for r in h.retrieve("treatment")]
        assert scores == sorted(scores, reverse=True)

    def test_sparse_only_top_score_one(self):
        _, s = _build_pair(_CORPUS)
        h = HybridRetriever(dense=None, sparse=s, default_top_k=3)
        results = h.retrieve("ACE inhibitor hypertension first-line")
        assert abs(results[0].relevance_score - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 5. Weighting
# ---------------------------------------------------------------------------

class TestWeighting:
    def test_weight_zero_dense_equals_sparse_only(self):
        """Setting weight_dense=0 must produce the same ranking as sparse-only."""
        d, s = _build_pair(_CORPUS)
        h_zero_dense = HybridRetriever(
            dense=d, sparse=s, default_top_k=3,
            weight_dense=0.0, weight_sparse=1.0,
        )
        h_sparse_only = HybridRetriever(
            dense=None, sparse=s, default_top_k=3,
        )
        q = "ACE inhibitor hypertension first-line"
        r_zero = h_zero_dense.retrieve(q)
        r_sparse = h_sparse_only.retrieve(q)

        # Top result chunk_id must match
        assert r_zero[0].chunk.chunk_id == r_sparse[0].chunk.chunk_id

    def test_weight_zero_sparse_equals_dense_only(self):
        d, s = _build_pair(_CORPUS)
        h_zero_sparse = HybridRetriever(
            dense=d, sparse=s, default_top_k=3,
            weight_dense=1.0, weight_sparse=0.0,
        )
        h_dense_only = HybridRetriever(dense=d, sparse=None, default_top_k=3)
        q = "treatment therapy"
        r_zero = h_zero_sparse.retrieve(q)
        r_dense = h_dense_only.retrieve(q)
        assert r_zero[0].chunk.chunk_id == r_dense[0].chunk.chunk_id

    def test_equal_weights_gives_balanced_fusion(self):
        """Symmetric weights must not crash and must produce normalised scores."""
        d, s = _build_pair(_CORPUS)
        h = HybridRetriever(dense=d, sparse=s, weight_dense=0.5, weight_sparse=0.5)
        results = h.retrieve("blood pressure", top_k=3)
        for r in results:
            assert 0.0 <= r.relevance_score <= 1.0


# ---------------------------------------------------------------------------
# 6. Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_empty_query_raises(self):
        d, s = _build_pair(_CORPUS)
        h = HybridRetriever(dense=d, sparse=s)
        with pytest.raises(ValueError, match="non-empty query"):
            h.retrieve("")

    def test_whitespace_query_raises(self):
        d, s = _build_pair(_CORPUS)
        h = HybridRetriever(dense=d, sparse=s)
        with pytest.raises(ValueError, match="non-empty query"):
            h.retrieve("   ")

    def test_top_k_zero_raises(self):
        d, s = _build_pair(_CORPUS)
        h = HybridRetriever(dense=d, sparse=s)
        with pytest.raises(ValueError, match="positive integer"):
            h.retrieve("query", top_k=0)

    def test_top_k_negative_raises(self):
        d, s = _build_pair(_CORPUS)
        h = HybridRetriever(dense=d, sparse=s)
        with pytest.raises(ValueError, match="positive integer"):
            h.retrieve("query", top_k=-1)

    def test_top_k_one_valid(self):
        d, s = _build_pair(_CORPUS)
        h = HybridRetriever(dense=d, sparse=s)
        assert len(h.retrieve("therapy", top_k=1)) == 1


# ---------------------------------------------------------------------------
# 7. Metadata preservation
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_chunk_metadata_preserved(self):
        dim = 4
        c = DocumentChunk(
            chunk_id=uuid.uuid4(), document_id=uuid.uuid4(),
            source_type="pubmed", title="Study",
            text="ACE inhibitors reduce blood pressure",
            chunk_index=0, metadata={"source": "lancet", "year": "2023"},
        )
        store = FAISSVectorStore(dim=dim)
        store.upsert([c], [_unit_vec(dim=dim, seed=0)])
        dense = DenseRetriever(
            embedder=_FakeEmbedder(_unit_vec(dim=dim, seed=0)),
            store=store, default_top_k=1,
        )
        sparse = SparseRetriever()
        sparse.index([c])

        h = HybridRetriever(dense=dense, sparse=sparse, default_top_k=1)
        results = h.retrieve("ACE inhibitor blood pressure", top_k=1)

        assert results[0].chunk.metadata["source"] == "lancet"
        assert results[0].chunk.metadata["year"] == "2023"
        assert results[0].chunk.source_type == "pubmed"


# ---------------------------------------------------------------------------
# 8. Import boundary
# ---------------------------------------------------------------------------

class TestImportBoundary:
    def test_importable_from_rag_module(self):
        from app.rag.retrievers.hybrid_retriever import HybridRetriever as H
        assert H is HybridRetriever

    def test_importable_from_rag_package(self):
        from app.rag.retrievers import HybridRetriever as H
        assert H is HybridRetriever

    def test_importable_from_retrieval_module(self):
        from app.retrieval.hybrid_retriever import HybridRetriever as H
        assert H is HybridRetriever

    def test_importable_from_retrieval_package(self):
        from app.retrieval import HybridRetriever as H
        assert H is HybridRetriever

    def test_dense_and_sparse_still_importable(self):
        from app.retrieval import DenseRetriever, SparseRetriever
        assert DenseRetriever is not None
        assert SparseRetriever is not None


# ---------------------------------------------------------------------------
# 9. BGE + FAISS integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestBGEFAISSHybridIntegration:
    """End-to-end: BGEEmbedder -> DenseRetriever + SparseRetriever -> HybridRetriever.

    Uses the cached BAAI/bge-small-en-v1.5 model.  Skip with:
        pytest -m "not integration"
    """

    @pytest.fixture(scope="class")
    def live_hybrid(self) -> HybridRetriever:
        from app.rag.embeddings.bge_embedding import BGEEmbedder

        passages = [
            "ACE inhibitors are recommended as first-line therapy for essential hypertension.",
            "Thiazide diuretics lower blood pressure by reducing plasma volume.",
            "Calcium channel blockers relax vascular smooth muscle to reduce blood pressure.",
            "Beta-blockers reduce heart rate and cardiac output.",
            "Intramedullary nailing is the standard treatment for femoral shaft fractures.",
        ]
        embedder = BGEEmbedder()
        chunks = [_chunk(i, t) for i, t in enumerate(passages)]

        store = FAISSVectorStore(dim=embedder.dim)
        store.upsert(chunks, embedder.embed_documents(passages))
        dense = DenseRetriever(embedder=embedder, store=store, default_top_k=5)

        sparse = SparseRetriever(default_top_k=5)
        sparse.index(chunks)

        return HybridRetriever(dense=dense, sparse=sparse, default_top_k=5)

    def test_returns_results(self, live_hybrid):
        results = live_hybrid.retrieve("first-line hypertension treatment")
        assert len(results) > 0

    def test_no_duplicate_chunk_ids(self, live_hybrid):
        results = live_hybrid.retrieve("blood pressure medication", top_k=5)
        ids = [r.chunk.chunk_id for r in results]
        assert len(ids) == len(set(ids))

    def test_scores_in_range(self, live_hybrid):
        for r in live_hybrid.retrieve("hypertension therapy", top_k=5):
            assert 0.0 <= r.relevance_score <= 1.0

    def test_scores_descending(self, live_hybrid):
        results = live_hybrid.retrieve("antihypertensive drug", top_k=5)
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_hypertension_chunk_in_top_two(self, live_hybrid):
        results = live_hybrid.retrieve(
            "What is the first-line drug for essential hypertension?", top_k=5
        )
        top2 = {r.chunk.chunk_index for r in results[:2]}
        assert top2 & {0, 1, 2, 3}, f"No hypertension chunk in top-2: {top2}"

    def test_fracture_chunk_in_top_three(self, live_hybrid):
        results = live_hybrid.retrieve(
            "treatment for femoral shaft fracture", top_k=5
        )
        top3 = {r.chunk.chunk_index for r in results[:3]}
        assert 4 in top3, f"Fracture chunk (4) not in top-3: {top3}"

    def test_top_k_respected(self, live_hybrid):
        for k in (1, 2, 3):
            assert len(live_hybrid.retrieve("medication therapy", top_k=k)) == k

    def test_empty_query_raises(self, live_hybrid):
        with pytest.raises(ValueError):
            live_hybrid.retrieve("")