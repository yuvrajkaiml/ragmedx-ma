"""Tests for DenseRetriever (app.rag.retrievers.dense_retriever).

Test design
-----------
* All unit tests use lightweight fake Embedder and VectorStore implementations
  so no model is loaded.  This keeps the unit suite fast and deterministic.
* The fake VectorStore wraps a real FAISSVectorStore with dim=4 to give the
  tests realistic search behaviour without downloading any weights.
* The integration test (``TestBGEFAISSIntegration``) loads the cached BGE
  model and a real FAISSVectorStore.  It is marked ``@pytest.mark.integration``
  so it can be excluded from fast runs:
      pytest -m "not integration"
* No files are written to the tracked project tree.  Persistence tests use
  ``tmp_path``.
"""

import uuid
from collections.abc import Sequence
from typing import Optional

import numpy as np
import pytest

from app.rag.interfaces import Embedder, VectorStore
from app.rag.retrievers.dense_retriever import DenseRetriever
from app.schemas.retrieval import DocumentChunk, RetrievalResult


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def _unit_vec(dim: int = 4, seed: int = 0) -> list[float]:
    """Deterministic L2-normalised float vector."""
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype("float32")
    v /= np.linalg.norm(v)
    return v.tolist()


def _make_chunk(index: int = 0) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        source_type="medqa",
        title=f"Chunk {index}",
        text=f"Medical evidence text {index}.",
        chunk_index=index,
        metadata={"page": str(index)},
    )


def _make_result(index: int = 0, score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(chunk=_make_chunk(index), relevance_score=score)


# ---------------------------------------------------------------------------
# Fakes — no model loading
# ---------------------------------------------------------------------------

class _FakeEmbedder(Embedder):
    """Returns a fixed query vector regardless of input text."""

    def __init__(self, query_vec: list[float]):
        self._vec = query_vec

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec


class _FakeStore(VectorStore):
    """Returns a pre-configured list of RetrievalResults."""

    def __init__(self, results: list[RetrievalResult]):
        self._results = results
        self.last_query_embedding: list[float] = []
        self.last_top_k: int = 0

    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        pass  # not needed for retriever unit tests

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> list[RetrievalResult]:
        self.last_query_embedding = list(query_embedding)
        self.last_top_k = top_k
        return self._results[:top_k]


def _make_retriever(
    results: Optional[list[RetrievalResult]] = None,
    query_vec: Optional[list[float]] = None,
    default_top_k: int = 5,
) -> DenseRetriever:
    """Build a DenseRetriever wired to fakes for unit testing."""
    if results is None:
        results = [_make_result(i, score=1.0 - i * 0.05) for i in range(10)]
    if query_vec is None:
        query_vec = _unit_vec()
    return DenseRetriever(
        embedder=_FakeEmbedder(query_vec),
        store=_FakeStore(results),
        default_top_k=default_top_k,
    )


# ---------------------------------------------------------------------------
# 1. Construction and type validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_accepts_valid_embedder_and_store(self):
        r = _make_retriever()
        assert r is not None

    def test_wrong_embedder_type_raises(self):
        store = _FakeStore([])
        with pytest.raises(TypeError, match="Embedder"):
            DenseRetriever(embedder="not_an_embedder", store=store)

    def test_wrong_store_type_raises(self):
        embedder = _FakeEmbedder(_unit_vec())
        with pytest.raises(TypeError, match="VectorStore"):
            DenseRetriever(embedder=embedder, store="not_a_store")

    def test_default_top_k_explicit(self):
        r = _make_retriever(default_top_k=7)
        assert r.default_top_k == 7

    def test_default_top_k_from_settings(self):
        """When no default_top_k is given it must come from Settings."""
        from app.config import get_settings
        r = DenseRetriever(
            embedder=_FakeEmbedder(_unit_vec()),
            store=_FakeStore([]),
        )
        assert r.default_top_k == get_settings().retrieval_top_k

    def test_embedder_property(self):
        emb = _FakeEmbedder(_unit_vec())
        r = DenseRetriever(embedder=emb, store=_FakeStore([]))
        assert r.embedder is emb

    def test_store_property(self):
        st = _FakeStore([])
        r = DenseRetriever(embedder=_FakeEmbedder(_unit_vec()), store=st)
        assert r.store is st


# ---------------------------------------------------------------------------
# 2. retrieve() — return type and basic behaviour
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_returns_list(self):
        results = _make_retriever().retrieve("hypertension treatment")
        assert isinstance(results, list)

    def test_returns_retrieval_results(self):
        results = _make_retriever().retrieve("query")
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_uses_default_top_k(self):
        store = _FakeStore([_make_result(i) for i in range(10)])
        r = DenseRetriever(
            embedder=_FakeEmbedder(_unit_vec()),
            store=store,
            default_top_k=3,
        )
        r.retrieve("query")
        assert store.last_top_k == 3

    def test_top_k_override(self):
        store = _FakeStore([_make_result(i) for i in range(10)])
        r = DenseRetriever(
            embedder=_FakeEmbedder(_unit_vec()),
            store=store,
            default_top_k=3,
        )
        r.retrieve("query", top_k=7)
        assert store.last_top_k == 7

    def test_passes_query_embedding_to_store(self):
        query_vec = _unit_vec(seed=99)
        store = _FakeStore([_make_result()])
        r = DenseRetriever(
            embedder=_FakeEmbedder(query_vec),
            store=store,
            default_top_k=1,
        )
        r.retrieve("some query")
        assert store.last_query_embedding == pytest.approx(query_vec, abs=1e-6)

    def test_empty_store_returns_empty_list(self):
        r = DenseRetriever(
            embedder=_FakeEmbedder(_unit_vec()),
            store=_FakeStore([]),
            default_top_k=5,
        )
        assert r.retrieve("query") == []

    def test_result_count_respects_store_clamping(self):
        """When the store has fewer items than top_k, all items are returned."""
        results = [_make_result(i) for i in range(2)]
        r = _make_retriever(results=results, default_top_k=10)
        returned = r.retrieve("query")
        assert len(returned) == 2

    def test_result_order_preserved(self):
        """Results must arrive in the same order as the store returns them."""
        results = [_make_result(i, score=1.0 - i * 0.1) for i in range(5)]
        r = _make_retriever(results=results, default_top_k=5)
        returned = r.retrieve("query")
        chunk_indices = [x.chunk.chunk_index for x in returned]
        assert chunk_indices == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# 3. Input validation / error handling
# ---------------------------------------------------------------------------

class TestValidation:
    def test_empty_query_raises(self):
        r = _make_retriever()
        with pytest.raises(ValueError, match="non-empty query"):
            r.retrieve("")

    def test_whitespace_only_query_raises(self):
        r = _make_retriever()
        with pytest.raises(ValueError, match="non-empty query"):
            r.retrieve("   ")

    def test_top_k_zero_raises(self):
        r = _make_retriever()
        with pytest.raises(ValueError, match="positive integer"):
            r.retrieve("query", top_k=0)

    def test_top_k_negative_raises(self):
        r = _make_retriever()
        with pytest.raises(ValueError, match="positive integer"):
            r.retrieve("query", top_k=-3)

    def test_valid_top_k_one(self):
        """top_k=1 is the minimum valid value and must not raise."""
        results = _make_retriever().retrieve("query", top_k=1)
        assert len(results) <= 1


# ---------------------------------------------------------------------------
# 4. Pipeline wiring — DenseRetriever + real FAISSVectorStore (no model)
# ---------------------------------------------------------------------------

class TestFAISSWiring:
    """Unit-level tests using a real FAISSVectorStore with dim=4 synthetic
    vectors.  No BGE model is loaded."""

    @pytest.fixture()
    def faiss_retriever(self) -> DenseRetriever:
        from app.rag.vectorstore.faiss_store import FAISSVectorStore

        dim = 4
        store = FAISSVectorStore(dim=dim)
        chunks = [_make_chunk(i) for i in range(5)]
        vecs = [_unit_vec(dim=dim, seed=i) for i in range(5)]
        store.upsert(chunks, vecs)

        # Embedder always returns the seed-0 vector (identical to chunk 0)
        embedder = _FakeEmbedder(_unit_vec(dim=dim, seed=0))
        return DenseRetriever(embedder=embedder, store=store, default_top_k=3)

    def test_retrieve_returns_results(self, faiss_retriever):
        results = faiss_retriever.retrieve("any query")
        assert len(results) > 0

    def test_exact_match_is_first(self, faiss_retriever):
        """Vector identical to chunk 0 must rank first."""
        results = faiss_retriever.retrieve("any query")
        assert results[0].chunk.chunk_index == 0

    def test_score_near_one_for_exact_match(self, faiss_retriever):
        results = faiss_retriever.retrieve("any query")
        assert abs(results[0].relevance_score - 1.0) < 1e-5

    def test_scores_descending(self, faiss_retriever):
        results = faiss_retriever.retrieve("any query")
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_override_respected(self, faiss_retriever):
        for k in (1, 2, 3):
            results = faiss_retriever.retrieve("query", top_k=k)
            assert len(results) == k

    def test_metadata_preserved(self, faiss_retriever):
        results = faiss_retriever.retrieve("any query", top_k=1)
        assert "page" in results[0].chunk.metadata

    def test_empty_store_returns_empty(self):
        from app.rag.vectorstore.faiss_store import FAISSVectorStore

        store = FAISSVectorStore(dim=4)
        r = DenseRetriever(
            embedder=_FakeEmbedder(_unit_vec(dim=4)),
            store=store,
            default_top_k=5,
        )
        assert r.retrieve("query") == []


# ---------------------------------------------------------------------------
# 5. Import boundary
# ---------------------------------------------------------------------------

class TestImportBoundary:
    def test_importable_from_rag_retrievers(self):
        from app.rag.retrievers.dense_retriever import DenseRetriever as D
        assert D is DenseRetriever

    def test_importable_from_rag_retrievers_package(self):
        from app.rag.retrievers import DenseRetriever as D
        assert D is DenseRetriever

    def test_importable_from_retrieval_dense_retriever(self):
        from app.retrieval.dense_retriever import DenseRetriever as D
        assert D is DenseRetriever

    def test_importable_from_retrieval_package(self):
        from app.retrieval import DenseRetriever as D
        assert D is DenseRetriever


# ---------------------------------------------------------------------------
# 6. BGE + FAISS integration test
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestBGEFAISSIntegration:
    """End-to-end: BGEEmbedder -> DenseRetriever -> FAISSVectorStore -> results.

    Uses the cached BAAI/bge-small-en-v1.5 model.  Skip in fast runs with:
        pytest -m "not integration"
    """

    @pytest.fixture(scope="class")
    def live_retriever(self) -> DenseRetriever:
        from app.rag.embeddings.bge_embedding import BGEEmbedder
        from app.rag.vectorstore.faiss_store import FAISSVectorStore

        passages = [
            "ACE inhibitors are recommended as first-line therapy for essential hypertension.",
            "Thiazide diuretics lower blood pressure by reducing plasma volume.",
            "Calcium channel blockers relax vascular smooth muscle to reduce blood pressure.",
            "Beta-blockers reduce heart rate and cardiac output.",
            "Intramedullary nailing is the standard treatment for femoral shaft fractures.",
        ]

        embedder = BGEEmbedder()
        store = FAISSVectorStore(dim=embedder.dim)
        chunks = [_make_chunk(i) for i in range(len(passages))]
        store.upsert(chunks, embedder.embed_documents(passages))

        return DenseRetriever(embedder=embedder, store=store, default_top_k=5)

    def test_retrieve_returns_list(self, live_retriever):
        results = live_retriever.retrieve("hypertension treatment")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_all_results_are_retrieval_results(self, live_retriever):
        for r in live_retriever.retrieve("heart medication", top_k=3):
            assert isinstance(r, RetrievalResult)

    def test_relevant_passage_ranks_in_top_two(self, live_retriever):
        """A hypertension query must surface a hypertension passage near top."""
        results = live_retriever.retrieve(
            "What is the first-line drug for essential hypertension?", top_k=5
        )
        top2_indices = {r.chunk.chunk_index for r in results[:2]}
        # Passages 0–3 are cardiovascular; 4 is orthopaedic
        assert top2_indices & {0, 1, 2, 3}, (
            f"Expected a cardiovascular passage in top-2, got {top2_indices}"
        )

    def test_orthopaedic_query_ranks_fracture_passage_high(self, live_retriever):
        """An orthopaedic query must rank the fracture passage above the median."""
        results = live_retriever.retrieve(
            "treatment for femoral shaft fracture", top_k=5
        )
        # Chunk index 4 is the fracture passage; it should appear in top 3
        top3_indices = {r.chunk.chunk_index for r in results[:3]}
        assert 4 in top3_indices, (
            f"Expected fracture passage (chunk 4) in top-3, got {top3_indices}"
        )

    def test_scores_bounded(self, live_retriever):
        for r in live_retriever.retrieve("medical evidence", top_k=5):
            assert 0.0 <= r.relevance_score <= 1.0

    def test_scores_descending(self, live_retriever):
        results = live_retriever.retrieve("blood pressure medication", top_k=5)
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self, live_retriever):
        for k in (1, 2, 3):
            assert len(live_retriever.retrieve("medication", top_k=k)) == k

    def test_empty_query_still_raises(self, live_retriever):
        with pytest.raises(ValueError):
            live_retriever.retrieve("")
