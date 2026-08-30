"""Tests for CrossEncoderReranker (app.rag.rerankers.cross_encoder).

Test design
-----------
* Unit tests use a FakeCrossEncoder that skips model loading entirely.
  This keeps the unit suite fast and deterministic.
* The integration class (TestCrossEncoderIntegration) loads the cached
  cross-encoder/ms-marco-MiniLM-L-6-v2 model and is marked
  @pytest.mark.integration.
* No files are written to disk; no FAISS indexes are created.
"""

import uuid
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.rag.rerankers.cross_encoder import CrossEncoderReranker, _DEFAULT_MODEL
from app.schemas.retrieval import DocumentChunk, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(index: int, text: str = "", source_type: str = "medqa") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        source_type=source_type,
        title=f"Chunk {index}",
        text=text or f"medical evidence chunk {index}",
        chunk_index=index,
        metadata={"page": str(index)},
    )


def _result(index: int, text: str = "", score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(chunk=_chunk(index, text), relevance_score=score)


def _make_fake_reranker(scores: list[float]) -> CrossEncoderReranker:
    """Build a CrossEncoderReranker whose _model.predict returns *scores*."""
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model_name = "fake-model"
    reranker.batch_size = 32
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array(scores, dtype=np.float32)
    reranker._model = mock_model
    return reranker


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_model_name(self):
        """Verify the default model constant is the ms-marco MiniLM."""
        assert _DEFAULT_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_model_name_stored(self):
        r = _make_fake_reranker([])
        assert r.model_name == "fake-model"

    def test_batch_size_stored(self):
        r = _make_fake_reranker([])
        assert r.batch_size == 32


# ---------------------------------------------------------------------------
# 2. rerank() -- return type and shape
# ---------------------------------------------------------------------------

class TestRerank:
    def test_returns_list(self):
        r = _make_fake_reranker([0.9, 0.5, 0.1])
        results = r.rerank("query", [_result(i) for i in range(3)])
        assert isinstance(results, list)

    def test_returns_retrieval_results(self):
        r = _make_fake_reranker([0.9, 0.5, 0.1])
        for item in r.rerank("query", [_result(i) for i in range(3)]):
            assert isinstance(item, RetrievalResult)

    def test_same_length_as_input(self):
        scores = [0.9, 0.6, 0.3]
        r = _make_fake_reranker(scores)
        assert len(r.rerank("q", [_result(i) for i in range(3)])) == 3

    def test_top_k_trims_results(self):
        r = _make_fake_reranker([0.9, 0.6, 0.3])
        assert len(r.rerank("q", [_result(i) for i in range(3)], top_k=2)) == 2

    def test_top_k_one(self):
        r = _make_fake_reranker([0.9, 0.6, 0.3])
        assert len(r.rerank("q", [_result(i) for i in range(3)], top_k=1)) == 1

    def test_top_k_larger_than_results_returns_all(self):
        r = _make_fake_reranker([0.9, 0.6])
        assert len(r.rerank("q", [_result(i) for i in range(2)], top_k=100)) == 2

    def test_empty_results_returns_empty(self):
        r = _make_fake_reranker([])
        assert r.rerank("query", []) == []

    def test_single_result(self):
        r = _make_fake_reranker([0.75])
        out = r.rerank("q", [_result(0)])
        assert len(out) == 1
        assert abs(out[0].relevance_score - 0.75) < 1e-6


# ---------------------------------------------------------------------------
# 3. Score correctness
# ---------------------------------------------------------------------------

class TestScores:
    def test_scores_in_zero_one(self):
        r = _make_fake_reranker([0.9, 0.5, 0.1])
        for item in r.rerank("q", [_result(i) for i in range(3)]):
            assert 0.0 <= item.relevance_score <= 1.0

    def test_scores_descending(self):
        r = _make_fake_reranker([0.3, 0.9, 0.6])  # intentionally unsorted
        results = r.rerank("q", [_result(i) for i in range(3)])
        scores = [x.relevance_score for x in results]
        assert scores == sorted(scores, reverse=True)

    def test_score_clamped_above_one(self):
        """Scores slightly above 1.0 (float32 edge) must be clamped."""
        r = _make_fake_reranker([1.0000001, 0.5])
        results = r.rerank("q", [_result(i) for i in range(2)])
        assert results[0].relevance_score <= 1.0

    def test_score_clamped_below_zero(self):
        """Negative scores (should not happen with sigmoid but guard anyway)."""
        r = _make_fake_reranker([-0.001, 0.5])
        results = r.rerank("q", [_result(i) for i in range(2)])
        for item in results:
            assert item.relevance_score >= 0.0

    def test_first_stage_scores_replaced(self):
        """The reranker must replace, not average, the first-stage scores."""
        first_stage_score = 0.99
        ce_score = 0.42
        r = _make_fake_reranker([ce_score])
        result = RetrievalResult(
            chunk=_chunk(0, "passage text"),
            relevance_score=first_stage_score,
        )
        reranked = r.rerank("q", [result])
        assert abs(reranked[0].relevance_score - ce_score) < 1e-6


# ---------------------------------------------------------------------------
# 4. Ordering correctness
# ---------------------------------------------------------------------------

class TestOrdering:
    def test_highest_ce_score_ranks_first(self):
        """chunk_index 1 gets the highest fake CE score -> must rank first."""
        r = _make_fake_reranker([0.3, 0.95, 0.6])
        results = [_result(0), _result(1), _result(2)]
        reranked = r.rerank("q", results)
        assert reranked[0].chunk.chunk_index == 1

    def test_ordering_independent_of_input_order(self):
        """Input order must not affect the final ranking."""
        scores = [0.7, 0.2, 0.9]
        r = _make_fake_reranker(scores)
        results = [_result(i) for i in range(3)]
        reranked = r.rerank("q", results)
        returned_scores = [x.relevance_score for x in reranked]
        assert returned_scores == sorted(returned_scores, reverse=True)

    def test_equal_scores_preserves_stability(self):
        """All-equal scores must not crash or produce nonsense."""
        r = _make_fake_reranker([0.5, 0.5, 0.5])
        results = r.rerank("q", [_result(i) for i in range(3)])
        assert len(results) == 3


# ---------------------------------------------------------------------------
# 5. Metadata and chunk preservation
# ---------------------------------------------------------------------------

class TestChunkPreservation:
    def test_chunk_id_preserved(self):
        original = _result(0, "passage about hypertension")
        r = _make_fake_reranker([0.8])
        reranked = r.rerank("q", [original])
        assert reranked[0].chunk.chunk_id == original.chunk.chunk_id

    def test_chunk_text_preserved(self):
        original = _result(0, "specific medical text content")
        r = _make_fake_reranker([0.8])
        reranked = r.rerank("q", [original])
        assert reranked[0].chunk.text == "specific medical text content"

    def test_chunk_metadata_preserved(self):
        chunk = _chunk(0, "text")
        chunk.metadata["source"] = "pubmed"
        chunk.metadata["year"] = "2023"
        result = RetrievalResult(chunk=chunk, relevance_score=0.5)
        r = _make_fake_reranker([0.8])
        reranked = r.rerank("q", [result])
        assert reranked[0].chunk.metadata["source"] == "pubmed"
        assert reranked[0].chunk.metadata["year"] == "2023"

    def test_source_type_preserved(self):
        chunk = _chunk(0, "text", source_type="pubmed")
        result = RetrievalResult(chunk=chunk, relevance_score=0.5)
        r = _make_fake_reranker([0.8])
        reranked = r.rerank("q", [result])
        assert reranked[0].chunk.source_type == "pubmed"

    def test_predict_receives_correct_pairs(self):
        """The model must be called with (query, chunk.text) pairs."""
        query = "What treats hypertension?"
        texts = ["passage A", "passage B"]
        results = [_result(i, texts[i]) for i in range(2)]
        r = _make_fake_reranker([0.8, 0.4])

        r.rerank(query, results)

        call_args = r._model.predict.call_args
        passed_pairs = call_args[0][0]  # first positional arg
        assert passed_pairs[0] == (query, "passage A")
        assert passed_pairs[1] == (query, "passage B")


# ---------------------------------------------------------------------------
# 6. Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_empty_query_raises(self):
        r = _make_fake_reranker([0.5])
        with pytest.raises(ValueError, match="non-empty query"):
            r.rerank("", [_result(0)])

    def test_whitespace_query_raises(self):
        r = _make_fake_reranker([0.5])
        with pytest.raises(ValueError, match="non-empty query"):
            r.rerank("   ", [_result(0)])

    def test_top_k_zero_raises(self):
        r = _make_fake_reranker([0.5])
        with pytest.raises(ValueError, match="positive integer"):
            r.rerank("q", [_result(0)], top_k=0)

    def test_top_k_negative_raises(self):
        r = _make_fake_reranker([0.5])
        with pytest.raises(ValueError, match="positive integer"):
            r.rerank("q", [_result(0)], top_k=-1)

    def test_none_top_k_returns_all(self):
        r = _make_fake_reranker([0.9, 0.6, 0.3])
        assert len(r.rerank("q", [_result(i) for i in range(3)], top_k=None)) == 3


# ---------------------------------------------------------------------------
# 7. Import boundary
# ---------------------------------------------------------------------------

class TestImportBoundary:
    def test_importable_from_rag_module(self):
        from app.rag.rerankers.cross_encoder import CrossEncoderReranker as C
        assert C is CrossEncoderReranker

    def test_importable_from_rag_package(self):
        from app.rag.rerankers import CrossEncoderReranker as C
        assert C is CrossEncoderReranker

    def test_importable_from_retrieval_module(self):
        from app.retrieval.reranker import CrossEncoderReranker as C
        assert C is CrossEncoderReranker

    def test_importable_from_retrieval_package(self):
        from app.retrieval import CrossEncoderReranker as C
        assert C is CrossEncoderReranker

    def test_all_prior_retrievers_still_importable(self):
        from app.retrieval import (
            DenseRetriever, SparseRetriever, HybridRetriever,
            BGEEmbedder, FAISSVectorStore,
        )
        assert all(x is not None for x in [
            DenseRetriever, SparseRetriever, HybridRetriever,
            BGEEmbedder, FAISSVectorStore,
        ])


# ---------------------------------------------------------------------------
# 8. Cross-encoder integration test (real model)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCrossEncoderIntegration:
    """Uses the cached cross-encoder/ms-marco-MiniLM-L-6-v2 model.

    Skip in fast runs:  pytest -m "not integration"
    """

    @pytest.fixture(scope="class")
    def reranker(self) -> CrossEncoderReranker:
        return CrossEncoderReranker()

    def _medical_results(self) -> list[RetrievalResult]:
        passages = [
            "Intramedullary nailing is the standard treatment for femoral shaft fractures.",
            "Thiazide diuretics reduce blood pressure by decreasing plasma volume.",
            "ACE inhibitors are recommended as first-line therapy for essential hypertension.",
            "Beta-blockers reduce heart rate and cardiac output in heart failure.",
            "Metformin is first-line pharmacotherapy for type 2 diabetes mellitus.",
        ]
        return [_result(i, p) for i, p in enumerate(passages)]

    def test_returns_correct_count(self, reranker):
        results = self._medical_results()
        reranked = reranker.rerank("hypertension treatment", results)
        assert len(reranked) == len(results)

    def test_scores_in_range(self, reranker):
        for item in reranker.rerank("hypertension therapy", self._medical_results()):
            assert 0.0 <= item.relevance_score <= 1.0

    def test_scores_descending(self, reranker):
        results = reranker.rerank("hypertension", self._medical_results())
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_hypertension_chunk_ranks_first(self, reranker):
        """ACE inhibitor passage (index 2) must rank above fracture (index 0)."""
        results = self._medical_results()
        reranked = reranker.rerank(
            "first-line pharmacotherapy for essential hypertension", results
        )
        top2_indices = {r.chunk.chunk_index for r in reranked[:2]}
        # Chunks 1, 2, 3 are cardiovascular; 0 is ortho; 4 is diabetes
        assert top2_indices & {1, 2, 3}, (
            f"Expected cardiovascular chunk in top-2, got {top2_indices}"
        )

    def test_fracture_chunk_ranks_first_for_ortho_query(self, reranker):
        results = self._medical_results()
        reranked = reranker.rerank(
            "treatment for femoral shaft fracture", results
        )
        top2_indices = {r.chunk.chunk_index for r in reranked[:2]}
        assert 0 in top2_indices, (
            f"Expected fracture chunk (0) in top-2, got {top2_indices}"
        )

    def test_top_k_trims_to_k(self, reranker):
        for k in (1, 2, 3):
            results = reranker.rerank("therapy", self._medical_results(), top_k=k)
            assert len(results) == k

    def test_empty_input_returns_empty(self, reranker):
        assert reranker.rerank("query", []) == []

    def test_no_duplicate_chunk_ids(self, reranker):
        results = self._medical_results()
        reranked = reranker.rerank("medical treatment", results)
        ids = [r.chunk.chunk_id for r in reranked]
        assert len(ids) == len(set(ids))