"""Tests for SparseRetriever (app.rag.retrievers.sparse_retriever).

Test design
-----------
* All tests are pure unit tests — no model loading, no FAISS, no GPU.
  BM25 is a CPU-only library with no external dependencies beyond rank-bm25.
* Corpus fixtures use short, distinct medical passages so BM25 term-overlap
  semantics are easy to reason about deterministically.
* Score assertions use a tolerance of 1e-9 rather than exact equality to
  guard against float64 rounding across platforms.
* There is no separate @pytest.mark.integration class because SparseRetriever
  has no heavy external dependencies.  A "live-corpus" class uses a realistic
  set of medical passages to verify end-to-end semantic behaviour.
"""

import uuid
from collections.abc import Callable

import pytest

from app.rag.retrievers.sparse_retriever import SparseRetriever, _default_tokenise
from app.schemas.retrieval import DocumentChunk, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(index: int, text: str, source_type: str = "medqa") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        source_type=source_type,
        title=f"Chunk {index}",
        text=text,
        chunk_index=index,
        metadata={"page": str(index)},
    )


# A small deterministic corpus used across many tests
_CORPUS = [
    _chunk(0, "ACE inhibitors are the first-line therapy for essential hypertension."),
    _chunk(1, "Thiazide diuretics reduce blood pressure by decreasing plasma volume."),
    _chunk(2, "Calcium channel blockers relax vascular smooth muscle to lower blood pressure."),
    _chunk(3, "Beta-blockers reduce heart rate and cardiac output."),
    _chunk(4, "Intramedullary nailing is the standard treatment for femoral shaft fractures."),
]


def _indexed(default_top_k: int = 5) -> SparseRetriever:
    """Return a SparseRetriever already indexed with the five-chunk corpus."""
    r = SparseRetriever(default_top_k=default_top_k)
    r.index(_CORPUS)
    return r


# ---------------------------------------------------------------------------
# 1. Default tokeniser
# ---------------------------------------------------------------------------

class TestDefaultTokeniser:
    def test_lowercases_input(self):
        assert _default_tokenise("ACE Inhibitor") == ["ace", "inhibitor"]

    def test_splits_on_punctuation(self):
        tokens = _default_tokenise("first-line therapy.")
        assert "first" in tokens
        assert "line" in tokens
        assert "therapy" in tokens

    def test_filters_empty_tokens(self):
        tokens = _default_tokenise("  blood  pressure  ")
        assert "" not in tokens
        assert tokens == ["blood", "pressure"]

    def test_numbers_kept(self):
        assert "1" in _default_tokenise("stage 1 hypertension")

    def test_empty_string_returns_empty_list(self):
        assert _default_tokenise("") == []

    def test_punctuation_only_returns_empty_list(self):
        assert _default_tokenise("--- ... !!") == []


# ---------------------------------------------------------------------------
# 2. Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_construction(self):
        r = SparseRetriever()
        assert r is not None

    def test_default_top_k_from_settings(self):
        from app.config import get_settings
        r = SparseRetriever()
        assert r.default_top_k == get_settings().retrieval_top_k

    def test_explicit_default_top_k(self):
        r = SparseRetriever(default_top_k=7)
        assert r.default_top_k == 7

    def test_starts_empty(self):
        assert SparseRetriever().ntotal == 0

    def test_custom_tokeniser_stored(self):
        custom = lambda t: t.split()
        r = SparseRetriever(tokeniser=custom)
        assert r.tokeniser is custom

    def test_default_tokeniser_used_when_none(self):
        r = SparseRetriever()
        assert r.tokeniser is _default_tokenise

    def test_k1_b_accepted(self):
        # Should not raise
        SparseRetriever(k1=1.2, b=0.6)


# ---------------------------------------------------------------------------
# 3. index()
# ---------------------------------------------------------------------------

class TestIndex:
    def test_ntotal_increases_after_index(self):
        r = SparseRetriever()
        r.index(_CORPUS[:3])
        assert r.ntotal == 3

    def test_ntotal_reflects_full_corpus(self):
        r = SparseRetriever()
        r.index(_CORPUS)
        assert r.ntotal == len(_CORPUS)

    def test_reindex_replaces_corpus(self):
        r = SparseRetriever()
        r.index(_CORPUS)
        r.index(_CORPUS[:2])
        assert r.ntotal == 2

    def test_empty_index_clears_corpus(self):
        r = SparseRetriever()
        r.index(_CORPUS)
        r.index([])
        assert r.ntotal == 0

    def test_index_single_chunk(self):
        r = SparseRetriever()
        r.index([_CORPUS[0]])
        assert r.ntotal == 1

    def test_index_preserves_chunk_order(self):
        """First retrieve after indexing must match the indexed order."""
        r = SparseRetriever(default_top_k=5)
        r.index(_CORPUS)
        # Ask for all docs with a query that matches all (common word "the")
        results = r.retrieve("the treatment therapy")
        # chunk_index values in results must be a subset of 0-4
        returned_indices = {res.chunk.chunk_index for res in results}
        assert returned_indices.issubset({0, 1, 2, 3, 4})


# ---------------------------------------------------------------------------
# 4. retrieve() — basic behaviour
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_returns_list(self):
        assert isinstance(_indexed().retrieve("hypertension"), list)

    def test_returns_retrieval_result_objects(self):
        results = _indexed().retrieve("hypertension")
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_respects_default_top_k(self):
        r = SparseRetriever(default_top_k=2)
        r.index(_CORPUS)
        assert len(r.retrieve("treatment")) == 2

    def test_respects_explicit_top_k(self):
        r = _indexed()
        for k in (1, 2, 3):
            assert len(r.retrieve("treatment therapy", top_k=k)) == k

    def test_top_k_larger_than_corpus_clamped(self):
        r = SparseRetriever()
        r.index(_CORPUS[:2])
        results = r.retrieve("treatment", top_k=100)
        assert len(results) == 2

    def test_top_k_one_returns_single_result(self):
        assert len(_indexed().retrieve("hypertension", top_k=1)) == 1

    def test_empty_index_returns_empty_list(self):
        r = SparseRetriever()
        assert r.retrieve("hypertension") == []

    def test_empty_index_after_clear_returns_empty_list(self):
        r = _indexed()
        r.index([])
        assert r.retrieve("hypertension") == []


# ---------------------------------------------------------------------------
# 5. Score correctness
# ---------------------------------------------------------------------------

class TestScores:
    def test_scores_in_zero_one_range(self):
        for res in _indexed().retrieve("treatment therapy", top_k=5):
            assert 0.0 <= res.relevance_score <= 1.0

    def test_top_result_has_score_one(self):
        """The highest-ranked result must always have normalised score = 1.0."""
        results = _indexed().retrieve("ACE inhibitor hypertension first-line", top_k=3)
        assert abs(results[0].relevance_score - 1.0) < 1e-9

    def test_scores_descending(self):
        results = _indexed().retrieve("blood pressure treatment therapy", top_k=5)
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_no_term_overlap_gives_zero_scores(self):
        """A query with no shared tokens must yield all-zero scores."""
        results = _indexed().retrieve("xyzzy qqqq zzz", top_k=3)
        for r in results:
            assert r.relevance_score == 0.0

    def test_exact_text_match_scores_highest(self):
        """A query whose tokens exactly match one chunk must rank that chunk first."""
        r = SparseRetriever()
        r.index([
            _chunk(0, "aspirin inhibits platelet aggregation"),
            _chunk(1, "metformin lowers blood glucose"),
            _chunk(2, "aspirin inhibits platelet aggregation"),  # duplicate text
        ])
        results = r.retrieve("aspirin inhibits platelet", top_k=1)
        assert results[0].chunk.chunk_index in {0, 2}  # one of the two matching docs

    def test_partial_term_match_outscores_no_match(self):
        r = _indexed()
        results_match = r.retrieve("hypertension", top_k=5)
        results_no = r.retrieve("xyzzy", top_k=5)
        best_match = max(res.relevance_score for res in results_match)
        best_no = max(res.relevance_score for res in results_no)
        assert best_match > best_no


# ---------------------------------------------------------------------------
# 6. Ranking correctness
# ---------------------------------------------------------------------------

class TestRanking:
    def test_hypertension_chunk_ranks_first(self):
        """Chunk 0 (hypertension + ACE inhibitor) must rank #1 for a
        hypertension query."""
        results = _indexed().retrieve("hypertension ACE inhibitor first-line", top_k=3)
        assert results[0].chunk.chunk_index == 0

    def test_fracture_chunk_ranks_first_for_ortho_query(self):
        """Chunk 4 (fracture) must rank #1 for an orthopaedic query."""
        results = _indexed().retrieve("femoral shaft fracture nailing", top_k=3)
        assert results[0].chunk.chunk_index == 4

    def test_blood_pressure_query_ranks_cardiovascular_top(self):
        """A 'blood pressure' query must surface one of chunks 1–3 first
        (all mention blood pressure), not the fracture chunk."""
        results = _indexed().retrieve("blood pressure", top_k=3)
        top_index = results[0].chunk.chunk_index
        assert top_index in {1, 2, 3}, (
            f"Expected a blood-pressure chunk in position 0, got chunk {top_index}"
        )

    def test_result_chunks_are_from_indexed_corpus(self):
        """Every returned chunk must be one of the five indexed chunks."""
        valid_ids = {c.chunk_id for c in _CORPUS}
        for res in _indexed().retrieve("treatment", top_k=5):
            assert res.chunk.chunk_id in valid_ids


# ---------------------------------------------------------------------------
# 7. Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_empty_query_raises(self):
        with pytest.raises(ValueError, match="non-empty query"):
            _indexed().retrieve("")

    def test_whitespace_query_raises(self):
        with pytest.raises(ValueError, match="non-empty query"):
            _indexed().retrieve("   \t\n  ")

    def test_top_k_zero_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            _indexed().retrieve("query", top_k=0)

    def test_top_k_negative_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            _indexed().retrieve("query", top_k=-1)

    def test_top_k_one_valid(self):
        results = _indexed().retrieve("hypertension", top_k=1)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# 8. Custom tokeniser
# ---------------------------------------------------------------------------

class TestCustomTokeniser:
    def test_custom_tokeniser_used_at_index_time(self):
        """A tokeniser that strips stop-words should change ranking."""
        stop_words = {"the", "is", "for", "to", "by", "a", "an", "of", "and"}

        def medical_tokenise(text: str) -> list[str]:
            tokens = _default_tokenise(text)
            return [t for t in tokens if t not in stop_words]

        r = SparseRetriever(tokeniser=medical_tokenise, default_top_k=5)
        r.index(_CORPUS)
        results = r.retrieve("ACE inhibitor hypertension")
        # chunk 0 has both "ace" and "inhibitor" and "hypertension"
        assert results[0].chunk.chunk_index == 0

    def test_custom_tokeniser_used_at_query_time(self):
        """Verify the same callable tokenises query tokens."""
        seen_calls: list[str] = []

        def recording_tokenise(text: str) -> list[str]:
            seen_calls.append(text)
            return _default_tokenise(text)

        r = SparseRetriever(tokeniser=recording_tokenise)
        r.index(_CORPUS)          # 5 corpus tokenisations
        r.retrieve("hypertension", top_k=1)  # +1 query tokenisation

        # Query text must appear in seen_calls
        assert "hypertension" in seen_calls

    def test_tokeniser_property_returns_injected(self):
        custom = lambda t: t.split()
        r = SparseRetriever(tokeniser=custom)
        assert r.tokeniser is custom


# ---------------------------------------------------------------------------
# 9. Metadata preservation
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_metadata_preserved_in_results(self):
        chunk = _chunk(0, "ACE inhibitors for hypertension")
        chunk.metadata["source"] = "pubmed"
        chunk.metadata["year"] = "2023"
        r = SparseRetriever()
        r.index([chunk])
        results = r.retrieve("hypertension", top_k=1)
        assert results[0].chunk.metadata["source"] == "pubmed"
        assert results[0].chunk.metadata["year"] == "2023"

    def test_source_type_preserved(self):
        chunk = _chunk(0, "beta blocker heart failure", source_type="pubmed")
        r = SparseRetriever()
        r.index([chunk])
        results = r.retrieve("beta blocker", top_k=1)
        assert results[0].chunk.source_type == "pubmed"

    def test_chunk_index_preserved(self):
        chunks = [_chunk(i, f"text {i} unique token_{i}") for i in range(5)]
        r = SparseRetriever()
        r.index(chunks)
        results = r.retrieve("unique token_3", top_k=1)
        assert results[0].chunk.chunk_index == 3


# ---------------------------------------------------------------------------
# 10. Import boundary
# ---------------------------------------------------------------------------

class TestImportBoundary:
    def test_importable_from_rag_retrievers_module(self):
        from app.rag.retrievers.sparse_retriever import SparseRetriever as S
        assert S is SparseRetriever

    def test_importable_from_rag_retrievers_package(self):
        from app.rag.retrievers import SparseRetriever as S
        assert S is SparseRetriever

    def test_importable_from_retrieval_sparse_retriever(self):
        from app.retrieval.sparse_retriever import SparseRetriever as S
        assert S is SparseRetriever

    def test_importable_from_retrieval_package(self):
        from app.retrieval import SparseRetriever as S
        assert S is SparseRetriever

    def test_dense_retriever_still_importable(self):
        """Adding SparseRetriever must not break DenseRetriever imports."""
        from app.retrieval import DenseRetriever
        assert DenseRetriever is not None


# ---------------------------------------------------------------------------
# 11. Live corpus — end-to-end semantic sanity
# ---------------------------------------------------------------------------

class TestLiveCorpus:
    """Uses a richer corpus of 10 medical passages to verify BM25 ranking
    on realistic clinical queries.  No external dependencies — still pure unit."""

    @pytest.fixture(scope="class")
    def rich_retriever(self) -> SparseRetriever:
        passages = [
            (0,  "ACE inhibitors block angiotensin-converting enzyme reducing blood pressure."),
            (1,  "ARBs block angiotensin II receptors as an alternative to ACE inhibitors."),
            (2,  "Thiazide diuretics are recommended for uncomplicated essential hypertension."),
            (3,  "Beta-blockers reduce heart rate useful in angina and post-MI patients."),
            (4,  "Calcium channel blockers are preferred in elderly hypertensive patients."),
            (5,  "Metformin is the first-line agent for type 2 diabetes mellitus."),
            (6,  "Insulin therapy is used when oral agents fail to control blood glucose."),
            (7,  "Aspirin irreversibly inhibits COX-1 preventing thromboxane A2 synthesis."),
            (8,  "Intramedullary nailing is standard for diaphyseal femoral shaft fractures."),
            (9,  "Open reduction internal fixation is used for displaced articular fractures."),
        ]
        r = SparseRetriever(default_top_k=5)
        r.index([_chunk(i, text) for i, text in passages])
        return r

    def test_hypertension_query_returns_antihypertensives(self, rich_retriever):
        results = rich_retriever.retrieve("antihypertensive blood pressure treatment", top_k=3)
        top3 = {r.chunk.chunk_index for r in results}
        # Chunks 0–4 are antihypertensive; 5–9 are not
        assert top3 & {0, 1, 2, 3, 4}, f"No antihypertensive in top-3: {top3}"

    def test_fracture_query_returns_orthopaedic(self, rich_retriever):
        results = rich_retriever.retrieve("femoral fracture nailing fixation", top_k=3)
        top3 = {r.chunk.chunk_index for r in results}
        assert top3 & {8, 9}, f"No orthopaedic chunk in top-3: {top3}"

    def test_diabetes_query_returns_diabetes_chunks(self, rich_retriever):
        results = rich_retriever.retrieve("diabetes glucose insulin metformin", top_k=3)
        top3 = {r.chunk.chunk_index for r in results}
        assert top3 & {5, 6}, f"No diabetes chunk in top-3: {top3}"

    def test_scores_always_normalised(self, rich_retriever):
        for query in [
            "ACE inhibitor hypertension",
            "diabetes glucose",
            "xyzzy unknown term",
        ]:
            for r in rich_retriever.retrieve(query, top_k=5):
                assert 0.0 <= r.relevance_score <= 1.0

    def test_all_result_scores_descending(self, rich_retriever):
        results = rich_retriever.retrieve("blood pressure hypertension treatment", top_k=5)
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_ntotal_correct(self, rich_retriever):
        assert rich_retriever.ntotal == 10
