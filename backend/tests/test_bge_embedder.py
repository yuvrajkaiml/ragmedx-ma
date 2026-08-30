"""Tests for BGEEmbedder (app.rag.embeddings.bge_embedding).

Design decisions
----------------
* The model is loaded once per test session via a session-scoped fixture.
  Loading BAAI/bge-small-en-v1.5 takes ~5 s on CPU; re-loading it for every
  test would make the suite impractically slow.
* All numerical assertions use tolerances appropriate for float32 arithmetic.
* Tests are grouped into four concerns:
    1. Interface conformance  — BGEEmbedder satisfies the Embedder ABC
    2. Output shape & dtype   — vectors have the expected length and type
    3. Normalisation          — all output vectors have L2-norm ≈ 1.0
    4. Semantic correctness   — a query is closer to a relevant passage than
                                to an unrelated one (smoke-test the model)
    5. Error handling         — empty inputs raise ValueError
    6. Import boundary        — the class is reachable via app.retrieval
"""

import math
from typing import Generator

import pytest

from app.rag.embeddings.bge_embedding import BGEEmbedder
from app.rag.interfaces import Embedder

# ---------------------------------------------------------------------------
# Session fixture — load the model once for the whole test run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def embedder() -> Generator[BGEEmbedder, None, None]:
    """Single BGEEmbedder instance shared across the entire test session."""
    yield BGEEmbedder()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# 1. Interface conformance
# ---------------------------------------------------------------------------

class TestInterfaceConformance:
    def test_is_embedder_subclass(self):
        """BGEEmbedder must be a subclass of the Embedder ABC."""
        assert issubclass(BGEEmbedder, Embedder)

    def test_instance_is_embedder(self, embedder: BGEEmbedder):
        """A live BGEEmbedder instance must pass isinstance(Embedder)."""
        assert isinstance(embedder, Embedder)

    def test_has_embed_documents(self, embedder: BGEEmbedder):
        assert callable(embedder.embed_documents)

    def test_has_embed_query(self, embedder: BGEEmbedder):
        assert callable(embedder.embed_query)

    def test_has_dim_property(self, embedder: BGEEmbedder):
        assert hasattr(embedder, "dim")


# ---------------------------------------------------------------------------
# 2. Output shape & dtype
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_dim_is_384(self, embedder: BGEEmbedder):
        """bge-small-en-v1.5 must produce 384-dimensional vectors."""
        assert embedder.dim == 384

    def test_embed_query_length(self, embedder: BGEEmbedder):
        vec = embedder.embed_query("What causes hypertension?")
        assert len(vec) == 384

    def test_embed_documents_single(self, embedder: BGEEmbedder):
        vecs = embedder.embed_documents(["ACE inhibitors lower blood pressure."])
        assert len(vecs) == 1
        assert len(vecs[0]) == 384

    def test_embed_documents_batch(self, embedder: BGEEmbedder):
        texts = [
            "Hypertension is a risk factor for stroke.",
            "Diabetes mellitus affects insulin secretion.",
            "Aspirin inhibits COX-1 and COX-2 enzymes.",
        ]
        vecs = embedder.embed_documents(texts)
        assert len(vecs) == len(texts)
        assert all(len(v) == 384 for v in vecs)

    def test_embed_query_returns_list_of_floats(self, embedder: BGEEmbedder):
        vec = embedder.embed_query("fever and cough differential diagnosis")
        assert isinstance(vec, list)
        assert all(isinstance(x, float) for x in vec)

    def test_embed_documents_returns_list_of_lists(self, embedder: BGEEmbedder):
        vecs = embedder.embed_documents(["sample passage"])
        assert isinstance(vecs, list)
        assert isinstance(vecs[0], list)
        assert all(isinstance(x, float) for x in vecs[0])


# ---------------------------------------------------------------------------
# 3. Normalisation — every vector must have L2-norm ≈ 1.0
# ---------------------------------------------------------------------------

class TestNormalisation:
    TOLERANCE = 1e-5

    def test_query_norm_is_one(self, embedder: BGEEmbedder):
        vec = embedder.embed_query("first-line treatment for essential hypertension")
        assert abs(_l2_norm(vec) - 1.0) < self.TOLERANCE

    def test_document_norms_are_one(self, embedder: BGEEmbedder):
        texts = [
            "Beta-blockers reduce heart rate.",
            "Thiazide diuretics decrease plasma volume.",
            "Calcium channel blockers relax vascular smooth muscle.",
        ]
        vecs = embedder.embed_documents(texts)
        for i, v in enumerate(vecs):
            assert abs(_l2_norm(v) - 1.0) < self.TOLERANCE, (
                f"Document {i} norm = {_l2_norm(v):.8f}, expected ≈ 1.0"
            )

    def test_large_batch_norms(self, embedder: BGEEmbedder):
        """Normalisation must hold for batches larger than one."""
        texts = [f"Medical sentence number {i}." for i in range(10)]
        vecs = embedder.embed_documents(texts)
        for i, v in enumerate(vecs):
            assert abs(_l2_norm(v) - 1.0) < self.TOLERANCE


# ---------------------------------------------------------------------------
# 4. Semantic correctness — dot product (= cosine for normalised vectors)
# ---------------------------------------------------------------------------

class TestSemanticCorrectness:
    def test_relevant_passage_scores_higher_than_irrelevant(
        self, embedder: BGEEmbedder
    ):
        """A query about hypertension must be closer to a hypertension passage
        than to an unrelated passage about fracture management."""
        query = "What is the first-line treatment for essential hypertension?"
        relevant = (
            "The first-line treatment for essential hypertension includes "
            "ACE inhibitors, ARBs, calcium channel blockers, and thiazide diuretics."
        )
        irrelevant = (
            "Management of a closed femoral shaft fracture involves intramedullary "
            "nailing with reamed technique in skeletally mature patients."
        )
        q_vec = embedder.embed_query(query)
        doc_vecs = embedder.embed_documents([relevant, irrelevant])

        score_relevant = _dot(q_vec, doc_vecs[0])
        score_irrelevant = _dot(q_vec, doc_vecs[1])

        assert score_relevant > score_irrelevant, (
            f"Relevant score ({score_relevant:.4f}) should exceed "
            f"irrelevant score ({score_irrelevant:.4f})"
        )

    def test_identical_documents_have_score_near_one(self, embedder: BGEEmbedder):
        """Two identical passages must have cosine similarity ≈ 1.0."""
        text = "ACE inhibitors are recommended as first-line therapy for hypertension."
        vecs = embedder.embed_documents([text, text])
        score = _dot(vecs[0], vecs[1])
        assert score > 0.999, f"Identical passage similarity = {score:.6f}, expected ≈ 1.0"

    def test_query_and_matching_passage_score_above_threshold(
        self, embedder: BGEEmbedder
    ):
        """A query should score ≥ 0.5 against its direct answer passage."""
        query = "What drug class is used first for hypertension?"
        passage = "Thiazide diuretics, ACE inhibitors, and calcium channel blockers are all first-line antihypertensive drug classes."
        q_vec = embedder.embed_query(query)
        doc_vecs = embedder.embed_documents([passage])
        score = _dot(q_vec, doc_vecs[0])
        assert score >= 0.5, (
            f"Query-passage similarity = {score:.4f}, expected ≥ 0.5"
        )


# ---------------------------------------------------------------------------
# 5. Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_embed_documents_empty_list_raises(self, embedder: BGEEmbedder):
        with pytest.raises(ValueError, match="at least one text"):
            embedder.embed_documents([])

    def test_embed_query_empty_string_raises(self, embedder: BGEEmbedder):
        with pytest.raises(ValueError, match="non-empty query"):
            embedder.embed_query("")

    def test_embed_query_whitespace_only_raises(self, embedder: BGEEmbedder):
        with pytest.raises(ValueError, match="non-empty query"):
            embedder.embed_query("   ")


# ---------------------------------------------------------------------------
# 6. Import boundary — reachable via app.retrieval
# ---------------------------------------------------------------------------

class TestImportBoundary:
    def test_importable_from_retrieval_embeddings(self):
        from app.retrieval.embeddings import BGEEmbedder as B
        assert B is BGEEmbedder

    def test_importable_from_retrieval_package(self):
        from app.retrieval import BGEEmbedder as B
        assert B is BGEEmbedder

    def test_importable_from_rag_embeddings_package(self):
        from app.rag.embeddings import BGEEmbedder as B
        assert B is BGEEmbedder
