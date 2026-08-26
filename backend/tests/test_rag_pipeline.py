"""Tests for RAGPipeline (app.rag.pipeline).

Test design
-----------
* Unit tests use fake HybridRetriever and CrossEncoderReranker so no model
  loads are needed.  The fakes are pure Python objects that return
  deterministic results.
* The integration class (TestRAGPipelineIntegration) loads real BGE + cross-
  encoder models and is marked @pytest.mark.integration.
* All tests verify the evidence dict schema that the orchestrator agents
  consume -- not just that the pipeline does not crash.
"""

import asyncio
import uuid
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.rag.pipeline import RAGPipeline
from app.schemas.retrieval import DocumentChunk, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_EVIDENCE_KEYS = {
    "chunk_id", "document_id", "source_type", "title", "cited_text",
    "relevance_score", "cross_encoder_score", "evidence_quality_score",
    "citation_marker",
}


def _chunk(index: int, text: str = "", source_type: str = "medqa",
           meta: Optional[dict] = None) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        source_type=source_type,
        title=f"Chunk {index}",
        text=text or f"medical evidence chunk {index}",
        chunk_index=index,
        metadata=meta or {},
    )


def _result(index: int, text: str = "", score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(chunk=_chunk(index, text), relevance_score=score)


class _FakeHybrid:
    """Returns a fixed list of RetrievalResults regardless of query."""
    def __init__(self, results: List[RetrievalResult]):
        self._results = results
        self.last_query: str = ""
        self.last_top_k: int = 0

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        self.last_query = query
        self.last_top_k = top_k
        return self._results[:top_k]


class _FakeReranker:
    """Returns results in reversed order (to verify reranking is called)."""
    def __init__(self):
        self.called_with_query: str = ""
        self.called_with_top_k: Optional[int] = None

    def rerank(self, query: str, results: List[RetrievalResult],
               top_k: Optional[int] = None) -> List[RetrievalResult]:
        self.called_with_query = query
        self.called_with_top_k = top_k
        ordered = sorted(results, key=lambda r: r.relevance_score, reverse=True)
        return ordered[:top_k] if top_k else ordered


def _make_pipeline(
    n_results: int = 3,
    with_reranker: bool = True,
    top_k_retrieval: int = 5,
    top_k_final: int = 3,
) -> tuple:
    results = [_result(i, score=1.0 - i * 0.1) for i in range(n_results)]
    hybrid = _FakeHybrid(results)
    reranker = _FakeReranker() if with_reranker else None
    pipeline = RAGPipeline(
        hybrid_retriever=hybrid,
        reranker=reranker,
        top_k_retrieval=top_k_retrieval,
        top_k_final=top_k_final,
    )
    return pipeline, hybrid, reranker


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_is_ready_with_hybrid(self):
        p, _, _ = _make_pipeline()
        assert p.is_ready is True

    def test_not_ready_without_hybrid(self):
        p = RAGPipeline(hybrid_retriever=None)
        assert p.is_ready is False

    def test_no_reranker_is_valid(self):
        p, _, _ = _make_pipeline(with_reranker=False)
        assert p.is_ready is True

    def test_both_none_is_degraded(self):
        p = RAGPipeline()
        assert p.is_ready is False


# ---------------------------------------------------------------------------
# 2. retrieve_evidence -- basic contract
# ---------------------------------------------------------------------------

class TestRetrieveEvidenceContract:
    def test_returns_tuple(self):
        p, _, _ = _make_pipeline()
        result = asyncio.run(p.retrieve_evidence("hypertension"))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_evidence_is_list(self):
        p, _, _ = _make_pipeline()
        evs, _ = asyncio.run(p.retrieve_evidence("hypertension"))
        assert isinstance(evs, list)

    def test_iterations_is_list(self):
        p, _, _ = _make_pipeline()
        _, iters = asyncio.run(p.retrieve_evidence("hypertension"))
        assert isinstance(iters, list)
        assert len(iters) == 1

    def test_iteration_has_status_completed(self):
        p, _, _ = _make_pipeline()
        _, iters = asyncio.run(p.retrieve_evidence("hypertension"))
        assert iters[0]["status"] == "completed"

    def test_iteration_count_matches_evidence(self):
        p, _, _ = _make_pipeline(n_results=3, top_k_final=2)
        evs, iters = asyncio.run(p.retrieve_evidence("query"))
        assert iters[0]["count"] == len(evs)

    def test_query_passed_to_hybrid(self):
        p, hybrid, _ = _make_pipeline()
        asyncio.run(p.retrieve_evidence("specific query text"))
        assert hybrid.last_query == "specific query text"

    def test_optimized_query_preferred(self):
        p, hybrid, _ = _make_pipeline()
        asyncio.run(p.retrieve_evidence("original", optimized_query="expanded query"))
        assert hybrid.last_query == "expanded query"


# ---------------------------------------------------------------------------
# 3. Evidence dict schema
# ---------------------------------------------------------------------------

class TestEvidenceDictSchema:
    def test_all_required_keys_present(self):
        p, _, _ = _make_pipeline(n_results=2, top_k_final=2)
        evs, _ = asyncio.run(p.retrieve_evidence("query"))
        for ev in evs:
            missing = _REQUIRED_EVIDENCE_KEYS - ev.keys()
            assert not missing, f"Missing evidence keys: {missing}"

    def test_citation_markers_sequential(self):
        p, _, _ = _make_pipeline(n_results=3, top_k_final=3)
        evs, _ = asyncio.run(p.retrieve_evidence("query"))
        for i, ev in enumerate(evs, start=1):
            assert ev["citation_marker"] == f"[E{i}]"

    def test_scores_in_range(self):
        p, _, _ = _make_pipeline(n_results=3, top_k_final=3)
        evs, _ = asyncio.run(p.retrieve_evidence("query"))
        for ev in evs:
            assert 0.0 <= ev["relevance_score"] <= 1.0
            assert 0.0 <= ev["cross_encoder_score"] <= 1.0
            assert 0.0 <= ev["evidence_quality_score"] <= 1.0

    def test_chunk_id_is_string(self):
        p, _, _ = _make_pipeline(n_results=1, top_k_final=1)
        evs, _ = asyncio.run(p.retrieve_evidence("query"))
        assert isinstance(evs[0]["chunk_id"], str)

    def test_metadata_fields_propagated(self):
        chunk = _chunk(0, "passage about ACE inhibitors",
                       meta={"journal": "Lancet", "publication_year": "2023",
                             "doi": "10.xxxx", "url": "https://example.com"})
        result = RetrievalResult(chunk=chunk, relevance_score=0.9)
        hybrid = _FakeHybrid([result])
        p = RAGPipeline(hybrid_retriever=hybrid, reranker=None,
                        top_k_retrieval=5, top_k_final=1)
        evs, _ = asyncio.run(p.retrieve_evidence("query"))
        assert evs[0]["journal"] == "Lancet"
        assert evs[0]["publication_year"] == 2023
        assert evs[0]["doi"] == "10.xxxx"
        assert evs[0]["url"] == "https://example.com"

    def test_missing_metadata_fields_are_none(self):
        p, _, _ = _make_pipeline(n_results=1, top_k_final=1)
        evs, _ = asyncio.run(p.retrieve_evidence("query"))
        assert evs[0]["journal"] is None
        assert evs[0]["publication_year"] is None


# ---------------------------------------------------------------------------
# 4. Empty / fallback behaviour
# ---------------------------------------------------------------------------

class TestFallbackBehaviour:
    def test_empty_query_returns_empty_evidence(self):
        p, _, _ = _make_pipeline()
        evs, iters = asyncio.run(p.retrieve_evidence(""))
        assert evs == []
        assert iters[0]["status"] == "empty_query"

    def test_whitespace_query_returns_empty_evidence(self):
        p, _, _ = _make_pipeline()
        evs, iters = asyncio.run(p.retrieve_evidence("   "))
        assert evs == []
        assert iters[0]["status"] == "empty_query"

    def test_no_retriever_returns_empty_with_status(self):
        p = RAGPipeline(hybrid_retriever=None)
        evs, iters = asyncio.run(p.retrieve_evidence("query"))
        assert evs == []
        assert iters[0]["status"] == "no_retriever"

    def test_no_retriever_does_not_raise(self):
        p = RAGPipeline(hybrid_retriever=None)
        evs, iters = asyncio.run(p.retrieve_evidence("query"))
        assert isinstance(evs, list)

    def test_empty_hybrid_results_returns_empty_evidence(self):
        hybrid = _FakeHybrid([])
        p = RAGPipeline(hybrid_retriever=hybrid, top_k_final=5)
        evs, iters = asyncio.run(p.retrieve_evidence("query"))
        assert evs == []
        assert iters[0]["count"] == 0


# ---------------------------------------------------------------------------
# 5. top_k behaviour
# ---------------------------------------------------------------------------

class TestTopK:
    def test_top_k_final_limits_output(self):
        p, _, _ = _make_pipeline(n_results=5, top_k_final=2)
        evs, _ = asyncio.run(p.retrieve_evidence("query"))
        assert len(evs) == 2

    def test_top_k_final_override(self):
        p, _, _ = _make_pipeline(n_results=5, top_k_final=3)
        evs, _ = asyncio.run(p.retrieve_evidence("query", top_k_final=1))
        assert len(evs) == 1

    def test_top_k_larger_than_results_returns_all(self):
        p, _, _ = _make_pipeline(n_results=2, top_k_final=10)
        evs, _ = asyncio.run(p.retrieve_evidence("query"))
        assert len(evs) == 2

    def test_top_k_retrieval_passed_to_hybrid(self):
        p, hybrid, _ = _make_pipeline(n_results=5, top_k_retrieval=7, top_k_final=3)
        asyncio.run(p.retrieve_evidence("query"))
        assert hybrid.last_top_k == 7


# ---------------------------------------------------------------------------
# 6. Reranker wiring
# ---------------------------------------------------------------------------

class TestRerankerWiring:
    def test_reranker_called_when_present(self):
        results = [_result(i, score=float(i) / 10) for i in range(3)]
        hybrid = _FakeHybrid(results)
        reranker = _FakeReranker()
        p = RAGPipeline(hybrid_retriever=hybrid, reranker=reranker,
                        top_k_retrieval=3, top_k_final=2)
        asyncio.run(p.retrieve_evidence("test query"))
        assert reranker.called_with_query == "test query"
        assert reranker.called_with_top_k == 2

    def test_no_reranker_uses_first_stage_order(self):
        results = [_result(i, score=1.0 - i * 0.1) for i in range(3)]
        hybrid = _FakeHybrid(results)
        p = RAGPipeline(hybrid_retriever=hybrid, reranker=None,
                        top_k_retrieval=3, top_k_final=3)
        evs, _ = asyncio.run(p.retrieve_evidence("query"))
        scores = [ev["relevance_score"] for ev in evs]
        assert scores == sorted(scores, reverse=True)

    def test_reranker_scores_reflected_in_output(self):
        """Reranker re-orders results; output scores must match reranker output."""
        original = [_result(0, score=0.3), _result(1, score=0.9)]
        hybrid = _FakeHybrid(original)
        # FakeReranker sorts descending -- chunk 1 (0.9) should be first
        reranker = _FakeReranker()
        p = RAGPipeline(hybrid_retriever=hybrid, reranker=reranker,
                        top_k_retrieval=2, top_k_final=2)
        evs, _ = asyncio.run(p.retrieve_evidence("query"))
        assert evs[0]["chunk_id"] == str(original[1].chunk.chunk_id)


# ---------------------------------------------------------------------------
# 7. Import boundary
# ---------------------------------------------------------------------------

class TestImportBoundary:
    def test_importable_from_rag_pipeline(self):
        from app.rag.pipeline import RAGPipeline as P
        assert P is RAGPipeline

    def test_importable_from_retrieval_package(self):
        from app.retrieval import RAGPipeline as P
        assert P is RAGPipeline

    def test_all_prior_components_still_importable(self):
        from app.retrieval import (
            DenseRetriever, SparseRetriever, HybridRetriever,
            CrossEncoderReranker, BGEEmbedder, FAISSVectorStore,
        )
        assert all(x is not None for x in [
            DenseRetriever, SparseRetriever, HybridRetriever,
            CrossEncoderReranker, BGEEmbedder, FAISSVectorStore,
        ])


# ---------------------------------------------------------------------------
# 8. Build factory (no-index path)
# ---------------------------------------------------------------------------

class TestBuildFactory:
    def test_build_without_index_returns_degraded_pipeline(self, tmp_path):
        p = RAGPipeline.build(index_dir=tmp_path)
        assert p.is_ready is False

    def test_build_returns_rag_pipeline(self, tmp_path):
        p = RAGPipeline.build(index_dir=tmp_path)
        assert isinstance(p, RAGPipeline)

    def test_degraded_pipeline_returns_empty_no_exception(self, tmp_path):
        p = RAGPipeline.build(index_dir=tmp_path)
        evs, iters = asyncio.run(p.retrieve_evidence("query"))
        assert evs == []
        assert iters[0]["status"] == "no_retriever"


# ---------------------------------------------------------------------------
# 9. Orchestrator integration (no model loading)
# ---------------------------------------------------------------------------

class TestOrchestratorIntegration:
    """Verify the pipeline can be injected into MultiAgentOrchestrator
    and the agent pipeline runs end-to-end with real evidence dicts."""

    def test_pipeline_injectable_as_retrieval_provider(self):
        """Pipeline's retrieve_evidence method must be accepted by the
        orchestrator constructor without error."""
        from app.agents.orchestrator import MultiAgentOrchestrator
        p, _, _ = _make_pipeline(n_results=2, top_k_final=2)
        # Capture the bound method once; Python creates a new object on each
        # attribute access so identity (is) is unreliable for bound methods.
        provider = p.retrieve_evidence
        orch = MultiAgentOrchestrator(retrieval_provider=provider)
        assert orch.retrieval_provider == provider
        assert callable(orch.retrieval_provider)

    def test_evidence_dicts_pass_orchestrator_normalisation(self):
        """The normaliser in _normalize_and_validate_evidence must accept
        evidence dicts produced by the pipeline without raising."""
        from app.agents.orchestrator import MultiAgentOrchestrator
        from app.agents.state import AgentState, create_deterministic_uuid

        p, _, _ = _make_pipeline(n_results=2, top_k_final=2)
        # Build evidence dicts directly (bypassing the async call)
        from app.schemas.retrieval import RetrievalResult
        results = [_result(i, f"evidence text {i}") for i in range(2)]
        evidence_dicts = [
            RAGPipeline._result_to_evidence_dict(r, i)
            for i, r in enumerate(results, start=1)
        ]

        state = AgentState(
            query_id=create_deterministic_uuid("norm_test"),
            original_question="What treats hypertension?",
            evidence=evidence_dicts,
        )
        orch = MultiAgentOrchestrator.__new__(MultiAgentOrchestrator)
        orch._normalize_and_validate_evidence(state)
        assert len(state.evidence) == 2
        assert state.evidence[0]["citation_marker"].startswith("[E")
        assert state.evidence[0]["chunk_id"] is not None


# ---------------------------------------------------------------------------
# 10. BGE + FAISS + CrossEncoder integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRAGPipelineIntegration:
    """End-to-end: BGEEmbedder -> HybridRetriever -> CrossEncoderReranker.

    Uses cached models.  Skip with:  pytest -m "not integration"
    """

    _passages = [
        "ACE inhibitors are first-line for essential hypertension.",
        "Thiazide diuretics reduce blood pressure by decreasing plasma volume.",
        "Calcium channel blockers relax vascular smooth muscle.",
        "Beta-blockers reduce heart rate in cardiac patients.",
        "Intramedullary nailing is standard for femoral shaft fractures.",
    ]

    @pytest.fixture(scope="class")
    def live_pipeline(self, tmp_path_factory) -> RAGPipeline:
        from app.rag.embeddings.bge_embedding import BGEEmbedder
        from app.rag.rerankers.cross_encoder import CrossEncoderReranker
        from app.rag.retrievers.dense_retriever import DenseRetriever
        from app.rag.retrievers.hybrid_retriever import HybridRetriever
        from app.rag.retrievers.sparse_retriever import SparseRetriever
        from app.rag.vectorstore.faiss_store import FAISSVectorStore

        passages = self.__class__._passages
        chunks = [_chunk(i, t, source_type="medqa") for i, t in enumerate(passages)]

        embedder = BGEEmbedder()
        store = FAISSVectorStore(dim=embedder.dim)
        store.upsert(chunks, embedder.embed_documents(passages))

        dense = DenseRetriever(embedder=embedder, store=store, default_top_k=5)
        sparse = SparseRetriever(default_top_k=5)
        sparse.index(chunks)
        hybrid = HybridRetriever(dense=dense, sparse=sparse, default_top_k=5)
        reranker = CrossEncoderReranker()

        return RAGPipeline(
            hybrid_retriever=hybrid,
            reranker=reranker,
            top_k_retrieval=5,
            top_k_final=3,
        )

    def test_returns_evidence(self, live_pipeline):
        evs, iters = asyncio.run(
            live_pipeline.retrieve_evidence("first-line treatment for hypertension")
        )
        assert len(evs) > 0
        assert iters[0]["status"] == "completed"

    def test_evidence_schema_complete(self, live_pipeline):
        evs, _ = asyncio.run(
            live_pipeline.retrieve_evidence("hypertension therapy")
        )
        for ev in evs:
            assert _REQUIRED_EVIDENCE_KEYS.issubset(ev.keys())

    def test_citation_markers_sequential(self, live_pipeline):
        evs, _ = asyncio.run(
            live_pipeline.retrieve_evidence("blood pressure medication")
        )
        for i, ev in enumerate(evs, start=1):
            assert ev["citation_marker"] == f"[E{i}]"

    def test_scores_in_range(self, live_pipeline):
        evs, _ = asyncio.run(
            live_pipeline.retrieve_evidence("antihypertensive drug")
        )
        for ev in evs:
            assert 0.0 <= ev["relevance_score"] <= 1.0

    def test_hypertension_evidence_returned_for_hypertension_query(self, live_pipeline):
        evs, _ = asyncio.run(
            live_pipeline.retrieve_evidence(
                "What is the first-line pharmacotherapy for essential hypertension?"
            )
        )
        assert len(evs) > 0
        indices = {ev["cited_text"] for ev in evs}
        hyp_passages = set(self.__class__._passages[:4])
        assert indices & hyp_passages, "No hypertension passage in results"

    def test_top_k_final_respected(self, live_pipeline):
        for k in (1, 2):
            evs, _ = asyncio.run(
                live_pipeline.retrieve_evidence("medication therapy", top_k_final=k)
            )
            assert len(evs) == k

    def test_empty_query_returns_empty(self, live_pipeline):
        evs, iters = asyncio.run(live_pipeline.retrieve_evidence(""))
        assert evs == []
        assert iters[0]["status"] == "empty_query"

