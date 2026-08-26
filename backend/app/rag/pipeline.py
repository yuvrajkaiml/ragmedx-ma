"""Medical RAG pipeline — composes retrieval, reranking, and evidence formatting.

This module is the single composition boundary between the retrieval stack
(``BGEEmbedder`` -> ``FAISSVectorStore`` -> ``HybridRetriever`` ->
``CrossEncoderReranker``) and the multi-agent orchestrator.

The orchestrator expects a *retrieval provider*: an async callable matching::

    async def retrieve_evidence(
        query: str,
        optimized_query: Optional[str] = None,
        max_iterations: int = 3,
        **kwargs,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]

This is a drop-in replacement for ``app.llm.temporary_retrieval_stub.retrieve_evidence``.

Evidence dict schema (matches what the agents consume)::

    {
        "chunk_id":               str   # UUID string
        "document_id":            str   # UUID string
        "source_type":            str
        "title":                  str
        "cited_text":             str   # chunk.text
        "relevance_score":        float # 0-1, from cross-encoder
        "cross_encoder_score":    float # same
        "evidence_quality_score": float # same
        "citation_marker":        str   # "[E1]", "[E2]", ...
        "journal":                Optional[str]
        "publication_year":       Optional[int]
        "doi":                    Optional[str]
        "url":                    Optional[str]
    }

Graceful degradation
--------------------
``RAGPipeline`` accepts ``None`` for any component.  ``retrieve_evidence``
returns empty results (not an error) when no retriever is configured, so the
orchestrator falls through to its abstention path cleanly.

Usage::

    from app.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline.build()   # loads saved FAISS index if present
    orchestrator = MultiAgentOrchestrator(
        retrieval_provider=pipeline.retrieve_evidence,
    )
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.utils.logger import logger


class RAGPipeline:
    """Compose hybrid retrieval + cross-encoder reranking into a single callable.

    Parameters
    ----------
    hybrid_retriever:
        Object with ``retrieve(query, top_k) -> list[RetrievalResult]``.
        Pass ``None`` to operate in fallback (empty-result) mode.
    reranker:
        Object with ``rerank(query, results, top_k) -> list[RetrievalResult]``.
        Pass ``None`` to skip reranking (first-stage order is preserved).
    top_k_retrieval:
        Candidates fetched from the hybrid retriever.  Default 20.
    top_k_final:
        Evidence items returned after reranking.  Default 5.
    """

    def __init__(
        self,
        hybrid_retriever=None,
        reranker=None,
        top_k_retrieval: int = 20,
        top_k_final: int = 5,
    ) -> None:
        self._hybrid = hybrid_retriever
        self._reranker = reranker
        self._top_k_retrieval = top_k_retrieval
        self._top_k_final = top_k_final

        logger.info(
            f"RAGPipeline initialised -- "
            f"hybrid={'yes' if self._hybrid else 'no'}, "
            f"reranker={'yes' if self._reranker else 'no'}, "
            f"top_k_retrieval={top_k_retrieval}, top_k_final={top_k_final}, "
            f"ready={self.is_ready}"
        )

    # ------------------------------------------------------------------
    # Public retrieval callable -- drop-in for the stub
    # ------------------------------------------------------------------

    async def retrieve_evidence(
        self,
        query: str,
        optimized_query: Optional[str] = None,
        max_iterations: int = 3,
        top_k_initial: int = 50,
        top_k_final: Optional[int] = None,
        enable_contradiction_check: bool = True,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Retrieve and rerank evidence for *query*.

        Matches the signature of ``temporary_retrieval_stub.retrieve_evidence``
        for drop-in injection into the orchestrator.
        """
        effective_top_k = top_k_final if top_k_final is not None else self._top_k_final
        search_query = str(optimized_query or query or "").strip()

        if not search_query:
            return [], [{"iteration": 1, "query_used": "", "count": 0,
                         "status": "empty_query"}]

        if self._hybrid is None:
            logger.warning(
                "RAGPipeline.retrieve_evidence: no hybrid retriever configured -- "
                "returning empty results."
            )
            return [], [{"iteration": 1, "query_used": search_query,
                         "count": 0, "status": "no_retriever"}]

        t0 = time.perf_counter()

        try:
            # Stage 1 -- hybrid retrieval (dense + sparse -> RRF)
            candidates = self._hybrid.retrieve(
                search_query,
                top_k=self._top_k_retrieval,
            )

            # Stage 2 -- cross-encoder reranking (optional)
            if self._reranker is not None and candidates:
                candidates = self._reranker.rerank(
                    search_query,
                    candidates,
                    top_k=effective_top_k,
                )
            else:
                candidates = candidates[:effective_top_k]

            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            evidence_dicts = [
                self._result_to_evidence_dict(result, index)
                for index, result in enumerate(candidates, start=1)
            ]

            iteration_meta = [{
                "iteration": 1,
                "query_used": search_query,
                "count": len(evidence_dicts),
                "status": "completed",
                "contradiction_detected": False,
                "elapsed_ms": elapsed_ms,
            }]

            logger.info(
                f"RAGPipeline.retrieve_evidence: '{search_query[:60]}' -> "
                f"{len(evidence_dicts)} results ({elapsed_ms}ms)"
            )
            return evidence_dicts, iteration_meta

        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            logger.error(f"RAGPipeline.retrieve_evidence failed: {exc}")
            return [], [{
                "iteration": 1,
                "query_used": search_query,
                "count": 0,
                "status": "error",
                "error": str(exc),
                "elapsed_ms": elapsed_ms,
            }]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        index_dir=None,
        model_name: Optional[str] = None,
        top_k_retrieval: int = 20,
        top_k_final: int = 5,
    ) -> "RAGPipeline":
        """Construct a fully wired pipeline from a persisted FAISS index.

        If the index does not exist, returns a degraded pipeline that returns
        empty results.  No exception is raised so the agent pipeline starts
        cleanly even before any documents are indexed.
        """
        from pathlib import Path
        from app.config import FAISS_INDEX_DIR

        if index_dir is None:
            index_dir = FAISS_INDEX_DIR

        index_dir = Path(index_dir)
        index_file = index_dir / "index.faiss"

        if not index_file.exists():
            logger.warning(
                f"RAGPipeline.build: FAISS index not found at {index_dir} -- "
                "pipeline will return empty results until the index is built."
            )
            return cls(
                hybrid_retriever=None,
                reranker=None,
                top_k_retrieval=top_k_retrieval,
                top_k_final=top_k_final,
            )

        try:
            from app.rag.embeddings.bge_embedding import BGEEmbedder
            from app.rag.rerankers.cross_encoder import CrossEncoderReranker
            from app.rag.retrievers.dense_retriever import DenseRetriever
            from app.rag.retrievers.hybrid_retriever import HybridRetriever
            from app.rag.retrievers.sparse_retriever import SparseRetriever
            from app.rag.vectorstore.faiss_store import FAISSVectorStore
            from app.schemas.retrieval import DocumentChunk

            embedder = BGEEmbedder(model_name=model_name)
            store = FAISSVectorStore.load(index_dir)

            dense = DenseRetriever(
                embedder=embedder,
                store=store,
                default_top_k=top_k_retrieval,
            )

            sparse = SparseRetriever(default_top_k=top_k_retrieval)
            if store.ntotal > 0:
                chunks = [DocumentChunk(**c) for c in store._chunks]
                sparse.index(chunks)

            hybrid = HybridRetriever(
                dense=dense,
                sparse=sparse,
                default_top_k=top_k_retrieval,
            )
            reranker = CrossEncoderReranker()

            logger.info(
                f"RAGPipeline.build: loaded {store.ntotal} vectors -- "
                f"BGE + HybridRetriever + CrossEncoderReranker ready."
            )
            return cls(
                hybrid_retriever=hybrid,
                reranker=reranker,
                top_k_retrieval=top_k_retrieval,
                top_k_final=top_k_final,
            )

        except Exception as exc:
            logger.error(f"RAGPipeline.build failed: {exc}")
            return cls(
                hybrid_retriever=None,
                reranker=None,
                top_k_retrieval=top_k_retrieval,
                top_k_final=top_k_final,
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True when a hybrid retriever is available."""
        return self._hybrid is not None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _result_to_evidence_dict(result, index: int) -> Dict[str, Any]:
        """Convert a RetrievalResult to the evidence dict schema agents expect."""
        chunk = result.chunk
        score = result.relevance_score
        return {
            "chunk_id":               str(chunk.chunk_id),
            "document_id":            str(chunk.document_id),
            "source_type":            chunk.source_type,
            "title":                  chunk.title,
            "cited_text":             chunk.text,
            "relevance_score":        score,
            "cross_encoder_score":    score,
            "evidence_quality_score": score,
            "citation_marker":        f"[E{index}]",
            "journal":                chunk.metadata.get("journal"),
            "publication_year":       (
                int(chunk.metadata["publication_year"])
                if "publication_year" in chunk.metadata else None
            ),
            "doi":                    chunk.metadata.get("doi"),
            "url":                    chunk.metadata.get("url"),
        }
