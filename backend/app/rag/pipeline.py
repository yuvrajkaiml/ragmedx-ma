"""
RAG retrieval pipeline for the RAGMedX-MA system.

Composes the retrieval-stage components into a single interface:

    User Query
        ↓
    Hybrid Retrieval
        ├── Query Expansion
        ├── Dense Retrieval (BGE-M3 + FAISS)
        ├── Sparse Retrieval (BM25)
        └── Reciprocal Rank Fusion
        ↓
    Cross-Encoder Reranking
        ↓
    Final Evidence Chunks

This module acts as the composition boundary for the retrieval
stage of RAGMedX-MA.

It intentionally does not implement:
    - LLM generation
    - multi-agent orchestration
    - answer verification
    - citation generation

Those responsibilities belong to later stages of the system.
"""

from __future__ import annotations

import time
from math import exp, isfinite
from pathlib import Path
from typing import Any, Sequence

from app.rag.chunking.models import MedicalChunk
from app.rag.citations.citation_generator import CitationGenerator
from app.rag.retrievers.hybrid_retriever import HybridRetriever
from app.rag.rerankers.cross_encoder_reranker import CrossEncoderReranker
from app.utils.logger import logger


class RAGRetrievalPipeline:
    """
    Compose hybrid retrieval and cross-encoder reranking.

    Pipeline:

        Query
          ↓
        HybridRetriever
          ↓
        Candidate Chunks
          ↓
        CrossEncoderReranker
          ↓
        Ranked Evidence
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        retrieval_top_k: int = 10,
        final_top_k: int = 5,
    ) -> None:
        """
        Initialize the retrieval pipeline.

        Args:
            hybrid_retriever:
                Configured hybrid retriever containing query
                expansion, dense retrieval, sparse retrieval,
                and RRF fusion.

            reranker:
                Cross-encoder reranker used to score the retrieved
                candidate chunks against the original query.

            retrieval_top_k:
                Number of candidates retrieved from the hybrid
                retrieval stage before reranking.

            final_top_k:
                Number of final evidence chunks returned after
                reranking.
        """

        if not isinstance(
            hybrid_retriever,
            HybridRetriever,
        ):
            raise TypeError(
                "hybrid_retriever must be a HybridRetriever."
            )

        if not isinstance(
            reranker,
            CrossEncoderReranker,
        ):
            raise TypeError(
                "reranker must be a CrossEncoderReranker."
            )

        if retrieval_top_k <= 0:
            raise ValueError(
                "retrieval_top_k must be greater than zero."
            )

        if final_top_k <= 0:
            raise ValueError(
                "final_top_k must be greater than zero."
            )

        if final_top_k > retrieval_top_k:
            raise ValueError(
                "final_top_k cannot be greater than "
                "retrieval_top_k."
            )

        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker

        self.retrieval_top_k = retrieval_top_k
        self.final_top_k = final_top_k

    # ------------------------------------------------------------------
    # INDEXING
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: Sequence[MedicalChunk],
    ) -> None:
        """
        Add medical chunks to the underlying hybrid retriever.

        The hybrid retriever handles indexing into both the dense
        and sparse retrieval systems.
        """

        if not chunks:
            return

        self.hybrid_retriever.add_chunks(chunks)

    # ------------------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        retrieval_top_k: int | None = None,
        final_top_k: int | None = None,
    ) -> list[tuple[MedicalChunk, float]]:
        """
        Execute the complete retrieval pipeline.

        Pipeline:

            Query
              ↓
            Query Expansion
              ↓
            Dense + Sparse Retrieval
              ↓
            RRF Fusion
              ↓
            Cross-Encoder Reranking
              ↓
            Final Evidence

        Args:
            query:
                Natural-language medical question.

            retrieval_top_k:
                Optional override for the number of candidates
                retrieved before reranking.

            final_top_k:
                Optional override for the number of final
                reranked results.

        Returns:
            List of:

                [(MedicalChunk, reranker_score), ...]

            ordered from highest relevance to lowest.
        """

        # --------------------------------------------------------------
        # Validate query
        # --------------------------------------------------------------

        if not isinstance(query, str):
            raise TypeError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        # --------------------------------------------------------------
        # Resolve configuration
        # --------------------------------------------------------------

        candidate_k = (
            self.retrieval_top_k
            if retrieval_top_k is None
            else retrieval_top_k
        )

        result_k = (
            self.final_top_k
            if final_top_k is None
            else final_top_k
        )

        if candidate_k <= 0:
            raise ValueError(
                "retrieval_top_k must be greater than zero."
            )

        if result_k <= 0:
            raise ValueError(
                "final_top_k must be greater than zero."
            )

        if result_k > candidate_k:
            raise ValueError(
                "final_top_k cannot be greater than "
                "retrieval_top_k."
            )

        # --------------------------------------------------------------
        # Hybrid retrieval
        # --------------------------------------------------------------

        hybrid_results = self.hybrid_retriever.retrieve(
            query=query,
            top_k=candidate_k,
        )

        if not hybrid_results:
            return []

        # --------------------------------------------------------------
        # Cross-encoder reranking
        # --------------------------------------------------------------

        reranked_results = self.reranker.rerank(
            query=query,
            results=hybrid_results,
            top_k=result_k,
        )

        return reranked_results

    # ------------------------------------------------------------------
    # INFORMATION
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """
        Return the number of indexed medical chunks.
        """

        return self.hybrid_retriever.size


class RAGPipeline:
    """Async evidence-provider facade over :class:`RAGRetrievalPipeline`.

    This preserves the public boundary consumed by ``RetrievalService`` and
    the agent orchestrator.  Retrieval itself remains entirely native: BGE-M3,
    ``MedicalChunk``, FAISS, BM25/RRF, and Sana's cross-encoder reranker.
    """

    def __init__(
        self,
        retrieval_pipeline: RAGRetrievalPipeline | None = None,
        top_k_final: int = 5,
    ) -> None:
        self._retrieval_pipeline = retrieval_pipeline
        self._top_k_final = top_k_final

    @classmethod
    def build(
        cls,
        index_dir: str | Path | None = None,
        model_name: str | None = None,
        top_k_retrieval: int = 20,
        top_k_final: int = 5,
    ) -> "RAGPipeline":
        """Load a native persisted index or return a safe empty facade."""
        from app.config import get_settings
        from app.rag.embeddings.bge_embedding import BGEEmbedding
        from app.rag.retrievers.dense_retriever import DenseRetriever
        from app.rag.retrievers.sparse_retriever import SparseRetriever
        from app.rag.vectorstore.faiss_store import FAISSVectorStore

        settings = get_settings()
        # Resolve relative paths against BASE_DIR (backend/) so the index is
        # found consistently regardless of the process working directory.
        from app.config import BASE_DIR
        if index_dir is not None:
            directory = Path(index_dir)
            if not directory.is_absolute():
                directory = BASE_DIR / directory
        else:
            directory = settings.faiss_index_dir
            if not directory.is_absolute():
                directory = BASE_DIR / directory
        if not (directory / "index.faiss").exists() or not (directory / "chunks.pkl").exists():
            return cls(top_k_final=top_k_final)

        try:
            store = FAISSVectorStore.load(directory)
            embedder = BGEEmbedding(model_name=model_name or BGEEmbedding.DEFAULT_MODEL)
            dense = DenseRetriever(embedder=embedder, vector_store=store)
            sparse = SparseRetriever()
            sparse.add_chunks(store.chunks)
            hybrid = HybridRetriever(
                dense_retriever=dense,
                sparse_retriever=sparse,
            )
            reranker = CrossEncoderReranker(device="cpu")
            native_pipeline = RAGRetrievalPipeline(
                hybrid_retriever=hybrid,
                reranker=reranker,
                retrieval_top_k=top_k_retrieval,
                final_top_k=top_k_final,
            )
            return cls(native_pipeline, top_k_final=top_k_final)
        except Exception as exc:
            if settings.llm_mode == "real":
                # In real mode a load failure must be loud: don't serve requests
                # with an empty retriever when a real index was expected.
                raise RuntimeError(
                    f"Failed to load FAISS index from '{directory}': {exc}. "
                    "Verify the index files are intact and all retrieval dependencies "
                    "(BGE-M3, bitsandbytes, CUDA) are installed."
                ) from exc
            # mock/dev mode: degrade to empty facade so dev workflows aren't broken.
            logger.warning(
                f"Failed to load FAISS index from '{directory}': {exc}. "
                "Falling back to empty retrieval facade (mock mode)."
            )
            return cls(top_k_final=top_k_final)

    @property
    def is_ready(self) -> bool:
        return self._retrieval_pipeline is not None

    @staticmethod
    def _normalize_cross_encoder_logit(score: Any) -> float:
        """Map native cross-encoder logits to the agent contract's [0, 1] range.

        ``CrossEncoderReranker`` returns the model's scalar relevance output,
        which is not constrained by its public interface.  Preserve that raw
        value separately and apply a documented sigmoid at the facade boundary
        before constructing ``EvidenceSnippet``-compatible dictionaries.
        """
        try:
            value = float(score)
        except (TypeError, ValueError):
            return 0.0
        if not isfinite(value):
            return 0.0
        if value >= 0:
            return 1.0 / (1.0 + exp(-value))
        positive = exp(value)
        return positive / (1.0 + positive)

    async def retrieve_evidence(
        self,
        query: str,
        optimized_query: str | None = None,
        max_iterations: int = 3,
        top_k_final: int | None = None,
        **_: Any,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return native retrieval results in the agent evidence schema."""
        search_query = (optimized_query or query or "").strip()
        if not search_query:
            return [], [{"iteration": 1, "query_used": "", "count": 0, "status": "empty_query"}]
        if self._retrieval_pipeline is None:
            return [], [{"iteration": 1, "query_used": search_query, "count": 0, "status": "no_retriever"}]

        started = time.perf_counter()
        try:
            results = self._retrieval_pipeline.retrieve(
                search_query,
                final_top_k=top_k_final or self._top_k_final,
            )
            citations_by_chunk = {
                citation.chunk_id: citation.citation_id
                for citation in CitationGenerator().generate(results)
            }
            evidence = [
                {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "source_type": chunk.source,
                    "title": chunk.title or "",
                    "cited_text": chunk.content,
                    "cross_encoder_logit": float(score),
                    "relevance_score": self._normalize_cross_encoder_logit(score),
                    "cross_encoder_score": self._normalize_cross_encoder_logit(score),
                    "evidence_quality_score": self._normalize_cross_encoder_logit(score),
                    # CitationGenerator is the authority for response-visible
                    # markers. The marker maps to this exact reranked chunk.
                    "citation_marker": citations_by_chunk.get(chunk.id, f"[Doc {index}]"),
                    "journal": chunk.journal,
                    "publication_year": chunk.publication_year,
                    "doi": chunk.metadata.get("doi"),
                    "url": chunk.metadata.get("url") or chunk.metadata.get("pmid_url"),
                }
                for index, (chunk, score) in enumerate(results, start=1)
            ]
            return evidence, [{
                "iteration": 1,
                "query_used": search_query,
                "count": len(evidence),
                "status": "completed",
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            }]
        except Exception as exc:
            return [], [{
                "iteration": 1,
                "query_used": search_query,
                "count": 0,
                "status": "error",
                "error": str(exc),
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            }]
