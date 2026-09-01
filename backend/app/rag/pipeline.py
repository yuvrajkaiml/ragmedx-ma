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

from typing import Sequence

from app.rag.chunking.models import MedicalChunk
from app.rag.retrievers.hybrid_retriever import HybridRetriever
from app.rag.rerankers.cross_encoder_reranker import CrossEncoderReranker


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