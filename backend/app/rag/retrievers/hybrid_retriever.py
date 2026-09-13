"""
Hybrid retrieval for the RAGMedX-MA retrieval engine.

Combines:
    - Dense semantic retrieval (BGE-M3 + FAISS)
    - Sparse lexical retrieval (BM25)
    - Query expansion
    - Reciprocal Rank Fusion (RRF)
    - Optional cross-encoder reranking

Pipeline:

    Original Query
          ↓
    Query Expansion
          ↓
    Dense + Sparse Retrieval
          ↓
    RRF Fusion
          ↓
    Candidate Pool
          ↓
    Cross-Encoder Reranking (optional)
          ↓
    Final Ranked Results

This component is intentionally independent of the
multi-agent and LLM orchestration layer.
"""

from __future__ import annotations

from typing import Sequence

from app.rag.chunking.models import MedicalChunk
from app.rag.retrievers.dense_retriever import DenseRetriever
from app.rag.retrievers.sparse_retriever import SparseRetriever
from app.rag.retrievers.query_expander import QueryExpander
from app.rag.rerankers.cross_encoder_reranker import CrossEncoderReranker


class HybridRetriever:
    """
    Hybrid retriever combining:

        Query Expansion
            ↓
        Dense Retrieval
            +
        Sparse Retrieval
            ↓
        RRF
            ↓
        Optional Cross-Encoder Reranking

    RRF:

        RRF(d) = Σ 1 / (k + rank(d))

    where:
        k is the RRF constant.
        rank(d) is the rank of document d in a retrieval list.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        query_expander: QueryExpander | None = None,
        reranker: CrossEncoderReranker | None = None,
        rrf_k: int = 60,
    ) -> None:

        if rrf_k <= 0:
            raise ValueError(
                "rrf_k must be greater than zero."
            )

        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever

        self.query_expander = (
            query_expander
            if query_expander is not None
            else QueryExpander()
        )

        self.reranker = reranker

        self.rrf_k = rrf_k

    # ------------------------------------------------------------------
    # INDEXING
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: Sequence[MedicalChunk],
    ) -> None:
        """
        Add chunks to both dense and sparse indexes.
        """

        if not chunks:
            return

        self.dense_retriever.add_chunks(chunks)
        self.sparse_retriever.add_chunks(chunks)

    # ------------------------------------------------------------------
    # RRF
    # ------------------------------------------------------------------

    def _rrf_score(
        self,
        rank: int,
    ) -> float:
        """
        Calculate the RRF contribution for one ranked result.

        Rank is 1-based.
        """

        return 1.0 / (
            self.rrf_k + rank
        )

    def _fuse_results(
        self,
        retrieval_lists: Sequence[
            list[tuple[MedicalChunk, float]]
        ],
    ) -> list[tuple[MedicalChunk, float]]:
        """
        Fuse multiple ranked retrieval lists using RRF.
        """

        scores: dict[str, float] = {}
        chunks: dict[str, MedicalChunk] = {}

        for results in retrieval_lists:

            for rank, (chunk, _) in enumerate(
                results,
                start=1,
            ):
                chunk_key = chunk.id

                chunks[chunk_key] = chunk

                scores[chunk_key] = (
                    scores.get(chunk_key, 0.0)
                    + self._rrf_score(rank)
                )

        ranked_ids = sorted(
            scores,
            key=lambda chunk_id: scores[chunk_id],
            reverse=True,
        )

        return [
            (
                chunks[chunk_id],
                float(scores[chunk_id]),
            )
            for chunk_id in ranked_ids
        ]

    # ------------------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        rerank_top_k: int | None = None,
    ) -> list[tuple[MedicalChunk, float]]:
        """
        Perform query-expanded hybrid retrieval with optional
        cross-encoder reranking.

        Args:
            query:
                Natural-language medical question.

            top_k:
                Number of final results returned.

            dense_top_k:
                Number of dense candidates retrieved for each
                expanded query.

            sparse_top_k:
                Number of sparse candidates retrieved for each
                expanded query.

            rerank_top_k:
                Number of RRF candidates passed to the cross-encoder.
                If None, defaults to max(top_k, 10).

        Returns:
            List of (MedicalChunk, score) tuples.

            If reranking is enabled, score is the cross-encoder
            relevance score.

            Otherwise, score is the RRF score.
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

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        # --------------------------------------------------------------
        # Configure candidate counts
        # --------------------------------------------------------------

        if dense_top_k is None:
            dense_top_k = max(
                top_k,
                10,
            )

        if sparse_top_k is None:
            sparse_top_k = max(
                top_k,
                10,
            )

        if rerank_top_k is None:
            rerank_top_k = max(
                top_k,
                10,
            )

        if dense_top_k <= 0:
            raise ValueError(
                "dense_top_k must be greater than zero."
            )

        if sparse_top_k <= 0:
            raise ValueError(
                "sparse_top_k must be greater than zero."
            )

        if rerank_top_k <= 0:
            raise ValueError(
                "rerank_top_k must be greater than zero."
            )

        # --------------------------------------------------------------
        # Query expansion
        # --------------------------------------------------------------

        expanded_queries = self.query_expander.expand(
            query
        )

        # --------------------------------------------------------------
        # Retrieve for every query variant
        # --------------------------------------------------------------

        retrieval_lists: list[
            list[tuple[MedicalChunk, float]]
        ] = []

        for expanded_query in expanded_queries:

            dense_results = (
                self.dense_retriever.retrieve(
                    query=expanded_query,
                    top_k=dense_top_k,
                )
            )

            retrieval_lists.append(
                dense_results
            )

            sparse_results = (
                self.sparse_retriever.retrieve(
                    query=expanded_query,
                    top_k=sparse_top_k,
                )
            )

            retrieval_lists.append(
                sparse_results
            )

        # --------------------------------------------------------------
        # RRF fusion
        # --------------------------------------------------------------

        fused_results = self._fuse_results(
            retrieval_lists
        )

        if not fused_results:
            return []

        # --------------------------------------------------------------
        # Cross-encoder reranking
        # --------------------------------------------------------------

        if self.reranker is not None:

            candidate_results = fused_results[
                :rerank_top_k
            ]

            return self.reranker.rerank(
                query=query,
                results=candidate_results,
                top_k=top_k,
            )

        # --------------------------------------------------------------
        # No reranker
        # --------------------------------------------------------------

        return fused_results[:top_k]

    # ------------------------------------------------------------------
    # INFORMATION
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """
        Return the number of chunks indexed.
        """

        return self.dense_retriever.size