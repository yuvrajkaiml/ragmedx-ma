"""
Hybrid retrieval for the RAGMedX-MA retrieval engine.

Combines dense semantic retrieval and sparse lexical retrieval
using Reciprocal Rank Fusion (RRF).

Dense retrieval:
    BGE-M3 + FAISS

Sparse retrieval:
    BM25

RRF combines the ranked lists without requiring the dense and
sparse scores to be on the same numerical scale.
"""

from __future__ import annotations

from typing import Sequence

from app.rag.chunking.models import MedicalChunk
from app.rag.retrievers.dense_retriever import DenseRetriever
from app.rag.retrievers.sparse_retriever import SparseRetriever


class HybridRetriever:
    """
    Hybrid retriever combining dense and sparse retrieval.

    Uses Reciprocal Rank Fusion (RRF):

        RRF(d) = Σ 1 / (k + rank(d))

    where:
        k is the RRF constant.
        rank(d) is the rank of document d in a retrieval list.

    The default RRF constant of 60 is commonly used because it
    prevents very high-ranked results from completely dominating
    the fused ranking.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError(
                "rrf_k must be greater than zero."
            )

        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
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

        The same MedicalChunk objects are indexed by both retrieval
        systems so that their results can be fused consistently.
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
        dense_results: list[
            tuple[MedicalChunk, float]
        ],
        sparse_results: list[
            tuple[MedicalChunk, float]
        ],
    ) -> list[
        tuple[MedicalChunk, float]
    ]:
        """
        Fuse dense and sparse ranked results using RRF.

        The original retrieval scores are intentionally not added
        directly because FAISS cosine similarity and BM25 scores
        have different numerical scales.

        Returns:
            List of (MedicalChunk, RRF score) tuples ordered from
            highest fused score to lowest.
        """

        scores: dict[str, float] = {}
        chunks: dict[str, MedicalChunk] = {}

        # --------------------------------------------------------------
        # Dense contribution
        # --------------------------------------------------------------

        for rank, (chunk, _) in enumerate(
            dense_results,
            start=1,
        ):
            chunk_key = chunk.id

            chunks[chunk_key] = chunk

            scores[chunk_key] = (
                scores.get(chunk_key, 0.0)
                + self._rrf_score(rank)
            )

        # --------------------------------------------------------------
        # Sparse contribution
        # --------------------------------------------------------------

        for rank, (chunk, _) in enumerate(
            sparse_results,
            start=1,
        ):
            chunk_key = chunk.id

            chunks[chunk_key] = chunk

            scores[chunk_key] = (
                scores.get(chunk_key, 0.0)
                + self._rrf_score(rank)
            )

        # --------------------------------------------------------------
        # Sort by fused RRF score
        # --------------------------------------------------------------

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
    ) -> list[
        tuple[MedicalChunk, float]
    ]:
        """
        Perform hybrid retrieval.

        Args:
            query:
                Natural-language medical question.

            top_k:
                Number of final hybrid results.

            dense_top_k:
                Number of candidates retrieved from dense search.
                Defaults to max(top_k, 10).

            sparse_top_k:
                Number of candidates retrieved from sparse search.
                Defaults to max(top_k, 10).

        Returns:
            List of (MedicalChunk, RRF score) tuples.
        """

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

        if dense_top_k <= 0:
            raise ValueError(
                "dense_top_k must be greater than zero."
            )

        if sparse_top_k <= 0:
            raise ValueError(
                "sparse_top_k must be greater than zero."
            )

        # --------------------------------------------------------------
        # Retrieve independently
        # --------------------------------------------------------------

        dense_results = (
            self.dense_retriever.retrieve(
                query=query,
                top_k=dense_top_k,
            )
        )

        sparse_results = (
            self.sparse_retriever.retrieve(
                query=query,
                top_k=sparse_top_k,
            )
        )

        # --------------------------------------------------------------
        # Fuse ranked lists
        # --------------------------------------------------------------

        fused_results = self._fuse_results(
            dense_results=dense_results,
            sparse_results=sparse_results,
        )

        return fused_results[:top_k]

    # ------------------------------------------------------------------
    # INFORMATION
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """
        Return the number of chunks indexed.

        Dense and sparse indexes should contain the same chunks.
        """

        return self.dense_retriever.size