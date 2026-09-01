"""
Hybrid retrieval for the RAGMedX-MA retrieval engine.

Combines:
    - Dense semantic retrieval (BGE-M3 + FAISS)
    - Sparse lexical retrieval (BM25)
    - Query expansion
    - Reciprocal Rank Fusion (RRF)

Query expansion generates multiple retrieval-oriented formulations
of the original query. Dense and sparse retrieval are performed for
each formulation, and the resulting ranked lists are fused using RRF.

This component is intentionally independent of the multi-agent
and LLM orchestration layer.
"""

from __future__ import annotations

from typing import Sequence

from app.rag.chunking.models import MedicalChunk
from app.rag.retrievers.dense_retriever import DenseRetriever
from app.rag.retrievers.sparse_retriever import SparseRetriever
from app.rag.retrievers.query_expander import QueryExpander


class HybridRetriever:
    """
    Hybrid retriever combining dense retrieval, sparse retrieval,
    optional query expansion, and Reciprocal Rank Fusion (RRF).

    Pipeline:

        Original Query
              |
        Query Expansion
              |
        Multiple Query Variants
              |
        +----------------------+
        | Dense + Sparse       |
        | Retrieval per query  |
        +----------+-----------+
                   |
              RRF Fusion
                   |
             Ranked Results

    RRF formula:

        RRF(d) = sum 1 / (k + rank(d))

    where:
        k is the RRF constant.
        rank(d) is the 1-based rank of document d
        in a retrieval list.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        query_expander: QueryExpander | None = None,
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
        systems so their results can be fused consistently.
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

        Each dense/sparse result list contributes independently
        to the final RRF score.

        Args:
            retrieval_lists:
                Multiple ranked lists of
                (MedicalChunk, score) tuples.

        Returns:
            List of (MedicalChunk, RRF score) tuples ordered by
            descending fused score.
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
        expand_query: bool = True,
    ) -> list[
        tuple[MedicalChunk, float]
    ]:
        """
        Perform hybrid retrieval with optional query expansion.

        Args:
            query:
                Natural-language medical question.

            top_k:
                Number of final hybrid results.

            dense_top_k:
                Number of candidates retrieved from dense search
                for each query variant.
                Defaults to max(top_k, 10).

            sparse_top_k:
                Number of candidates retrieved from sparse search
                for each query variant.
                Defaults to max(top_k, 10).

            expand_query:
                Whether to generate multiple query formulations
                before retrieval.

                True:
                    Original query + expanded variants are retrieved.

                False:
                    Only the original query is retrieved.

        Returns:
            List of (MedicalChunk, RRF score) tuples ordered by
            descending fused score.
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

        if dense_top_k <= 0:
            raise ValueError(
                "dense_top_k must be greater than zero."
            )

        if sparse_top_k <= 0:
            raise ValueError(
                "sparse_top_k must be greater than zero."
            )

        # --------------------------------------------------------------
        # Query expansion
        # --------------------------------------------------------------

        if expand_query:
            queries = self.query_expander.expand(query)
        else:
            queries = [query]

        # --------------------------------------------------------------
        # Retrieve for every query variant
        # --------------------------------------------------------------

        retrieval_lists: list[
            list[tuple[MedicalChunk, float]]
        ] = []

        for retrieval_query in queries:

            # Dense retrieval
            dense_results = (
                self.dense_retriever.retrieve(
                    query=retrieval_query,
                    top_k=dense_top_k,
                )
            )

            retrieval_lists.append(
                dense_results
            )

            # Sparse retrieval
            sparse_results = (
                self.sparse_retriever.retrieve(
                    query=retrieval_query,
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