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

import re
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
        dedup_jaccard_threshold: float = 0.90,
    ) -> None:

        if rrf_k <= 0:
            raise ValueError(
                "rrf_k must be greater than zero."
            )

        if not 0.0 < dedup_jaccard_threshold <= 1.0:
            raise ValueError(
                "dedup_jaccard_threshold must be in (0, 1]."
            )

        self.dedup_jaccard_threshold = dedup_jaccard_threshold

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

    # ------------------------------------------------------------------
    # DEDUPLICATION HELPERS
    # ------------------------------------------------------------------

    _WHITESPACE_RE = re.compile(r"\s+")
    _TOKEN_RE = re.compile(r"[a-z0-9]+")

    @staticmethod
    def _doc_key(chunk: MedicalChunk) -> str:
        """
        Grouping key for dedup.

        Uses ``metadata['pmid']`` (or ``'PMID'``) when present so that
        same-document / different-snippet chunks collapse together.
        Otherwise falls back to the chunk's unique ``document_id`` (or
        ``id`` if that is empty) so PMID-less chunks are never merged with
        anything but themselves.
        """

        metadata = getattr(chunk, "metadata", None) or {}

        for pmid_field in ("pmid", "PMID"):
            value = metadata.get(pmid_field)
            if value:
                return f"pmid:{value}"

        return f"docid:{chunk.document_id or chunk.id}"

    @classmethod
    def _norm_text(cls, chunk: MedicalChunk) -> str:
        """
        Normalize chunk text: lowercased, stripped, internal whitespace
        collapsed to a single space.
        """

        text = (chunk.content or "").lower().strip()
        return cls._WHITESPACE_RE.sub(" ", text)

    @classmethod
    def _token_set(cls, normalized_text: str) -> set[str]:
        """Lowercased alphanumeric token set of normalized text."""

        return set(cls._TOKEN_RE.findall(normalized_text))

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        """
        Jaccard overlap of two token sets.

        Empty vs empty is defined as 1.0; one-empty as 0.0.
        """

        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0

        union = len(a | b)
        if not union:
            return 0.0
        return len(a & b) / union

    def _dedup_fused(
        self,
        fused: list[tuple[MedicalChunk, float]],
    ) -> list[tuple[MedicalChunk, float]]:
        """
        Collapse near-duplicate candidates in an RRF-ordered list.

        Two candidates are duplicates when they share the same document key
        (:meth:`_doc_key`) AND have near-identical text (exact normalized-text
        equality OR token Jaccard >= ``dedup_jaccard_threshold``). The input is
        assumed to be ordered best-first; the best-ranked (earliest) copy of
        each duplicate group is kept and the relative order of everything kept
        is preserved. Genuinely distinct passages are never merged.
        """

        kept: list[tuple[MedicalChunk, float]] = []
        # doc_key -> list of (normalized_text, token_set) already kept.
        kept_by_key: dict[str, list[tuple[str, set[str]]]] = {}

        for chunk, score in fused:
            key = self._doc_key(chunk)
            normalized = self._norm_text(chunk)
            tokens = self._token_set(normalized)

            is_duplicate = False
            for kept_text, kept_tokens in kept_by_key.get(key, []):
                if (
                    normalized == kept_text
                    or self._jaccard(tokens, kept_tokens)
                    >= self.dedup_jaccard_threshold
                ):
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            kept.append((chunk, score))
            kept_by_key.setdefault(key, []).append(
                (normalized, tokens)
            )

        return kept

    def _fuse_results(
        self,
        retrieval_lists: Sequence[
            list[tuple[MedicalChunk, float]]
        ],
    ) -> list[tuple[MedicalChunk, float]]:
        """
        Fuse multiple ranked retrieval lists using RRF, then collapse
        same-document near-duplicate candidates.
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

        fused = [
            (
                chunks[chunk_id],
                float(scores[chunk_id]),
            )
            for chunk_id in ranked_ids
        ]

        return self._dedup_fused(fused)

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