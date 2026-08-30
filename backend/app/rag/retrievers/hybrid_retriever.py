"""Hybrid retriever — Reciprocal Rank Fusion of dense + sparse results.

Combines :class:`~app.rag.retrievers.dense_retriever.DenseRetriever` (vector
similarity) with :class:`~app.rag.retrievers.sparse_retriever.SparseRetriever`
(BM25 keyword) via **Reciprocal Rank Fusion (RRF)**.

RRF formula
-----------
For each document *d* that appears in at least one ranked list::

    rrf_score(d) = Σ  weight_i / (k + rank_i(d))

where:

* ``rank_i(d)`` is the 1-based position of *d* in ranker *i*'s result list
  (documents absent from a list contribute 0).
* ``k = 60`` is the standard smoothing constant (Cormack et al., 2009).
* ``weight_i`` scales the contribution of each ranker (default 1.0 each).

Final scores are normalised to ``[0, 1]`` by dividing by the maximum RRF
score across the candidate pool, so they remain compatible with the
``RetrievalResult.relevance_score`` schema constraint.

Over-retrieval
--------------
To improve fusion recall, each sub-retriever fetches
``min(ntotal, top_k * retrieval_multiplier)`` candidates before fusion.
The default multiplier of 3 is a practical balance between recall and
latency.

Graceful degradation
--------------------
Either ``dense`` or ``sparse`` may be ``None`` at construction time.  If one
is absent, ``retrieve`` falls back transparently to the available ranker with
no fusion overhead.  Both being ``None`` raises ``ValueError`` at construction.

Typical wiring::

    from app.rag.retrievers.hybrid_retriever import HybridRetriever
    from app.rag.retrievers.dense_retriever import DenseRetriever
    from app.rag.retrievers.sparse_retriever import SparseRetriever

    hybrid = HybridRetriever(dense=dense_retriever, sparse=sparse_retriever)
    results = hybrid.retrieve("ACE inhibitor hypertension therapy", top_k=5)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional
from uuid import UUID

from app.config import get_settings
from app.rag.retrievers.dense_retriever import DenseRetriever
from app.rag.retrievers.sparse_retriever import SparseRetriever
from app.schemas.retrieval import DocumentChunk, RetrievalResult
from app.utils.logger import logger

# Standard RRF smoothing constant (Cormack et al., SIGIR 2009).
_RRF_K: int = 60


class HybridRetriever:
    """Fuse dense and sparse retrieval results using Reciprocal Rank Fusion.

    Parameters
    ----------
    dense:
        A pre-built :class:`DenseRetriever`.  Pass ``None`` to use only the
        sparse ranker.
    sparse:
        A pre-built :class:`SparseRetriever`.  Pass ``None`` to use only the
        dense ranker.
    default_top_k:
        Final number of results returned by :meth:`retrieve` when *top_k* is
        not supplied.  Defaults to ``Settings.retrieval_top_k`` (5).
    weight_dense:
        Multiplicative weight applied to each dense rank contribution.
        Default 1.0.  Increase to favour semantic matches.
    weight_sparse:
        Multiplicative weight applied to each sparse rank contribution.
        Default 1.0.  Increase to favour keyword matches.
    rrf_k:
        RRF smoothing constant.  Default 60.  Larger values reduce the
        advantage of very high ranks; smaller values amplify it.
    retrieval_multiplier:
        Each sub-retriever fetches ``top_k * retrieval_multiplier`` candidates
        before fusion.  Default 3.  Raise if final recall is low.
    """

    def __init__(
        self,
        dense: Optional[DenseRetriever] = None,
        sparse: Optional[SparseRetriever] = None,
        default_top_k: Optional[int] = None,
        weight_dense: float = 1.0,
        weight_sparse: float = 1.0,
        rrf_k: int = _RRF_K,
        retrieval_multiplier: int = 3,
    ) -> None:
        if dense is None and sparse is None:
            raise ValueError(
                "HybridRetriever requires at least one of 'dense' or 'sparse'."
            )
        if dense is not None and not isinstance(dense, DenseRetriever):
            raise TypeError(
                f"'dense' must be a DenseRetriever, got {type(dense).__name__}."
            )
        if sparse is not None and not isinstance(sparse, SparseRetriever):
            raise TypeError(
                f"'sparse' must be a SparseRetriever, got {type(sparse).__name__}."
            )
        if weight_dense < 0 or weight_sparse < 0:
            raise ValueError("Retriever weights must be non-negative.")
        if rrf_k <= 0:
            raise ValueError(f"rrf_k must be positive, got {rrf_k}.")
        if retrieval_multiplier < 1:
            raise ValueError(
                f"retrieval_multiplier must be >= 1, got {retrieval_multiplier}."
            )

        self._dense = dense
        self._sparse = sparse
        self._default_top_k: int = (
            default_top_k
            if default_top_k is not None
            else get_settings().retrieval_top_k
        )
        self._weight_dense = weight_dense
        self._weight_sparse = weight_sparse
        self._rrf_k = rrf_k
        self._retrieval_multiplier = retrieval_multiplier

        active = []
        if dense is not None:
            active.append(f"dense(w={weight_dense})")
        if sparse is not None:
            active.append(f"sparse(w={weight_sparse})")

        logger.info(
            f"HybridRetriever ready — "
            f"rankers=[{', '.join(active)}], "
            f"default_top_k={self._default_top_k}, "
            f"rrf_k={rrf_k}, multiplier={retrieval_multiplier}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """Run both rankers, fuse via RRF, and return the top-K results.

        Parameters
        ----------
        query:
            Free-text search query.
        top_k:
            Final number of results to return.  Falls back to
            ``default_top_k`` when omitted.

        Returns
        -------
        list[RetrievalResult]
            Ordered by descending fused RRF score, normalised to ``[0, 1]``.
            Empty list when both retrievers produce no results.

        Raises
        ------
        ValueError
            If *query* is empty/whitespace-only, or *top_k* is not positive.
        """
        if not query or not query.strip():
            raise ValueError(
                "HybridRetriever.retrieve requires a non-empty query string."
            )

        effective_top_k = top_k if top_k is not None else self._default_top_k
        if effective_top_k <= 0:
            raise ValueError(
                f"top_k must be a positive integer, got {effective_top_k}."
            )

        # Number of candidates to pull from each sub-retriever
        candidate_k = effective_top_k * self._retrieval_multiplier

        logger.debug(
            f"HybridRetriever.retrieve: "
            f"query='{query[:60]}{'...' if len(query) > 60 else ''}', "
            f"top_k={effective_top_k}, candidate_k={candidate_k}"
        )

        # Fallback: only one ranker active
        if self._dense is None:
            return self._sparse.retrieve(query, top_k=effective_top_k)
        if self._sparse is None:
            return self._dense.retrieve(query, top_k=effective_top_k)

        # Both active — over-retrieve then fuse
        dense_results = self._dense.retrieve(query, top_k=candidate_k)
        sparse_results = self._sparse.retrieve(query, top_k=candidate_k)

        fused = self._rrf_fuse(dense_results, sparse_results)
        final = fused[:effective_top_k]

        logger.debug(
            f"HybridRetriever.retrieve: "
            f"dense={len(dense_results)}, sparse={len(sparse_results)}, "
            f"fused={len(fused)}, returned={len(final)}"
        )
        return final

    # ------------------------------------------------------------------
    # RRF internals
    # ------------------------------------------------------------------

    def _rrf_fuse(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Merge two ranked lists with Reciprocal Rank Fusion.

        Uses ``chunk_id`` as the deduplication key so the same chunk returned
        by both rankers is merged into a single fused result.  The
        ``DocumentChunk`` object is taken from whichever list ranks it higher
        (lower position index), preserving all metadata.

        Returns a list sorted by descending normalised RRF score.
        """
        # Map chunk_id -> accumulated RRF score
        rrf_scores: dict[UUID, float] = defaultdict(float)
        # Map chunk_id -> best DocumentChunk (from the higher-ranked list)
        best_chunk: dict[UUID, DocumentChunk] = {}

        for rank, result in enumerate(dense_results, start=1):
            cid = result.chunk.chunk_id
            rrf_scores[cid] += self._weight_dense / (self._rrf_k + rank)
            if cid not in best_chunk:
                best_chunk[cid] = result.chunk

        for rank, result in enumerate(sparse_results, start=1):
            cid = result.chunk.chunk_id
            rrf_scores[cid] += self._weight_sparse / (self._rrf_k + rank)
            if cid not in best_chunk:
                best_chunk[cid] = result.chunk

        if not rrf_scores:
            return []

        # Sort by descending RRF score
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Normalise to [0, 1]
        max_score = sorted_items[0][1]

        results: list[RetrievalResult] = []
        for cid, raw_score in sorted_items:
            normalised = (raw_score / max_score) if max_score > 0.0 else 0.0
            score = max(0.0, min(1.0, normalised))
            results.append(
                RetrievalResult(chunk=best_chunk[cid], relevance_score=score)
            )

        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dense(self) -> Optional[DenseRetriever]:
        """The injected DenseRetriever, or None."""
        return self._dense

    @property
    def sparse(self) -> Optional[SparseRetriever]:
        """The injected SparseRetriever, or None."""
        return self._sparse

    @property
    def default_top_k(self) -> int:
        """Default number of results returned by :meth:`retrieve`."""
        return self._default_top_k

    @property
    def rrf_k(self) -> int:
        """RRF smoothing constant."""
        return self._rrf_k

    @property
    def weight_dense(self) -> float:
        """Weight applied to dense rank contributions."""
        return self._weight_dense

    @property
    def weight_sparse(self) -> float:
        """Weight applied to sparse rank contributions."""
        return self._weight_sparse
