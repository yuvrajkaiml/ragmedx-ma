"""Sparse retriever using BM25 (rank-bm25).

Implements lexical keyword retrieval over a collection of
:class:`~app.schemas.retrieval.DocumentChunk` objects using
``BM25Okapi`` from the ``rank-bm25`` library.

Design
------
``SparseRetriever`` is intentionally parallel in API to
:class:`~app.rag.retrievers.dense_retriever.DenseRetriever`:

* :meth:`index`    — add chunks to the BM25 corpus (equivalent to
                     ``FAISSVectorStore.upsert``)
* :meth:`retrieve` — tokenise query, score all indexed chunks, return top-K

Tokenisation
------------
The default tokeniser lowercases and whitespace-splits each document / query.
A custom callable tokeniser can be injected at construction time for
domain-specific pre-processing (e.g. medical term normalisation, stop-word
removal).

Score normalisation
-------------------
Raw BM25 scores are non-negative floats with no fixed upper bound.
:meth:`retrieve` normalises them to ``[0, 1]`` by dividing by the maximum
score in the result set before constructing :class:`RetrievalResult` objects.
When every score is 0 (the query shares no tokens with any document), all
returned results carry ``relevance_score = 0.0``.

Thread safety
-------------
``BM25Okapi`` rebuilds its IDF index on every call to :meth:`index` because
the corpus must be provided up-front.  Keep one ``SparseRetriever`` per
process and call :meth:`index` once after loading all chunks.

Typical wiring::

    from app.rag.retrievers.sparse_retriever import SparseRetriever

    retriever = SparseRetriever()
    retriever.index(chunks)              # call once after loading documents
    results = retriever.retrieve("first-line hypertension therapy", top_k=5)
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Optional

from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.schemas.retrieval import DocumentChunk, RetrievalResult
from app.utils.logger import logger

# ---------------------------------------------------------------------------
# Default tokeniser — lowercase + split on non-alphanumeric boundaries
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[^\w]+")


def _default_tokenise(text: str) -> list[str]:
    """Lowercase and split *text* on non-word character boundaries.

    Empty tokens produced by leading / trailing punctuation are filtered.
    """
    return [t for t in _TOKEN_RE.split(text.lower()) if t]


# ---------------------------------------------------------------------------
# SparseRetriever
# ---------------------------------------------------------------------------


class SparseRetriever:
    """BM25 lexical retriever over a collection of document chunks.

    Parameters
    ----------
    tokeniser:
        Callable ``(str) -> list[str]``.  Defaults to a simple
        lowercase + word-boundary split.  Inject a custom callable for
        domain-specific pre-processing (stop-word removal, stemming, etc.).
    default_top_k:
        Number of results returned by :meth:`retrieve` when *top_k* is not
        supplied.  Defaults to ``Settings.retrieval_top_k`` (5).
    k1:
        BM25 term-saturation parameter.  Default 1.5 (rank-bm25 default).
    b:
        BM25 length-normalisation parameter.  Default 0.75 (rank-bm25 default).
    """

    def __init__(
        self,
        tokeniser: Optional[Callable[[str], list[str]]] = None,
        default_top_k: Optional[int] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._tokenise: Callable[[str], list[str]] = (
            tokeniser if tokeniser is not None else _default_tokenise
        )
        self._default_top_k: int = (
            default_top_k
            if default_top_k is not None
            else get_settings().retrieval_top_k
        )
        self._k1 = k1
        self._b = b

        # Populated by index()
        self._chunks: list[DocumentChunk] = []
        self._bm25: Optional[BM25Okapi] = None

        logger.info(
            f"SparseRetriever initialised — "
            f"default_top_k={self._default_top_k}, k1={k1}, b={b}"
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        """Build the BM25 index from *chunks*.

        Parameters
        ----------
        chunks:
            Document chunks to index.  Each chunk's ``text`` field is
            tokenised and added to the BM25 corpus.

        Notes
        -----
        Calling :meth:`index` with an empty sequence clears any existing
        index (safe no-op for retrieval — :meth:`retrieve` will return
        ``[]``).

        Calling :meth:`index` a second time **replaces** the existing index
        entirely.  There is no incremental add because ``BM25Okapi`` requires
        the full corpus at construction time to compute IDF correctly.
        """
        if len(chunks) == 0:
            logger.debug("SparseRetriever.index called with empty sequence — clearing index.")
            self._chunks = []
            self._bm25 = None
            return

        self._chunks = list(chunks)
        tokenised_corpus = [self._tokenise(chunk.text) for chunk in self._chunks]
        self._bm25 = BM25Okapi(tokenised_corpus, k1=self._k1, b=self._b)

        logger.info(
            f"SparseRetriever.index: indexed {len(self._chunks)} chunk(s)"
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """Score all indexed chunks against *query* and return the top-K.

        Parameters
        ----------
        query:
            Free-text search query.  Tokenised with the same tokeniser used
            during :meth:`index`.
        top_k:
            Number of results to return.  Falls back to ``default_top_k``
            when omitted.  Clamped to the number of indexed chunks if the
            corpus is smaller.

        Returns
        -------
        list[RetrievalResult]
            Ordered by descending BM25 score (normalised to ``[0, 1]``).
            Empty list when no chunks are indexed.

        Raises
        ------
        ValueError
            If *query* is empty or whitespace-only, or if *top_k* is not a
            positive integer.
        """
        if not query or not query.strip():
            raise ValueError(
                "SparseRetriever.retrieve requires a non-empty query string."
            )

        effective_top_k = top_k if top_k is not None else self._default_top_k
        if effective_top_k <= 0:
            raise ValueError(
                f"top_k must be a positive integer, got {effective_top_k}."
            )

        if self._bm25 is None or not self._chunks:
            logger.debug("SparseRetriever.retrieve: index is empty, returning [].")
            return []

        query_tokens = self._tokenise(query)
        logger.debug(
            f"SparseRetriever.retrieve: "
            f"query='{query[:60]}{'...' if len(query) > 60 else ''}', "
            f"tokens={query_tokens[:8]}, top_k={effective_top_k}"
        )

        raw_scores = self._bm25.get_scores(query_tokens)  # ndarray, len == len(corpus)

        # Clamp to available corpus size
        effective_top_k = min(effective_top_k, len(self._chunks))

        # Indices of top-K scores in descending order
        # argsort gives ascending; slice with [-effective_top_k:][::-1] for top-K desc
        top_indices = raw_scores.argsort()[-effective_top_k:][::-1]

        # Normalise scores to [0, 1] by max score in the *full* corpus
        max_score = float(raw_scores.max())

        results: list[RetrievalResult] = []
        for idx in top_indices:
            raw = float(raw_scores[idx])
            normalised = (raw / max_score) if max_score > 0.0 else 0.0
            # Clamp to [0, 1] to guard against floating-point edge cases
            score = max(0.0, min(1.0, normalised))
            results.append(
                RetrievalResult(
                    chunk=self._chunks[int(idx)],
                    relevance_score=score,
                )
            )

        logger.debug(
            f"SparseRetriever.retrieve: returned {len(results)} result(s)"
        )
        return results

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def ntotal(self) -> int:
        """Number of chunks currently indexed."""
        return len(self._chunks)

    @property
    def default_top_k(self) -> int:
        """Default number of results returned by :meth:`retrieve`."""
        return self._default_top_k

    @property
    def tokeniser(self) -> Callable[[str], list[str]]:
        """The tokeniser callable used for both indexing and querying."""
        return self._tokenise
