"""Dense retriever.

Connects :class:`~app.rag.interfaces.Embedder` and
:class:`~app.rag.interfaces.VectorStore` into a single callable that takes a
free-text query and returns a ranked list of
:class:`~app.schemas.retrieval.RetrievalResult` objects.

Design
------
``DenseRetriever`` is intentionally thin.  It owns:

* query → embedding (via the injected Embedder)
* embedding → top-K results  (via the injected VectorStore)

It does **not** own the indexing step.  Callers are responsible for calling
:meth:`~app.rag.vectorstore.faiss_store.FAISSVectorStore.upsert` before
using this class.

Dependency injection
--------------------
Both dependencies (``Embedder`` and ``VectorStore``) are injected at
construction time.  This makes the class trivially unit-testable with fakes
and allows the agent pipeline to share long-lived singleton instances.

Typical wiring::

    from app.rag.embeddings.bge_embedding import BGEEmbedder
    from app.rag.vectorstore.faiss_store import FAISSVectorStore
    from app.rag.retrievers.dense_retriever import DenseRetriever

    embedder = BGEEmbedder()
    store    = FAISSVectorStore.load(index_dir)
    retriever = DenseRetriever(embedder=embedder, store=store)

    results = retriever.retrieve("first-line treatment for hypertension", top_k=5)

Default top_k
-------------
If *top_k* is not supplied, ``Settings.retrieval_top_k`` (default 5) is used.
The constant ``TOP_K_FINAL = 5`` in ``app.config`` is also available for
callers that want a module-level default.
"""

from __future__ import annotations

from typing import Optional

from app.config import TOP_K_FINAL, get_settings
from app.rag.interfaces import Embedder, VectorStore
from app.schemas.retrieval import RetrievalResult
from app.utils.logger import logger


class DenseRetriever:
    """Retrieve evidence chunks using dense vector similarity.

    Parameters
    ----------
    embedder:
        Any concrete :class:`~app.rag.interfaces.Embedder` implementation.
        Typically :class:`~app.rag.embeddings.bge_embedding.BGEEmbedder`.
    store:
        Any concrete :class:`~app.rag.interfaces.VectorStore` implementation.
        Typically :class:`~app.rag.vectorstore.faiss_store.FAISSVectorStore`.
    default_top_k:
        Number of results to return when *top_k* is not passed to
        :meth:`retrieve`.  Defaults to ``Settings.retrieval_top_k``.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        default_top_k: Optional[int] = None,
    ) -> None:
        if not isinstance(embedder, Embedder):
            raise TypeError(
                f"embedder must be an Embedder instance, got {type(embedder).__name__}."
            )
        if not isinstance(store, VectorStore):
            raise TypeError(
                f"store must be a VectorStore instance, got {type(store).__name__}."
            )

        self._embedder = embedder
        self._store = store
        self._default_top_k: int = (
            default_top_k
            if default_top_k is not None
            else get_settings().retrieval_top_k
        )

        logger.info(
            f"DenseRetriever ready — "
            f"embedder={type(embedder).__name__}, "
            f"store={type(store).__name__}, "
            f"default_top_k={self._default_top_k}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """Embed *query* and return the top-K most similar indexed chunks.

        Parameters
        ----------
        query:
            Free-text search query (e.g. a clinical question).
        top_k:
            Number of results to return.  Falls back to ``default_top_k``
            when omitted.  Clamped to the number of indexed vectors if the
            index is smaller.

        Returns
        -------
        list[RetrievalResult]
            Ordered by descending relevance score.  Empty list if the index
            contains no vectors.

        Raises
        ------
        ValueError
            If *query* is empty or whitespace-only, or if *top_k* is not a
            positive integer.
        """
        if not query or not query.strip():
            raise ValueError(
                "DenseRetriever.retrieve requires a non-empty query string."
            )

        effective_top_k = top_k if top_k is not None else self._default_top_k
        if effective_top_k <= 0:
            raise ValueError(
                f"top_k must be a positive integer, got {effective_top_k}."
            )

        logger.debug(
            f"DenseRetriever.retrieve: query='{query[:60]}{'...' if len(query) > 60 else ''}', "
            f"top_k={effective_top_k}"
        )

        query_embedding = self._embedder.embed_query(query)
        results = self._store.search(query_embedding, top_k=effective_top_k)

        logger.debug(
            f"DenseRetriever.retrieve: returned {len(results)} result(s)"
        )
        return results

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def embedder(self) -> Embedder:
        """The injected Embedder instance."""
        return self._embedder

    @property
    def store(self) -> VectorStore:
        """The injected VectorStore instance."""
        return self._store

    @property
    def default_top_k(self) -> int:
        """Default number of results returned by :meth:`retrieve`."""
        return self._default_top_k
