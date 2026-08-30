"""DenseRetriever re-export for the retrieval boundary.

The canonical implementation lives in ``app.rag.retrievers.dense_retriever``.
This module re-exports it so retrieval-layer code can import from one stable
location without depending directly on the rag internals package.

Usage::

    from app.retrieval.dense_retriever import DenseRetriever
    from app.retrieval import BGEEmbedder, FAISSVectorStore

    retriever = DenseRetriever(
        embedder=BGEEmbedder(),
        store=FAISSVectorStore.load(index_dir),
    )
    results = retriever.retrieve("first-line antihypertensive therapy", top_k=5)
"""

from app.rag.retrievers.dense_retriever import DenseRetriever  # noqa: F401

__all__ = ["DenseRetriever"]
