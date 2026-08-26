"""HybridRetriever re-export for the retrieval boundary.

The canonical implementation lives in ``app.rag.retrievers.hybrid_retriever``.
This module re-exports it so retrieval-layer code can import from one stable
location without depending directly on the rag internals package.

Usage::

    from app.retrieval.hybrid_retriever import HybridRetriever
    from app.retrieval import DenseRetriever, SparseRetriever

    hybrid = HybridRetriever(
        dense=DenseRetriever(embedder=embedder, store=store),
        sparse=sparse_retriever,
    )
    results = hybrid.retrieve("ACE inhibitor hypertension therapy", top_k=5)
"""

from app.rag.retrievers.hybrid_retriever import HybridRetriever  # noqa: F401

__all__ = ["HybridRetriever"]