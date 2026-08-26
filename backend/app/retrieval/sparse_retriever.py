"""SparseRetriever re-export for the retrieval boundary.

The canonical implementation lives in ``app.rag.retrievers.sparse_retriever``.
This module re-exports it so retrieval-layer code can import from one stable
location without depending directly on the rag internals package.

Usage::

    from app.retrieval.sparse_retriever import SparseRetriever

    retriever = SparseRetriever()
    retriever.index(chunks)
    results = retriever.retrieve("hypertension first-line therapy", top_k=5)
"""

from app.rag.retrievers.sparse_retriever import SparseRetriever  # noqa: F401

__all__ = ["SparseRetriever"]
