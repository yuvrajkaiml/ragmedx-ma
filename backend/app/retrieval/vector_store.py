"""Vector-store re-export for the retrieval boundary.

The canonical implementation lives in ``app.rag.vectorstore.faiss_store``.
This module re-exports it so retrieval-layer code can import from one stable
location without depending directly on the rag internals package.

Usage::

    from app.retrieval.vector_store import FAISSVectorStore
    store = FAISSVectorStore()          # uses FAISS_INDEX_DIR from config
    store = FAISSVectorStore(dim=384, index_dir=Path("/tmp/test_idx"))
"""

from app.rag.vectorstore.faiss_store import FAISSVectorStore  # noqa: F401

__all__ = ["FAISSVectorStore"]