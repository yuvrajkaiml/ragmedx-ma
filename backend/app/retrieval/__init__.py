"""
app.retrieval — retrieval pipeline boundary package.

Public surface
--------------
``DocumentLoader``  — unified ingestion entry-point (delegates to app.rag.loaders).
``Embedder``        — abstract base class for embedding implementations.
``VectorStore``     — abstract base class for vector store implementations.

Concrete implementations live in ``app.rag`` (the canonical home for all
RAG internals).  The stubs in this package (embeddings.py, vector_store.py,
hybrid_search.py, reranker.py) will be removed once the corresponding
``app.rag`` implementations are complete.
"""

from app.retrieval.document_loader import DocumentLoader
from app.retrieval.interface import Embedder, VectorStore

__all__ = ["DocumentLoader", "Embedder", "VectorStore"]
