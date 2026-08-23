"""
app.retrieval -- retrieval pipeline boundary package.

Public surface
--------------
``DocumentLoader``     -- unified ingestion entry-point (delegates to app.rag.loaders).
``BGEEmbedder``        -- BAAI/bge-small-en-v1.5 sentence-transformers adapter.
``FAISSVectorStore``   -- FAISS inner-product vector store.
``DenseRetriever``     -- query -> embedding -> vector search -> RetrievalResult list.
``Embedder``           -- abstract base class for embedding implementations.
``VectorStore``        -- abstract base class for vector store implementations.

Concrete implementations live in ``app.rag`` (the canonical home for all
RAG internals).  The stubs in this package (hybrid_search.py, reranker.py)
will be removed once the corresponding ``app.rag`` implementations are
complete.
"""

from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.document_loader import DocumentLoader
from app.retrieval.embeddings import BGEEmbedder
from app.retrieval.interface import Embedder, VectorStore
from app.retrieval.vector_store import FAISSVectorStore

__all__ = [
    "DocumentLoader",
    "BGEEmbedder",
    "FAISSVectorStore",
    "DenseRetriever",
    "Embedder",
    "VectorStore",
]