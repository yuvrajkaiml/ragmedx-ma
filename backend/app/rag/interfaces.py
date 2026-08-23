"""Abstract base classes for the RAG pipeline's retrieval contracts.

These interfaces decouple the API layer and the clinical-safety layer from any
specific vendor or model choice.  All concrete implementations (embedders,
vector stores) must satisfy these contracts.

History
-------
Originally defined in ``app.retrieval.interface``.  Moved here because
``app.rag`` is the canonical home for all RAG internals.  The old module
re-exports these classes for backwards compatibility until all stubs in
``app.retrieval`` are retired.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.schemas.retrieval import DocumentChunk, RetrievalResult


class Embedder(ABC):
    """Contract for text embedding implementations.

    Concrete adapters (e.g. ``app.rag.embeddings.bge_embedding.BGEEmbedder``)
    must subclass this and implement both methods.
    """

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one normalised vector per input text."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Return a normalised vector for a single search query."""


class VectorStore(ABC):
    """Contract for vector store implementations.

    Concrete adapters (e.g. ``app.rag.vectorstore.faiss_store.FAISSStore``)
    must subclass this and implement both methods.
    """

    @abstractmethod
    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Persist *chunks* together with their pre-computed *embeddings*."""

    @abstractmethod
    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Return the *top_k* most relevant chunks for *query_embedding*."""
