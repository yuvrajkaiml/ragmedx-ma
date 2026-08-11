"""Interfaces for the retrieval implementation.

Concrete vector stores and embedders must keep these contracts so the API and
clinical safety layer remain independent of a specific vendor.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.schemas.retrieval import DocumentChunk, RetrievalResult


class Embedder(ABC):
    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one normalized vector for each input text."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Return a normalized vector for a search query."""


class VectorStore(ABC):
    @abstractmethod
    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Persist chunks with their corresponding embeddings."""

    @abstractmethod
    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Return the most relevant indexed chunks for a query vector."""
