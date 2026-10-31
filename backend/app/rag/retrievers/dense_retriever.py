"""
Dense retrieval for the RAGMedX-MA retrieval engine.

Uses BGE-M3 embeddings and FAISS vector similarity search
to retrieve semantically relevant medical document chunks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from app.rag.chunking.models import MedicalChunk
from app.rag.embeddings.bge_embedding import BGEEmbedding
from app.rag.vectorstore.faiss_store import FAISSVectorStore


class DenseRetriever:
    """
    Dense semantic retriever using BGE-M3 + FAISS.

    The retriever is responsible for:
    1. Converting a natural-language query into a BGE-M3 embedding.
    2. Searching the FAISS vector store.
    3. Returning ranked MedicalChunk objects with similarity scores.
    """

    def __init__(
        self,
        embedder: BGEEmbedding,
        vector_store: FAISSVectorStore,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    # ------------------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[MedicalChunk, float]]:
        """
        Retrieve the most semantically relevant medical chunks.

        Args:
            query:
                Natural-language medical question.

            top_k:
                Number of results to return.

        Returns:
            List of (MedicalChunk, similarity_score) tuples,
            ordered from highest similarity to lowest similarity.
        """

        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        # Generate BGE-M3 query embedding.
        query_embedding = self.embedder.embed_query(query)

        # Search FAISS.
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        return results

    # ------------------------------------------------------------------
    # INDEXING
    # ------------------------------------------------------------------

    
    def add_chunks(
        self,
        chunks: Sequence[MedicalChunk],
        batch_size: int | None = None,
        on_batch_complete=None,
    ) -> None:
        """
        Embed and index chunks in batches.

        Args:
            chunks: Medical chunks to index.
            batch_size: Maximum number of chunks per batch.
                Defaults to the embedder's configured batch size.
            on_batch_complete: Optional callback invoked after each
                successfully indexed batch with (processed, total).
        """
        if not chunks:
            return

        if batch_size is None:
            batch_size = self.embedder.batch_size

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        total = len(chunks)

        for start in range(0, total, batch_size):
            batch = chunks[start : start + batch_size]
            texts = [chunk.content for chunk in batch]

            embeddings = self.embedder.embed_documents(
                texts,
                batch_size=batch_size,
            )

            self.vector_store.add(
                embeddings=embeddings,
                chunks=batch,
            )

            processed = min(start + len(batch), total)

            if on_batch_complete is not None:
                on_batch_complete(processed, total)


    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """
        Save the underlying FAISS vector store.
        """

        self.vector_store.save(directory)

    @classmethod
    def load(
        cls,
        directory: str | Path,
        embedder: BGEEmbedding,
    ) -> "DenseRetriever":
        """
        Load a persisted dense retriever.

        The embedding model must still be supplied because it is
        required for converting future queries into vectors.
        """

        vector_store = FAISSVectorStore.load(directory)

        return cls(
            embedder=embedder,
            vector_store=vector_store,
        )

    # ------------------------------------------------------------------
    # INFORMATION
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Return the number of indexed chunks."""

        return self.vector_store.size