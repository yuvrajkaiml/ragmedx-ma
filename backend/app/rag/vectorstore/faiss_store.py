"""
FAISS vector store for the RAGMedX-MA retrieval engine.

Stores dense embeddings generated from MedicalChunk objects and
maintains the corresponding chunk metadata for retrieval.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

from app.rag.chunking.models import MedicalChunk


class FAISSVectorStore:
    """
    Persistent FAISS vector store for medical document chunks.

    Uses IndexFlatIP (inner product). Since BGE-M3 embeddings are
    L2-normalized, inner product is equivalent to cosine similarity.
    """

    def __init__(self, dimension: int = 1024) -> None:
        self.dimension = dimension

        self.index = faiss.IndexFlatIP(self.dimension)

        # Position in this list corresponds directly to the FAISS vector ID.
        self.chunks: list[MedicalChunk] = []

    @property
    def size(self) -> int:
        """Return the number of vectors currently stored."""

        return self.index.ntotal

    def add(
        self,
        embeddings: np.ndarray,
        chunks: Sequence[MedicalChunk],
    ) -> None:
        """
        Add embeddings and their corresponding MedicalChunk objects.

        Args:
            embeddings: Array of shape (n, dimension).
            chunks: MedicalChunk objects corresponding to each embedding.
        """

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be 2-dimensional. "
                f"Received shape: {embeddings.shape}"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch. "
                f"Expected {self.dimension}, "
                f"received {embeddings.shape[1]}."
            )

        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Number of embeddings ({len(embeddings)}) does not match "
                f"number of chunks ({len(chunks)})."
            )

        if len(embeddings) == 0:
            return

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[MedicalChunk, float]]:
        """
        Search the vector store using a query embedding.

        Returns:
            List of (MedicalChunk, similarity_score) tuples,
            ordered from most similar to least similar.
        """

        if self.size == 0:
            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding dimension mismatch. "
                f"Expected {self.dimension}, "
                f"received {query_embedding.shape[1]}."
            )

        top_k = min(top_k, self.size)

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results: list[tuple[MedicalChunk, float]] = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            results.append(
                (
                    self.chunks[index],
                    float(score),
                )
            )

        return results

    def save(self, directory: str | Path) -> None:
        """
        Persist the FAISS index and chunk metadata to disk.
        """

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        index_path = directory / "index.faiss"
        metadata_path = directory / "chunks.pkl"

        faiss.write_index(
            self.index,
            str(index_path),
        )

        with open(metadata_path, "wb") as file:
            pickle.dump(
                self.chunks,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(
        cls,
        directory: str | Path,
    ) -> "FAISSVectorStore":
        """
        Load a previously persisted vector store.
        """

        directory = Path(directory)

        index_path = directory / "index.faiss"
        metadata_path = directory / "chunks.pkl"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Chunk metadata not found: {metadata_path}"
            )

        index = faiss.read_index(str(index_path))

        with open(metadata_path, "rb") as file:
            chunks = pickle.load(file)

        store = cls(dimension=index.d)
        store.index = index
        store.chunks = chunks

        if store.index.ntotal != len(store.chunks):
            raise ValueError(
                "FAISS index and chunk metadata are inconsistent: "
                f"{store.index.ntotal} vectors vs "
                f"{len(store.chunks)} chunks."
            )

        return store