
"""
FAISS vector store for the RAGMedX-MA retrieval engine.

Features:
- Exact inner-product search using a flat FAISS index.
- Optional IVF indexing for larger knowledge bases.
- Stable chunk IDs and explicit FAISS ID mapping.
- Incremental insertion with duplicate-ID protection.
- Persistent index and metadata with backward-compatible loading.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

from app.rag.chunking.models import MedicalChunk


class FAISSVectorStore:
    """Persistent FAISS store with explicit chunk-to-vector ID mapping."""

    def __init__(
        self,
        dimension: int = 1024,
        index_type: str = "flat",
        nlist: int = 100,
        nprobe: int = 10,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero.")
        if index_type not in {"flat", "ivf"}:
            raise ValueError("index_type must be 'flat' or 'ivf'.")
        if nlist <= 0 or nprobe <= 0:
            raise ValueError("nlist and nprobe must be greater than zero.")

        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = min(nprobe, nlist)

        self.chunks: list[MedicalChunk] = []
        self._id_to_position: dict[int, int] = {}
        self._chunk_id_to_faiss_id: dict[str, int] = {}
        self._next_faiss_id = 0

        self.index = self._create_index()

    def _create_index(self):
        """Create a flat index or an IVF index wrapped with explicit IDs."""
        metric = faiss.METRIC_INNER_PRODUCT

        if self.index_type == "flat":
            base_index = faiss.IndexFlatIP(self.dimension)
        else:
            quantizer = faiss.IndexFlatIP(self.dimension)
            base_index = faiss.IndexIVFFlat(
                quantizer,
                self.dimension,
                self.nlist,
                metric,
            )

        index = faiss.IndexIDMap2(base_index)

        if self.index_type == "ivf":
            faiss.extract_index_ivf(index).nprobe = self.nprobe

        return index

    @property
    def size(self) -> int:
        """Number of indexed vectors."""
        return int(self.index.ntotal)

    @property
    def is_trained(self) -> bool:
        """Whether the underlying FAISS index is ready for insertion."""
        return bool(self.index.is_trained)

    def train(self, training_embeddings: np.ndarray) -> None:
        """Train an IVF index using representative embeddings."""
        if self.index_type != "ivf":
            return

        vectors = self._validate_embeddings(training_embeddings)

        if len(vectors) < self.nlist:
            raise ValueError(
                f"IVF training needs at least nlist={self.nlist} vectors; "
                f"received {len(vectors)}."
            )

        if self.size:
            raise ValueError(
                "Train the IVF index before adding vectors."
            )

        self.index.train(vectors)

    def _validate_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError(
                f"Embeddings must be 2-dimensional; got {vectors.shape}."
            )

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension {self.dimension}; "
                f"received {vectors.shape[1]}."
            )

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain NaN or infinite values.")

        return np.ascontiguousarray(vectors)

    @staticmethod
    def _chunk_id(chunk: MedicalChunk) -> str:
        """Prefer a supplied ID; otherwise derive a reproducible ID."""
        supplied_id = getattr(chunk, "id", None)

        if supplied_id is not None and str(supplied_id).strip():
            return str(supplied_id)

        metadata = getattr(chunk, "metadata", None) or {}

        if isinstance(metadata, dict):
            for key in ("chunk_id", "id"):
                value = metadata.get(key)
                if value is not None and str(value).strip():
                    return str(value)

        content = str(getattr(chunk, "content", ""))
        document_id = ""

        if isinstance(metadata, dict):
            document_id = str(
                metadata.get("document_id")
                or metadata.get("source")
                or metadata.get("pmid")
                or ""
            )

        raw = f"{document_id}\0{content}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def add(
        self,
        embeddings: np.ndarray,
        chunks: Sequence[MedicalChunk],
    ) -> None:
        """Incrementally add new chunks, skipping IDs already indexed."""
        vectors = self._validate_embeddings(embeddings)

        if len(vectors) != len(chunks):
            raise ValueError(
                f"Received {len(vectors)} embeddings for "
                f"{len(chunks)} chunks."
            )

        if not len(vectors):
            return

        if self.index_type == "ivf" and not self.is_trained:
            raise RuntimeError(
                "IVF index is not trained. Call train() with representative "
                "embeddings, or use index_type='flat' for small datasets."
            )

        new_vectors = []
        new_chunks = []
        new_ids = []
        seen_in_batch = set(self._chunk_id_to_faiss_id)

        for vector, chunk in zip(vectors, chunks):
            chunk_id = self._chunk_id(chunk)

            if chunk_id in seen_in_batch:
                continue

            seen_in_batch.add(chunk_id)
            new_vectors.append(vector)
            new_chunks.append(chunk)
            new_ids.append(self._next_faiss_id)
            self._next_faiss_id += 1

        if not new_vectors:
            return

        vector_array = np.ascontiguousarray(
            np.stack(new_vectors),
            dtype=np.float32,
        )
        ids_array = np.asarray(new_ids, dtype=np.int64)

        self.index.add_with_ids(vector_array, ids_array)

        for chunk, faiss_id in zip(new_chunks, new_ids):
            position = len(self.chunks)
            self.chunks.append(chunk)
            self._id_to_position[faiss_id] = position
            self._chunk_id_to_faiss_id[self._chunk_id(chunk)] = faiss_id

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[MedicalChunk, float]]:
        """Return ranked (chunk, similarity) pairs."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.size == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)

        if query.ndim == 1:
            query = query.reshape(1, -1)

        query = self._validate_embeddings(query)

        if query.shape[0] != 1:
            raise ValueError("Search expects exactly one query embedding.")

        scores, ids = self.index.search(
            query,
            min(top_k, self.size),
        )

        results = []

        for score, faiss_id in zip(scores[0], ids[0]):
            if faiss_id < 0:
                continue

            position = self._id_to_position.get(int(faiss_id))
            if position is None:
                continue

            results.append((self.chunks[position], float(score)))

        return results

    def save(self, directory: str | Path) -> None:
        """Persist index and metadata, using temporary files before replace."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        index_path = directory / "index.faiss"
        metadata_path = directory / "chunks.pkl"
        config_path = directory / "index_config.json"

        metadata = {
            "version": 2,
            "chunks": self.chunks,
            "id_to_position": self._id_to_position,
            "chunk_id_to_faiss_id": self._chunk_id_to_faiss_id,
            "next_faiss_id": self._next_faiss_id,
            "index_type": self.index_type,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "dimension": self.dimension,
        }

        temp_paths = []

        try:
            for destination, writer in (
                (index_path, lambda path: faiss.write_index(self.index, str(path))),
                (
                    metadata_path,
                    lambda path: self._write_pickle(path, metadata),
                ),
                (
                    config_path,
                    lambda path: path.write_text(
                        json.dumps(
                            {
                                "version": 2,
                                "dimension": self.dimension,
                                "index_type": self.index_type,
                                "nlist": self.nlist,
                                "nprobe": self.nprobe,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    ),
                ),
            ):
                fd, temp_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.",
                    suffix=".tmp",
                    dir=directory,
                )
                os.close(fd)
                temp_path = Path(temp_name)
                temp_paths.append((temp_path, destination))
                writer(temp_path)

            for temp_path, destination in temp_paths:
                os.replace(temp_path, destination)

        finally:
            for temp_path, _ in temp_paths:
                if temp_path.exists():
                    temp_path.unlink()

    @staticmethod
    def _write_pickle(path: Path, value) -> None:
        with path.open("wb") as file:
            pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, directory: str | Path) -> "FAISSVectorStore":
        """Load a current-format or legacy flat-index store."""
        directory = Path(directory)
        index_path = directory / "index.faiss"
        metadata_path = directory / "chunks.pkl"
        config_path = directory / "index_config.json"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Chunk metadata not found: {metadata_path}"
            )

        index = faiss.read_index(str(index_path))

        with metadata_path.open("rb") as file:
            metadata = pickle.load(file)

        # Older versions stored only a list of chunks and used positional IDs.
        if isinstance(metadata, list):
            chunks = metadata

            if index.ntotal != len(chunks):
                raise ValueError(
                    "FAISS index and chunk metadata are inconsistent: "
                    f"{index.ntotal} vectors vs {len(chunks)} chunks."
                )

            # Legacy indexes used sequential positional IDs.
            # Rebuild the index as IndexIDMap2 so future additions
            # can safely use add_with_ids().
            store = cls(dimension=index.d, index_type="flat")

            if index.ntotal:
                vectors = index.reconstruct_n(0, index.ntotal)
                ids = np.arange(index.ntotal, dtype=np.int64)
                store.index.add_with_ids(
                    np.ascontiguousarray(vectors, dtype=np.float32),
                    ids,
                )

            store.chunks = chunks
            store._id_to_position = {
                i: i for i in range(len(chunks))
            }
            store._chunk_id_to_faiss_id = {
                store._chunk_id(chunk): i
                for i, chunk in enumerate(chunks)
            }
            store._next_faiss_id = len(chunks)

            return store

        config = {}
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))

        store = cls(
            dimension=int(metadata.get("dimension", index.d)),
            index_type=metadata.get(
                "index_type", config.get("index_type", "flat")
            ),
            nlist=int(metadata.get("nlist", config.get("nlist", 100))),
            nprobe=int(metadata.get("nprobe", config.get("nprobe", 10))),
        )
        store.index = index
        store.chunks = metadata["chunks"]
        store._id_to_position = {
            int(key): int(value)
            for key, value in metadata["id_to_position"].items()
        }
        store._chunk_id_to_faiss_id = {
            str(key): int(value)
            for key, value in metadata["chunk_id_to_faiss_id"].items()
        }
        store._next_faiss_id = int(metadata["next_faiss_id"])

        if store.size != len(store.chunks):
            raise ValueError(
                "FAISS index and chunk metadata are inconsistent: "
                f"{store.size} vectors vs {len(store.chunks)} chunks."
            )

        return store
