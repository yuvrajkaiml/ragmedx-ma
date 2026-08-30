"""FAISS vector-store adapter.

Implements the canonical :class:`~app.rag.interfaces.VectorStore` interface
using ``faiss.IndexFlatIP`` (inner-product search).

Normalisation contract
----------------------
``BGEEmbedder`` already produces L2-normalised vectors, so inner-product
search is equivalent to cosine similarity and no further normalisation is
performed inside this class.  **Embeddings passed to :meth:`upsert` and
:meth:`search` must already be L2-normalised** (unit vectors).  If they are
not, similarity scores will be incorrect but no exception is raised -- the
caller is responsible for normalising before calling this class.

Persistence layout
------------------
:meth:`save` writes two files to *index_dir*::

    <index_dir>/
        index.faiss      -- raw FAISS binary written by faiss.write_index()
        metadata.json    -- JSON mapping: dim, ntotal, and DocumentChunk dicts

:meth:`load` restores both and validates that the stored dimension matches
the dimension this instance was constructed with.

Thread safety
-------------
``faiss.IndexFlatIP`` is not thread-safe for concurrent writes.  Use one
``FAISSVectorStore`` per process and protect concurrent access externally if
needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from collections.abc import Sequence

import faiss
import numpy as np

from app.rag.interfaces import VectorStore
from app.schemas.retrieval import DocumentChunk, RetrievalResult
from app.utils.logger import logger

_INDEX_FILENAME = "index.faiss"
_METADATA_FILENAME = "metadata.json"


class FAISSVectorStore(VectorStore):
    """FAISS-backed vector store using inner-product (cosine) similarity.

    Parameters
    ----------
    dim:
        Embedding dimension.  Must match the output dimension of the
        :class:`~app.rag.embeddings.bge_embedding.BGEEmbedder` (384).
    index_dir:
        Optional directory for :meth:`save` / :meth:`load`.  Defaults to
        ``app.config.FAISS_INDEX_DIR``.  The directory is created on first
        :meth:`save` if it does not exist.
    """

    def __init__(
        self,
        dim: int = 384,
        index_dir: Optional[Path] = None,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"Embedding dimension must be positive, got {dim}.")

        self._dim = dim
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        # Ordered list of DocumentChunk dicts -- position i corresponds to
        # FAISS internal id i.
        self._chunks: list[dict] = []

        if index_dir is None:
            from app.config import FAISS_INDEX_DIR
            index_dir = FAISS_INDEX_DIR
        self._index_dir = Path(index_dir)

        logger.info(
            f"FAISSVectorStore initialised -- dim={dim}, "
            f"index_dir={self._index_dir}"
        )

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Add *chunks* with their pre-computed *embeddings* to the index.

        Parameters
        ----------
        chunks:
            Document chunks to store.
        embeddings:
            One L2-normalised embedding vector per chunk.  Must have the same
            length as *chunks* and each vector must have ``self.dim`` elements.

        Raises
        ------
        ValueError
            If *chunks* and *embeddings* lengths differ, or if any embedding
            has the wrong dimension.

        Notes
        -----
        Calling :meth:`upsert` with an empty sequence is a documented no-op.
        """
        if len(chunks) == 0:
            logger.debug("FAISSVectorStore.upsert called with empty sequence -- no-op.")
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings must have the same length, "
                f"got {len(chunks)} chunks and {len(embeddings)} embeddings."
            )

        # Validate and convert to float32 numpy array
        matrix = self._to_float32_matrix(embeddings)

        # Store serialisable chunk dicts in insertion order
        for chunk in chunks:
            self._chunks.append(chunk.model_dump(mode="json"))

        self._index.add(matrix)
        logger.info(
            f"FAISSVectorStore.upsert: added {len(chunks)} vector(s), "
            f"total={self._index.ntotal}"
        )

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Return the top-K most similar chunks for *query_embedding*.

        Parameters
        ----------
        query_embedding:
            An L2-normalised query vector of length ``self.dim``.
        top_k:
            Number of results to return.  If the index contains fewer vectors
            than *top_k*, all indexed vectors are returned (no padding).

        Returns
        -------
        list[RetrievalResult]
            Ordered by descending similarity score.  Empty list if the index
            contains no vectors.

        Raises
        ------
        ValueError
            If *top_k* <= 0 or if *query_embedding* has the wrong dimension.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")

        if self._index.ntotal == 0:
            logger.debug("FAISSVectorStore.search: index is empty, returning [].")
            return []

        query_matrix = self._to_float32_matrix([query_embedding])

        # Clamp top_k to the actual index size
        effective_k = min(top_k, self._index.ntotal)

        distances, indices = self._index.search(query_matrix, effective_k)

        results: list[RetrievalResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                # FAISS returns -1 as a sentinel for unfilled slots
                continue

            chunk_dict = self._chunks[int(idx)]
            chunk = DocumentChunk(**chunk_dict)
            # Clamp score to [0, 1] -- inner-product of unit vectors is in
            # [-1, 1]; cosine similarity of 1 = identical direction.
            score = float(max(0.0, min(1.0, dist)))
            results.append(RetrievalResult(chunk=chunk, relevance_score=score))

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, index_dir: Optional[Path] = None) -> None:
        """Persist the FAISS index and metadata mapping to *index_dir*.

        Parameters
        ----------
        index_dir:
            Target directory.  Defaults to the directory passed at
            construction time.  Created if it does not exist.

        Raises
        ------
        IOError
            If writing either file fails.
        """
        target = Path(index_dir) if index_dir is not None else self._index_dir
        target.mkdir(parents=True, exist_ok=True)

        index_path = target / _INDEX_FILENAME
        meta_path = target / _METADATA_FILENAME

        faiss.write_index(self._index, str(index_path))
        meta_path.write_text(
            json.dumps(
                {
                    "dim": self._dim,
                    "ntotal": self._index.ntotal,
                    "chunks": self._chunks,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(
            f"FAISSVectorStore.save: {self._index.ntotal} vector(s) -> {target}"
        )

    @classmethod
    def load(cls, index_dir: Path) -> "FAISSVectorStore":
        """Restore a previously saved index from *index_dir*.

        Parameters
        ----------
        index_dir:
            Directory containing ``index.faiss`` and ``metadata.json``.

        Returns
        -------
        FAISSVectorStore
            A fully initialised store with all previously added chunks.

        Raises
        ------
        FileNotFoundError
            If either persistence file is missing.
        ValueError
            If the stored dimension is incompatible with the loaded index.
        """
        index_dir = Path(index_dir)
        index_path = index_dir / _INDEX_FILENAME
        meta_path = index_dir / _METADATA_FILENAME

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index file not found: {index_path}"
            )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"FAISS metadata file not found: {meta_path}"
            )

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stored_dim: int = meta["dim"]
        chunks: list[dict] = meta["chunks"]

        raw_index = faiss.read_index(str(index_path))

        # Validate dimension consistency
        if raw_index.d != stored_dim:
            raise ValueError(
                f"Dimension mismatch: index.faiss has d={raw_index.d} but "
                f"metadata.json records dim={stored_dim}."
            )

        store = cls.__new__(cls)
        store._dim = stored_dim
        store._index = raw_index
        store._chunks = chunks
        store._index_dir = index_dir

        logger.info(
            f"FAISSVectorStore.load: {raw_index.ntotal} vector(s) from {index_dir}"
        )
        return store

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Expected embedding dimension."""
        return self._dim

    @property
    def ntotal(self) -> int:
        """Number of vectors currently in the index."""
        return self._index.ntotal

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_float32_matrix(
        self, embeddings: Sequence[Sequence[float]]
    ) -> np.ndarray:
        """Validate and convert embeddings to a contiguous float32 matrix.

        Raises
        ------
        ValueError
            If any vector has the wrong dimension.
        """
        matrix = np.array(embeddings, dtype=np.float32)

        if matrix.ndim != 2:
            raise ValueError(
                f"Expected a 2-D array of embeddings, got shape {matrix.shape}."
            )
        if matrix.shape[1] != self._dim:
            raise ValueError(
                f"Expected embedding dimension {self._dim}, "
                f"received {matrix.shape[1]}."
            )

        # FAISS requires C-contiguous float32 arrays
        return np.ascontiguousarray(matrix, dtype=np.float32)
