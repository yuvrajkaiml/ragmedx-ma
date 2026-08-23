"""BGE embedding adapter.

Wraps ``BAAI/bge-small-en-v1.5`` (or any model configured via
``Settings.embedding_model_name``) behind the ``Embedder`` interface so the
rest of the pipeline never imports sentence-transformers directly.

BGE-specific details
--------------------
* All vectors are L2-normalised (``normalize_embeddings=True``) so dot-product
  and cosine similarity are equivalent — this is required for FAISS IndexFlatIP.
* Query encoding prepends the BGE query instruction prefix
  ``"Represent this sentence for searching relevant passages: "``
  as recommended in the BAAI model card for asymmetric retrieval.
* Document encoding does *not* use a prefix (symmetric encoding).
* Batch size defaults to 64 and can be tuned per deployment via the constructor.

Thread / process safety
-----------------------
``SentenceTransformer`` is not thread-safe for concurrent ``encode()`` calls on
the same instance.  Keep one ``BGEEmbedder`` per process and, if you need
parallelism, use ``batch_size`` rather than threading.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.rag.interfaces import Embedder
from app.utils.logger import logger

# BGE asymmetric retrieval instruction — prepended to queries only.
# Passages / documents are encoded without a prefix.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class BGEEmbedder(Embedder):
    """Sentence-Transformers adapter for ``BAAI/bge-small-en-v1.5``.

    Parameters
    ----------
    model_name:
        HuggingFace model id or local path.  Defaults to
        ``Settings.embedding_model_name`` (``BAAI/bge-small-en-v1.5``).
    device:
        ``"cpu"``, ``"cuda"``, or ``None`` (auto-detect).
    batch_size:
        Number of texts encoded per forward pass.  64 is a safe default for
        CPU; raise to 256+ on GPU.
    cache_folder:
        Optional local directory for the downloaded model weights.
        Useful for air-gapped deployments.  ``None`` → HuggingFace default
        (``~/.cache/huggingface/hub``).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 64,
        cache_folder: Optional[str] = None,
    ) -> None:
        # Resolve model name: explicit arg → settings default
        if model_name is None:
            from app.config import get_settings
            model_name = get_settings().embedding_model_name

        self.model_name = model_name
        self.batch_size = batch_size

        logger.info(f"BGEEmbedder: loading '{model_name}' (device={device or 'auto'})")
        self._model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_folder,
        )
        # Retrieve output dimension once so callers can inspect it cheaply.
        # ST 5.x renamed get_sentence_embedding_dimension → get_embedding_dimension.
        getter = getattr(
            self._model,
            "get_embedding_dimension",
            getattr(self._model, "get_sentence_embedding_dimension", None),
        )
        self._dim: int = getter()
        logger.info(
            f"BGEEmbedder ready — model='{model_name}', dim={self._dim}, "
            f"batch_size={batch_size}"
        )

    # ------------------------------------------------------------------
    # Embedder interface
    # ------------------------------------------------------------------

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a batch of document passages.

        Passages are encoded *without* an instruction prefix (symmetric
        representation).  All vectors are L2-normalised.

        Parameters
        ----------
        texts:
            Non-empty sequence of passage strings.

        Returns
        -------
        list[list[float]]
            One float vector per input text, shape ``(len(texts), dim)``.

        Raises
        ------
        ValueError
            If *texts* is empty.
        """
        if not texts:
            raise ValueError("embed_documents requires at least one text.")

        texts = list(texts)
        logger.debug(f"BGEEmbedder.embed_documents: encoding {len(texts)} passage(s)")

        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Encode a single search query with the BGE instruction prefix.

        Parameters
        ----------
        text:
            The user's search query string.

        Returns
        -------
        list[float]
            A single L2-normalised float vector of length ``self.dim``.

        Raises
        ------
        ValueError
            If *text* is empty or whitespace-only.
        """
        if not text or not text.strip():
            raise ValueError("embed_query requires a non-empty query string.")

        logger.debug(f"BGEEmbedder.embed_query: '{text[:60]}{'...' if len(text) > 60 else ''}'")

        embedding: np.ndarray = self._model.encode(
            _QUERY_PREFIX + text,
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Output embedding dimension (384 for bge-small-en-v1.5)."""
        return self._dim
