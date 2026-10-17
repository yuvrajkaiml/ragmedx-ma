
"""
Embedding services for RAGMedX-MA.

Supports:
- BGE-M3: 1024-dimensional dense embeddings.
- PubMedBERT: 768-dimensional mean-pooled token embeddings.

Both implementations support batch inference, document embeddings,
query embeddings, optional L2 normalization, and CPU/GPU execution.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class _TransformerEmbedding:
    """Shared transformer-based dense embedding implementation."""

    DEFAULT_MODEL = ""
    EXPECTED_DIMENSION: int | None = None

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        normalize: bool = True,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name or self.DEFAULT_MODEL
        self.normalize = normalize

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if max_length < 1:
            raise ValueError("max_length must be at least 1.")

        self.batch_size = batch_size
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        print(
            f"Loading embedding model: {self.model_name} "
            f"on {self.device}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.embedding_dimension = int(
            self.model.config.hidden_size
        )

        if (
            self.EXPECTED_DIMENSION is not None
            and self.embedding_dimension != self.EXPECTED_DIMENSION
        ):
            raise ValueError(
                f"{self.model_name} returned hidden size "
                f"{self.embedding_dimension}; expected "
                f"{self.EXPECTED_DIMENSION}."
            )

        print(
            f"Embedding model loaded: {self.model_name} "
            f"({self.embedding_dimension} dimensions)."
        )

    @staticmethod
    def _mean_pooling(
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool token vectors while excluding padding tokens."""

        expanded_mask = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .to(dtype=token_embeddings.dtype)
        )

        summed = torch.sum(
            token_embeddings * expanded_mask,
            dim=1,
        )
        counts = expanded_mask.sum(dim=1).clamp(min=1e-9)

        return summed / counts

    def embed(
        self,
        texts: str | Sequence[str],
        batch_size: int | None = None,
    ) -> np.ndarray:
        """
        Encode one text or a sequence of texts.

        Returns a float32 array with shape:
        (number_of_texts, embedding_dimension).
        """

        if isinstance(texts, str):
            texts = [texts]
        else:
            texts = list(texts)

        if not texts:
            return np.empty(
                (0, self.embedding_dimension),
                dtype=np.float32,
            )

        if any(not isinstance(text, str) for text in texts):
            raise TypeError("Every input text must be a string.")

        if batch_size is None:
            batch_size = self.batch_size

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        all_embeddings: list[np.ndarray] = []

        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]

                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                encoded = {
                    key: value.to(self.device)
                    for key, value in encoded.items()
                }

                output = self.model(**encoded)

                pooled = self._mean_pooling(
                    output.last_hidden_state,
                    encoded["attention_mask"],
                )

                if self.normalize:
                    pooled = torch.nn.functional.normalize(
                        pooled,
                        p=2,
                        dim=1,
                    )

                all_embeddings.append(
                    pooled.cpu().numpy().astype(
                        np.float32,
                        copy=False,
                    )
                )

        return np.concatenate(all_embeddings, axis=0)

    def embed_documents(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Generate embeddings for document chunks."""

        return self.embed(texts, batch_size=batch_size)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate a one-dimensional embedding for a query."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        return self.embed(query)[0]


class BGEEmbedding(_TransformerEmbedding):
    """Dense embeddings using BAAI/bge-m3."""

    DEFAULT_MODEL = "BAAI/bge-m3"
    EXPECTED_DIMENSION = 1024


class PubMedBERTEmbedding(_TransformerEmbedding):
    """
    Dense biomedical embeddings using PubMedBERT.

    Uses mean pooling over the model's last hidden state. This provides
    fixed-size vectors, but the base PubMedBERT model is not necessarily
    optimized for semantic similarity without task-specific fine-tuning.
    """

    DEFAULT_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    EXPECTED_DIMENSION = 768
