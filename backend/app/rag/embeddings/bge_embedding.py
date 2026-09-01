"""
BGE-M3 embedding adapter for the RAGMedX-MA retrieval engine.

Generates dense vector embeddings for medical document chunks
and user queries using BAAI/bge-m3.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class BGEEmbedding:
    """Dense embedding generator using BGE-M3."""

    DEFAULT_MODEL = "BAAI/bge-m3"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.normalize = normalize

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

        print("BGE-M3 embedding model loaded successfully.")

    @staticmethod
    def _mean_pooling(
        model_output: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool token embeddings while ignoring padding tokens."""

        token_embeddings = model_output

        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        return torch.sum(
            token_embeddings * input_mask_expanded,
            dim=1,
        ) / torch.clamp(
            input_mask_expanded.sum(dim=1),
            min=1e-9,
        )

    def embed(
        self,
        texts: str | Sequence[str],
        batch_size: int = 8,
    ) -> np.ndarray:
        """
        Generate embeddings for one or more texts.

        Returns:
            np.ndarray of shape (n_texts, embedding_dimension)
        """

        if isinstance(texts, str):
            texts = [texts]

        texts = list(texts)

        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        embeddings: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]

                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                encoded = {
                    key: value.to(self.device)
                    for key, value in encoded.items()
                }

                model_output = self.model(**encoded)

                pooled = self._mean_pooling(
                    model_output.last_hidden_state,
                    encoded["attention_mask"],
                )

                if self.normalize:
                    pooled = torch.nn.functional.normalize(
                        pooled,
                        p=2,
                        dim=1,
                    )

                embeddings.append(
                    pooled.cpu().numpy().astype(np.float32)
                )

        return np.vstack(embeddings)

    def embed_documents(
        self,
        texts: Sequence[str],
        batch_size: int = 8,
    ) -> np.ndarray:
        """Generate embeddings for document chunks."""

        return self.embed(texts, batch_size=batch_size)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate an embedding for a user query."""

        embedding = self.embed(query)

        return embedding[0]
