"""
Cross-encoder reranker for the RAGMedX-MA retrieval engine.

Uses BAAI/bge-reranker-v2-m3 to rerank candidate medical chunks
returned by the hybrid retrieval stage.

Pipeline:

    Hybrid Retrieval
        ↓
    Candidate Chunks
        ↓
    Cross-Encoder
        ↓
    Reranked Chunks
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import CrossEncoder

from app.rag.chunking.models import MedicalChunk


class CrossEncoderReranker:
    """
    Reranks retrieved medical chunks using a cross-encoder.

    The cross-encoder jointly processes:

        (query, chunk.content)

    and produces a relevance score for each pair.
    """

    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        """
        Initialize the cross-encoder reranker.

        Args:
            model_name:
                Hugging Face model identifier.

            device:
                Device used for inference.
                Examples: "cpu", "cuda".
                If None, Sentence Transformers selects automatically.

            max_length:
                Maximum sequence length passed to the reranker.
        """

        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        print(
            f"Loading reranker model: {model_name}"
            + (f" on {device}" if device else "")
        )

        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
        )

        print("Cross-encoder reranker loaded successfully.")

    # ------------------------------------------------------------------
    # RERANKING
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        results: Sequence[tuple[MedicalChunk, float]],
        top_k: int = 5,
    ) -> list[tuple[MedicalChunk, float]]:
        """
        Rerank candidate chunks for a query.

        Args:
            query:
                Natural-language user query.

            results:
                Candidate retrieval results in the form:

                    [(MedicalChunk, retrieval_score), ...]

                The original retrieval score is not used to calculate
                the cross-encoder score.

            top_k:
                Number of reranked results to return.

        Returns:
            List of:

                [(MedicalChunk, reranker_score), ...]

            ordered from highest reranker score to lowest.
        """

        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not results:
            return []

        # --------------------------------------------------------------
        # Limit candidates before reranking.
        # --------------------------------------------------------------

        candidate_results = list(results)

        top_k = min(top_k, len(candidate_results))

        # --------------------------------------------------------------
        # Build query-document pairs.
        # --------------------------------------------------------------

        pairs = [
            (query, chunk.content)
            for chunk, _ in candidate_results
        ]

        # --------------------------------------------------------------
        # Generate cross-encoder relevance scores.
        # --------------------------------------------------------------

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        scores = np.asarray(scores).reshape(-1)

        if len(scores) != len(candidate_results):
            raise RuntimeError(
                "Reranker returned an unexpected number of scores: "
                f"{len(scores)} scores for "
                f"{len(candidate_results)} candidates."
            )

        # --------------------------------------------------------------
        # Combine chunks with reranker scores.
        # --------------------------------------------------------------

        reranked = [
            (candidate_results[index][0], float(scores[index]))
            for index in range(len(candidate_results))
        ]

        # --------------------------------------------------------------
        # Sort by reranker score.
        # --------------------------------------------------------------

        reranked.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return reranked[:top_k]

    # ------------------------------------------------------------------
    # CONVENIENCE METHOD
    # ------------------------------------------------------------------

    def rerank_chunks(
        self,
        query: str,
        chunks: Sequence[MedicalChunk],
        top_k: int = 5,
    ) -> list[tuple[MedicalChunk, float]]:
        """
        Rerank raw MedicalChunk objects.

        Useful when no previous retrieval scores are available.
        """

        results = [
            (chunk, 0.0)
            for chunk in chunks
        ]

        return self.rerank(
            query=query,
            results=results,
            top_k=top_k,
        )