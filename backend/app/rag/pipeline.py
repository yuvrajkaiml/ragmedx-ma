
"""
RAG retrieval pipeline for the RAGMedX-MA system.

Composes hybrid retrieval and cross-encoder reranking, with optional
medical evidence contradiction detection.

Contradiction detection reports potentially conflicting evidence. It
does not establish medical truth and does not remove retrieved chunks.
"""

from __future__ import annotations

from typing import Sequence

from app.rag.chunking.models import MedicalChunk
from app.rag.retrievers.hybrid_retriever import HybridRetriever
from app.rag.rerankers.cross_encoder_reranker import CrossEncoderReranker
from app.rag.verification.contradiction_detector import (
    ContradictionResult,
    MedicalContradictionDetector,
)


class RAGRetrievalPipeline:
    """Compose hybrid retrieval, reranking, and optional contradiction detection."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        retrieval_top_k: int = 10,
        final_top_k: int = 5,
        contradiction_detector: MedicalContradictionDetector | None = None,
    ) -> None:
        if not isinstance(hybrid_retriever, HybridRetriever):
            raise TypeError(
                "hybrid_retriever must be a HybridRetriever."
            )

        if not isinstance(reranker, CrossEncoderReranker):
            raise TypeError(
                "reranker must be a CrossEncoderReranker."
            )

        if retrieval_top_k <= 0:
            raise ValueError(
                "retrieval_top_k must be greater than zero."
            )

        if final_top_k <= 0:
            raise ValueError(
                "final_top_k must be greater than zero."
            )

        if final_top_k > retrieval_top_k:
            raise ValueError(
                "final_top_k cannot be greater than retrieval_top_k."
            )

        if (
            contradiction_detector is not None
            and not isinstance(
                contradiction_detector,
                MedicalContradictionDetector,
            )
        ):
            raise TypeError(
                "contradiction_detector must be a "
                "MedicalContradictionDetector or None."
            )

        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.contradiction_detector = contradiction_detector

        self.retrieval_top_k = retrieval_top_k
        self.final_top_k = final_top_k

    def add_chunks(self, chunks: Sequence[MedicalChunk]) -> None:
        """Add chunks to the underlying hybrid retrieval system."""
        if chunks:
            self.hybrid_retriever.add_chunks(chunks)

    def retrieve(
        self,
        query: str,
        retrieval_top_k: int | None = None,
        final_top_k: int | None = None,
    ) -> list[tuple[MedicalChunk, float]]:
        """Retrieve and rerank evidence, preserving the existing return format."""
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty.")

        candidate_k = (
            self.retrieval_top_k
            if retrieval_top_k is None
            else retrieval_top_k
        )
        result_k = (
            self.final_top_k
            if final_top_k is None
            else final_top_k
        )

        if candidate_k <= 0:
            raise ValueError(
                "retrieval_top_k must be greater than zero."
            )

        if result_k <= 0:
            raise ValueError(
                "final_top_k must be greater than zero."
            )

        if result_k > candidate_k:
            raise ValueError(
                "final_top_k cannot be greater than retrieval_top_k."
            )

        hybrid_results = self.hybrid_retriever.retrieve(
            query=query,
            top_k=candidate_k,
        )

        if not hybrid_results:
            return []

        return self.reranker.rerank(
            query=query,
            results=hybrid_results,
            top_k=result_k,
        )

    def detect_contradictions(
        self,
        results: Sequence[tuple[MedicalChunk, float]],
    ) -> list[ContradictionResult]:
        """Report contradictions among retrieved evidence.

        This method does not modify, reorder, or remove any evidence.
        If no detector is configured, it returns an empty list.
        """
        if self.contradiction_detector is None:
            return []

        return self.contradiction_detector.detect(results)

    @property
    def size(self) -> int:
        """Return the number of indexed medical chunks."""
        return self.hybrid_retriever.size
