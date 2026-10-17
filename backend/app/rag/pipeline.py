"""
RAG retrieval pipeline for the RAGMedX-MA system.

Composes hybrid retrieval and cross-encoder reranking, with optional
evidence-quality scoring and contradiction detection.

Evidence-quality scoring is a ranking aid, not a clinical risk-of-bias
assessment or a validated GRADE assessment. Contradiction detection reports
potentially conflicting evidence; it does not establish medical truth.
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
from app.rag.verification.evidence_quality_scorer import (
    EvidenceQualityScorer,
)


class RAGRetrievalPipeline:
    """Compose retrieval, reranking, and optional evidence verification."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        retrieval_top_k: int = 10,
        final_top_k: int = 5,
        contradiction_detector: MedicalContradictionDetector | None = None,
        evidence_quality_scorer: EvidenceQualityScorer | None = None,
        quality_weight: float = 0.15,
    ) -> None:
        if not isinstance(hybrid_retriever, HybridRetriever):
            raise TypeError("hybrid_retriever must be a HybridRetriever.")

        if not isinstance(reranker, CrossEncoderReranker):
            raise TypeError("reranker must be a CrossEncoderReranker.")

        if retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be greater than zero.")

        if final_top_k <= 0:
            raise ValueError("final_top_k must be greater than zero.")

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

        if (
            evidence_quality_scorer is not None
            and not isinstance(
                evidence_quality_scorer,
                EvidenceQualityScorer,
            )
        ):
            raise TypeError(
                "evidence_quality_scorer must be an "
                "EvidenceQualityScorer or None."
            )

        if not 0.0 <= quality_weight <= 1.0:
            raise ValueError("quality_weight must be between 0 and 1.")

        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.contradiction_detector = contradiction_detector
        self.evidence_quality_scorer = evidence_quality_scorer
        self.quality_weight = float(quality_weight)

        self.retrieval_top_k = retrieval_top_k
        self.final_top_k = final_top_k

    def add_chunks(self, chunks: Sequence[MedicalChunk]) -> None:
        """Add chunks to the underlying hybrid retrieval system."""
        if chunks:
            self.hybrid_retriever.add_chunks(chunks)

    @staticmethod
    def _normalize_relevance_scores(
        results: Sequence[tuple[MedicalChunk, float]],
    ) -> list[float]:
        """
        Normalize reranker scores to [0, 1].

        If all scores are equal, use the original reranker ordering so
        that ties do not cause arbitrary ranking changes.
        """
        scores = [float(score) for _, score in results]

        if not scores:
            return []

        minimum = min(scores)
        maximum = max(scores)
        score_range = maximum - minimum

        if score_range > 0.0:
            return [
                (score - minimum) / score_range
                for score in scores
            ]

        count = len(scores)

        if count == 1:
            return [1.0]

        return [
            (count - rank - 1) / (count - 1)
            for rank in range(count)
        ]

    def retrieve(
        self,
        query: str,
        retrieval_top_k: int | None = None,
        final_top_k: int | None = None,
    ) -> list[tuple[MedicalChunk, float]]:
        """
        Retrieve and rerank evidence.

        The returned tuple format remains (MedicalChunk, score).
        When quality scoring is enabled, quality affects the ordering,
        but the returned score remains the original cross-encoder score.
        """
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
            raise ValueError("retrieval_top_k must be greater than zero.")

        if result_k <= 0:
            raise ValueError("final_top_k must be greater than zero.")

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

        # Preserve the original pipeline behavior when quality scoring
        # is disabled or its contribution is zero.
        if (
            self.evidence_quality_scorer is None
            or self.quality_weight == 0.0
        ):
            return self.reranker.rerank(
                query=query,
                results=hybrid_results,
                top_k=result_k,
            )

        # Rerank all retrieved candidates first. Otherwise, a document
        # outside the reranker's final top-k could never be promoted by
        # the quality score.
        reranked_results = self.reranker.rerank(
            query=query,
            results=hybrid_results,
            top_k=len(hybrid_results),
        )

        if not reranked_results:
            return []

        if len(reranked_results) == 1:
            return reranked_results[:result_k]

        normalized_scores = self._normalize_relevance_scores(
            reranked_results
        )

        ranked_results: list[
            tuple[MedicalChunk, float, float, int]
        ] = []

        for original_rank, (
            chunk,
            relevance_score,
        ) in enumerate(reranked_results):
            quality_result = self.evidence_quality_scorer.score(chunk)

            try:
                quality_score = float(
                    quality_result["quality_score"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "EvidenceQualityScorer.score() must return a "
                    "numeric 'quality_score'."
                ) from exc

            if not 0.0 <= quality_score <= 1.0:
                raise ValueError(
                    "Evidence quality scores must be between 0 and 1."
                )

            normalized_relevance = normalized_scores[original_rank]

            combined_score = (
                (1.0 - self.quality_weight) * normalized_relevance
                + self.quality_weight * quality_score
            )

            ranked_results.append(
                (
                    chunk,
                    float(relevance_score),
                    combined_score,
                    original_rank,
                )
            )

        # Sort by combined score. The original rank breaks ties and
        # preserves the cross-encoder's preference where possible.
        ranked_results.sort(
            key=lambda item: (-item[2], item[3])
        )

        # Preserve the public return format and original reranker scores.
        return [
            (chunk, relevance_score)
            for chunk, relevance_score, _, _ in ranked_results[:result_k]
        ]

    def detect_contradictions(
        self,
        results: Sequence[tuple[MedicalChunk, float]],
    ) -> list[ContradictionResult]:
        """Report potential contradictions without changing evidence."""
        if self.contradiction_detector is None:
            return []

        return self.contradiction_detector.detect(results)

    @property
    def size(self) -> int:
        """Return the number of indexed medical chunks."""
        return self.hybrid_retriever.size
