import unittest
from unittest.mock import Mock

from app.rag.chunking.models import MedicalChunk
from app.rag.pipeline import RAGRetrievalPipeline
from app.rag.rerankers.cross_encoder_reranker import CrossEncoderReranker
from app.rag.retrievers.hybrid_retriever import HybridRetriever
from app.rag.verification.evidence_quality_scorer import (
    EvidenceQualityScorer,
)


def make_chunk(chunk_id, study_type):
    return MedicalChunk(
        id=chunk_id,
        document_id=f"doc-{chunk_id}",
        content=f"Medical evidence {chunk_id}",
        title=f"Study {chunk_id}",
        source="PMC",
        journal="Example Journal",
        publication_year=2025,
        metadata={"study_type": study_type},
    )


class TestRAGPipelineQuality(unittest.TestCase):
    def setUp(self):
        self.hybrid = Mock()
        self.reranker = Mock()

        self.pipeline = RAGRetrievalPipeline.__new__(
            RAGRetrievalPipeline
        )
        self.pipeline.hybrid_retriever = self.hybrid
        self.pipeline.reranker = self.reranker
        self.pipeline.retrieval_top_k = 10
        self.pipeline.final_top_k = 2
        self.pipeline.contradiction_detector = None
        self.pipeline.evidence_quality_scorer = None
        self.pipeline.quality_weight = 0.15

        self.high_quality = make_chunk("review", "systematic_review")
        self.low_quality = make_chunk("case", "case_report")

        self.hybrid.retrieve.return_value = [
            (self.low_quality, 0.9),
            (self.high_quality, 0.8),
        ]
        self.reranker.rerank.return_value = [
            (self.low_quality, 2.0),
            (self.high_quality, 1.0),
        ]

    def test_quality_scoring_disabled_preserves_existing_behavior(self):
        results = self.pipeline.retrieve("medical question")

        self.assertEqual(
            [chunk.id for chunk, _ in results],
            ["case", "review"],
        )
        self.reranker.rerank.assert_called_once_with(
            query="medical question",
            results=self.hybrid.retrieve.return_value,
            top_k=2,
        )

    def test_quality_scoring_changes_ranking_when_enabled(self):
        self.pipeline.evidence_quality_scorer = EvidenceQualityScorer(
            current_year=2025
        )
        self.pipeline.quality_weight = 0.80

        results = self.pipeline.retrieve("medical question")

        self.assertEqual(
            [chunk.id for chunk, _ in results],
            ["review", "case"],
        )

        # Returned scores remain the original cross-encoder scores.
        self.assertEqual(
            dict((chunk.id, score) for chunk, score in results),
            {"review": 1.0, "case": 2.0},
        )

        self.reranker.rerank.assert_called_once_with(
            query="medical question",
            results=self.hybrid.retrieve.return_value,
            top_k=2,
        )

    def test_quality_weight_zero_preserves_reranker_order(self):
        self.pipeline.evidence_quality_scorer = EvidenceQualityScorer(
            current_year=2025
        )
        self.pipeline.quality_weight = 0.0

        results = self.pipeline.retrieve("medical question")

        self.assertEqual(
            [chunk.id for chunk, _ in results],
            ["case", "review"],
        )

    def test_invalid_quality_weight_is_rejected(self):
        with self.assertRaises(ValueError):
            RAGRetrievalPipeline(
                hybrid_retriever=Mock(spec=HybridRetriever),
                reranker=Mock(spec=CrossEncoderReranker),
                quality_weight=1.5,
            )


if __name__ == "__main__":
    unittest.main()