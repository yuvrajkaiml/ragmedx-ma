
import unittest
from unittest.mock import Mock

from app.rag.chunking.models import MedicalChunk
from app.rag.pipeline import RAGRetrievalPipeline
from app.rag.verification.contradiction_detector import (
    ContradictionResult,
    MedicalContradictionDetector,
)


def make_chunk(chunk_id: str, content: str) -> MedicalChunk:
    return MedicalChunk(
        id=chunk_id,
        document_id=f"doc-{chunk_id}",
        content=content,
        title=f"Study {chunk_id}",
        source="PMC",
        journal="Example Journal",
        publication_year=2024,
    )


class TestRAGPipelineContradictions(unittest.TestCase):
    def setUp(self):
        self.hybrid = Mock()
        self.reranker = Mock()

        # Bypass model-heavy constructor checks. These tests focus only
        # on the new integration method.
        self.pipeline = RAGRetrievalPipeline.__new__(
            RAGRetrievalPipeline
        )
        self.pipeline.hybrid_retriever = self.hybrid
        self.pipeline.reranker = self.reranker
        self.pipeline.retrieval_top_k = 10
        self.pipeline.final_top_k = 5
        self.pipeline.contradiction_detector = None

    def test_without_detector_returns_empty_list(self):
        a = make_chunk("a", "Treatment A improves the outcome.")
        b = make_chunk("b", "Treatment A has no effect.")

        results = [(a, 0.9), (b, 0.8)]

        contradictions = self.pipeline.detect_contradictions(results)

        self.assertEqual(contradictions, [])
        self.assertEqual(len(results), 2)

    def test_with_detector_returns_detected_contradictions(self):
        a = make_chunk("a", "Treatment A improves the outcome.")
        b = make_chunk("b", "Treatment A has no effect.")

        expected = ContradictionResult(
            chunk_a=a,
            chunk_b=b,
            score=0.95,
            source_a=a.source,
            source_b=b.source,
            title_a=a.title,
            title_b=b.title,
            publication_year_a=a.publication_year,
            publication_year_b=b.publication_year,
        )

        detector = Mock(spec=MedicalContradictionDetector)
        detector.detect.return_value = [expected]
        self.pipeline.contradiction_detector = detector

        results = [(a, 0.9), (b, 0.8)]
        original_ids = [chunk.id for chunk, _ in results]

        contradictions = self.pipeline.detect_contradictions(results)

        self.assertEqual(contradictions, [expected])
        detector.detect.assert_called_once_with(results)

        # Contradiction detection must not remove or reorder evidence.
        self.assertEqual(
            [chunk.id for chunk, _ in results],
            original_ids,
        )

    def test_empty_evidence_returns_no_contradictions(self):
        detector = Mock(spec=MedicalContradictionDetector)
        detector.detect.return_value = []
        self.pipeline.contradiction_detector = detector

        self.assertEqual(
            self.pipeline.detect_contradictions([]),
            [],
        )
        detector.detect.assert_called_once_with([])


if __name__ == "__main__":
    unittest.main()
