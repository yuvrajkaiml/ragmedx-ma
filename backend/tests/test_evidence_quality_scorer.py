
import unittest

from app.rag.chunking.models import MedicalChunk
from app.rag.verification.evidence_quality_scorer import (
    EvidenceQualityScorer,
)


class TestEvidenceQualityScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = EvidenceQualityScorer(current_year=2026)

    def make_chunk(self, **overrides):
        values = {
            "id": "chunk_1",
            "document_id": "doc_1",
            "content": "Medical evidence content.",
            "title": "Example medical study",
            "section": "Results",
            "source": "PMC",
            "journal": "Example Journal",
            "publication_year": 2024,
            "metadata": {
                "study_type": "randomized_controlled_trial"
            },
        }
        values.update(overrides)
        return MedicalChunk(**values)

    def test_score_is_between_zero_and_one(self):
        result = self.scorer.score(self.make_chunk())
        self.assertGreaterEqual(result["quality_score"], 0.0)
        self.assertLessEqual(result["quality_score"], 1.0)

    def test_known_study_type_is_recognized(self):
        result = self.scorer.score(self.make_chunk())
        self.assertEqual(
            result["study_type"],
            "randomized_controlled_trial",
        )
        self.assertEqual(result["study_type_score"], 0.90)

    def test_newer_publication_has_higher_recency_score(self):
        recent = self.scorer.score(
            self.make_chunk(publication_year=2025)
        )
        older = self.scorer.score(
            self.make_chunk(publication_year=2010)
        )
        self.assertGreater(
            recent["recency_score"],
            older["recency_score"],
        )

    def test_missing_study_type_is_not_invented(self):
        result = self.scorer.score(
            self.make_chunk(metadata={})
        )
        self.assertIsNone(result["study_type"])
        self.assertIsNone(result["study_type_score"])

    def test_invalid_future_year_is_not_scored_for_recency(self):
        result = self.scorer.score(
            self.make_chunk(publication_year=2030)
        )
        self.assertIsNone(result["recency_score"])

    def test_score_many_preserves_chunk_ids(self):
        chunks = [
            self.make_chunk(id="chunk_a"),
            self.make_chunk(id="chunk_b"),
        ]
        results = self.scorer.score_many(chunks)
        self.assertEqual(
            [item["chunk_id"] for item in results],
            ["chunk_a", "chunk_b"],
        )


if __name__ == "__main__":
    unittest.main()
