
import unittest

from app.rag.chunking.models import MedicalChunk
from app.rag.verification.contradiction_detector import (
    MedicalContradictionDetector,
)


def make_chunk(
    chunk_id: str,
    content: str,
    year: int = 2024,
) -> MedicalChunk:
    return MedicalChunk(
        id=chunk_id,
        document_id=f"doc-{chunk_id}",
        content=content,
        title=f"Study {chunk_id}",
        source="PMC",
        journal="Example Medical Journal",
        publication_year=year,
    )


class TestMedicalContradictionDetector(unittest.TestCase):
    def test_contradictory_pair_is_flagged(self):
        def predictor(premise, hypothesis):
            return {"label": "CONTRADICTION", "score": 0.95}

        detector = MedicalContradictionDetector(
            predictor=predictor,
            contradiction_threshold=0.80,
        )

        a = make_chunk("a", "Treatment A reduces blood pressure.")
        b = make_chunk("b", "Treatment A does not reduce blood pressure.")

        results = detector.detect([(a, 0.9), (b, 0.8)])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_a.id, "a")
        self.assertEqual(results[0].chunk_b.id, "b")
        self.assertAlmostEqual(results[0].score, 0.95)
        self.assertEqual(results[0].source_a, "PMC")
        self.assertEqual(results[0].publication_year_b, 2024)

    def test_entailment_is_not_flagged(self):
        def predictor(premise, hypothesis):
            return {"label": "ENTAILMENT", "score": 0.93}

        detector = MedicalContradictionDetector(predictor=predictor)

        a = make_chunk("a", "The treatment reduced blood pressure.")
        b = make_chunk("b", "Blood pressure decreased after treatment.")

        self.assertEqual(detector.detect([(a, 0.9), (b, 0.8)]), [])

    def test_neutral_pair_is_not_flagged(self):
        def predictor(premise, hypothesis):
            return {"label": "NEUTRAL", "score": 0.91}

        detector = MedicalContradictionDetector(predictor=predictor)

        a = make_chunk("a", "The patient received medication A.")
        b = make_chunk("b", "The study recruited 120 participants.")

        self.assertEqual(detector.detect([(a, 0.9), (b, 0.8)]), [])

    def test_low_contradiction_score_is_not_flagged(self):
        def predictor(premise, hypothesis):
            return {"label": "CONTRADICTION", "score": 0.60}

        detector = MedicalContradictionDetector(
            predictor=predictor,
            contradiction_threshold=0.80,
        )

        a = make_chunk("a", "Treatment A is effective.")
        b = make_chunk("b", "Treatment A is ineffective.")

        self.assertEqual(detector.detect([(a, 0.9), (b, 0.8)]), [])

    def test_empty_text_is_rejected(self):
        detector = MedicalContradictionDetector(
            predictor=lambda p, h: {
                "label": "NEUTRAL",
                "score": 0.5,
            }
        )

        with self.assertRaises(ValueError):
            detector.predict_pair("", "Some evidence.")

    def test_max_pairs_limit_is_respected(self):
        calls = []

        def predictor(premise, hypothesis):
            calls.append((premise, hypothesis))
            return {"label": "CONTRADICTION", "score": 0.99}

        detector = MedicalContradictionDetector(
            predictor=predictor,
            max_pairs=1,
        )

        chunks = [
            make_chunk("a", "Evidence A"),
            make_chunk("b", "Evidence B"),
            make_chunk("c", "Evidence C"),
        ]

        detector.detect([(chunk, 0.9) for chunk in chunks])

        # One pair, tested in both directions.
        self.assertEqual(len(calls), 2)

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            MedicalContradictionDetector(
                predictor=lambda p, h: {},
                contradiction_threshold=1.5,
            )


if __name__ == "__main__":
    unittest.main()
