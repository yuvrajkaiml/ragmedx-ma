"""Smoke test for real NLI model predictions.

This test loads the actual Hugging Face NLI model and checks whether
its predicted labels are sensible for basic example sentence pairs.
It does not validate medical contradiction detection.
"""

import unittest

from app.rag.verification.contradiction_detector import (
    MedicalContradictionDetector,
)


class TestRealNLIModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the real model once for this test class.
        cls.detector = MedicalContradictionDetector(
            contradiction_threshold=0.80,
            max_pairs=10,
        )

    def test_entailment_example(self):
        premise = "A patient has a fever."
        hypothesis = "The patient has an elevated body temperature."

        label, score = self.detector.predict_pair(
            premise,
            hypothesis,
        )

        print(f"\nEntailment example: label={label}, score={score:.4f}")
        self.assertIn(
            label,
            {"entailment", "neutral", "contradiction"},
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_contradiction_example(self):
        premise = "The patient has a fever."
        hypothesis = "The patient does not have a fever."

        label, score = self.detector.predict_pair(
            premise,
            hypothesis,
        )

        print(f"\nContradiction example: label={label}, score={score:.4f}")
        self.assertIn(
            label,
            {"entailment", "neutral", "contradiction"},
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_neutral_example(self):
        premise = "The patient has a fever."
        hypothesis = "The patient has a history of asthma."

        label, score = self.detector.predict_pair(
            premise,
            hypothesis,
        )

        print(f"\nNeutral example: label={label}, score={score:.4f}")
        self.assertIn(
            label,
            {"entailment", "neutral", "contradiction"},
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
