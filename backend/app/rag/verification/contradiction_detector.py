
"""NLI-based contradiction detection for retrieved medical evidence.

This component flags potentially conflicting evidence. It does not establish
medical truth and must not automatically discard evidence.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from app.rag.chunking.models import MedicalChunk


@dataclass(frozen=True)
class EvidenceRelation:
    """Relationship predicted between two evidence passages."""

    chunk_a: MedicalChunk
    chunk_b: MedicalChunk
    relation: str
    score: float


@dataclass(frozen=True)
class ContradictionResult:
    """A potentially conflicting pair of evidence passages."""

    chunk_a: MedicalChunk
    chunk_b: MedicalChunk
    score: float
    source_a: str
    source_b: str
    title_a: Optional[str]
    title_b: Optional[str]
    publication_year_a: Optional[int]
    publication_year_b: Optional[int]


class MedicalContradictionDetector:
    """Detect potentially contradictory evidence using Natural Language Inference.

    Args:
        model_name: Hugging Face NLI model identifier.
        device: Hugging Face pipeline device (-1 for CPU, 0 for CUDA GPU 0).
        contradiction_threshold: Minimum contradiction probability to flag a pair.
        max_pairs: Maximum number of chunk pairs to evaluate per call.
        predictor: Optional callable for testing. Receives premise and hypothesis
            and returns a mapping containing 'label' and 'score'.
    """

    VALID_RELATIONS = {"contradiction", "entailment", "neutral"}

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        device: int = -1,
        contradiction_threshold: float = 0.80,
        max_pairs: int = 100,
        predictor: Optional[Callable] = None,
    ) -> None:
        if not 0.0 <= contradiction_threshold <= 1.0:
            raise ValueError("contradiction_threshold must be between 0 and 1.")
        if max_pairs < 1:
            raise ValueError("max_pairs must be at least 1.")

        self.model_name = model_name
        self.device = device
        self.contradiction_threshold = contradiction_threshold
        self.max_pairs = max_pairs
        self._predictor = predictor
        self._classifier = None

    def _load_classifier(self) -> None:
        """Load the NLI model only when inference is first requested."""
        if self._classifier is not None:
            return

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Contradiction detection requires transformers. "
                "Install the project's declared dependencies first."
            ) from exc

        self._classifier = pipeline(
            "text-classification",
            model=self.model_name,
            tokenizer=self.model_name,
            device=self.device,
        )

    @staticmethod
    def _normalise_label(label: str) -> str:
        """Normalize common NLI label spellings."""
        normalized = label.strip().lower().replace("-", "_").replace(" ", "_")

        aliases = {
            "contradiction": "contradiction",
            "contradictory": "contradiction",
            "entailment": "entailment",
            "entailed": "entailment",
            "neutral": "neutral",
        }

        if normalized in aliases:
            return aliases[normalized]

        # LABEL_0, LABEL_1, etc. are deliberately not guessed here:
        # their meanings depend on the model's label configuration.
        raise ValueError(
            f"Unrecognized NLI label {label!r}. "
            "Check the selected model's id2label configuration."
        )

    def predict_pair(
        self,
        premise: str,
        hypothesis: str,
    ) -> tuple[str, float]:
        """Classify one text pair.

        The returned relation is directional: premise entails hypothesis
        is not necessarily equivalent to hypothesis entails premise.
        """
        if not premise or not premise.strip():
            raise ValueError("premise must not be empty.")
        if not hypothesis or not hypothesis.strip():
            raise ValueError("hypothesis must not be empty.")

        if self._predictor is not None:
            prediction = self._predictor(premise, hypothesis)
        else:
            self._load_classifier()
            prediction = self._classifier(
                {
                    "text": premise,
                    "text_pair": hypothesis,
                },
                truncation=True,
                max_length=512,
            )

        if isinstance(prediction, list):
            if not prediction:
                raise RuntimeError("The NLI model returned no predictions.")
            prediction = prediction[0]

        if not isinstance(prediction, dict):
            raise TypeError("The NLI predictor must return a dictionary.")

        if "label" not in prediction or "score" not in prediction:
            raise ValueError("The NLI prediction must contain label and score.")

        raw_label = str(prediction["label"])

        if raw_label.upper().startswith("LABEL_") and self._classifier is not None:
            try:
                label_index = int(raw_label.split("_", 1)[1])
                id2label = self._classifier.model.config.id2label
                raw_label = id2label.get(
                    label_index,
                    id2label.get(str(label_index), raw_label),
                )
            except (ValueError, AttributeError):
                pass

        relation = self._normalise_label(raw_label)
        score = float(prediction["score"])

        if not 0.0 <= score <= 1.0:
            raise ValueError("The prediction score must be between 0 and 1.")

        # Use lightweight placeholder chunks only for the relation result.
        # detect() below returns the original, complete MedicalChunk objects.
        return relation, score

    def detect(
        self,
        results: Sequence[tuple[MedicalChunk, float]],
    ) -> list[ContradictionResult]:
        """Flag pairs of retrieved chunks with sufficiently strong contradictions.

        The retrieval scores in the input are accepted for compatibility with
        the retrieval pipeline. NLI confidence, not retrieval score, is used
        for the contradiction threshold.
        """
        chunks = [chunk for chunk, _score in results]
        contradictions: list[ContradictionResult] = []
        pairs_checked = 0

        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                if pairs_checked >= self.max_pairs:
                    return contradictions

                chunk_a = chunks[i]
                chunk_b = chunks[j]
                pairs_checked += 1

                # Check both directions because NLI entailment is directional.
                forward = self.predict_pair(
                    chunk_a.content,
                    chunk_b.content,
                )
                reverse = self.predict_pair(
                    chunk_b.content,
                    chunk_a.content,
                )

                # predict_pair returns (relation, score).
                # Conservatively flag a contradiction only when both
                # sentence orders predict contradiction above the threshold.
                if (
                    forward[0] != "contradiction"
                    or reverse[0] != "contradiction"
                ):
                    continue

                # Use the lower score so both directions must be confident.
                contradiction_score = min(
                    forward[1],
                    reverse[1],
                )

                if contradiction_score < self.contradiction_threshold:
                    continue

                best_score = contradiction_score

                contradictions.append(
                    ContradictionResult(
                        chunk_a=chunk_a,
                        chunk_b=chunk_b,
                        score=best_score,
                        source_a=chunk_a.source,
                        source_b=chunk_b.source,
                        title_a=chunk_a.title,
                        title_b=chunk_b.title,
                        publication_year_a=chunk_a.publication_year,
                        publication_year_b=chunk_b.publication_year,
                    )
                )

        return contradictions
