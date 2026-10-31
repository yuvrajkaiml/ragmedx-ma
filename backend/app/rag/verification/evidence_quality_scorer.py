
"""Heuristic evidence-quality scoring for medical retrieval."""

from datetime import datetime
from typing import Any, Mapping, Optional

from app.rag.chunking.models import MedicalChunk


class EvidenceQualityScorer:
    """
    Assigns a transparent heuristic quality score to medical evidence.

    This is not a clinical risk-of-bias assessment or a validated GRADE
    assessment. Missing metadata is treated as unknown, not as proof
    of poor-quality research.
    """

    # Heuristic study-type weights, not validated clinical GRADE ratings.
    STUDY_TYPE_WEIGHTS = {
        "systematic_review": 1.00,
        "meta_analysis": 1.00,
        "randomized_controlled_trial": 0.90,
        "rct": 0.90,
        "cohort_study": 0.75,
        "case_control_study": 0.70,
        "cross_sectional_study": 0.60,
        "case_report": 0.40,
        "case_series": 0.45,
        "animal_study": 0.35,
        "in_vitro_study": 0.30,
    }

    def __init__(
        self,
        current_year: Optional[int] = None,
        recency_half_life_years: float = 10.0,
    ) -> None:
        if recency_half_life_years <= 0:
            raise ValueError(
                "recency_half_life_years must be greater than zero"
            )

        self.current_year = current_year or datetime.now().year
        self.recency_half_life_years = recency_half_life_years

    @staticmethod
    def _normalize_study_type(value: Any) -> Optional[str]:
        if not isinstance(value, str) or not value.strip():
            return None

        normalized = value.lower().strip()
        normalized = normalized.replace("-", "_").replace(" ", "_")

        aliases = {
            "systematic_review_and_meta_analysis": "systematic_review",
            "randomised_controlled_trial": "randomized_controlled_trial",
            "randomized_trial": "randomized_controlled_trial",
            "clinical_trial": "randomized_controlled_trial",
            "case_series_report": "case_series",
            "in_vitro": "in_vitro_study",
            "in_vivo_animal_study": "animal_study",
        }

        return aliases.get(normalized, normalized)

    @staticmethod
    def _get_study_type(chunk: MedicalChunk) -> Optional[str]:
        metadata = chunk.metadata or {}

        # Check explicit metadata only. Do not infer a study type
        # merely from the title or document source.
        for key in ("study_type", "publication_type", "article_type"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return EvidenceQualityScorer._normalize_study_type(value)

        return None

    def _recency_score(self, publication_year: Optional[int]) -> Optional[float]:
        if publication_year is None:
            return None

        if not isinstance(publication_year, int):
            return None

        if publication_year < 1000 or publication_year > self.current_year:
            return None

        age = self.current_year - publication_year

        # Exponential decay: newer evidence gets a higher recency score.
        return 2 ** (-age / self.recency_half_life_years)

    @staticmethod
    def _metadata_completeness(chunk: MedicalChunk) -> float:
        fields = [
            bool(chunk.title and chunk.title.strip()),
            bool(chunk.journal and chunk.journal.strip()),
            chunk.publication_year is not None,
            bool(chunk.source and chunk.source.strip()),
        ]
        return sum(fields) / len(fields)

    def score(self, chunk: MedicalChunk) -> dict[str, Any]:
        """
        Return component scores and a combined score between 0 and 1.

        Unknown study type and publication year are excluded from the
        weighted average rather than assigned arbitrary low scores.
        """

        if not isinstance(chunk, MedicalChunk):
            raise TypeError("chunk must be a MedicalChunk instance")

        study_type = self._get_study_type(chunk)
        recency = self._recency_score(chunk.publication_year)
        metadata_completeness = self._metadata_completeness(chunk)

        components = [
            (self.STUDY_TYPE_WEIGHTS.get(study_type), 0.60),
            (recency, 0.25),
            (metadata_completeness, 0.15),
        ]

        available = [
            (value, weight)
            for value, weight in components
            if value is not None
        ]

        total_weight = sum(weight for _, weight in available)

        quality_score = (
            sum(value * weight for value, weight in available)
            / total_weight
            if total_weight > 0
            else metadata_completeness
        )

        return {
            "quality_score": round(quality_score, 4),
            "study_type": study_type,
            "study_type_score": self.STUDY_TYPE_WEIGHTS.get(study_type),
            "recency_score": (
                round(recency, 4) if recency is not None else None
            ),
            "metadata_completeness": round(metadata_completeness, 4),
            "assessment": (
                "heuristic"
                if study_type is not None or recency is not None
                else "metadata_only"
            ),
        }

    def score_many(
        self, chunks: list[MedicalChunk]
    ) -> list[dict[str, Any]]:
        """Score multiple chunks while preserving their input order."""

        return [
            {
                "chunk_id": chunk.id,
                **self.score(chunk),
            }
            for chunk in chunks
        ]
