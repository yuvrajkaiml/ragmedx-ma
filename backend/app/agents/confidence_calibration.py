from datetime import datetime, timezone
from typing import Any, Dict, List

from .base import BaseAgent
from .state import ConfidenceCalibrationOutput


class ConfidenceCalibrationAgent(BaseAgent):
    """
    Agent 7: Confidence Calibration.

    Converts evidence and critic signals into a
    calibrated confidence score.

    This agent does not generate medical content.
    It estimates how trustworthy the generated answer is.
    """

    def __init__(self):
        super().__init__(
            agent_name="Confidence Calibration",
            agent_type="confidence_calibration",
            llm_wrapper=None,
        )

    async def run(
        self,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:

        started_at = datetime.now(timezone.utc)

        try:
            evidence = state.get(
                "evidence",
                []
            )

            critic = state.get(
                "critic_verdict"
            )

            if critic is None:
                raise ValueError(
                    "Confidence calibration requires "
                    "'critic_verdict'."
                )

            evidence_score = (
                self._calculate_evidence_score(
                    evidence
                )
            )

            factual_consistency = self._score(
                critic.get(
                    "factual_consistency",
                    0.0
                )
            )

            citation_support = self._score(
                critic.get(
                    "citation_support",
                    0.0
                )
            )

            hallucination_risk = self._score(
                critic.get(
                    "hallucination_risk",
                    1.0
                )
            )

            hallucination_safety = (
                1.0 - hallucination_risk
            )

            critic_confidence = self._score(
                critic.get(
                    "confidence",
                    0.0
                )
            )

            calibrated_confidence = (
                0.25 * evidence_score
                + 0.25 * factual_consistency
                + 0.20 * citation_support
                + 0.20 * hallucination_safety
                + 0.10 * critic_confidence
            )

            calibrated_confidence = round(
                calibrated_confidence,
                4,
            )

            confidence_band = (
                self._confidence_band(
                    calibrated_confidence
                )
            )

            requires_abstention = (
                evidence_score <= 0.0
                or calibrated_confidence < 0.50
                or hallucination_risk >= 0.50
                or factual_consistency < 0.50
                or citation_support < 0.50
            )

            reasoning = (
                self._build_reasoning(
                    evidence_score=evidence_score,
                    factual_consistency=
                        factual_consistency,
                    citation_support=
                        citation_support,
                    hallucination_safety=
                        hallucination_safety,
                    critic_confidence=
                        critic_confidence,
                    calibrated_confidence=
                        calibrated_confidence,
                    requires_abstention=
                        requires_abstention,
                )
            )

            output = ConfidenceCalibrationOutput(
                calibrated_confidence=
                    calibrated_confidence,
                evidence_score=evidence_score,
                factual_consistency_score=
                    factual_consistency,
                citation_support_score=
                    citation_support,
                hallucination_safety_score=
                    hallucination_safety,
                confidence_band=
                    confidence_band,
                requires_abstention=
                    requires_abstention,
                reasoning=reasoning,
            )

            completed_at = datetime.now(
                timezone.utc
            )

            trace_step = self.create_trace_step(
                status="completed",
                started_at=started_at,
                completed_at=completed_at,
                details={
                    "calibrated_confidence":
                        calibrated_confidence,
                    "confidence_band":
                        confidence_band,
                    "requires_abstention":
                        requires_abstention,
                },
            )

            return {
                "confidence_calibration":
                    output.model_dump(),

                "agent_trace": [
                    trace_step
                ],
            }

        except Exception as exc:

            completed_at = datetime.now(
                timezone.utc
            )

            trace_step = self.create_trace_step(
                status="failed",
                started_at=started_at,
                completed_at=completed_at,
                details={
                    "error": str(exc)
                },
            )

            raise RuntimeError(
                "Confidence Calibration Agent failed."
            ) from exc

    @staticmethod
    def _score(value: Any) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0

        return max(
            0.0,
            min(1.0, value)
        )

    @classmethod
    def _calculate_evidence_score(
        cls,
        evidence: List[Any],
    ) -> float:

        if not evidence:
            return 0.0

        scores = []

        for item in evidence:

            if isinstance(item, dict):

                relevance = item.get(
                    "relevance_score",
                    0.0,
                )

                cross_encoder = item.get(
                    "cross_encoder_score",
                    0.0,
                )

                quality = item.get(
                    "evidence_quality_score",
                    0.0,
                )

            else:

                relevance = getattr(
                    item,
                    "relevance_score",
                    0.0,
                )

                cross_encoder = getattr(
                    item,
                    "cross_encoder_score",
                    0.0,
                )

                quality = getattr(
                    item,
                    "evidence_quality_score",
                    0.0,
                )

            item_score = (
                0.40 * cls._score(relevance)
                + 0.30 * cls._score(cross_encoder)
                + 0.30 * cls._score(quality)
            )

            scores.append(item_score)

        if not scores:
            return 0.0

        return round(
            sum(scores) / len(scores),
            4,
        )

    @staticmethod
    def _confidence_band(
        confidence: float,
    ) -> str:

        if confidence >= 0.80:
            return "high"

        if confidence >= 0.60:
            return "moderate"

        if confidence >= 0.40:
            return "low"

        return "very_low"

    @staticmethod
    def _build_reasoning(
        evidence_score: float,
        factual_consistency: float,
        citation_support: float,
        hallucination_safety: float,
        critic_confidence: float,
        calibrated_confidence: float,
        requires_abstention: bool,
    ) -> str:

        decision = (
            "abstention is recommended"
            if requires_abstention
            else "answer may proceed"
        )

        return (
            "Calibrated confidence is based on "
            f"evidence={evidence_score:.2f}, "
            f"factual_consistency="
            f"{factual_consistency:.2f}, "
            f"citation_support="
            f"{citation_support:.2f}, "
            f"hallucination_safety="
            f"{hallucination_safety:.2f}, "
            f"critic_confidence="
            f"{critic_confidence:.2f}. "
            f"Final confidence="
            f"{calibrated_confidence:.2f}; "
            f"{decision}."
        )