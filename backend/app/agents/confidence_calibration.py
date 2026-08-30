from datetime import datetime, timezone
from typing import Any, Dict, List

from .base import BaseAgent
from .state import ConfidenceCalibrationOutput

# Configuration Constants
CONFIDENCE_THRESHOLD = 0.65
RETRIEVAL_WEIGHT = 0.25
EVIDENCE_QUALITY_WEIGHT = 0.30
ANSWER_CONSISTENCY_WEIGHT = 0.25
INTER_SOURCE_AGREEMENT_WEIGHT = 0.20


class ConfidenceCalibrationAgent(BaseAgent):
    """
    Agent 7: Confidence Calibration.

    Converts evidence and critic signals into a calibrated confidence score
    using the 4 configured components and weights:

    composite = 0.25 * retrieval_confidence
              + 0.30 * evidence_quality_score
              + 0.25 * answer_consistency_score
              + 0.20 * inter_source_agreement
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
            evidence = state.get("evidence", [])
            critic = state.get("critic_verdict")

            if critic is None:
                raise ValueError("Confidence calibration requires 'critic_verdict'.")

            # 1. Retrieval Confidence (from relevance scores)
            retrieval_confidence = self._calculate_retrieval_confidence(evidence)

            # 2. Evidence Quality Score (from cross encoder & evidence quality)
            evidence_quality = self._calculate_evidence_quality_score(evidence)

            # 3. Answer Consistency Score (from critic factual consistency & hallucination safety)
            factual_consistency = self._score(critic.get("factual_consistency", 0.0))
            hallucination_risk = self._score(critic.get("hallucination_risk", 1.0))
            hallucination_safety = 1.0 - hallucination_risk
            answer_consistency = self._score(0.5 * factual_consistency + 0.5 * hallucination_safety)

            # 4. Inter-Source Agreement (from citation support and multi-source consensus)
            citation_support = self._score(critic.get("citation_support", 0.0))
            inter_source_agreement = citation_support

            # 4-Component Composite Formula
            composite = (
                RETRIEVAL_WEIGHT * retrieval_confidence
                + EVIDENCE_QUALITY_WEIGHT * evidence_quality
                + ANSWER_CONSISTENCY_WEIGHT * answer_consistency
                + INTER_SOURCE_AGREEMENT_WEIGHT * inter_source_agreement
            )

            # Clamp composite score to [0.0, 1.0]
            calibrated_confidence = round(self._score(composite), 4)

            # Categorize Confidence
            confidence_band = self._confidence_band(calibrated_confidence)

            # Check Abstention Thresholds
            requires_abstention = (
                calibrated_confidence < CONFIDENCE_THRESHOLD
                or not evidence
                or factual_consistency < 0.50
                or hallucination_risk >= 0.50
                or citation_support < 0.50
            )

            reasoning = self._build_reasoning(
                retrieval_confidence=retrieval_confidence,
                evidence_quality=evidence_quality,
                answer_consistency=answer_consistency,
                inter_source_agreement=inter_source_agreement,
                calibrated_confidence=calibrated_confidence,
                requires_abstention=requires_abstention,
            )

            output = ConfidenceCalibrationOutput(
                calibrated_confidence=calibrated_confidence,
                retrieval_confidence=retrieval_confidence,
                evidence_quality_score=evidence_quality,
                answer_consistency_score=answer_consistency,
                inter_source_agreement=inter_source_agreement,
                evidence_score=evidence_quality,
                factual_consistency_score=factual_consistency,
                citation_support_score=citation_support,
                hallucination_safety_score=hallucination_safety,
                confidence_band=confidence_band,
                requires_abstention=requires_abstention,
                reasoning=reasoning,
            )

            completed_at = datetime.now(timezone.utc)

            trace_step = self.create_trace_step(
                status="completed",
                started_at=started_at,
                completed_at=completed_at,
                details={
                    "calibrated_confidence": calibrated_confidence,
                    "retrieval_confidence": retrieval_confidence,
                    "evidence_quality_score": evidence_quality,
                    "answer_consistency_score": answer_consistency,
                    "inter_source_agreement": inter_source_agreement,
                    "confidence_band": confidence_band,
                    "requires_abstention": requires_abstention,
                },
            )

            return {
                "confidence_calibration": output.model_dump(),
                "agent_trace": [trace_step],
            }

        except Exception as exc:
            completed_at = datetime.now(timezone.utc)

            trace_step = self.create_trace_step(
                status="failed",
                started_at=started_at,
                completed_at=completed_at,
                details={"error": str(exc)},
            )

            raise RuntimeError(f"Confidence Calibration Agent failed: {exc}") from exc

    @staticmethod
    def _score(value: Any) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, value))

    @classmethod
    def _calculate_retrieval_confidence(cls, evidence: List[Any]) -> float:
        if not evidence:
            return 0.0
        scores = []
        for item in evidence:
            if isinstance(item, dict):
                relevance = item.get("relevance_score", 0.0)
            else:
                relevance = getattr(item, "relevance_score", 0.0)
            scores.append(cls._score(relevance))
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    @classmethod
    def _calculate_evidence_quality_score(cls, evidence: List[Any]) -> float:
        if not evidence:
            return 0.0
        scores = []
        for item in evidence:
            if isinstance(item, dict):
                quality = item.get("evidence_quality_score", item.get("relevance_score", 0.0))
                cross = item.get("cross_encoder_score", quality)
            else:
                quality = getattr(item, "evidence_quality_score", getattr(item, "relevance_score", 0.0))
                cross = getattr(item, "cross_encoder_score", quality)

            item_score = 0.5 * cls._score(quality) + 0.5 * cls._score(cross)
            scores.append(item_score)
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    @staticmethod
    def _confidence_band(confidence: float) -> str:
        if confidence >= 0.80:
            return "high"
        if confidence >= 0.65:
            return "moderate"
        return "low"

    @staticmethod
    def _build_reasoning(
        retrieval_confidence: float,
        evidence_quality: float,
        answer_consistency: float,
        inter_source_agreement: float,
        calibrated_confidence: float,
        requires_abstention: bool,
    ) -> str:
        decision = "abstention is recommended" if requires_abstention else "answer may proceed"
        return (
            f"Calibrated confidence is computed from retrieval={retrieval_confidence:.2f}, "
            f"evidence_quality={evidence_quality:.2f}, answer_consistency={answer_consistency:.2f}, "
            f"inter_source_agreement={inter_source_agreement:.2f}. "
            f"Composite confidence={calibrated_confidence:.2f}; {decision}."
        )