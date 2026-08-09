from datetime import datetime, timezone
from typing import Any, Dict

from .base import BaseAgent
from .state import AbstentionDecisionOutput

class AbstentionDecisionAgent(BaseAgent):
    """
    Agent 8: Abstention Decision.

    Consumes the output of Agent 7 (Confidence Calibration)
    and makes the final safety decision:

        PROCEED
            or
        ABSTAIN

    This agent does not recalculate confidence.
    It applies explicit safety rules to the calibrated
    confidence output.
    """

    def __init__(self):
        super().__init__(
            agent_name="Abstention Decision",
            agent_type="abstention_decision",
            llm_wrapper=None,
        )

    async def run(
        self,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:

        started_at = datetime.now(timezone.utc)

        try:
            calibration = state.get(
                "confidence_calibration"
            )

            if calibration is None:
                raise ValueError(
                    "Abstention decision requires "
                    "'confidence_calibration'."
                )

            decision, reasons = self._make_decision(
                calibration
            )

            reasoning = self._build_reasoning(
                calibration=calibration,
                decision=decision,
                reasons=reasons,
            )

            output = AbstentionDecisionOutput(
                decision=decision,
                should_abstain=decision == "ABSTAIN",
                reasons=reasons,
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
                    "decision": decision,
                    "should_abstain":
                        decision == "ABSTAIN",
                    "reasons": reasons,
                },
            )

            return {
                    "abstention_decision": output.model_dump(),
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
                "Abstention Decision Agent failed."
            ) from exc

    @staticmethod
    def _score(
        value: Any,
    ) -> float:
        """
        Normalize a score into the [0, 1] range.
        """
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0

        return max(
            0.0,
            min(1.0, value)
        )

    @classmethod
    def _make_decision(
        cls,
        calibration: Dict[str, Any],
    ) -> tuple[str, list[str]]:

        reasons = []

        # Rule 1:
        # Always respect A7's explicit abstention flag.
        if calibration.get(
            "requires_abstention",
            False,
        ):
            reasons.append(
                "A7 requires abstention."
            )

        calibrated_confidence = cls._score(
            calibration.get(
                "calibrated_confidence",
                0.0,
            )
        )

        evidence_score = cls._score(
            calibration.get(
                "evidence_score",
                0.0,
            )
        )

        factual_consistency = cls._score(
            calibration.get(
                "factual_consistency_score",
                0.0,
            )
        )

        citation_support = cls._score(
            calibration.get(
                "citation_support_score",
                0.0,
            )
        )

        hallucination_safety = cls._score(
            calibration.get(
                "hallucination_safety_score",
                0.0,
            )
        )

        # Rule 2:
        # Very low calibrated confidence.
        if calibrated_confidence < 0.40:
            reasons.append(
                "Calibrated confidence is below "
                "the minimum safety threshold."
            )

        # Rule 3:
        # Insufficient evidence.
        if evidence_score < 0.40:
            reasons.append(
                "Evidence support is insufficient."
            )

        # Rule 4:
        # Poor factual consistency.
        if factual_consistency < 0.50:
            reasons.append(
                "Factual consistency is below "
                "the minimum safety threshold."
            )

        # Rule 5:
        # Poor citation support.
        if citation_support < 0.50:
            reasons.append(
                "Citation support is below "
                "the minimum safety threshold."
            )

        # Rule 6:
        # Poor hallucination safety.
        if hallucination_safety < 0.50:
            reasons.append(
                "Hallucination safety is below "
                "the minimum safety threshold."
            )

        if reasons:
            return "ABSTAIN", reasons

        return "PROCEED", []

    @staticmethod
    def _build_reasoning(
        calibration: Dict[str, Any],
        decision: str,
        reasons: list[str],
    ) -> str:

        confidence = calibration.get(
            "calibrated_confidence",
            0.0,
        )

        band = calibration.get(
            "confidence_band",
            "unknown",
        )

        if decision == "PROCEED":
            return (
                "A7 confidence calibration passed "
                "all A8 safety gates. "
                f"Calibrated confidence="
                f"{float(confidence):.2f}, "
                f"confidence band={band}. "
                "Answer may proceed."
            )

        return (
            "A8 determined that the answer should "
            "not proceed because one or more safety "
            "conditions were not satisfied. "
            f"Calibrated confidence="
            f"{float(confidence):.2f}, "
            f"confidence band={band}. "
            f"Reasons: {' '.join(reasons)}"
        )