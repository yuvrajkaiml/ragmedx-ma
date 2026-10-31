from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from .base import BaseAgent
from .state import AbstentionDecisionOutput

# Standard Exact Copy Definitions
COPY_INVALID_QUERY = "Please enter a valid medical question."
COPY_RETRIEVAL_FAILURE = "No relevant medical literature found."
COPY_MODEL_FAILURE = "Unable to generate a response. Please try again."
COPY_CONTRADICTORY_EVIDENCE = "Sources contain conflicting information."
COPY_OUT_OF_DOMAIN = "Outside the scope of medical literature."
COPY_TERMINAL_REVIEW_FAILURE = "The answer did not pass the required evidence review."


class AbstentionDecisionAgent(BaseAgent):
    """
    Agent 8: Abstention Decision Agent.

    Consumes the output of Agent 7 (Confidence Calibration) and pipeline safety flags
    to make the definitive safety decision:

        PROCEED
            or
        ABSTAIN

    Enforces standard exact copy for medical safety conditions.
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
            calibration = state.get("confidence_calibration")

            if calibration is None:
                raise ValueError("Abstention decision requires 'confidence_calibration'.")

            decision, reasons = self._make_decision(calibration, state)

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

            completed_at = datetime.now(timezone.utc)

            trace_step = self.create_trace_step(
                status="completed",
                started_at=started_at,
                completed_at=completed_at,
                details={
                    "decision": decision,
                    "should_abstain": decision == "ABSTAIN",
                    "reasons": reasons,
                },
            )

            return {
                "abstention_decision": output.model_dump(),
                "agent_trace": [trace_step],
            }

        except Exception as exc:
            completed_at = datetime.now(timezone.utc)

            trace_step = self.create_trace_step(
                status="failed",
                started_at=started_at,
                completed_at=completed_at,
                details={
                    "error": str(exc),
                },
            )

            raise RuntimeError(f"Abstention Decision Agent failed: {exc}") from exc

    @staticmethod
    def _score(value: Any) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, value))

    @classmethod
    def _make_decision(
        cls,
        calibration: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Tuple[str, List[str]]:
        reasons = []

        # Check explicit pipeline safety flags first
        if state.get("contradictory_evidence", False):
            reasons.append(COPY_CONTRADICTORY_EVIDENCE)

        if state.get("out_of_domain", False):
            reasons.append(COPY_OUT_OF_DOMAIN)

        if state.get("retrieval_failure", False):
            reasons.append(COPY_RETRIEVAL_FAILURE)

        if state.get("model_failure", False):
            reasons.append(COPY_MODEL_FAILURE)

        if state.get("force_abstention", False):
            reasons.append(COPY_TERMINAL_REVIEW_FAILURE)

        # Rule 1: Always respect A7's explicit abstention flag
        if calibration.get("requires_abstention", False):
            reasons.append("A7 requires abstention.")

        calibrated_confidence = cls._score(calibration.get("calibrated_confidence", 0.0))
        evidence_score = cls._score(calibration.get("evidence_score", calibration.get("evidence_quality_score", 0.0)))
        factual_consistency = cls._score(calibration.get("factual_consistency_score", 0.0))
        citation_support = cls._score(calibration.get("citation_support_score", 0.0))
        hallucination_safety = cls._score(calibration.get("hallucination_safety_score", 0.0))

        # Rule 2: Calibrated confidence below minimum safety threshold (0.65)
        if calibrated_confidence < 0.65:
            reasons.append("Calibrated confidence is below the minimum safety threshold (0.65).")

        # Rule 3: Insufficient evidence support
        if evidence_score < 0.40:
            reasons.append("Evidence support is insufficient.")

        # Rule 4: Poor factual consistency
        if factual_consistency < 0.50:
            reasons.append("Factual consistency is below the minimum safety threshold.")

        # Rule 5: Poor citation support
        if citation_support < 0.50:
            reasons.append("Citation support is below the minimum safety threshold.")

        # Rule 6: Poor hallucination safety
        if hallucination_safety < 0.50:
            reasons.append("Hallucination safety is below the minimum safety threshold.")

        # Deduplicate reasons while preserving order
        unique_reasons = list(dict.fromkeys(reasons))

        if unique_reasons:
            return "ABSTAIN", unique_reasons

        return "PROCEED", []

    @staticmethod
    def _build_reasoning(
        calibration: Dict[str, Any],
        decision: str,
        reasons: List[str],
    ) -> str:
        confidence = calibration.get("calibrated_confidence", 0.0)
        band = calibration.get("confidence_band", "unknown")

        if decision == "PROCEED":
            return (
                f"Confidence calibration passed all safety gates. "
                f"Calibrated confidence={float(confidence):.2f}, confidence band={band}. "
                "Answer may proceed."
            )

        return (
            f"Abstention determined because one or more safety conditions were not satisfied. "
            f"Calibrated confidence={float(confidence):.2f}, confidence band={band}. "
            f"Reasons: {' '.join(reasons)}"
        )
