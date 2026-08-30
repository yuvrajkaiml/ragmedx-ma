from datetime import datetime, timezone
from typing import Any, Dict, List

from app.llm.parser import LLMParseError, extract_raw_text, parse_json_from_llm
from .base import BaseAgent
from .state import CriticVerdict


class CriticAgent(BaseAgent):
    """
    Agent 6: Medical Answer Critic.

    Evaluates a generated answer against the supplied
    question and retrieved evidence.

    The Critic does NOT generate a replacement answer.
    It evaluates factual consistency, citation support, hallucination risk,
    and determines whether revision is required.
    """

    def __init__(self, llm_wrapper=None):
        super().__init__(
            agent_name="Medical Answer Critic",
            agent_type="critic",
            llm_wrapper=llm_wrapper,
        )

    async def run(
        self,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        started_at = datetime.now(timezone.utc)

        try:
            question = state.get("original_question") or state.get("question")
            answer = state.get("answer")
            evidence = state.get("evidence", [])

            if not question or not str(question).strip():
                raise ValueError("State must contain 'original_question' or 'question'.")

            if not answer or not str(answer).strip():
                raise ValueError("State must contain 'answer'.")

            if not evidence:
                raise ValueError("Critic requires evidence.")

            evidence_text = self._format_evidence(evidence)

            prompt = self._build_prompt(
                question=str(question).strip(),
                answer=str(answer).strip(),
                evidence_text=evidence_text,
            )

            if self.llm_wrapper is None:
                raise RuntimeError("CriticAgent requires an LLM wrapper.")

            raw_response = self.llm_wrapper.generate(
                prompt,
                temperature=0.0,
            )

            data = self._parse_response(raw_response)

            verdict = CriticVerdict(
                passed=bool(data.get("passed", False)),
                factual_consistency=float(data.get("factual_consistency", 0.0)),
                citation_support=float(data.get("citation_support", 0.0)),
                hallucination_risk=float(data.get("hallucination_risk", 0.0)),
                issues=data.get("issues", []),
                revision_instructions=data.get("revision_instructions"),
                confidence=float(data.get("confidence", 0.0)),
            )

            self._validate_scores(verdict)

            # A failed verdict must contain actionable revision information
            if not verdict.passed:
                if not verdict.revision_instructions:
                    raise ValueError("Failed critic verdict requires revision instructions.")

            completed_at = datetime.now(timezone.utc)

            trace_step = self.create_trace_step(
                status="completed",
                started_at=started_at,
                completed_at=completed_at,
                details={
                    "passed": verdict.passed,
                    "factual_consistency": verdict.factual_consistency,
                    "citation_support": verdict.citation_support,
                    "hallucination_risk": verdict.hallucination_risk,
                    "issue_count": len(verdict.issues),
                },
            )

            return {
                "critic_verdict": verdict.model_dump(),
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

            raise RuntimeError(f"Critic Agent failed: {exc}") from exc

    @staticmethod
    def _build_prompt(
        question: str,
        answer: str,
        evidence_text: str,
    ) -> str:
        return f"""You are the Medical Answer Critic in a medical question-answering system.

Evaluate the generated answer against ONLY the supplied evidence.

Do NOT rewrite the answer.

Question:
{question}

Generated answer:
{answer}

Evidence:
{evidence_text}

Evaluate:
1. Factual consistency
   Are the answer's factual claims supported by the evidence?
2. Citation support
   Do citations actually support the claims they are attached to?
3. Hallucination risk
   Does the answer contain claims that are unsupported by the evidence?

Return ONLY valid JSON:
{{
    "passed": true,
    "factual_consistency": 0.0,
    "citation_support": 0.0,
    "hallucination_risk": 0.0,
    "issues": [],
    "revision_instructions": null,
    "confidence": 0.0
}}

Rules:
- Scores must be between 0.0 and 1.0.
- passed=true only when the answer is adequately supported by the evidence.
- If passed=false, provide actionable revision instructions.
- Do not invent evidence.
- Do not introduce new medical facts."""

    @staticmethod
    def _format_evidence(evidence: List[Any]) -> str:
        formatted = []
        for index, item in enumerate(evidence, start=1):
            if isinstance(item, dict):
                citation = item.get("citation_marker", f"[E{index}]")
                text = item.get("cited_text", "")
            else:
                citation = getattr(item, "citation_marker", f"[E{index}]")
                text = getattr(item, "cited_text", "")

            marker_str = str(citation)
            if not marker_str.startswith("["):
                marker_str = f"[{marker_str}]"

            formatted.append(f"{marker_str}: {text}")

        return "\n\n".join(formatted)

    @staticmethod
    def _parse_response(raw_response: Any) -> Dict[str, Any]:
        try:
            return parse_json_from_llm(raw_response)
        except LLMParseError as exc:
            raise ValueError(f"Critic returned invalid JSON: {exc}") from exc

    @staticmethod
    def _validate_scores(verdict: CriticVerdict) -> None:
        scores = {
            "factual_consistency": verdict.factual_consistency,
            "citation_support": verdict.citation_support,
            "hallucination_risk": verdict.hallucination_risk,
            "confidence": verdict.confidence,
        }

        for name, score in scores.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0.")