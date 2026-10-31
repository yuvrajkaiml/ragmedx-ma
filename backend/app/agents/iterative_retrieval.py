from datetime import datetime, timezone
from typing import Any, Dict, List

from app.llm.parser import LLMParseError, extract_raw_text, parse_json_from_llm
from app.llm.prompt_templates import render_prompt
from .base import BaseAgent
from .state import GapAnalysisOutput


class IterativeRetrievalAgent(BaseAgent):
    """
    Agent 2: Medical Gap Analysis Agent.

    Determines whether retrieved evidence is sufficient
    to answer the user's question safely.

    This agent does NOT perform vector search.
    It generates a focused sub-query when evidence is insufficient.
    """

    def __init__(self, llm_wrapper=None):
        super().__init__(
            agent_name="Medical Gap Analysis Agent",
            agent_type="iterative_retrieval",
            llm_wrapper=llm_wrapper,
        )

    async def run(
        self,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        started_at = datetime.now(timezone.utc)

        try:
            question = state.get("original_question") or state.get("question")

            if not question or not str(question).strip():
                raise ValueError("State must contain a non-empty 'original_question' or 'question'.")

            evidence = state.get("evidence", [])
            evidence_text = self._format_evidence(evidence)

            prompt = render_prompt(
                "iterative_retrieval",
                {"question": str(question).strip(), "evidence_text": evidence_text},
            )

            if self.llm_wrapper is None:
                raise RuntimeError("IterativeRetrievalAgent requires an LLM wrapper.")

            raw_response = self.llm_wrapper.generate(
                prompt,
                temperature=0.0,
            )

            data = self._parse_response(raw_response)

            confidence_val = float(data.get("confidence", 0.0))
            confidence_val = max(0.0, min(1.0, confidence_val))

            result = GapAnalysisOutput(
                sufficient=bool(data.get("sufficient", False)),
                reasoning=str(data.get("reasoning", "")),
                missing_information=data.get("missing_information", []),
                sub_query=data.get("sub_query"),
                confidence=confidence_val,
            )

            # Safety validation:
            if result.sufficient:
                result.sub_query = None
            else:
                if not result.sub_query:
                    raise ValueError("Insufficient evidence requires a retrieval sub-query.")

            completed_at = datetime.now(timezone.utc)

            trace_step = self.create_trace_step(
                status="completed",
                started_at=started_at,
                completed_at=completed_at,
                details={
                    "sufficient": result.sufficient,
                    "missing_information": result.missing_information,
                    "sub_query": result.sub_query,
                    "confidence": result.confidence,
                },
            )

            return {
                "gap_analysis": result.model_dump(),
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

            raise RuntimeError(f"Medical Gap Analysis Agent failed: {exc}") from exc

    @staticmethod
    def _format_evidence(evidence: list) -> str:
        if not evidence:
            return "No evidence was retrieved."

        formatted = []
        for index, item in enumerate(evidence, start=1):
            if isinstance(item, dict):
                citation = item.get("citation_marker", f"[E{index}]")
                text = item.get("cited_text", "")
            else:
                citation = getattr(item, "citation_marker", f"[E{index}]")
                text = getattr(item, "cited_text", "")

            formatted.append(f"{citation}: {text}")

        return "\n\n".join(formatted)

    @staticmethod
    def _parse_response(raw_response: Any) -> Dict[str, Any]:
        try:
            return parse_json_from_llm(raw_response)
        except LLMParseError as exc:
            raise ValueError(f"Medical Gap Analysis Agent returned invalid JSON: {exc}") from exc
