from datetime import datetime, timezone
from typing import Any, Dict

from app.llm.parser import LLMParseError, extract_raw_text, parse_json_from_llm
from app.llm.prompt_templates import render_prompt
from .base import BaseAgent
from .state import QueryUnderstandingOutput


class QueryUnderstandingAgent(BaseAgent):
    """
    Agent 1: Query Understanding Agent.

    Responsibilities:
    - Identify medical specialty.
    - Classify user intent.
    - Resolve important abbreviations.
    - Expand medical terminology.
    - Produce an optimized retrieval query.
    """

    def __init__(self, llm_wrapper=None):
        super().__init__(
            agent_name="Query Understanding Agent",
            agent_type="query_understanding",
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

            prompt = render_prompt("query_understanding", {"question": str(question).strip()})

            if self.llm_wrapper is None:
                raise RuntimeError("QueryUnderstandingAgent requires an LLM wrapper.")

            raw_response = self.llm_wrapper.generate(
                prompt,
                temperature=0.0,
            )

            data = self._parse_response(raw_response)

            confidence_val = float(data.get("confidence", 0.0))
            confidence_val = max(0.0, min(1.0, confidence_val))

            result = QueryUnderstandingOutput(
                original_question=str(question).strip(),
                medical_specialty=data.get("medical_specialty"),
                intent=data.get("intent"),
                expanded_query=data.get("expanded_query"),
                abbreviations=data.get("abbreviations", {}),
                confidence=confidence_val,
            )

            completed_at = datetime.now(timezone.utc)

            trace_step = self.create_trace_step(
                status="completed",
                started_at=started_at,
                completed_at=completed_at,
                details={
                    "input_question": question,
                    "specialty": result.medical_specialty,
                    "intent": result.intent,
                    "confidence": result.confidence,
                },
            )

            return {
                "processed_query": result.model_dump(),
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

            raise RuntimeError(f"Query Understanding Agent failed: {exc}") from exc

    @staticmethod
    def _parse_response(raw_response: Any) -> Dict[str, Any]:
        """Parse the JSON returned by LLM using robust parser."""
        try:
            return parse_json_from_llm(raw_response)
        except LLMParseError as exc:
            raise ValueError(f"Query Understanding Agent returned invalid JSON: {exc}") from exc
