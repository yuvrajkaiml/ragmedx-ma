import json
from datetime import datetime, timezone
from typing import Any, Dict

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
            question = state.get("original_question")

            if not question:
                question = state.get("question")

            if not question:
                raise ValueError(
                    "State must contain 'original_question' "
                    "or 'question'."
                )

            prompt = f"""
You are the Query Understanding Agent
in a medical question answering system.

Analyze the user's medical question.

Identify:

1. Medical specialty
2. User intent
3. Important abbreviations
4. Expanded medical terminology
5. An optimized retrieval query

Do NOT answer the medical question.

Question:
{question}

Return ONLY valid JSON.

Required schema:

{{
    "medical_specialty": "",
    "intent": "",
    "expanded_query": "",
    "abbreviations": {{}},
    "confidence": 0.0
}}

The confidence must be a number between 0.0 and 1.0.
"""

            if self.llm_wrapper is None:
                raise RuntimeError(
                    "QueryUnderstandingAgent requires "
                    "an LLM wrapper."
                )

            raw_response = self.llm_wrapper.generate(
                prompt,
                temperature=0.0,
            )

            data = self._parse_response(raw_response)

            result = QueryUnderstandingOutput(
                original_question=question,
                medical_specialty=data.get(
                    "medical_specialty"
                ),
                intent=data.get("intent"),
                expanded_query=data.get(
                    "expanded_query"
                ),
                abbreviations=data.get(
                    "abbreviations",
                    {},
                ),
                confidence=float(
                    data.get("confidence", 0.0)
                ),
            )

            completed_at = datetime.now(timezone.utc)

            trace_step = self.create_trace_step(
                status="completed",
                started_at=started_at,
                completed_at=completed_at,
                details={
                    "input_question": question,
                    "specialty":
                        result.medical_specialty,
                    "intent": result.intent,
                    "confidence":
                        result.confidence,
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

            raise RuntimeError(
                "Query Understanding Agent failed."
            ) from exc

    @staticmethod
    def _parse_response(raw_response: str) -> Dict[str, Any]:
        """
        Parse the JSON returned by the LLM.

        Handles accidental markdown code fences.
        """

        if not isinstance(raw_response, str):
            raise ValueError(
                "LLM response must be a string."
            )

        text = raw_response.strip()

        if text.startswith("```"):
            lines = text.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Query Understanding Agent returned "
                "invalid JSON."
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "Query Understanding response must "
                "be a JSON object."
            )

        return data