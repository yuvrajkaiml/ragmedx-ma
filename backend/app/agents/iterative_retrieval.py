import json
from datetime import datetime, timezone
from typing import Any, Dict

from .base import BaseAgent
from .state import GapAnalysisOutput


class IterativeRetrievalAgent(BaseAgent):
    """
    Agent 2: Medical Gap Analysis Agent.

    Determines whether retrieved evidence is sufficient
    to answer the user's question safely.

    This agent does NOT perform vector search.

    It may generate a focused sub-query when evidence
    is insufficient.
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
            question = (
                state.get("original_question")
                or state.get("question")
            )

            if not question:
                raise ValueError(
                    "State must contain "
                    "'original_question' or 'question'."
                )

            evidence = state.get(
                "evidence",
                [],
            )

            evidence_text = (
                self._format_evidence(evidence)
            )

            prompt = f"""
You are the Medical Gap Analysis Agent
in a medical question answering system.

Your task is to determine whether the supplied
medical evidence is sufficient to safely answer
the user's question.

Do NOT answer the question.

Analyze:

1. Whether the evidence directly addresses
   the question.
2. What important information is missing.
3. Whether there are contradictions.
4. Whether another retrieval iteration is needed.
5. If another iteration is needed, create ONE
   focused medical retrieval sub-query.

Question:
{question}

Retrieved evidence:
{evidence_text}

Return ONLY valid JSON:

{{
    "sufficient": true,
    "reasoning": "",
    "missing_information": [],
    "sub_query": null,
    "confidence": 0.0
}}

Rules:

- "sufficient" must be true or false.
- "missing_information" must be a JSON array.
- "sub_query" must be null when evidence
  is sufficient.
- If evidence is insufficient, sub_query must
  be specific and directly address the missing
  information.
- confidence must be between 0.0 and 1.0.
"""

            if self.llm_wrapper is None:
                raise RuntimeError(
                    "IterativeRetrievalAgent requires "
                    "an LLM wrapper."
                )

            raw_response = self.llm_wrapper.generate(
                prompt,
                temperature=0.0,
            )

            data = self._parse_response(
                raw_response
            )

            result = GapAnalysisOutput(
                sufficient=bool(
                    data.get("sufficient", False)
                ),
                reasoning=str(
                    data.get("reasoning", "")
                ),
                missing_information=data.get(
                    "missing_information",
                    [],
                ),
                sub_query=data.get(
                    "sub_query"
                ),
                confidence=float(
                    data.get("confidence", 0.0)
                ),
            )

            # Safety validation:
            if result.sufficient:
                result.sub_query = None

            if not result.sufficient:
                if not result.sub_query:
                    raise ValueError(
                        "Insufficient evidence requires "
                        "a retrieval sub-query."
                    )

            completed_at = datetime.now(
                timezone.utc
            )

            trace_step = self.create_trace_step(
                status="completed",
                started_at=started_at,
                completed_at=completed_at,
                details={
                    "sufficient":
                        result.sufficient,
                    "missing_information":
                        result.missing_information,
                    "sub_query":
                        result.sub_query,
                    "confidence":
                        result.confidence,
                },
            )

            return {
                "gap_analysis":
                    result.model_dump(),
                "agent_trace": [trace_step],
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
                    "error": str(exc),
                },
            )

            raise RuntimeError(
                "Medical Gap Analysis Agent failed."
            ) from exc

    @staticmethod
    def _format_evidence(
        evidence: list,
    ) -> str:

        if not evidence:
            return "No evidence was retrieved."

        formatted = []

        for index, item in enumerate(
            evidence,
            start=1,
        ):

            if isinstance(item, dict):

                citation = item.get(
                    "citation_marker",
                    f"[Evidence {index}]",
                )

                text = item.get(
                    "cited_text",
                    "",
                )

            else:

                citation = getattr(
                    item,
                    "citation_marker",
                    f"[Evidence {index}]",
                )

                text = getattr(
                    item,
                    "cited_text",
                    "",
                )

            formatted.append(
                f"{citation}: {text}"
            )

        return "\n\n".join(formatted)

    @staticmethod
    def _parse_response(
        raw_response: str,
    ) -> Dict[str, Any]:

        if not isinstance(
            raw_response,
            str,
        ):
            raise ValueError(
                "LLM response must be a string."
            )

        text = raw_response.strip()

        # Handle accidental Markdown fences.
        if text.startswith("```"):

            lines = text.splitlines()

            if lines and lines[0].startswith(
                "```"
            ):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)

        except json.JSONDecodeError as exc:

            raise ValueError(
                "Medical Gap Analysis Agent "
                "returned invalid JSON."
            ) from exc

        if not isinstance(data, dict):

            raise ValueError(
                "Gap analysis response must "
                "be a JSON object."
            )

        return data