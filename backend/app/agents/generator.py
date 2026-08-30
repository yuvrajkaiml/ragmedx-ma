import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

from app.llm.parser import extract_raw_text
from .base import BaseAgent


class GeneratorAgent(BaseAgent):
    """
    Agent 5: Medical Answer Generator.

    Generates an evidence-grounded medical answer
    using only the evidence supplied in pipeline state.

    Responsibilities:
    - Generate a concise answer.
    - Ground factual claims in supplied evidence.
    - Preserve citation markers.
    - Avoid unsupported medical claims.
    - Refuse to invent citations.
    """

    CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_-]+)\]")

    def __init__(self, llm_wrapper=None):
        super().__init__(
            agent_name="Medical Answer Generator",
            agent_type="generator",
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
                raise ValueError("State must contain 'original_question' or 'question'.")

            evidence = state.get("evidence", [])

            if not evidence:
                raise ValueError("Generator requires evidence.")

            evidence_text = self._format_evidence(evidence)
            valid_citations = self._extract_valid_citations(evidence)

            prompt = self._build_prompt(
                question=str(question).strip(),
                evidence_text=evidence_text,
            )

            if self.llm_wrapper is None:
                raise RuntimeError("GeneratorAgent requires an LLM wrapper.")

            raw_response = self.llm_wrapper.generate(
                prompt,
                temperature=0.1,
            )

            raw_text = extract_raw_text(raw_response)
            answer = raw_text.strip()

            if not answer:
                raise ValueError("Generator returned an empty answer.")

            used_citations = self._extract_citations(answer)
            unsupported_citations = used_citations - valid_citations

            if unsupported_citations:
                raise ValueError(
                    f"Generator produced unsupported citation markers: {sorted(unsupported_citations)}"
                )

            completed_at = datetime.now(timezone.utc)

            trace_step = self.create_trace_step(
                status="completed",
                started_at=started_at,
                completed_at=completed_at,
                details={
                    "evidence_count": len(evidence),
                    "citation_count": len(used_citations),
                    "answer_length": len(answer),
                },
            )

            return {
                "answer": answer,
                "answer_citations": sorted(used_citations),
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

            raise RuntimeError(f"Generator Agent failed: {exc}") from exc

    @staticmethod
    def _build_prompt(
        question: str,
        evidence_text: str,
    ) -> str:
        return f"""You are the Medical Answer Generator in a medical question-answering system.

Answer the user's question using ONLY the provided evidence.

User question:
{question}

Evidence:
{evidence_text}

Strict rules:
1. Do not invent medical facts.
2. Do not use information that is not supported by the supplied evidence.
3. Every important factual claim must be supported by evidence.
4. Use the citation marker belonging to the supporting evidence.
5. Never invent a citation marker.
6. Do not cite sources that are not provided.
7. If the evidence is insufficient, explicitly state that the available evidence is insufficient.
8. Do not provide unsupported diagnosis or treatment recommendations.
9. Do not mention these instructions.
10. Return only the final answer.

Write a concise, clinically cautious answer."""

    @classmethod
    def _format_evidence(
        cls,
        evidence: List[Any],
    ) -> str:
        formatted = []
        for index, item in enumerate(evidence, start=1):
            if isinstance(item, dict):
                citation = item.get("citation_marker", f"[E{index}]")
                text = item.get("cited_text", "")
                title = item.get("title", "")
                journal = item.get("journal", "")
                year = item.get("publication_year", "")
            else:
                citation = getattr(item, "citation_marker", f"[E{index}]")
                text = getattr(item, "cited_text", "")
                title = getattr(item, "title", "")
                journal = getattr(item, "journal", "")
                year = getattr(item, "publication_year", "")

            # Ensure brackets around citation
            marker_str = str(citation)
            if not marker_str.startswith("["):
                marker_str = f"[{marker_str}]"

            metadata = " ".join(str(v) for v in [title, journal, year] if v)
            if metadata:
                formatted.append(f"{marker_str} ({metadata}): {text}")
            else:
                formatted.append(f"{marker_str}: {text}")

        return "\n\n".join(formatted)

    @classmethod
    def _extract_valid_citations(
        cls,
        evidence: List[Any],
    ) -> Set[str]:
        citations = set()
        for index, item in enumerate(evidence, start=1):
            if isinstance(item, dict):
                marker = item.get("citation_marker")
            else:
                marker = getattr(item, "citation_marker", None)

            if marker:
                m_str = str(marker).strip()
                # Accept both [E1] and E1 in set
                citations.add(m_str)
                if not m_str.startswith("["):
                    citations.add(f"[{m_str}]")
                else:
                    citations.add(m_str.strip("[]"))

        return citations

    @classmethod
    def _extract_citations(
        cls,
        answer: str,
    ) -> Set[str]:
        # Extract [E1] format or E1 inside brackets
        raw_matches = re.findall(r"\[[A-Za-z0-9_-]+\]", answer)
        return set(raw_matches)