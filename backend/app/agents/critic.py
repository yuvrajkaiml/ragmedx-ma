import re
from datetime import datetime, timezone
from typing import Any, Dict, List

from app.llm.parser import LLMParseError, extract_raw_text, parse_json_from_llm
from app.llm.prompt_templates import render_prompt
from app.rag.chunking.models import MedicalChunk
from app.rag.verification.evidence_verifier import EvidenceVerifier
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

            verification = self._verify_claim_citations(answer=str(answer), evidence=evidence)
            verdict.evidence_verification = verification
            if verification["blocking_issues"]:
                verdict.passed = False
                verdict.citation_support = 0.0
                verdict.hallucination_risk = max(verdict.hallucination_risk, 0.5)
                verdict.issues = list(dict.fromkeys([*verdict.issues, *verification["blocking_issues"]]))
                verdict.revision_instructions = (
                    "Revise the answer so every cited claim uses only a valid [Doc X] "
                    "marker mapped to the supplied evidence span."
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
                    "evidence_verification": verification,
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
        return render_prompt(
            "critic",
            {"question": question, "answer": answer, "evidence_text": evidence_text},
        )

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
    def _verify_claim_citations(answer: str, evidence: List[Any]) -> Dict[str, Any]:
        """Verify that answer citations resolve to usable retrieved evidence spans.

        ``EvidenceVerifier`` owns structural provenance checks on native chunks.
        This method adds the agent-boundary check: every answer marker must map to
        one supplied chunk span. It deliberately does not claim semantic entailment;
        the LLM critic remains responsible for that judgement.
        """
        marker_to_item: Dict[str, Dict[str, Any]] = {}
        chunks: List[MedicalChunk] = []
        strict_provenance = True
        for index, raw_item in enumerate(evidence, start=1):
            item = dict(raw_item) if isinstance(raw_item, dict) else raw_item.__dict__
            marker = str(item.get("citation_marker") or f"[Doc {index}]")
            if not marker.startswith("["):
                marker = f"[{marker}]"
            marker_to_item[marker] = item
            required = ("chunk_id", "document_id", "title", "journal", "publication_year")
            strict_provenance = strict_provenance and all(item.get(field) not in (None, "") for field in required)
            chunks.append(MedicalChunk(
                id=str(item.get("chunk_id") or f"legacy-{index}"),
                document_id=str(item.get("document_id") or f"legacy-document-{index}"),
                content=str(item.get("cited_text") or ""),
                title=item.get("title"),
                journal=item.get("journal"),
                publication_year=item.get("publication_year"),
                source=str(item.get("source_type") or "unknown"),
            ))

        referenced = set(re.findall(r"\[(?:Doc\s+\d+|[A-Za-z0-9_-]+)\]", answer))
        unresolved = sorted(referenced - set(marker_to_item))
        report = EvidenceVerifier(min_content_length=1).verify(
            [(chunk, 1.0) for chunk in chunks]
        )
        invalid_markers = [
            marker for marker, result in zip(marker_to_item, report)
            if not result.is_valid
        ]
        blocking_issues = []
        if unresolved:
            blocking_issues.append(f"Answer cites markers without evidence spans: {unresolved}.")
        if strict_provenance and invalid_markers:
            blocking_issues.append(
                f"Cited evidence failed deterministic provenance validation: {invalid_markers}."
            )
        return {
            "referenced_markers": sorted(referenced),
            "mapped_markers": sorted(set(referenced) & set(marker_to_item)),
            "unresolved_markers": unresolved,
            "invalid_evidence_markers": invalid_markers,
            "blocking_issues": blocking_issues,
        }

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
