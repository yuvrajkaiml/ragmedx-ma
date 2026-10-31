"""Agent 3: evidence re-ranking and safety screening."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional, Set

from app.llm.parser import LLMParseError, parse_json_from_llm
from app.llm.prompt_templates import render_prompt

from .base import BaseAgent

logger = logging.getLogger(__name__)

# Maximum number of LLM-parse retries before falling back to original order
_MAX_PARSE_RETRIES = 2


class EvidenceRerankingAgent(BaseAgent):
    """Screen and rank native retrieval evidence before answer generation.

    Sana's retrieval pipeline remains the owner of dense/BM25/RRF retrieval and
    cross-encoder scoring.  This agent is the workflow-stage owner: it consumes
    those scores, applies a transparent evidence-hierarchy adjustment, and asks
    for a bounded contradiction assessment over the retrieved evidence.

    Phase 3 fixes:
    - Unknown IDs from the LLM are rejected; only IDs in the input set are accepted
    - Generated explanation/reasoning text is tagged as non_source_text (never a passage)
    - Malformed LLM output triggers bounded retry (max 2), then safe failure (original order)
    - Unvalidated evidence is never treated as verified
    """

    _GRADE_TERMS = (
        ("systematic review", 1.00, "systematic_review"),
        ("meta-analysis", 1.00, "systematic_review"),
        ("randomized", 0.90, "randomized_trial"),
        ("randomised", 0.90, "randomized_trial"),
        ("cohort", 0.75, "cohort"),
        ("case report", 0.45, "case_report"),
    )

    def __init__(self, llm_wrapper=None):
        super().__init__(
            agent_name="Evidence Re-Ranking Agent",
            agent_type="evidence_reranking",
            llm_wrapper=llm_wrapper,
        )

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        started_at = datetime.now(timezone.utc)
        try:
            evidence = [dict(item) for item in state.get("evidence", []) if isinstance(item, dict)]
            if not evidence:
                raise ValueError("Evidence Re-Ranking Agent requires evidence.")
            if self.llm_wrapper is None:
                raise RuntimeError("Evidence Re-Ranking Agent requires an LLM wrapper.")

            ranked = self._rank_evidence(evidence)
            # Record the set of valid canonical markers BEFORE calling the LLM
            valid_markers: Set[str] = {item["citation_marker"] for item in ranked}

            evidence_text = self._format_evidence(ranked)
            prompt = render_prompt(
                "evidence_reranking",
                {
                    "question": state.get("original_question", ""),
                    "evidence_text": evidence_text,
                },
            )

            # --- Bounded retry on LLM parse failure ---
            assessment: Optional[Dict[str, Any]] = None
            parse_error_logged = False
            for attempt in range(1 + _MAX_PARSE_RETRIES):
                try:
                    raw = self.llm_wrapper.generate(prompt, temperature=0.0)
                    assessment = self._parse_assessment(raw)
                    break
                except (ValueError, LLMParseError) as exc:
                    if attempt < _MAX_PARSE_RETRIES:
                        logger.warning(
                            "EvidenceRerankingAgent: LLM parse attempt %d failed (%s); retrying.",
                            attempt + 1,
                            exc,
                        )
                    else:
                        logger.warning(
                            "EvidenceRerankingAgent: all %d parse attempts failed; "
                            "falling back to original retrieval order (unvalidated).",
                            _MAX_PARSE_RETRIES + 1,
                        )
                        parse_error_logged = True

            if assessment is None:
                # Safe failure: return original ranked order, mark as unvalidated
                completed_at = datetime.now(timezone.utc)
                trace = self.create_trace_step(
                    status="completed",
                    started_at=started_at,
                    completed_at=completed_at,
                    details={
                        "input_evidence_count": len(evidence),
                        "selected_evidence_count": len(ranked),
                        "contradictory_evidence": False,
                        "assessment": "parse_failure_fallback_to_original_order",
                        "validated": False,
                    },
                )
                return {
                    "evidence": ranked,
                    "contradictory_evidence": False,
                    "evidence_reranking": {
                        "selected_markers": [item["citation_marker"] for item in ranked],
                        "contradictory": False,
                        "reasoning": "parse_failure_fallback",
                        # non_source_text: the reasoning here is a system note, not a passage
                        "non_source_text": True,
                        "validated": False,
                    },
                    "agent_trace": [trace],
                }

            # --- Reject unknown IDs: LLM may only reference items in the input set ---
            selected_markers = self._resolve_selected_markers(
                assessment.get("selected_markers", []), ranked
            )
            # Filter to items whose marker was in the input and the LLM selected
            if selected_markers:
                ranked = [item for item in ranked if item["citation_marker"] in selected_markers]
            if not ranked:
                raise ValueError("Evidence Re-Ranking Agent selected no usable evidence.")

            # Tag the LLM's reasoning as non-source text so it's never treated as
            # a retrieved passage or cited evidence
            reasoning_text = assessment["reasoning"]
            assessment["_reasoning_is_non_source_text"] = True

            completed_at = datetime.now(timezone.utc)
            trace = self.create_trace_step(
                status="completed",
                started_at=started_at,
                completed_at=completed_at,
                details={
                    "input_evidence_count": len(evidence),
                    "selected_evidence_count": len(ranked),
                    "contradictory_evidence": bool(assessment["contradictory"]),
                    "assessment": reasoning_text,
                    "validated": True,
                    "parse_attempts": 1 if not parse_error_logged else _MAX_PARSE_RETRIES + 1,
                },
            )
            return {
                "evidence": ranked,
                "contradictory_evidence": bool(assessment["contradictory"]),
                "evidence_reranking": {
                    "selected_markers": [item["citation_marker"] for item in ranked],
                    "contradictory": bool(assessment["contradictory"]),
                    "reasoning": reasoning_text,
                    # Explicitly tag that reasoning is NOT a source passage
                    "non_source_text": True,
                    "validated": True,
                },
                "agent_trace": [trace],
            }
        except Exception as exc:
            raise RuntimeError(f"Evidence Re-Ranking Agent failed: {exc}") from exc

    @classmethod
    def _rank_evidence(cls, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ranked = []
        for index, item in enumerate(evidence, start=1):
            marker = str(item.get("citation_marker") or f"[E{index}]")
            if not marker.startswith("["):
                marker = f"[{marker}]"
            item["citation_marker"] = marker
            text = f"{item.get('title', '')} {item.get('cited_text', '')}".lower()
            grade_weight, grade = 0.60, "unclassified"
            for term, weight, label in cls._GRADE_TERMS:
                if term in text:
                    grade_weight, grade = weight, label
                    break
            try:
                relevance = float(item.get("cross_encoder_score", item.get("relevance_score", 0.0)))
            except (TypeError, ValueError):
                relevance = 0.0
            relevance = max(0.0, min(1.0, relevance))
            item["evidence_grade"] = grade
            item["grade_weight"] = grade_weight
            item["reranking_score"] = round(0.8 * relevance + 0.2 * grade_weight, 6)
            ranked.append(item)
        return sorted(ranked, key=lambda item: item["reranking_score"], reverse=True)

    @staticmethod
    def _format_evidence(evidence: List[Dict[str, Any]]) -> str:
        return "\n\n".join(
            f"{item['citation_marker']} ({item['evidence_grade']}): {item.get('cited_text', '')}"
            for item in evidence
        )

    @staticmethod
    def _parse_assessment(raw: Any) -> Dict[str, Any]:
        try:
            data = parse_json_from_llm(raw)
        except LLMParseError as exc:
            raise ValueError(f"Agent 3 returned invalid JSON: {exc}") from exc
        selected = data.get("selected_markers", [])
        if not isinstance(selected, list) or not all(isinstance(marker, str) for marker in selected):
            raise ValueError("Agent 3 selected_markers must be a list of citation markers.")
        return {
            "contradictory": bool(data.get("contradictory", False)),
            "selected_markers": selected,
            "reasoning": str(data.get("reasoning", "")),
        }

    @staticmethod
    def _resolve_selected_markers(
        selected_markers: List[str],
        ranked: List[Dict[str, Any]],
    ) -> set[str]:
        """Map an unambiguous legacy ordinal to the current canonical marker.

        Current retrieval emits `[Doc X]`, but pre-existing prompt/model behavior
        can return `[EX]`.  Only translate a marker that identifies one ranked
        item by ordinal; unknown markers remain unknown and cannot select evidence.

        Phase 3 fix: markers not in the input set are silently dropped.
        Unknown IDs injected by the LLM are rejected here — they cannot
        select evidence items that were not part of the retrieved set.
        """
        canonical = {item["citation_marker"] for item in ranked}
        resolved = set()
        for marker in selected_markers:
            if marker in canonical:
                resolved.add(marker)
                continue
            match = re.fullmatch(r"\[E(\d+)\]", marker.strip(), flags=re.IGNORECASE)
            if match:
                ordinal = int(match.group(1))
                if 1 <= ordinal <= len(ranked):
                    resolved.add(ranked[ordinal - 1]["citation_marker"])
            # Unknown markers (not in canonical, not a valid ordinal) are rejected — no action
        return resolved
