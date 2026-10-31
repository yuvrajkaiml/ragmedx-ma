"""
Evidence sufficiency assessment for RAGMedX-MA.

Provides SufficiencyAssessor: heuristic assessment of whether retrieved
evidence is sufficient to answer a medical query.

IMPORTANT DISCLAIMER: This is a heuristic assessment. It has not been
clinically validated.

Evidence tier: SOFTWARE TEST (heuristic, not clinically validated)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


def _jaccard(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class SufficiencyAssessor:
    """
    Heuristic evidence sufficiency assessor.

    This is a heuristic assessment. It has not been clinically validated.

    assess() returns a structured dict with all required fields:
      - coverage, relevance_score, source_quality, claim_support_rate,
        contradictions_present, missing_info, sufficient, confidence, reasoning.

    NEVER returns bare sufficient=True without all other fields populated.
    Does NOT claim heuristics are clinically validated.
    """

    # Minimum thresholds for sufficiency decision
    MIN_COVERAGE = 0.3
    MIN_RELEVANCE = 0.3
    MIN_CLAIM_SUPPORT_RATE = 0.5
    MIN_CONFIDENCE = 0.4

    def assess(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        verified_claims: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Assess whether retrieved evidence is sufficient to answer the query.

        This is a heuristic assessment. It has not been clinically validated.

        Args:
            query: The original medical query string.
            retrieved_docs: List of retrieved evidence dicts, each with at minimum:
                - 'cited_text' or 'text': the passage text
                - (optional) 'relevance_score': float
                - (optional) 'source_type': str
            verified_claims: List of verify_claim() output dicts, each with:
                - 'label': one of 'supported'|'partial'|'unsupported'|'contradicted'|'unverifiable'

        Returns:
            {
                'coverage': float,           # fraction of query aspects addressed (0-1)
                'relevance_score': float,    # mean relevance of retrieved docs (0-1)
                'source_quality': str,       # 'high'|'medium'|'low'|'mixed'
                'claim_support_rate': float, # fraction of claims supported
                'contradictions_present': bool,
                'missing_info': list[str],   # specific missing information items
                'sufficient': bool,
                'confidence': float,
                'reasoning': str,            # required, non-empty
            }
        """
        # --- Coverage: fraction of query keywords found in any doc text ---
        coverage = self._compute_coverage(query, retrieved_docs)

        # --- Relevance: mean relevance score across docs ---
        relevance_score = self._compute_mean_relevance(retrieved_docs)

        # --- Source quality ---
        source_quality = self._assess_source_quality(retrieved_docs)

        # --- Claim support rate ---
        claim_support_rate = self._compute_claim_support_rate(verified_claims)

        # --- Contradictions present ---
        contradictions_present = any(
            vc.get("label") == "contradicted" for vc in verified_claims
        )

        # --- Missing info ---
        missing_info = self._identify_missing_info(query, retrieved_docs, verified_claims)

        # --- Sufficiency decision ---
        sufficient = (
            coverage >= self.MIN_COVERAGE
            and relevance_score >= self.MIN_RELEVANCE
            and claim_support_rate >= self.MIN_CLAIM_SUPPORT_RATE
            and not contradictions_present
            and len(retrieved_docs) > 0
        )

        # --- Confidence ---
        if not retrieved_docs:
            confidence = 0.0
        else:
            components = [
                min(coverage / max(self.MIN_COVERAGE, 1e-9), 1.0) * 0.3,
                min(relevance_score / max(self.MIN_RELEVANCE, 1e-9), 1.0) * 0.3,
                claim_support_rate * 0.3,
                (0.0 if contradictions_present else 0.1),
            ]
            confidence = round(min(sum(components), 1.0), 4)

        # --- Reasoning (required, non-empty) ---
        reasoning = self._build_reasoning(
            coverage=coverage,
            relevance_score=relevance_score,
            source_quality=source_quality,
            claim_support_rate=claim_support_rate,
            contradictions_present=contradictions_present,
            missing_info=missing_info,
            sufficient=sufficient,
            confidence=confidence,
        )

        return {
            "coverage": round(coverage, 4),
            "relevance_score": round(relevance_score, 4),
            "source_quality": source_quality,
            "claim_support_rate": round(claim_support_rate, 4),
            "contradictions_present": contradictions_present,
            "missing_info": missing_info,
            "sufficient": sufficient,
            "confidence": confidence,
            "reasoning": reasoning,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_doc_text(doc: Dict[str, Any]) -> str:
        return doc.get("cited_text") or doc.get("text") or ""

    def _compute_coverage(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> float:
        """Fraction of query keywords (>3 chars) found in at least one doc."""
        if not retrieved_docs:
            return 0.0
        keywords = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", query)}
        if not keywords:
            return 0.5  # no keywords to check — assume partial coverage
        all_text = " ".join(self._get_doc_text(d) for d in retrieved_docs).lower()
        found = sum(1 for kw in keywords if kw in all_text)
        return found / len(keywords)

    @staticmethod
    def _compute_mean_relevance(retrieved_docs: List[Dict[str, Any]]) -> float:
        """Mean relevance_score across docs."""
        if not retrieved_docs:
            return 0.0
        scores = []
        for d in retrieved_docs:
            score = d.get("relevance_score", d.get("cross_encoder_score", 0.0))
            try:
                scores.append(max(0.0, min(1.0, float(score))))
            except (TypeError, ValueError):
                scores.append(0.0)
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    @staticmethod
    def _assess_source_quality(retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Classify overall source quality based on evidence grades and source types.
        Returns 'high', 'medium', 'low', or 'mixed'.
        """
        if not retrieved_docs:
            return "low"
        grades = []
        for d in retrieved_docs:
            grade = d.get("evidence_grade", "")
            if grade in ("systematic_review", "randomized_trial"):
                grades.append(2)
            elif grade in ("cohort",):
                grades.append(1)
            elif grade in ("case_report",):
                grades.append(0)
            else:
                grades.append(1)  # default: medium
        if not grades:
            return "low"
        mean = sum(grades) / len(grades)
        if len(set(grades)) > 1:
            return "mixed"
        if mean >= 1.5:
            return "high"
        if mean >= 0.8:
            return "medium"
        return "low"

    @staticmethod
    def _compute_claim_support_rate(verified_claims: List[Dict[str, Any]]) -> float:
        """Fraction of claims with label 'supported'."""
        if not verified_claims:
            return 0.0
        supported = sum(
            1 for vc in verified_claims if vc.get("label") == "supported"
        )
        return supported / len(verified_claims)

    def _identify_missing_info(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        verified_claims: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Identify specific types of missing information.

        Returns a list of human-readable strings describing gaps.
        """
        missing: List[str] = []

        # No documents retrieved at all
        if not retrieved_docs:
            missing.append("No documents were retrieved for this query.")
            return missing

        # Coverage gaps
        keywords = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", query)}
        all_text = " ".join(self._get_doc_text(d) for d in retrieved_docs).lower()
        missing_keywords = [kw for kw in keywords if kw not in all_text]
        if missing_keywords:
            missing.append(f"Query keywords not found in retrieved evidence: {', '.join(sorted(missing_keywords[:5]))}.")

        # Unsupported or unverifiable claims
        unsupported = [
            vc for vc in verified_claims
            if vc.get("label") in ("unsupported", "unverifiable")
        ]
        if unsupported:
            missing.append(
                f"{len(unsupported)} claim(s) could not be verified against the retrieved evidence."
            )

        # Contradicted claims
        contradicted = [vc for vc in verified_claims if vc.get("label") == "contradicted"]
        if contradicted:
            missing.append(
                f"{len(contradicted)} claim(s) are contradicted by retrieved evidence."
            )

        return missing

    @staticmethod
    def _build_reasoning(
        coverage: float,
        relevance_score: float,
        source_quality: str,
        claim_support_rate: float,
        contradictions_present: bool,
        missing_info: List[str],
        sufficient: bool,
        confidence: float,
    ) -> str:
        """Build a human-readable reasoning string. Always non-empty."""
        verdict = "SUFFICIENT" if sufficient else "INSUFFICIENT"
        parts = [
            f"Sufficiency verdict: {verdict} (confidence={confidence:.2f}).",
            f"Coverage={coverage:.2f}, mean_relevance={relevance_score:.2f}, "
            f"source_quality={source_quality}, claim_support_rate={claim_support_rate:.2f}.",
        ]
        if contradictions_present:
            parts.append("Contradictions detected in retrieved evidence.")
        if missing_info:
            parts.append("Missing information: " + " ".join(missing_info))
        parts.append(
            "Note: This is a heuristic assessment and has not been clinically validated."
        )
        return " ".join(parts)
