"""
Contradiction detection for RAGMedX-MA retrieved passages.

Detects contradictions between passage pairs using:
  - Lexical negation patterns (direct negation)
  - Numeric/dosage/date conflict detection
  - Scope-difference classification (population, time, study design)

Does NOT assume the newest source is correct.
Unresolved material conflicts trigger abstention_recommended=True.

Evidence tier: SOFTWARE TEST (heuristic, not clinically validated)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Numeric extraction helpers
# ---------------------------------------------------------------------------

_NUMBER_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s*(%|mg|mcg|µg|g|kg|ml|l|mmhg|iu|units?|days?|weeks?|months?|years?|hours?|mins?)\b", re.IGNORECASE)
_YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")
_NEGATION_PATTERN = re.compile(
    r"\b(not|no|never|neither|nor|contraindicated|ineffective|does not|do not|did not|cannot|won't|doesn't|isn't|aren't|wasn't|weren't|failed to)\b",
    re.IGNORECASE,
)


def _extract_numbers_with_units(text: str) -> List[Tuple[float, str]]:
    """Return list of (value, unit) pairs from text."""
    results = []
    for match in _NUMBER_PATTERN.finditer(text):
        try:
            results.append((float(match.group(1)), match.group(2).lower()))
        except ValueError:
            pass
    return results


def _extract_years(text: str) -> List[int]:
    return [int(m.group(1)) for m in _YEAR_PATTERN.finditer(text)]


def _has_negation(text: str) -> bool:
    return bool(_NEGATION_PATTERN.search(text))


def _key_terms(text: str) -> set:
    """Extract salient medical key terms (lowercased alphabetic words > 4 chars)."""
    return {w.lower() for w in re.findall(r"[a-zA-Z]{5,}", text)}


def _overlap_ratio(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


# ---------------------------------------------------------------------------
# Scope difference heuristics
# ---------------------------------------------------------------------------

_POPULATION_TERMS = re.compile(
    r"\b(children|pediatric|adult|elderly|geriatric|pregnant|neonatal|infant|obese|diabetic|hypertensive)\b",
    re.IGNORECASE,
)
_DESIGN_TERMS = re.compile(
    r"\b(retrospective|prospective|randomized|observational|case.control|cross.sectional|systematic review|meta.analysis|cohort|rct)\b",
    re.IGNORECASE,
)


def _populations(text: str) -> set:
    return {m.group(0).lower() for m in _POPULATION_TERMS.finditer(text)}


def _study_designs(text: str) -> set:
    return {m.group(0).lower() for m in _DESIGN_TERMS.finditer(text)}


# ---------------------------------------------------------------------------
# ContradictionDetector
# ---------------------------------------------------------------------------

class ContradictionDetector:
    """
    Detect contradictions between pairs of retrieved passages.

    Supported contradiction types:
      - true_contradiction: direct negation of the same claim
      - scope_difference: same topic, different populations or study designs
      - temporal_difference: conflicting numeric or directional claims across time periods
      - population_difference: same claim applies to different patient groups

    Does NOT use an LLM. Does NOT assume newer sources are correct.
    """

    # Minimum topic overlap to consider passages as potentially contradictory
    MIN_OVERLAP_FOR_COMPARISON = 0.15

    def detect_contradictions(
        self,
        passages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Check all passage pairs for contradictions.

        Args:
            passages: List of passage dicts, each with at minimum:
                - 'id' or 'doc_id': identifier
                - 'text' or 'cited_text': the passage text
                - (optional) 'publication_year': int

        Returns:
            List of contradiction records:
            {
                'pair': (id_a, id_b),
                'type': str,
                'description': str,
                'severity': 'high'|'medium'|'low',
                'abstention_recommended': bool,
            }
        """
        results = []
        normalised = [self._normalise_passage(p) for p in passages]

        for i in range(len(normalised)):
            for j in range(i + 1, len(normalised)):
                p_a = normalised[i]
                p_b = normalised[j]
                contradictions = self._compare_pair(p_a, p_b)
                results.extend(contradictions)
        return results

    # ------------------------------------------------------------------
    # Internal: normalise and compare
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_passage(p: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the fields we need for comparison."""
        pid = p.get("id") or p.get("doc_id") or p.get("chunk_id") or "unknown"
        text = p.get("text") or p.get("cited_text") or ""
        year = p.get("publication_year")
        return {"id": str(pid), "text": text, "year": year}

    def _compare_pair(
        self,
        a: Dict[str, Any],
        b: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Compare a single pair of passages and return all detected contradictions."""
        text_a = a["text"]
        text_b = b["text"]
        id_a = a["id"]
        id_b = b["id"]

        terms_a = _key_terms(text_a)
        terms_b = _key_terms(text_b)
        overlap = _overlap_ratio(terms_a, terms_b)

        if overlap < self.MIN_OVERLAP_FOR_COMPARISON:
            return []

        results = []

        # --- 1. Direct negation check ---
        neg_a = _has_negation(text_a)
        neg_b = _has_negation(text_b)
        if neg_a != neg_b and overlap >= 0.25:
            results.append({
                "pair": (id_a, id_b),
                "type": "true_contradiction",
                "description": (
                    f"One passage contains negation while the other does not, "
                    f"on overlapping topic (overlap={overlap:.2f}). "
                    f"Negation in {'A' if neg_a else 'B'}, absent in {'B' if neg_a else 'A'}."
                ),
                "severity": "high",
                "abstention_recommended": True,
            })

        # --- 2. Contradictory numbers/dosages ---
        nums_a = _extract_numbers_with_units(text_a)
        nums_b = _extract_numbers_with_units(text_b)
        numeric_conflict = self._check_numeric_conflict(nums_a, nums_b)
        if numeric_conflict:
            results.append({
                "pair": (id_a, id_b),
                "type": "true_contradiction",
                "description": f"Conflicting numeric values: {numeric_conflict}",
                "severity": "high",
                "abstention_recommended": True,
            })

        # --- 3. Temporal difference (dates / years) ---
        years_a = _extract_years(text_a)
        years_b = _extract_years(text_b)
        if years_a and years_b:
            min_gap = 5
            max_a = max(years_a)
            max_b = max(years_b)
            if abs(max_a - max_b) >= min_gap and overlap >= 0.20:
                results.append({
                    "pair": (id_a, id_b),
                    "type": "temporal_difference",
                    "description": (
                        f"Passages reference different time periods "
                        f"({max_a} vs {max_b}, gap={abs(max_a - max_b)} years). "
                        "Recommendations may differ across eras."
                    ),
                    "severity": "medium",
                    "abstention_recommended": False,
                })

        # --- 4. Population difference ---
        pop_a = _populations(text_a)
        pop_b = _populations(text_b)
        if pop_a and pop_b and not (pop_a & pop_b) and overlap >= 0.20:
            results.append({
                "pair": (id_a, id_b),
                "type": "population_difference",
                "description": (
                    f"Passages describe different patient populations "
                    f"(A: {sorted(pop_a)}, B: {sorted(pop_b)}). "
                    "Findings may not generalise across groups."
                ),
                "severity": "low",
                "abstention_recommended": False,
            })

        # --- 5. Study design / scope difference ---
        design_a = _study_designs(text_a)
        design_b = _study_designs(text_b)
        if design_a and design_b and not (design_a & design_b) and overlap >= 0.20:
            results.append({
                "pair": (id_a, id_b),
                "type": "scope_difference",
                "description": (
                    f"Passages use different study designs "
                    f"(A: {sorted(design_a)}, B: {sorted(design_b)}), "
                    "which may explain conflicting conclusions."
                ),
                "severity": "low",
                "abstention_recommended": False,
            })

        return results

    @staticmethod
    def _check_numeric_conflict(
        nums_a: List[Tuple[float, str]],
        nums_b: List[Tuple[float, str]],
    ) -> Optional[str]:
        """
        Detect conflicting numeric values for the same unit across passages.
        Returns a description string if a conflict is detected, else None.
        """
        # Group by unit
        by_unit_a: Dict[str, List[float]] = {}
        for val, unit in nums_a:
            by_unit_a.setdefault(unit, []).append(val)
        for val, unit in nums_b:
            if unit in by_unit_a:
                vals_a = by_unit_a[unit]
                # Conflict if the numeric ranges don't overlap AND both are medical-scale
                min_a, max_a = min(vals_a), max(vals_a)
                if val < min_a * 0.5 or val > max_a * 2.0:
                    return (
                        f"Passage A reports {unit} values in [{min_a}, {max_a}], "
                        f"Passage B reports {val} {unit}."
                    )
        return None
