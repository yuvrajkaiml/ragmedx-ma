"""
Query expansion for the RAGMedX-MA retrieval engine.

Provides retrieval-oriented query reformulation without introducing
an additional LLM dependency.

The expander generates lightweight alternative formulations by:
- preserving the original query
- expanding common medical abbreviations
- adding terminology-aware variants
- generating keyword-focused formulations

This component is intentionally independent of the multi-agent
and LLM orchestration layer.
"""

from __future__ import annotations

import re


class QueryExpander:
    """
    Generates alternative formulations of a medical query.

    The original query is always preserved as the first query.
    """

    # Common medical abbreviations encountered in the knowledge base.
    MEDICAL_EXPANSIONS = {
        "fdg-pet": "18F-FDG PET",
        "fdg pet": "18F-FDG PET",
        "pet-ct": "PET/CT",
        "pet ct": "PET/CT",
        "icpi": "immune checkpoint inhibitor",
        "icpis": "immune checkpoint inhibitors",
        "ctla-4": "cytotoxic T-lymphocyte-associated antigen 4",
        "pd-1": "programmed cell death protein 1",
        "pd-l1": "programmed death ligand 1",
        "nsclc": "non-small cell lung cancer",
        "ct": "computed tomography",
        "mri": "magnetic resonance imaging",
    }

    def __init__(self, max_queries: int = 4) -> None:
        """
        Initialize the query expander.

        Args:
            max_queries:
                Maximum number of queries returned, including
                the original query.
        """

        if max_queries <= 0:
            raise ValueError(
                "max_queries must be greater than zero."
            )

        self.max_queries = max_queries

    # ------------------------------------------------------------------
    # NORMALIZATION
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(query: str) -> str:
        """Normalize whitespace while preserving medical terminology."""

        return re.sub(r"\s+", " ", query).strip()

    # ------------------------------------------------------------------
    # TERMINOLOGY EXPANSION
    # ------------------------------------------------------------------

    def _expand_medical_terms(self, query: str) -> str:
        """
        Expand recognized medical abbreviations.

        The original query is not modified; this method produces
        an alternative formulation.
        """

        expanded = query

        # --------------------------------------------------------------
        # Protect already-expanded FDG-PET terminology.
        # --------------------------------------------------------------

        expanded = re.sub(
            r"\b18F-18F-FDG PET\b",
            "18F-FDG PET",
            expanded,
            flags=re.IGNORECASE,
        )

        # --------------------------------------------------------------
        # Expand medical abbreviations.
        # --------------------------------------------------------------

        for abbreviation, full_term in self.MEDICAL_EXPANSIONS.items():

            pattern = re.compile(
                rf"(?<!18F-)\b{re.escape(abbreviation)}\b",
                flags=re.IGNORECASE,
            )

            expanded = pattern.sub(
                full_term,
                expanded,
            )

        # --------------------------------------------------------------
        # Final safety cleanup for accidental duplicate prefixes.
        # --------------------------------------------------------------

        expanded = re.sub(
            r"\b18F-(?:18F-)+FDG PET\b",
            "18F-FDG PET",
            expanded,
            flags=re.IGNORECASE,
        )

        return expanded

    # ------------------------------------------------------------------
    # QUESTION REFORMULATION
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_question_marker(query: str) -> str:
        """Remove a trailing question mark."""

        return query.rstrip().rstrip("?").strip()

    def _create_keyword_query(self, query: str) -> str:
        """
        Create a concise keyword-oriented formulation.

        This is deliberately lightweight and deterministic.
        """

        stopwords = {
            "what",
            "are",
            "is",
            "the",
            "a",
            "an",
            "of",
            "for",
            "to",
            "in",
            "on",
            "when",
            "how",
            "does",
            "do",
            "can",
            "with",
            "and",
        }

        words = re.findall(
            r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*",
            query,
        )

        keywords = [
            word
            for word in words
            if word.lower() not in stopwords
        ]

        return " ".join(keywords)

    def _create_context_query(self, query: str) -> str:
        """
        Convert a question into a retrieval-oriented formulation.
        """

        base = self._remove_question_marker(query)

        if not base:
            return ""

        return (
            f"medical evidence regarding {base}"
        )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def expand(self, query: str) -> list[str]:
        """
        Generate alternative retrieval queries.

        Args:
            query:
                Original natural-language medical question.

        Returns:
            A list of unique query formulations. The original query
            is always returned first.
        """

        if not isinstance(query, str):
            raise TypeError(
                "Query must be a string."
            )

        query = self._normalize(query)

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        candidates: list[str] = []

        def add_candidate(candidate: str) -> None:
            """Add a candidate if it is non-empty and unique."""

            candidate = self._normalize(candidate)

            if not candidate:
                return

            existing_queries = {
                existing.lower()
                for existing in candidates
            }

            if candidate.lower() in existing_queries:
                return

            candidates.append(candidate)

        # --------------------------------------------------------------
        # 1. Always preserve the original query.
        # --------------------------------------------------------------

        add_candidate(query)

        # --------------------------------------------------------------
        # 2. Medical terminology expansion.
        # --------------------------------------------------------------

        expanded = self._expand_medical_terms(query)

        if expanded.lower() != query.lower():
            add_candidate(expanded)

        # --------------------------------------------------------------
        # 3. Context-oriented formulation.
        # --------------------------------------------------------------

        context_query = self._create_context_query(query)

        add_candidate(context_query)

        # --------------------------------------------------------------
        # 4. Keyword-oriented formulation.
        # --------------------------------------------------------------

        keyword_query = self._create_keyword_query(query)

        add_candidate(keyword_query)

        return candidates[: self.max_queries]

    # ------------------------------------------------------------------
    # MULTI-QUERY EXPANSION
    # ------------------------------------------------------------------

    def expand_many(
        self,
        queries: list[str],
    ) -> list[str]:
        """
        Expand multiple queries and return one deduplicated list.
        """

        if not isinstance(queries, list):
            raise TypeError(
                "queries must be a list."
            )

        expanded_queries: list[str] = []

        for query in queries:

            for expanded in self.expand(query):

                existing_queries = {
                    existing.lower()
                    for existing in expanded_queries
                }

                if expanded.lower() not in existing_queries:
                    expanded_queries.append(expanded)

        return expanded_queries