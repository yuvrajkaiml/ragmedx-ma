
"""Iterative medical evidence retrieval with bounded gap analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from app.rag.chunking.models import MedicalChunk
from app.rag.retrievers.hybrid_retriever import HybridRetriever


@dataclass(frozen=True)
class RetrievalIteration:
    """Summary of one retrieval iteration."""

    iteration: int
    queries: tuple[str, ...]
    retrieved_count: int
    new_evidence_count: int
    remaining_gaps: tuple[str, ...]


@dataclass
class IterativeRetrievalResult:
    """Evidence accumulated by iterative retrieval."""

    query: str
    evidence: list[tuple[MedicalChunk, float]]
    iterations: list[RetrievalIteration] = field(default_factory=list)
    stop_reason: str = "max_iterations_reached"

    @property
    def iteration_count(self) -> int:
        return len(self.iterations)


GapAnalyzer = Callable[
    [str, Sequence[tuple[MedicalChunk, float]]],
    Sequence[str],
]


class MedicalGapAnalyzer:
    """
    Lightweight, deterministic gap analyzer.

    Identifies potentially missing aspects of a medical question.
    This is a baseline heuristic, not clinical claim verification.
    A stronger analyzer can be injected through the gap_analyzer
    argument of IterativeRetriever.
    """

    ASPECTS = {
        "population": (
            "population", "patients", "adults", "children",
            "elderly", "pregnant",
        ),
        "treatment": (
            "treatment", "therapy", "drug", "medication",
            "intervention", "dose",
        ),
        "outcomes": (
            "outcome", "effectiveness", "efficacy", "survival",
            "mortality", "risk", "adverse effect", "safety",
        ),
        "diagnosis": (
            "diagnosis", "diagnostic", "sensitivity", "specificity",
            "screening", "biomarker",
        ),
        "limitations": (
            "limitation", "limitations", "uncertainty", "bias",
            "challenge", "drawback",
        ),
    }

    def analyze(
        self,
        query: str,
        evidence: Sequence[tuple[MedicalChunk, float]],
    ) -> list[str]:
        if not query.strip():
            raise ValueError("query cannot be empty.")

        text = " ".join(
            chunk.content.lower()
            for chunk, _ in evidence
        )

        query_lower = query.lower()
        gaps: list[str] = []

        for aspect, terms in self.ASPECTS.items():
            requested = any(term in query_lower for term in terms)
            covered = any(term in text for term in terms)

            if requested and not covered:
                gaps.append(aspect)

        return gaps


class IterativeRetriever:
    """
    Runs hybrid retrieval repeatedly to improve evidence coverage.

    Args:
        retriever: Existing HybridRetriever instance.
        max_iterations: Maximum retrieval rounds; capped at three.
        top_k: Number of results requested per retrieval round.
        gap_analyzer: Optional callable that returns unresolved gaps.
        max_follow_up_queries: Maximum follow-up queries per round.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        max_iterations: int = 3,
        top_k: int = 5,
        gap_analyzer: GapAnalyzer | None = None,
        max_follow_up_queries: int = 3,
    ) -> None:
        if not 1 <= max_iterations <= 3:
            raise ValueError("max_iterations must be between 1 and 3.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if max_follow_up_queries <= 0:
            raise ValueError(
                "max_follow_up_queries must be greater than zero."
            )

        self.retriever = retriever
        self.max_iterations = max_iterations
        self.top_k = top_k
        self.gap_analyzer = gap_analyzer or MedicalGapAnalyzer().analyze
        self.max_follow_up_queries = max_follow_up_queries

    @staticmethod
    def _key(chunk: MedicalChunk) -> str:
        return chunk.id.strip() or (
            f"{chunk.document_id}:{chunk.content.strip()}"
        )

    @staticmethod
    def _follow_up_queries(
        original_query: str,
        gaps: Sequence[str],
        existing_queries: set[str],
        limit: int,
    ) -> list[str]:
        queries: list[str] = []

        for gap in gaps:
            follow_up = f"{original_query.strip()} evidence about {gap}"
            normalized = re.sub(r"\s+", " ", follow_up).casefold()

            if normalized not in existing_queries:
                existing_queries.add(normalized)
                queries.append(follow_up)

            if len(queries) >= limit:
                break

        return queries

    def retrieve(self, query: str) -> IterativeRetrievalResult:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty.")

        evidence_by_id: dict[str, tuple[MedicalChunk, float]] = {}
        history: list[RetrievalIteration] = []
        queries_this_round = [query]
        seen_queries = {query.casefold()}
        stop_reason = "max_iterations_reached"

        for iteration in range(1, self.max_iterations + 1):
            retrieved: list[tuple[MedicalChunk, float]] = []

            for current_query in queries_this_round:
                retrieved.extend(
                    self.retriever.retrieve(
                        query=current_query,
                        top_k=self.top_k,
                    )
                )

            new_count = 0
            for chunk, score in retrieved:
                key = self._key(chunk)
                previous = evidence_by_id.get(key)

                if previous is None:
                    evidence_by_id[key] = (chunk, float(score))
                    new_count += 1
                elif score > previous[1]:
                    evidence_by_id[key] = (chunk, float(score))

            evidence = list(evidence_by_id.values())
            gaps = list(self.gap_analyzer(query, evidence))

            history.append(
                RetrievalIteration(
                    iteration=iteration,
                    queries=tuple(queries_this_round),
                    retrieved_count=len(retrieved),
                    new_evidence_count=new_count,
                    remaining_gaps=tuple(gaps),
                )
            )

            if not gaps:
                stop_reason = "gaps_resolved"
                break

            if new_count == 0:
                stop_reason = "no_new_evidence"
                break

            if iteration == self.max_iterations:
                stop_reason = "max_iterations_reached"
                break

            queries_this_round = self._follow_up_queries(
                original_query=query,
                gaps=gaps,
                existing_queries=seen_queries,
                limit=self.max_follow_up_queries,
            )

            if not queries_this_round:
                stop_reason = "no_new_queries"
                break

        final_evidence = sorted(
            evidence_by_id.values(),
            key=lambda item: item[1],
            reverse=True,
        )

        return IterativeRetrievalResult(
            query=query,
            evidence=final_evidence,
            iterations=history,
            stop_reason=stop_reason,
        )
