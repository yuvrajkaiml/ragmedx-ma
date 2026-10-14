"""
Citation generation for the RAGMedX-MA retrieval engine.

Converts verified medical evidence chunks into structured citation
records that can later be consumed by the generation and
multi-agent verification layers.

This component does NOT generate medical claims and does NOT use
an LLM.

Supported input formats:
    1. (MedicalChunk, score) tuples
    2. EvidenceVerificationResult objects

Pipeline:

    Verified Evidence
          ↓
    CitationGenerator
          ↓
    Structured Citations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.rag.chunking.models import MedicalChunk
from app.rag.verification.evidence_verifier import (
    EvidenceVerificationResult,
)


@dataclass(frozen=True)
class Citation:
    """
    Structured citation associated with one medical evidence chunk.
"""

    citation_id: str
    chunk_id: str
    document_id: str
    title: str
    journal: str
    publication_year: int | None
    section: str
    source_type: str


class CitationGenerator:
    """
    Generate deterministic citation records from medical evidence.

    Citation generation is independent from the LLM layer.
    Bibliographic metadata is extracted from MedicalChunk.
    """

    DEFAULT_SOURCE_TYPE = "medical_literature"

    def _build_citation_id(self, index: int) -> str:
        """Generate a human-readable citation identifier."""
        return f"[{index}]"

    def _create_citation(
        self,
        chunk: MedicalChunk,
        index: int,
    ) -> Citation:
        """Create a citation from one MedicalChunk."""

        if not isinstance(chunk, MedicalChunk):
            raise TypeError("chunk must be a MedicalChunk.")

        return Citation(
            citation_id=self._build_citation_id(index),
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            title=chunk.title.strip(),
            journal=chunk.journal.strip(),
            publication_year=chunk.publication_year,
            section=chunk.section.strip(),
            source_type=self.DEFAULT_SOURCE_TYPE,
        )

    def _extract_evidence(
        self,
        item: tuple[MedicalChunk, float]
        | EvidenceVerificationResult,
    ) -> tuple[MedicalChunk, float] | None:
        """
        Normalize supported evidence formats.

        Returns None for verification results that are invalid.
        """

        if isinstance(item, EvidenceVerificationResult):
            if not item.is_valid:
                return None

            chunk = item.chunk
            score = item.score

        elif isinstance(item, tuple) and len(item) == 2:
            chunk, score = item

        else:
            raise TypeError(
                "Each evidence item must be either a "
                "(MedicalChunk, score) tuple or an "
                "EvidenceVerificationResult."
            )

        if not isinstance(chunk, MedicalChunk):
            raise TypeError(
                "Each evidence item must contain a MedicalChunk."
            )

        if not isinstance(score, (int, float)):
            raise TypeError(
                "Each evidence score must be numeric."
            )

        return chunk, float(score)

    def generate(
        self,
        evidence: Sequence[
            tuple[MedicalChunk, float]
            | EvidenceVerificationResult
        ],
    ) -> list[Citation]:
        """
        Generate citations for evidence while preserving input order.

        Invalid EvidenceVerificationResult objects are skipped.
        Duplicate chunk IDs produce only one citation.
        """

        if evidence is None:
            raise TypeError("evidence cannot be None.")

        citations: list[Citation] = []
        seen_chunks: set[str] = set()

        for item in evidence:
            normalized = self._extract_evidence(item)

            if normalized is None:
                continue

            chunk, _score = normalized

            if chunk.id in seen_chunks:
                continue

            seen_chunks.add(chunk.id)

            citations.append(
                self._create_citation(
                    chunk=chunk,
                    index=len(citations) + 1,
                )
            )

        return citations

    def citation_map(
        self,
        evidence: Sequence[
            tuple[MedicalChunk, float]
            | EvidenceVerificationResult
        ],
    ) -> dict[str, Citation]:
        """Create a mapping from chunk ID to Citation."""

        citations = self.generate(evidence)

        return {
            citation.chunk_id: citation
            for citation in citations
        }

    def format_citation(
        self,
        citation: Citation,
    ) -> str:
        """Format a citation for human-readable output."""

        if not isinstance(citation, Citation):
            raise TypeError(
                "citation must be a Citation."
            )

        year = (
            str(citation.publication_year)
            if citation.publication_year is not None
            else "Year unavailable"
        )

        return (
            f"{citation.citation_id} "
            f"{citation.title} — "
            f"{citation.journal} ({year})"
        )

    def format_citations(
        self,
        citations: Sequence[Citation],
    ) -> list[str]:
        """Format multiple citations for display."""

        return [
            self.format_citation(citation)
            for citation in citations
        ]
