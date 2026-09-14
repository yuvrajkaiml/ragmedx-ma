"""
Citation generation for the RAGMedX-MA retrieval engine.

Converts verified medical evidence chunks into structured citation
records that can later be consumed by the generation and
multi-agent verification layers.

This component does NOT generate medical claims and does NOT use
an LLM.

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


@dataclass(frozen=True)
class Citation:
    """
    Structured citation associated with one medical evidence chunk.

    The citation contains bibliographic metadata together with the
    originating chunk identifier so that generated answers can be
    traced back to the exact retrieved evidence.
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

    Citation generation is intentionally independent from the LLM
    layer. The generator only extracts and normalizes metadata that
    already exists in MedicalChunk.
    """

    DEFAULT_SOURCE_TYPE = "medical_literature"

    # ------------------------------------------------------------------
    # CITATION ID
    # ------------------------------------------------------------------

    def _build_citation_id(
        self,
        index: int,
    ) -> str:
        """
        Generate a human-readable citation identifier.

        Example:

            [Doc 1]
            [Doc 2]
            [Doc 3]
        """

        return f"[Doc {index}]"

    # ------------------------------------------------------------------
    # INDIVIDUAL CITATION
    # ------------------------------------------------------------------

    def _create_citation(
        self,
        chunk: MedicalChunk,
        index: int,
    ) -> Citation:
        """
        Create a citation from one MedicalChunk.
        """

        if not isinstance(chunk, MedicalChunk):
            raise TypeError(
                "chunk must be a MedicalChunk."
            )

        return Citation(
            citation_id=self._build_citation_id(index),
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            title=(chunk.title or "Untitled source").strip(),
            journal=(chunk.journal or "Journal unavailable").strip(),
            publication_year=chunk.publication_year,
            section=(chunk.section or "").strip(),
            source_type=self.DEFAULT_SOURCE_TYPE,
        )

    # ------------------------------------------------------------------
    # GENERATION
    # ------------------------------------------------------------------

    def generate(
        self,
        evidence: Sequence[
            tuple[MedicalChunk, float]
        ],
    ) -> list[Citation]:
        """
        Generate citations for retrieved evidence.

        Args:
            evidence:
                Evidence chunks in the form:

                    [(MedicalChunk, score), ...]

                The evidence is expected to already have passed
                the evidence verification layer.

        Returns:
            List of Citation objects preserving the evidence order.
        """

        if evidence is None:
            raise TypeError(
                "evidence cannot be None."
            )

        citations: list[Citation] = []

        seen_chunks: set[str] = set()

        for index, (chunk, _) in enumerate(
            evidence,
            start=1,
        ):
            if not isinstance(chunk, MedicalChunk):
                raise TypeError(
                    "Each evidence item must contain a MedicalChunk."
                )

            # ----------------------------------------------------------
            # Avoid duplicate citations for the same chunk.
            # ----------------------------------------------------------

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

    # ------------------------------------------------------------------
    # CITATION LOOKUP
    # ------------------------------------------------------------------

    def citation_map(
        self,
        evidence: Sequence[
            tuple[MedicalChunk, float]
        ],
    ) -> dict[str, Citation]:
        """
        Create a mapping from chunk ID to Citation.

        Useful when the generation layer needs to associate a
        retrieved chunk with its citation metadata.
        """

        citations = self.generate(evidence)

        return {
            citation.chunk_id: citation
            for citation in citations
        }

    # ------------------------------------------------------------------
    # FORMATTING
    # ------------------------------------------------------------------

    def format_citation(
        self,
        citation: Citation,
    ) -> str:
        """
        Format a citation for human-readable output.

        Example:

            [1] Evaluation of response to immune checkpoint
            inhibitors... — Nat Rev Cancer (2017)
        """

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
        """
        Format multiple citations for display.
        """

        return [
            self.format_citation(citation)
            for citation in citations
        ]
