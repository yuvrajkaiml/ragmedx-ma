"""
Evidence verification for the RAGMedX-MA retrieval engine.

Performs deterministic quality checks on reranked medical evidence
before it is passed to the generation / multi-agent layer.

This component does NOT perform medical claim verification.
It verifies that retrieved evidence is structurally usable,
non-empty, sufficiently informative, and free from duplicates.

Pipeline:

    Reranked Results
          ↓
    EvidenceVerifier
          ↓
    Validated Evidence
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.rag.chunking.models import MedicalChunk


@dataclass(frozen=True)
class EvidenceVerificationResult:
    """
    Result of verifying one retrieved evidence chunk.
    """

    chunk: MedicalChunk
    score: float
    is_valid: bool
    reasons: tuple[str, ...]


class EvidenceVerifier:
    """
    Deterministic evidence-quality verifier.

    Checks:

        1. Chunk identity
        2. Document identity
        3. Content presence
        4. Minimum content length
        5. Source metadata
        6. Publication year
        7. Duplicate evidence

    This class intentionally does not use an LLM.

    Medical claim-level verification will be handled later by
    the multi-agent verification layer.
    """

    DEFAULT_MIN_CONTENT_LENGTH = 100

    def __init__(
        self,
        min_content_length: int = DEFAULT_MIN_CONTENT_LENGTH,
    ) -> None:

        if min_content_length <= 0:
            raise ValueError(
                "min_content_length must be greater than zero."
            )

        self.min_content_length = min_content_length

    # ------------------------------------------------------------------
    # INDIVIDUAL CHECKS
    # ------------------------------------------------------------------

    def _validate_chunk(
        self,
        chunk: MedicalChunk,
    ) -> list[str]:
        """
        Validate the structural quality of one MedicalChunk.

        Returns:
            List of validation failure reasons.
            Empty list means the chunk passed all checks.
        """

        reasons: list[str] = []

        # --------------------------------------------------------------
        # Chunk ID
        # --------------------------------------------------------------

        if not isinstance(chunk.id, str) or not chunk.id.strip():
            reasons.append(
                "Missing chunk ID."
            )

        # --------------------------------------------------------------
        # Document ID
        # --------------------------------------------------------------

        if (
            not isinstance(chunk.document_id, str)
            or not chunk.document_id.strip()
        ):
            reasons.append(
                "Missing document ID."
            )

        # --------------------------------------------------------------
        # Content
        # --------------------------------------------------------------

        if not isinstance(chunk.content, str):
            reasons.append(
                "Chunk content is not a string."
            )
        else:
            content = chunk.content.strip()

            if not content:
                reasons.append(
                    "Chunk content is empty."
                )

            elif len(content) < self.min_content_length:
                reasons.append(
                    "Chunk content is shorter than the minimum "
                    f"required length ({self.min_content_length})."
                )

        # --------------------------------------------------------------
        # Title
        # --------------------------------------------------------------

        if (
            not isinstance(chunk.title, str)
            or not chunk.title.strip()
        ):
            reasons.append(
                "Missing source title."
            )

        # --------------------------------------------------------------
        # Publication year
        # --------------------------------------------------------------

        if chunk.publication_year is None:
            reasons.append(
                "Missing publication year."
            )

        # --------------------------------------------------------------
        # Journal
        # --------------------------------------------------------------

        if (
            not isinstance(chunk.journal, str)
            or not chunk.journal.strip()
        ):
            reasons.append(
                "Missing journal/source information."
            )

        return reasons

    # ------------------------------------------------------------------
    # DUPLICATE DETECTION
    # ------------------------------------------------------------------

    def _duplicate_key(
        self,
        chunk: MedicalChunk,
    ) -> str:
        """
        Generate a stable key for duplicate detection.

        Chunk ID is preferred because it uniquely identifies a chunk
        within the indexed corpus.
        """

        if isinstance(chunk.id, str) and chunk.id.strip():
            return chunk.id.strip()

        return chunk.content.strip()

    # ------------------------------------------------------------------
    # VERIFICATION
    # ------------------------------------------------------------------

    def verify(
        self,
        results: Sequence[
            tuple[MedicalChunk, float]
        ],
    ) -> list[EvidenceVerificationResult]:
        """
        Verify retrieved evidence candidates.

        Args:
            results:
                Retrieved/reranked results in the form:

                    [(MedicalChunk, score), ...]

        Returns:
            Verification results in the same input order.

        Duplicate chunks are marked invalid rather than silently
        removed so that the caller can inspect what happened.
        """

        if results is None:
            raise TypeError(
                "results cannot be None."
            )

        seen: set[str] = set()

        verification_results: list[
            EvidenceVerificationResult
        ] = []

        for chunk, score in results:

            if not isinstance(chunk, MedicalChunk):
                raise TypeError(
                    "Each result must contain a MedicalChunk."
                )

            if not isinstance(score, (float, int)):
                raise TypeError(
                    "Each evidence score must be numeric."
                )

            reasons = self._validate_chunk(chunk)

            # ----------------------------------------------------------
            # Duplicate detection
            # ----------------------------------------------------------

            duplicate_key = self._duplicate_key(chunk)

            if duplicate_key in seen:
                reasons.append(
                    "Duplicate evidence chunk."
                )

            seen.add(duplicate_key)

            verification_results.append(
                EvidenceVerificationResult(
                    chunk=chunk,
                    score=float(score),
                    is_valid=len(reasons) == 0,
                    reasons=tuple(reasons),
                )
            )

        return verification_results

    # ------------------------------------------------------------------
    # VALID EVIDENCE ONLY
    # ------------------------------------------------------------------

    def filter_valid(
        self,
        results: Sequence[
            tuple[MedicalChunk, float]
        ],
    ) -> list[tuple[MedicalChunk, float]]:
        """
        Return only evidence chunks that pass verification.

        Original scores are preserved.
        """

        verified = self.verify(results)

        return [
            (
                result.chunk,
                result.score,
            )
            for result in verified
            if result.is_valid
        ]

    # ------------------------------------------------------------------
    # CONVENIENCE
    # ------------------------------------------------------------------

    def verify_and_filter(
        self,
        results: Sequence[
            tuple[MedicalChunk, float]
        ],
    ) -> tuple[
        list[tuple[MedicalChunk, float]],
        list[EvidenceVerificationResult],
    ]:
        """
        Verify evidence and return both valid evidence and the
        complete verification report.
        """

        verification_report = self.verify(results)

        valid_results = [
            (
                result.chunk,
                result.score,
            )
            for result in verification_report
            if result.is_valid
        ]

        return valid_results, verification_report