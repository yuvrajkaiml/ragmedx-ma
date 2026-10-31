"""
Evidence-grounded answer generation for the RAGMedX-MA system.

The AnswerGenerator converts validated medical evidence and citation
metadata into a grounded natural-language answer using the existing
LocalLLMWrapper.

Pipeline:

    Validated Evidence
            +
        Citation Metadata
            ↓
       AnswerGenerator
            ↓
     Grounded Medical Answer

This component does not perform retrieval, reranking, or verification.
Those responsibilities belong to their respective layers.
"""

from __future__ import annotations

from typing import Any, Sequence

from app.llm.local_llm import LocalLLMWrapper
from app.rag.prompts.qa_prompt import render_prompt


class AnswerGenerator:
    """
    Generate evidence-grounded answers using the existing local LLM.

    The generator is intentionally independent of retrieval and
    verification. It expects evidence to have already passed through
    the verification layer.
    """

    def __init__(
        self,
        llm_wrapper: LocalLLMWrapper | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> None:
        """
        Initialize the answer generator.

        Args:
            llm_wrapper:
                Existing LocalLLMWrapper instance.

            max_new_tokens:
                Maximum number of tokens generated.

            temperature:
                Generation temperature.
        """

        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than zero."
            )

        if temperature < 0:
            raise ValueError(
                "temperature cannot be negative."
            )

        self.llm_wrapper = (
            llm_wrapper
            if llm_wrapper is not None
            else LocalLLMWrapper()
        )

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    # ------------------------------------------------------------------
    # EVIDENCE PREPARATION
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_evidence(
        evidence: Sequence[Any],
    ) -> list[dict[str, Any]]:
        """
        Convert evidence records into the format expected by the
        Agent 4 generator prompt.

        Supports dictionary-based records as well as objects containing
        MedicalChunk-like attributes.
        """

        prepared: list[dict[str, Any]] = []

        for index, item in enumerate(evidence, start=1):

            # ----------------------------------------------------------
            # Dictionary-based evidence
            # ----------------------------------------------------------

            if isinstance(item, dict):

                chunk = item.get("chunk", item)

                citation_marker = (
                    item.get("citation_marker")
                    or item.get("marker")
                    or f"[{index}]"
                )

                title = (
                    item.get("title")
                    or getattr(chunk, "title", "")
                )

                journal = (
                    item.get("journal")
                    or getattr(chunk, "journal", "")
                )

                publication_year = (
                    item.get("publication_year")
                    or getattr(chunk, "publication_year", "")
                )

                cited_text = (
                    item.get("cited_text")
                    or item.get("content")
                    or getattr(chunk, "content", "")
                )

            # ----------------------------------------------------------
            # Object-based evidence
            # ----------------------------------------------------------

            else:

                chunk = getattr(item, "chunk", item)

                citation_marker = (
                    getattr(item, "citation_marker", None)
                    or getattr(item, "marker", None)
                    or f"[{index}]"
                )

                title = (
                    getattr(item, "title", None)
                    or getattr(chunk, "title", "")
                )

                journal = (
                    getattr(item, "journal", None)
                    or getattr(chunk, "journal", "")
                )

                publication_year = (
                    getattr(item, "publication_year", None)
                    or getattr(chunk, "publication_year", "")
                )

                cited_text = (
                    getattr(item, "cited_text", None)
                    or getattr(item, "content", None)
                    or getattr(chunk, "content", "")
                )

            prepared.append(
                {
                    "citation_marker": str(citation_marker),
                    "title": str(title),
                    "journal": str(journal),
                    "publication_year": publication_year,
                    "cited_text": str(cited_text),
                }
            )

        return prepared

    # ------------------------------------------------------------------
    # GENERATION
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        evidence: Sequence[Any],
    ) -> str:
        """
        Generate an evidence-grounded medical answer.

        Args:
            query:
                Original user medical question.

            evidence:
                Evidence that has already passed verification.

        Returns:
            Generated answer as a string.
        """

        # --------------------------------------------------------------
        # Validate query
        # --------------------------------------------------------------

        if not isinstance(query, str):
            raise TypeError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        # --------------------------------------------------------------
        # Validate evidence
        # --------------------------------------------------------------

        if not evidence:
            raise ValueError(
                "At least one validated evidence item is required."
            )

        prepared_evidence = self._prepare_evidence(
            evidence
        )

        if not prepared_evidence:
            raise ValueError(
                "No usable evidence was provided."
            )

        # --------------------------------------------------------------
        # Render Agent 4 generator prompt
        # --------------------------------------------------------------

        prompt = render_prompt(
            "generator",
            {
                "question": query,
                "evidence": prepared_evidence,
            },
        )

        # --------------------------------------------------------------
        # Generate using existing LocalLLMWrapper
        # --------------------------------------------------------------

        answer = self.llm_wrapper.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )

        if answer is None:
            raise RuntimeError(
                "LLM returned no answer."
            )

        answer = str(answer).strip()

        if not answer:
            raise RuntimeError(
                "LLM returned an empty answer."
            )

        return answer

    # ------------------------------------------------------------------
    # CONVENIENCE METHOD
    # ------------------------------------------------------------------

    def generate_from_verified_evidence(
        self,
        query: str,
        verified_evidence: Sequence[Any],
    ) -> str:
        """
        Generate an answer specifically from verified evidence.

        This method makes the intended architecture explicit:

            Verification → Generation
        """

        return self.generate(
            query=query,
            evidence=verified_evidence,
        )