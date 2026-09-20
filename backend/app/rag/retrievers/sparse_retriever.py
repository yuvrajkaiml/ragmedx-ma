"""
Sparse lexical retrieval for the RAGMedX-MA retrieval engine.

Implements Okapi BM25 retrieval over MedicalChunk objects.

BM25 complements dense semantic retrieval by matching important
medical terms, abbreviations, drug names, disease names, and other
lexical signals that may not always be captured by semantic similarity.
"""

from __future__ import annotations

import math
import pickle
import re
from pathlib import Path
from typing import Sequence

from app.rag.chunking.models import MedicalChunk


class SparseRetriever:
    """
    Sparse lexical retriever using the Okapi BM25 ranking function.

    The retriever stores MedicalChunk objects and builds an inverted
    lexical representation of their content.

    BM25 parameters:
        k1:
            Controls term-frequency saturation.

        b:
            Controls document-length normalization.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative.")

        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")

        self.k1 = k1
        self.b = b

        # Position in this list corresponds to the internal
        # document ID used by the BM25 index.
        self.chunks: list[MedicalChunk] = []

        # Tokenized representation of every chunk.
        self.documents: list[list[str]] = []

        # Term frequencies for every document.
        #
        # Example:
        # {
        #     "cancer": 4,
        #     "therapy": 2,
        # }
        self.term_frequencies: list[dict[str, int]] = []

        # Document frequency:
        # number of documents containing each term.
        self.document_frequency: dict[str, int] = {}

        # Document lengths in tokens.
        self.document_lengths: list[int] = []

        self.total_documents = 0
        self.average_document_length = 0.0

    # ------------------------------------------------------------------
    # TOKENIZATION
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Convert text into normalized lexical tokens.

        The tokenizer:
        - converts text to lowercase
        - preserves alphanumeric medical terms
        - handles terms containing numbers
        - removes punctuation
        """

        if not text:
            return []

        text = text.lower()

        # Keep alphabetic and numeric sequences together.
        #
        # Examples:
        #   FDG-PET -> fdg, pet
        #   CTLA-4  -> ctla, 4
        #   PD-1    -> pd, 1
        #   18F     -> 18f
        tokens = re.findall(
            r"[a-z0-9]+",
            text,
        )

        return tokens

    # ------------------------------------------------------------------
    # INDEXING
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: Sequence[MedicalChunk],
    ) -> None:
        """
        Add MedicalChunk objects to the BM25 index.

        Existing chunks remain indexed, allowing the knowledge base
        to be extended incrementally.
        """

        if not chunks:
            return

        for chunk in chunks:
            if not isinstance(chunk, MedicalChunk):
                raise TypeError(
                    "All items must be MedicalChunk objects."
                )

            tokens = self._tokenize(chunk.content)

            term_frequency: dict[str, int] = {}

            for token in tokens:
                term_frequency[token] = (
                    term_frequency.get(token, 0) + 1
                )

            document_id = len(self.chunks)

            self.chunks.append(chunk)
            self.documents.append(tokens)
            self.term_frequencies.append(term_frequency)
            self.document_lengths.append(len(tokens))

            # Update document frequency only once per term
            # within the current document.
            for term in term_frequency:
                self.document_frequency[term] = (
                    self.document_frequency.get(term, 0) + 1
                )

            self.total_documents += 1

        self._update_average_document_length()

    def _update_average_document_length(self) -> None:
        """Recalculate the average indexed document length."""

        if self.total_documents == 0:
            self.average_document_length = 0.0
            return

        self.average_document_length = (
            sum(self.document_lengths)
            / self.total_documents
        )

    # ------------------------------------------------------------------
    # BM25 SCORING
    # ------------------------------------------------------------------

    def _idf(self, term: str) -> float:
        """
        Calculate BM25 inverse document frequency.

        Uses the standard smoothed BM25 formulation:

            IDF = ln(
                1 + (N - df + 0.5) / (df + 0.5)
            )
        """

        document_frequency = self.document_frequency.get(
            term,
            0,
        )

        if document_frequency == 0:
            return 0.0

        numerator = (
            self.total_documents
            - document_frequency
            + 0.5
        )

        denominator = document_frequency + 0.5

        return math.log(
            1.0 + numerator / denominator
        )

    def _score_document(
        self,
        query_terms: list[str],
        document_index: int,
    ) -> float:
        """
        Calculate the BM25 score for one document.
        """

        if self.total_documents == 0:
            return 0.0

        document_length = self.document_lengths[
            document_index
        ]

        if self.average_document_length == 0:
            return 0.0

        term_frequency = self.term_frequencies[
            document_index
        ]

        score = 0.0

        for term in query_terms:

            if term not in term_frequency:
                continue

            tf = term_frequency[term]
            idf = self._idf(term)

            normalization = (
                self.k1
                * (
                    1
                    - self.b
                    + self.b
                    * (
                        document_length
                        / self.average_document_length
                    )
                )
            )

            numerator = (
                tf * (self.k1 + 1)
            )

            denominator = (
                tf + normalization
            )

            score += (
                idf
                * numerator
                / denominator
            )

        return score

    # ------------------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[MedicalChunk, float]]:
        """
        Retrieve the most lexically relevant medical chunks.

        Args:
            query:
                Natural-language medical question.

            top_k:
                Number of results to return.

        Returns:
            List of (MedicalChunk, BM25 score) tuples ordered from
            highest score to lowest score.
        """

        if not isinstance(query, str):
            raise TypeError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if self.total_documents == 0:
            return []

        query_terms = self._tokenize(query)

        if not query_terms:
            return []

        # Calculate BM25 score for every indexed chunk.
        scored_results: list[
            tuple[MedicalChunk, float]
        ] = []

        for document_index in range(
            self.total_documents
        ):
            score = self._score_document(
                query_terms=query_terms,
                document_index=document_index,
            )

            # Only return documents with at least one
            # matching query term.
            if score > 0:
                scored_results.append(
                    (
                        self.chunks[document_index],
                        float(score),
                    )
                )

        # Highest BM25 score first.
        scored_results.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return scored_results[:top_k]

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Persist the sparse retriever to disk.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        state = {
            "k1": self.k1,
            "b": self.b,
            "chunks": self.chunks,
            "documents": self.documents,
            "term_frequencies": self.term_frequencies,
            "document_frequency": self.document_frequency,
            "document_lengths": self.document_lengths,
            "total_documents": self.total_documents,
            "average_document_length": (
                self.average_document_length
            ),
        }

        with open(path, "wb") as file:
            pickle.dump(
                state,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "SparseRetriever":
        """
        Load a previously persisted sparse retriever.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Sparse retriever not found: {path}"
            )

        with open(path, "rb") as file:
            state = pickle.load(file)

        retriever = cls(
            k1=state["k1"],
            b=state["b"],
        )

        retriever.chunks = state["chunks"]
        retriever.documents = state["documents"]
        retriever.term_frequencies = (
            state["term_frequencies"]
        )
        retriever.document_frequency = (
            state["document_frequency"]
        )
        retriever.document_lengths = (
            state["document_lengths"]
        )
        retriever.total_documents = (
            state["total_documents"]
        )
        retriever.average_document_length = (
            state["average_document_length"]
        )

        return retriever

    # ------------------------------------------------------------------
    # INFORMATION
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Return the number of indexed chunks."""

        return self.total_documents

    @property
    def vocabulary_size(self) -> int:
        """Return the number of unique indexed terms."""

        return len(self.document_frequency)