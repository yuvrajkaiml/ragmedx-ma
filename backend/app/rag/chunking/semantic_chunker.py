import re
from typing import List

from app.rag.loaders.models import MedicalDocument
from app.rag.chunking.models import MedicalChunk
from app.utils.logger import logger


class MedicalChunker:
    """
    Splits medical documents into retrieval-friendly chunks.

    The chunker attempts to:
    - preserve section boundaries
    - remove unwanted article metadata
    - avoid extremely small chunks
    - maintain contextual overlap
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        min_chunk_size: int = 150,
        overlap: int | None = None,
    ):
        """
        Initialize the medical chunker.

        Args:
            chunk_size: Maximum target chunk size in characters.
            chunk_overlap: Number of characters to preserve as overlap.
            min_chunk_size: Minimum accepted chunk size.
            overlap: Backward-compatible alias for chunk_overlap.
        """

        # Support both `chunk_overlap` and the older `overlap`
        # parameter used by some tests.
        if overlap is not None:
            chunk_overlap = overlap

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if min_chunk_size < 0:
            raise ValueError("min_chunk_size cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        logger.info(
            "Medical Chunker initialized | "
            f"chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}, "
            f"min_chunk_size={min_chunk_size}"
        )

    @classmethod
    def from_settings(cls) -> "MedicalChunker":
        """Construct a MedicalChunker from the application Settings.

        Reads ``retrieval_chunk_size``, ``retrieval_chunk_overlap``, and
        ``retrieval_chunk_min_chars`` from the pydantic Settings singleton.
        Constructor-arg overrides used by tests are unaffected because they
        bypass this factory method and call ``__init__`` directly.
        """
        from app.config import get_settings  # local import avoids circular deps
        s = get_settings()
        return cls(
            chunk_size=s.retrieval_chunk_size,
            chunk_overlap=s.retrieval_chunk_overlap,
            min_chunk_size=s.retrieval_chunk_min_chars,
        )

    # ---------------------------------------------------------
    # TEXT CLEANING
    # ---------------------------------------------------------

    def _clean_text(self, text: str) -> str:
        """
        Normalize whitespace and remove obvious XML/article noise.
        """

        if not text:
            return ""

        # Normalize line endings.
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Normalize spaces and tabs.
        text = re.sub(r"[ \t]+", " ", text)

        # Normalize excessive blank lines.
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove common article markers.
        text = re.sub(
            r"^\s*(TEXT|ARTICLE)\s*$",
            "",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        return text.strip()

    # ---------------------------------------------------------
    # REMOVE REFERENCES / ARTICLE END MATTER
    # ---------------------------------------------------------

    def _remove_article_end_matter(self, text: str) -> str:
        """
        Remove references and publication-related material that
        should not normally participate in semantic retrieval.
        """

        patterns = [
            r"\n\s*References\s*\n",
            r"\n\s*REFERENCES\s*\n",
            r"\n\s*Conflict[- ]of[- ]interest statement\s*:",
            r"\n\s*Conflict of interest\s*:",
            r"\n\s*Manuscript source\s*:",
            r"\n\s*Peer-review report classification\s*",
            r"\n\s*Peer review report classification\s*",
        ]

        earliest_index = None

        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                if (
                    earliest_index is None
                    or match.start() < earliest_index
                ):
                    earliest_index = match.start()

        if earliest_index is not None:
            text = text[:earliest_index]

        return text.strip()

    # ---------------------------------------------------------
    # SECTION DETECTION
    # ---------------------------------------------------------

    def _looks_like_heading(self, paragraph: str) -> bool:
        """
        Heuristic detection of medical article section headings.
        """

        paragraph = paragraph.strip()

        if not paragraph:
            return False

        # Headings should generally be relatively short.
        if len(paragraph) > 150:
            return False

        # Avoid treating normal sentences as headings.
        if paragraph.endswith("."):
            return False

        # Common heading characteristics.
        heading_keywords = [
            "introduction",
            "background",
            "methods",
            "materials and methods",
            "results",
            "discussion",
            "conclusion",
            "conclusions",
            "evaluation",
            "limitations",
            "treatment",
            "diagnosis",
            "pathogenesis",
            "clinical",
            "management",
            "imaging",
            "diagnostic",
            "references",
        ]

        lowered = paragraph.lower()

        if lowered in heading_keywords:
            return True

        # Question-style headings are common in medical papers.
        if paragraph.endswith("?"):
            return True

        # Short title-like paragraphs.
        words = paragraph.split()

        if len(words) <= 12:
            return True

        return False

    # ---------------------------------------------------------
    # PARAGRAPH EXTRACTION
    # ---------------------------------------------------------

    def _extract_paragraphs(self, text: str) -> List[str]:
        """
        Convert article text into clean paragraphs.
        """

        text = self._clean_text(text)

        paragraphs = re.split(
            r"\n\s*\n",
            text,
        )

        cleaned = []

        for paragraph in paragraphs:
            paragraph = re.sub(
                r"\s+",
                " ",
                paragraph,
            ).strip()

            if paragraph:
                cleaned.append(paragraph)

        return cleaned

    # ---------------------------------------------------------
    # SECTION-AWARE CHUNKING
    # ---------------------------------------------------------

    def _create_chunks(
        self,
        paragraphs: List[str],
    ) -> List[tuple[str, str]]:
        """
        Create section-aware text chunks.

        Returns:
            List of (section, chunk_text)
        """

        chunks = []

        current_section = "Unknown"
        current_paragraphs = []
        current_length = 0

        def flush_chunk():
            nonlocal current_paragraphs
            nonlocal current_length

            if not current_paragraphs:
                return

            content = " ".join(
                current_paragraphs
            ).strip()

            if len(content) >= self.min_chunk_size:
                chunks.append(
                    (
                        current_section,
                        content,
                    )
                )

            current_paragraphs = []
            current_length = 0

        for paragraph in paragraphs:

            # Detect a new medical section.
            if self._looks_like_heading(paragraph):
                flush_chunk()

                current_section = paragraph

                continue

            paragraph_length = len(paragraph)

            # If adding the paragraph exceeds the chunk limit,
            # finalize the current chunk first.
            if (
                current_length + paragraph_length
                > self.chunk_size
                and current_paragraphs
            ):
                flush_chunk()

                # Add contextual overlap from the previous chunk.
                if chunks and self.chunk_overlap > 0:
                    previous_text = chunks[-1][1]

                    # Approximate character overlap.
                    overlap_text = previous_text[
                        -self.chunk_overlap:
                    ]

                    if overlap_text:
                        current_paragraphs = [
                            overlap_text
                        ]

                        current_length = len(
                            overlap_text
                        )

            current_paragraphs.append(paragraph)
            current_length += paragraph_length

        # Flush the final chunk.
        flush_chunk()

        return chunks

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------

    def chunk_document(
        self,
        document: MedicalDocument,
    ) -> List[MedicalChunk]:
        """
        Convert a MedicalDocument into retrieval-ready chunks.
        """

        logger.info(
            f"Chunking document: {document.id}"
        )

        # Clean document text.
        text = self._clean_text(
            document.content
        )

        # Remove references and publication metadata.
        text = self._remove_article_end_matter(
            text
        )

        # Extract paragraphs.
        paragraphs = self._extract_paragraphs(
            text
        )

        # Create raw section-aware chunks.
        raw_chunks = self._create_chunks(
            paragraphs
        )

        chunks = []

        for index, (section, content) in enumerate(
            raw_chunks
        ):
            chunk_id = (
                f"{document.id}_chunk_{index}"
            )

            chunk = MedicalChunk(
                id=chunk_id,
                document_id=document.id,
                content=content,
                title=document.title,
                section=section,
                source=document.source,
                journal=document.journal,
                publication_year=document.publication_year,
                metadata={
                    **document.metadata,
                    "chunk_index": index,
                    "total_chunks": len(raw_chunks),
                },
            )

            chunks.append(chunk)

        logger.info(
            f"Created {len(chunks)} chunks "
            f"from {document.id}"
        )

        return chunks

    def chunk_documents(
        self,
        documents: List[MedicalDocument],
    ) -> List[MedicalChunk]:
        """
        Chunk multiple medical documents.
        """

        all_chunks = []

        logger.info(
            f"Starting chunking for "
            f"{len(documents)} document(s)"
        )

        for document in documents:
            try:
                chunks = self.chunk_document(
                    document
                )

                all_chunks.extend(chunks)

            except Exception as e:
                logger.error(
                    f"Failed to chunk "
                    f"{document.id}: {e}"
                )

        logger.success(
            f"Created {len(all_chunks)} total "
            f"medical chunks."
        )

        return all_chunks