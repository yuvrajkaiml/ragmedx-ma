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
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        logger.info(
            "Medical Chunker initialized | "
            f"chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}, "
            f"min_chunk_size={min_chunk_size}"
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

        # Normalize line endings
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Remove repeated whitespace
        text = re.sub(r"[ \t]+", " ", text)

        # Normalize excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove common article markers
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

        # Common heading characteristics
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

            content = " ".join(current_paragraphs).strip()

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

            if self._looks_like_heading(paragraph):
                flush_chunk()

                current_section = paragraph

                continue

            paragraph_length = len(paragraph)

            # If adding the paragraph exceeds the limit,
            # finalize the current chunk.
            if (
                current_length + paragraph_length
                > self.chunk_size
                and current_paragraphs
            ):
                flush_chunk()

                # Add overlap from previous paragraph.
                if chunks:
                    previous_text = chunks[-1][1]

                    overlap_text = previous_text[
                        -self.chunk_overlap * 4:
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

        text = self._clean_text(
            document.content
        )

        text = self._remove_article_end_matter(
            text
        )

        paragraphs = self._extract_paragraphs(
            text
        )

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