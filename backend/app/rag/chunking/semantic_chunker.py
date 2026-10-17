import re
from typing import List, Tuple

from app.rag.loaders.models import MedicalDocument
from app.rag.chunking.models import MedicalChunk
from app.utils.logger import logger


class MedicalChunker:
    """Split medical documents into retrieval-ready text chunks."""

    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 400,
        min_chunk_size: int = 150,
        overlap: int | None = None,
    ):
        # Preserve compatibility with callers using `overlap`.
        if overlap is not None:
            chunk_overlap = overlap

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if min_chunk_size < 0:
            raise ValueError("min_chunk_size cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        logger.info(
            "Medical Chunker initialized | "
            f"chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}, "
            f"min_chunk_size={min_chunk_size}"
        )

    def _clean_text(self, text: str) -> str:
        """Normalize whitespace and remove common article markers."""
        if not text:
            return ""

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(
            r"^\s*(TEXT|ARTICLE)\s*$",
            "",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        return text.strip()

    def _remove_article_end_matter(self, text: str) -> str:
        """Remove references and common publication end matter."""
        patterns = [
            r"\n\s*References\s*\n",
            r"\n\s*Conflict[- ]of[- ]interest statement\s*:",
            r"\n\s*Conflict of interest\s*:",
            r"\n\s*Manuscript source\s*:",
            r"\n\s*Peer[- ]review report classification\s*",
        ]

        earliest_index = len(text)

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                earliest_index = min(earliest_index, match.start())

        return text[:earliest_index].strip()

    def _looks_like_heading(self, paragraph: str) -> bool:
        """Identify likely standalone medical section headings."""
        paragraph = paragraph.strip()

        if not paragraph or len(paragraph) > 120:
            return False

        if paragraph.endswith((".", "!", ";", ",")):
            return False

        known_headings = {
            "abstract",
            "introduction",
            "background",
            "methods",
            "materials and methods",
            "methodology",
            "results",
            "discussion",
            "conclusion",
            "conclusions",
            "limitations",
            "case presentation",
            "case report",
            "clinical presentation",
            "treatment",
            "diagnosis",
            "management",
            "references",
            "acknowledgments",
            "acknowledgements",
            "data availability",
            "ethical approval",
            "patient consent",
        }

        normalized = re.sub(r"\s+", " ", paragraph).strip().lower()
        normalized = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", normalized)

        if normalized in known_headings:
            return True

        # Permit numbered or short question-style headings.
        if paragraph.endswith("?") and len(paragraph.split()) <= 12:
            return True

        return False

    def _extract_paragraphs(self, text: str) -> List[str]:
        """Extract non-empty paragraphs without losing heading boundaries."""
        text = self._clean_text(text)
        paragraphs = re.split(r"\n\s*\n", text)
        result = []

        for paragraph in paragraphs:
            paragraph = re.sub(r"\s+", " ", paragraph).strip()
            if paragraph:
                result.append(paragraph)

        return result

    def _split_long_text(self, text: str) -> List[str]:
        """Split long text into bounded chunks with word-safe overlap."""
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)

            if end < text_length:
                # Prefer a sentence boundary, then a word boundary.
                search_start = start + max(1, self.chunk_size // 2)
                sentence_boundary = max(
                    text.rfind(". ", search_start, end),
                    text.rfind("? ", search_start, end),
                    text.rfind("! ", search_start, end),
                )

                if sentence_boundary >= search_start:
                    end = sentence_boundary + 1
                else:
                    space_boundary = text.rfind(" ", search_start, end)
                    if space_boundary > start:
                        end = space_boundary

            piece = text[start:end].strip()

            if piece:
                chunks.append(piece)

            if end >= text_length:
                break

            # Move backward from the end to retain contextual overlap.
            next_start = max(start + 1, end - self.chunk_overlap)

            # Start on a word boundary where possible.
            while next_start < end and text[next_start].isspace():
                next_start += 1

            if (
                next_start > 0
                and next_start < text_length
                and not text[next_start - 1].isspace()
            ):
                next_space = text.find(" ", next_start, end)
                if next_space != -1:
                    next_start = next_space + 1

            # Guarantee forward progress.
            if next_start <= start:
                next_start = end

            start = next_start

        # Defensive bound check.
        for chunk in chunks:
            if len(chunk) > self.chunk_size:
                raise RuntimeError(
                    "Chunk size limit exceeded: "
                    f"{len(chunk)} > {self.chunk_size}"
                )

        return chunks

    def _create_chunks(
        self,
        paragraphs: List[str],
    ) -> List[Tuple[str, str]]:
        """Create section-aware chunks and split oversized paragraphs."""
        chunks: List[Tuple[str, str]] = []
        current_section = "Unknown"
        current_parts: List[str] = []
        current_length = 0

        def flush():
            nonlocal current_parts, current_length

            if not current_parts:
                return

            content = " ".join(current_parts).strip()

            if content:
                for piece in self._split_long_text(content):
                    if len(piece) >= self.min_chunk_size or not chunks:
                        chunks.append((current_section, piece))

            current_parts = []
            current_length = 0

        for paragraph in paragraphs:
            if self._looks_like_heading(paragraph):
                flush()
                current_section = paragraph
                continue

            # Split an individual oversized paragraph before grouping.
            if len(paragraph) > self.chunk_size:
                flush()
                for piece in self._split_long_text(paragraph):
                    if len(piece) >= self.min_chunk_size or not chunks:
                        chunks.append((current_section, piece))
                continue

            separator_length = 1 if current_parts else 0
            proposed_length = current_length + separator_length + len(paragraph)

            if current_parts and proposed_length > self.chunk_size:
                previous_content = " ".join(current_parts).strip()
                flush()

                # Carry a bounded tail into the next chunk for context.
                if self.chunk_overlap and previous_content:
                    tail = previous_content[-self.chunk_overlap:].strip()

                    # Avoid starting overlap in the middle of a word.
                    first_space = tail.find(" ")
                    if first_space != -1:
                        tail = tail[first_space + 1:]

                    if tail and len(tail) < self.chunk_size:
                        current_parts = [tail]
                        current_length = len(tail)

            separator_length = 1 if current_parts else 0
            if current_length + separator_length + len(paragraph) <= self.chunk_size:
                current_parts.append(paragraph)
                current_length += separator_length + len(paragraph)
            else:
                flush()
                current_parts = [paragraph]
                current_length = len(paragraph)

        flush()
        return chunks

    def chunk_document(self, document: MedicalDocument) -> List[MedicalChunk]:
        """Convert one MedicalDocument into retrieval-ready chunks."""
        logger.info(f"Chunking document: {document.id}")

        text = self._clean_text(document.content)

        # Remove a duplicated leading title when the source includes it.
        title = (document.title or "").strip()
        if title and text[:len(title)].casefold() == title.casefold():
            text = text[len(title):].lstrip(" \n:-")

        text = self._remove_article_end_matter(text)
        paragraphs = self._extract_paragraphs(text)
        raw_chunks = self._create_chunks(paragraphs)

        chunks = []
        total_chunks = len(raw_chunks)

        for index, (section, content) in enumerate(raw_chunks):
            chunk = MedicalChunk(
                id=f"{document.id}_chunk_{index}",
                document_id=document.id,
                content=content,
                title=document.title,
                section=section,
                source=document.source,
                journal=document.journal,
                publication_year=document.publication_year,
                metadata={
                    **(document.metadata or {}),
                    "chunk_index": index,
                    "total_chunks": total_chunks,
                },
            )
            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks from {document.id}")
        return chunks

    def chunk_documents(
        self,
        documents: List[MedicalDocument],
    ) -> List[MedicalChunk]:
        """Chunk multiple medical documents."""
        all_chunks = []

        logger.info(f"Starting chunking for {len(documents)} document(s)")

        for document in documents:
            try:
                all_chunks.extend(self.chunk_document(document))
            except Exception as exc:
                logger.error(f"Failed to chunk {document.id}: {exc}")

        logger.success(f"Created {len(all_chunks)} total medical chunks.")
        return all_chunks
