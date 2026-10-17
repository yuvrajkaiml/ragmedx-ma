
"""Recursive text chunking for medical document retrieval."""

import re

from app.config import CHUNK_OVERLAP, CHUNK_SIZE


class RecursiveChunker:
    """Split text into bounded chunks while preserving useful context."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks without exceeding chunk_size."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)

            if end < text_length:
                window = text[start:end]

                # Prefer paragraph, sentence, then word boundaries.
                boundaries = [
                    window.rfind("\n\n"),
                    window.rfind(". "),
                    window.rfind("? "),
                    window.rfind("! "),
                    window.rfind(" "),
                ]

                boundary = next(
                    (position for position in boundaries
                     if position > self.chunk_size // 2),
                    -1,
                )

                if boundary > 0:
                    end = start + boundary + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            # Advance while retaining the configured context overlap.
            next_start = max(start + 1, end - self.chunk_overlap)

            # Avoid starting the next chunk in the middle of a word.
            if next_start < text_length and text[next_start].isalnum():
                next_space = text.find(" ", next_start, end)
                if next_space != -1:
                    next_start = next_space + 1

            start = next_start

        return chunks

    def chunk_text(self, text: str) -> list[str]:
        """Compatibility alias for split_text()."""
        return self.split_text(text)
