"""DocumentLoader — unified ingestion entry-point for the retrieval pipeline.

This class is the single public surface that the retrieval layer uses to pull
documents from disk.  It delegates to the specialised loaders that live in
``app.rag.loaders`` so that format-specific parsing stays in one place.

Supported sources
-----------------
* **MedQA** — JSON/JSONL question-answer files under ``<datasets_dir>/medqa/``
* **PDF**   — any ``.pdf`` file passed to :meth:`load_file`
* **Text**  — any ``.txt`` file passed to :meth:`load_file`

Usage::

    from app.config import DATASETS_DIR
    from app.retrieval.document_loader import DocumentLoader

    loader = DocumentLoader(DATASETS_DIR)

    # Load every bundled dataset that has a recognised sub-directory:
    all_docs = loader.load_all()

    # Load a single arbitrary file:
    docs = loader.load_file(Path("path/to/paper.pdf"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from app.rag.loaders.medqa_loader import MedQALoader
from app.rag.loaders.models import MedicalDocument
from app.rag.loaders.pdf_loader import PDFLoader
from app.rag.loaders.text_loader import TextLoader
from app.utils.logger import logger

# Sub-directory names inside ``datasets_dir`` that are auto-discovered.
_KNOWN_SUBDIRS: dict[str, str] = {
    "medqa": "MedQA",
}


class DocumentLoader:
    """Unified document loader for the RAG retrieval pipeline.

    Parameters
    ----------
    datasets_dir:
        Root directory that contains dataset sub-directories
        (e.g. ``backend/data/datasets``).  The loader auto-discovers any
        recognised sub-directory (``medqa/``) and will load it when
        :meth:`load_all` is called.
    """

    def __init__(self, datasets_dir: Path) -> None:
        self.datasets_dir = Path(datasets_dir)
        logger.info(f"DocumentLoader initialised — datasets_dir={self.datasets_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(self) -> list[MedicalDocument]:
        """Load every known dataset sub-directory and return a flat list.

        Sub-directories that do not exist are silently skipped so that a
        partial datasets installation does not break the whole pipeline.
        """
        documents: list[MedicalDocument] = []

        for subdir, source_name in _KNOWN_SUBDIRS.items():
            path = self.datasets_dir / subdir
            if not path.exists():
                logger.warning(
                    f"Dataset sub-directory not found, skipping: {path}"
                )
                continue
            docs = self._load_subdir(path, source_name)
            logger.info(
                f"Loaded {len(docs)} document(s) from '{subdir}'"
            )
            documents.extend(docs)

        logger.info(f"DocumentLoader.load_all — {len(documents)} total documents")
        return documents

    def load_file(self, path: str | Path) -> list[MedicalDocument]:
        """Load a single PDF or plain-text file.

        Returns a list of :class:`~app.rag.loaders.models.MedicalDocument`
        objects — one per page for PDFs, one for text files.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file extension is not ``.pdf`` or ``.txt``.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()

        if ext == ".pdf":
            return self._load_pdf(path)
        if ext == ".txt":
            return self._load_text(path)

        raise ValueError(
            f"Unsupported file type '{ext}'. "
            "DocumentLoader supports .pdf and .txt files."
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_subdir(
        self, path: Path, source_name: str
    ) -> list[MedicalDocument]:
        """Dispatch to the correct specialised loader for *path*."""
        if source_name == "MedQA":
            return MedQALoader(path).load()
        # Future sources (BioASQ, PubMed, PMC, …) will be dispatched here.
        logger.warning(f"No loader registered for source '{source_name}', skipping.")
        return []

    def _load_pdf(self, path: Path) -> list[MedicalDocument]:
        """Load a PDF and convert each page to a MedicalDocument."""
        raw_pages = PDFLoader().load(path)
        documents: list[MedicalDocument] = []
        for page_doc in raw_pages:
            documents.append(
                MedicalDocument(
                    id=f"{path.stem}_p{page_doc.metadata.page}",
                    title=f"{path.name} — page {page_doc.metadata.page}",
                    content=page_doc.text,
                    source=path.name,
                    metadata={"page": str(page_doc.metadata.page)},
                )
            )
        return documents

    def _load_text(self, path: Path) -> list[MedicalDocument]:
        """Load a plain-text file as a single MedicalDocument."""
        raw = TextLoader().load(path)
        return [
            MedicalDocument(
                id=path.stem,
                title=path.name,
                content=raw[0].text,
                source=path.name,
                metadata={"page": "1"},
            )
        ]
