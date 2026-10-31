# Owner: Pranav Sukale
"""Per-document ingestion + retrieval for user-uploaded PDFs.

Each uploaded PDF is turned into its own isolated FAISS index so that user
documents never pollute the shared biomedical knowledge base and can be
deleted cleanly. The pipeline mirrors ``scripts/build_index.py`` but is scoped
to a single document:

    PDF file
      -> PDFLoader            (page text)
      -> MedicalDocument      (one per page)
      -> MedicalChunker       (retrieval-ready chunks)
      -> BGEEmbedding         (BGE-M3, 1024-dim, L2-normalised)
      -> FAISSVectorStore     (IndexFlatIP)
      -> data/uploads/{document_id}/index.faiss + chunks.pkl

The BGE-M3 embedder is expensive to load (~2 GB on CPU), so a single instance
is shared across all uploads via :func:`get_shared_embedder`. In production the
same embedder already loaded by ``RetrievalService`` can be injected instead.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

from app.config import get_settings
from app.rag.chunking.models import MedicalChunk
from app.rag.chunking.semantic_chunker import MedicalChunker
from app.rag.embeddings.bge_embedding import BGEEmbedding
from app.rag.loaders.models import MedicalDocument
from app.rag.loaders.pdf_loader import PDFLoader
from app.rag.vectorstore.faiss_store import FAISSVectorStore
from app.utils.logger import logger

_EMBED_DIM = 1024  # BGE-M3 output dimension — must match FAISSVectorStore.

# Process-wide singleton embedder guarded by a lock so concurrent uploads do
# not each trigger a multi-GB model load.
_shared_embedder: Optional[BGEEmbedding] = None
_embedder_lock = threading.Lock()


def get_shared_embedder(embedder: Optional[BGEEmbedding] = None) -> BGEEmbedding:
    """Return a shared BGE-M3 embedder, creating one on first use.

    Passing an explicit ``embedder`` (e.g. the one already loaded by
    ``RetrievalService``) adopts it as the shared instance so no second copy is
    loaded into memory.
    """
    global _shared_embedder
    if embedder is not None:
        with _embedder_lock:
            _shared_embedder = embedder
        return embedder
    if _shared_embedder is None:
        with _embedder_lock:
            if _shared_embedder is None:
                logger.info("Loading shared BGE-M3 embedder for uploaded-document indexing …")
                _shared_embedder = BGEEmbedding(model_name=BGEEmbedding.DEFAULT_MODEL)
    return _shared_embedder


class UploadedDocumentIndexService:
    """Build and query an isolated FAISS index for a single uploaded PDF."""

    def __init__(
        self,
        uploads_dir: Optional[Path] = None,
        embedder: Optional[BGEEmbedding] = None,
    ) -> None:
        settings = get_settings()
        base = uploads_dir or settings.upload_data_dir
        base = Path(base)
        if not base.is_absolute():
            from app.config import BASE_DIR

            base = BASE_DIR / base
        self.uploads_dir = base
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        # Do not force-load the embedder at construction; only when indexing.
        self._injected_embedder = embedder

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def index_dir(self, document_id: UUID | str) -> Path:
        """Directory that holds this document's isolated FAISS artifacts."""
        return self.uploads_dir / str(document_id)

    def pdf_path(self, document_id: UUID | str) -> Path:
        return self.uploads_dir / f"{document_id}.pdf"

    def is_indexed(self, document_id: UUID | str) -> bool:
        directory = self.index_dir(document_id)
        return (directory / "index.faiss").is_file() and (directory / "chunks.pkl").is_file()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_index(self, document_id: UUID | str, filename: str | None = None) -> int:
        """Parse -> chunk -> embed -> persist an isolated FAISS index.

        Returns
        -------
        int
            The number of chunks indexed.

        Raises
        ------
        FileNotFoundError
            If the source PDF is missing.
        ValueError
            If the PDF yields no extractable text or no chunks.
        """
        pdf_path = self.pdf_path(document_id)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Uploaded PDF not found: {pdf_path}")

        # 1. Parse PDF into one MedicalDocument per page.
        pages = PDFLoader().load(pdf_path)
        if not pages:
            raise ValueError("The PDF contains no extractable text.")

        source_name = filename or pdf_path.name
        documents = [
            MedicalDocument(
                id=f"{document_id}_p{page.metadata.page}",
                title=f"{source_name} — page {page.metadata.page}",
                content=page.text,
                source="uploaded_document",
                metadata={
                    "upload_document_id": str(document_id),
                    "filename": source_name,
                    "page": str(page.metadata.page),
                },
            )
            for page in pages
        ]

        # 2. Chunk.
        chunker = MedicalChunker.from_settings()
        chunks: list[MedicalChunk] = chunker.chunk_documents(documents)
        if not chunks:
            raise ValueError(
                "The PDF produced no chunks (text may be too short after cleaning)."
            )

        # 3. Embed.
        embedder = get_shared_embedder(self._injected_embedder)
        texts = [chunk.content for chunk in chunks]
        embeddings = embedder.embed_documents(texts)

        # 4. Persist an isolated store.
        store = FAISSVectorStore(dimension=_EMBED_DIM)
        store.add(embeddings, chunks)
        directory = self.index_dir(document_id)
        store.save(directory)

        logger.info(
            f"Uploaded document {document_id} indexed — "
            f"{store.size} chunks saved to {directory}"
        )
        return store.size

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        document_id: UUID | str,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[MedicalChunk, float]]:
        """Search a single document's index. Returns [] if not indexed."""
        if not self.is_indexed(document_id):
            return []
        store = FAISSVectorStore.load(self.index_dir(document_id))
        embedder = get_shared_embedder(self._injected_embedder)
        query_embedding = embedder.embed_query(query)
        return store.search(query_embedding=query_embedding, top_k=top_k)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def evidence_provider(self, document_id: UUID | str):
        """Return an async ``retrieve_evidence`` callable scoped to one document.

        The returned callable matches the signature the orchestrator expects
        (``query``, ``optimized_query``, ``max_iterations``, ``**kwargs``) and
        yields evidence dicts in the same schema as ``RAGPipeline`` so the agent
        pipeline treats uploaded-document evidence identically to KB evidence.
        """

        async def _retrieve_evidence(
            query: str,
            optimized_query: Optional[str] = None,
            max_iterations: int = 3,
            top_k_final: Optional[int] = None,
            **_: Any,
        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            search_query = (optimized_query or query or "").strip()
            if not search_query:
                return [], [{"iteration": 1, "query_used": "", "count": 0, "status": "empty_query"}]
            if not self.is_indexed(document_id):
                return [], [{
                    "iteration": 1,
                    "query_used": search_query,
                    "count": 0,
                    "status": "document_not_indexed",
                }]
            results = self.retrieve(document_id, search_query, top_k=top_k_final or 5)
            evidence = []
            for index, (chunk, score) in enumerate(results, start=1):
                # BGE-M3 embeddings are L2-normalised, so the FAISS inner-product
                # score is cosine similarity in [-1, 1]; clamp to [0, 1] for the
                # agent contract's relevance range.
                relevance = max(0.0, min(1.0, float(score)))
                evidence.append({
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "source_type": "uploaded_document",
                    "title": chunk.title or "",
                    "cited_text": chunk.content,
                    "relevance_score": relevance,
                    "cross_encoder_score": relevance,
                    "evidence_quality_score": relevance,
                    "citation_marker": f"[Doc {index}]",
                    "journal": chunk.journal,
                    "publication_year": chunk.publication_year,
                    "doi": (chunk.metadata or {}).get("doi"),
                    "url": (chunk.metadata or {}).get("url"),
                })
            return evidence, [{
                "iteration": 1,
                "query_used": search_query,
                "count": len(evidence),
                "status": "completed",
                "scope": "uploaded_document",
            }]

        return _retrieve_evidence

    def delete_index(self, document_id: UUID | str) -> None:
        """Remove the isolated index directory for a document, if present."""
        directory = self.index_dir(document_id)
        if directory.exists():
            for child in directory.iterdir():
                child.unlink(missing_ok=True)
            directory.rmdir()
            logger.info(f"Deleted uploaded-document index: {directory}")
