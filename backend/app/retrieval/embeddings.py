"""Embedding adapter re-export for the retrieval boundary.

The canonical implementation lives in ``app.rag.embeddings.bge_embedding``.
This module re-exports it so retrieval-layer code can import from one stable
location without depending directly on the rag internals package.

Usage::

    from app.retrieval.embeddings import BGEEmbedder
    embedder = BGEEmbedder()               # alias of native BGE-M3 adapter
    embedder = BGEEmbedder("BAAI/bge-m3", device="cuda")
"""

from app.rag.embeddings.bge_embedding import BGEEmbedder  # noqa: F401

__all__ = ["BGEEmbedder"]
