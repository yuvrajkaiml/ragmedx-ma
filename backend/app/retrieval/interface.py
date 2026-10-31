"""Backwards-compatibility re-export of the RAG retrieval interfaces.

The canonical definitions have moved to ``app.rag.interfaces``.
Import from there for new code.  This shim exists so that any code that still
references ``app.retrieval.interface`` continues to work without changes.
"""

# Re-export unchanged — do not add logic here.
from app.rag.interfaces import Embedder, VectorStore  # noqa: F401

__all__ = ["Embedder", "VectorStore"]
