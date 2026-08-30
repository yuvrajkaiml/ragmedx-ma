"""CrossEncoderReranker re-export for the retrieval boundary.

The canonical implementation lives in ``app.rag.rerankers.cross_encoder``.
This module re-exports it so retrieval-layer code can import from one stable
location without depending directly on the rag internals package.

Usage::

    from app.retrieval.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query, first_stage_results, top_k=5)
"""

from app.rag.rerankers.cross_encoder import CrossEncoderReranker  # noqa: F401

__all__ = ["CrossEncoderReranker"]
