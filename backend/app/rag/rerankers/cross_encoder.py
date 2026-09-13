"""Backward-compatible import for the canonical cross-encoder reranker.

The implementation lives in :mod:`cross_encoder_reranker`; this module exists
only for callers that imported the pre-merge path.
"""

from app.rag.rerankers.cross_encoder_reranker import CrossEncoderReranker

__all__ = ["CrossEncoderReranker"]
