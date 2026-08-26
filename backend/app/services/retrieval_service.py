"""Retrieval service — constructs and exposes the RAG pipeline.

This service is the single place that decides which retrieval provider
the orchestrator uses.  It can be injected into ``AgentService`` or
``MultiAgentOrchestrator`` as the ``retrieval_provider`` argument.

Usage::

    from app.services.retrieval_service import RetrievalService

    # In production (requires a built FAISS index):
    service = RetrievalService.build()
    orchestrator = MultiAgentOrchestrator(
        retrieval_provider=service.retrieve_evidence,
    )

    # Fallback (no index yet -- returns empty evidence, agents abstain):
    service = RetrievalService()   # pipeline.is_ready == False
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.rag.pipeline import RAGPipeline
from app.utils.logger import logger


class RetrievalService:
    """Thin service wrapper around ``RAGPipeline``.

    Parameters
    ----------
    pipeline:
        A pre-built :class:`~app.rag.pipeline.RAGPipeline`.  When ``None``
        (default), a degraded pipeline is created that returns empty results.
    """

    def __init__(self, pipeline: Optional[RAGPipeline] = None) -> None:
        self._pipeline = pipeline or RAGPipeline()
        logger.info(
            f"RetrievalService ready -- pipeline.is_ready={self._pipeline.is_ready}"
        )

    @classmethod
    def build(
        cls,
        index_dir: Optional[Path] = None,
        model_name: Optional[str] = None,
        top_k_retrieval: int = 20,
        top_k_final: int = 5,
    ) -> "RetrievalService":
        """Build a fully wired retrieval service from a persisted FAISS index.

        Delegates to :meth:`RAGPipeline.build`.  Gracefully degrades if no
        index exists yet.
        """
        pipeline = RAGPipeline.build(
            index_dir=index_dir,
            model_name=model_name,
            top_k_retrieval=top_k_retrieval,
            top_k_final=top_k_final,
        )
        return cls(pipeline=pipeline)

    async def retrieve_evidence(
        self,
        query: str,
        optimized_query: Optional[str] = None,
        max_iterations: int = 3,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Delegate to the underlying pipeline.  Drop-in for the stub."""
        return await self._pipeline.retrieve_evidence(
            query=query,
            optimized_query=optimized_query,
            max_iterations=max_iterations,
            **kwargs,
        )

    @property
    def pipeline(self) -> RAGPipeline:
        """The underlying RAGPipeline instance."""
        return self._pipeline

    @property
    def is_ready(self) -> bool:
        """True when the pipeline has a loaded retriever."""
        return self._pipeline.is_ready
