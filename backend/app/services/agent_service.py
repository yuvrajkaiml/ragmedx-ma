"""
Agent service boundary coordinating multi-agent medical workflows.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import UUID

from app.agents import run_agent_pipeline
from app.schemas.response import AgentTraceOutput, MedicalResponse


class AgentService:
    """Service wrapper for executing the multi-agent QA pipeline."""

    def __init__(
        self,
        llm_wrapper: Optional[Any] = None,
        retrieval_provider: Optional[Callable] = None,
    ):
        self.llm_wrapper = llm_wrapper
        self.retrieval_provider = retrieval_provider

    async def execute_query(
        self,
        question: str,
        query_id: Optional[Union[UUID, str]] = None,
        conversation_id: Optional[Union[UUID, str]] = None,
        include_trace: bool = False,
        processed_query: Optional[Dict[str, Any]] = None,
        evidence: Optional[List[Dict[str, Any]]] = None,
        retrieval_provider: Optional[Callable] = None,
    ) -> Tuple[MedicalResponse, Optional[AgentTraceOutput]]:
        """
        Execute the multi-agent question-answering workflow.

        A per-call ``retrieval_provider`` overrides the service default, which
        lets a single query be scoped to an uploaded document's isolated index
        without disturbing the shared knowledge-base provider.
        """
        return await run_agent_pipeline(
            original_question=question,
            query_id=str(query_id) if query_id else None,
            conversation_id=str(conversation_id) if conversation_id else None,
            processed_query=processed_query,
            evidence=evidence,
            include_trace=include_trace,
            llm_wrapper=self.llm_wrapper,
            retrieval_provider=retrieval_provider or self.retrieval_provider,
        )
