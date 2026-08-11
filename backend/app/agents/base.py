import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class BaseAgent(ABC):
    """
    Abstract Base Class for all RAGMedX-MA Agents.
    Provides standard lifecycle hooks, logging, and trace generation capabilities.
    """

    def __init__(self, agent_name: str, agent_type: str, llm_wrapper: Optional[Any] = None):
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.llm_wrapper = llm_wrapper

    @abstractmethod
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent logic given the current pipeline state.

        Args:
            state: The current LangGraph pipeline state dictionary.

        Returns:
            Dict containing state updates to be merged into the pipeline state.
        """
        pass

    def create_trace_step(
        self,
        status: str,
        started_at: datetime,
        completed_at: Optional[datetime] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Helper method to generate an AgentTraceStep dictionary conforming to Pydantic schemas.

        Args:
            status: Execution status ('running', 'completed', 'failed').
            started_at: Datetime when step started.
            completed_at: Datetime when step ended.
            details: Optional dictionary containing agent execution metadata.
        """
        completed = completed_at or datetime.now(timezone.utc)
        duration_ms = int((completed - started_at).total_seconds() * 1000)

        return {
            "agent_type": self.agent_type,
            "status": status,
            "started_at": started_at.isoformat(),
            "completed_at": completed.isoformat(),
            "duration_ms": duration_ms,
            "details": details or {}
        }
