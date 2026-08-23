# Owner: Pranav Sukale — integrated with Yuvraj AI Pipeline via AgentService
from app.schemas.query import MedicalQueryRequest
from app.schemas.response import MedicalResponse
from app.services.agent_service import AgentService


class QueryService:
    """Query service delegating medical QA execution to AgentService."""

    def __init__(self, agent_service: AgentService | None = None):
        self.agent_service = agent_service or AgentService()

    async def ask(self, request: MedicalQueryRequest) -> MedicalResponse:
        response, _ = await self.agent_service.execute_query(
            question=request.question,
            conversation_id=request.conversation_id,
            include_trace=request.include_agent_trace,
        )
        return response