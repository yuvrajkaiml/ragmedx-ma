# Owner: Pranav Sukale
# Query service: API-facing service layer for the medical QA pipeline.
# Integrates with the AI pipeline through AgentService.

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.query import MedicalQueryRequest, QueryAcceptedResponse
from app.services.query_execution_service import QueryExecutionService, query_execution_service


class QueryService:
    """API service that starts one observable medical query execution."""

    def __init__(self, execution_service: QueryExecutionService | None = None):
        self.execution_service = execution_service or query_execution_service

    async def start(
        self,
        request: MedicalQueryRequest,
        user_id: UUID,
        session: AsyncSession,
    ) -> QueryAcceptedResponse:
        """Schedule the pipeline once and return its subscription identifier."""
        return await self.execution_service.start(request, user_id, session)
