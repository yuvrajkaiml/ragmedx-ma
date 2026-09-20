# Owner: Pranav Sukale — not yet implemented, see Task_Assignment.md
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.db import get_session
from app.db.models import User
from app.schemas.query import MedicalQueryRequest, QueryAcceptedResponse, QueryExecutionResponse
from app.services.query_service import QueryService

router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "/ask",
    response_model=QueryAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def ask_medical_question(
    request: MedicalQueryRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    try:
        return await QueryService().start(request, current_user.id, session)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{query_id}", response_model=QueryExecutionResponse)
async def get_query_execution(
    query_id: UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    execution = await QueryService().execution_service.get_persisted(
        query_id,
        current_user.id,
        session,
    )
    if execution is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Query not found")
    return execution
