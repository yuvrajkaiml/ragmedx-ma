from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.db import get_session
from app.db.models import QueryExecutionRecord, QueryFeedback, User
from app.schemas.feedback import FeedbackCreate, FeedbackResponse

router = APIRouter(prefix="/feedback", tags=["Feedback"])


@router.post("", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
async def submit_query_feedback(
    request: FeedbackCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    execution = await session.get(QueryExecutionRecord, request.query_id)
    if execution is None or execution.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Completed query not found",
        )
    if execution.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Feedback can only be submitted for completed queries",
        )

    feedback = QueryFeedback(
        query_id=request.query_id,
        user_id=current_user.id,
        rating=request.rating,
        comment=request.comment,
    )
    session.add(feedback)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Feedback has already been submitted for this query",
        ) from exc
    await session.refresh(feedback)
    return feedback
