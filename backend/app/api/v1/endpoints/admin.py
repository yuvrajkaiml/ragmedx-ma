# Owner: Pranav Sukale — not yet implemented, see Task_Assignment.md
from datetime import datetime, timezone
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_admin
from app.db import get_session
from app.db.models import Conversation, ConversationMessage, User
from app.schemas.admin import (
    AdminMetricsResponse,
    AdminUserStatusUpdate,
    AdminUserSummary,
)

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/metrics", response_model=AdminMetricsResponse)
async def get_metrics(
    _: User = Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    total_users = await session.scalar(select(func.count()).select_from(User))
    active_users = await session.scalar(
        select(func.count()).select_from(User).where(User.is_active.is_(True))
    )
    total_conversations = await session.scalar(
        select(func.count()).select_from(Conversation)
    )
    total_messages = await session.scalar(
        select(func.count()).select_from(ConversationMessage)
    )

    return AdminMetricsResponse(
        total_users=total_users or 0,
        active_users=active_users or 0,
        total_conversations=total_conversations or 0,
        total_messages=total_messages or 0,
        generated_at=datetime.now(timezone.utc),
    )

@router.get("/users", response_model=list[AdminUserSummary])
async def list_users(
    _: User = Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    users = (
        await session.execute(
            select(User).order_by(User.created_at.desc())
        )
    ).scalars().all()

    return [
        AdminUserSummary(
            id=user.id,
            email=user.email,
            username=user.username,
            role=user.role,
            is_active=user.is_active,
            created_at=user.created_at,
        )
        for user in users
    ]

@router.patch(
    "/users/{user_id}/status",
    response_model=AdminUserSummary,
)
async def update_user_status(
    user_id: UUID,
    request: AdminUserStatusUpdate,
    current_admin: User = Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
):
    user = await session.get(User, user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    if user.id == current_admin.id and not request.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot deactivate your own account",
        )

    user.is_active = request.is_active
    await session.commit()
    await session.refresh(user)

    return AdminUserSummary(
        id=user.id,
        email=user.email,
        username=user.username,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
    )