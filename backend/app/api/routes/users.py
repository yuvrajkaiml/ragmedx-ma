from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.db import get_session
from app.db.models import User
from app.schemas.user import UserProfileUpdate, UserResponse

router = APIRouter(prefix="/api/v1/users")


@router.get("/me", response_model=UserResponse)
async def get_my_profile(
    current_user: User = Depends(get_current_user),
):
    return current_user


@router.patch("/me", response_model=UserResponse)
async def update_my_profile(
    request: UserProfileUpdate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    for field, value in request.model_dump(exclude_unset=True).items():
        setattr(current_user, field, value)

    await session.commit()
    await session.refresh(current_user)

    return current_user