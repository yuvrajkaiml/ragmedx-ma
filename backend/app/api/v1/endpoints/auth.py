from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.schemas.auth import RefreshTokenRequest, TokenResponse, UserLoginRequest, UserRegisterRequest
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(request: UserRegisterRequest, session: AsyncSession = Depends(get_session)):
    service = AuthService(session)
    try:
        user = await service.register(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return service.tokens_for(user)


@router.post("/login", response_model=TokenResponse)
async def login(request: UserLoginRequest, session: AsyncSession = Depends(get_session)):
    service = AuthService(session)
    try:
        user = await service.login(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    return service.tokens_for(user)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(request: RefreshTokenRequest, session: AsyncSession = Depends(get_session)):
    try:
        return await AuthService(session).refresh(request.refresh_token)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
