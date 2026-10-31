from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.schemas.auth import (
    EmailActionRequest,
    EmailTokenRequest,
    PasswordResetRequest,
    RefreshTokenRequest,
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
)
from app.services.auth_service import AuthService
from app.services.email_service import EmailDeliveryError, is_email_delivery_configured

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(request: UserRegisterRequest, session: AsyncSession = Depends(get_session)):
    service = AuthService(session)
    try:
        user = await service.register(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if get_settings().require_email_verification:
        try:
            await service.send_verification_email(user)
        except EmailDeliveryError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Email verification delivery is unavailable",
            ) from exc
    return service.tokens_for(user)
@router.post("/verification-email", status_code=status.HTTP_202_ACCEPTED)
async def request_verification_email(
    request: EmailActionRequest,
    session: AsyncSession = Depends(get_session),
):
    if not is_email_delivery_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Email delivery is not configured",
        )
    try:
        await AuthService(session).request_email_verification(str(request.email))
    except EmailDeliveryError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Email delivery is unavailable",
        ) from exc
    return {"message": "If the account needs verification, an email has been sent"}

@router.post("/verify-email")
async def verify_email(
    request: EmailTokenRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        await AuthService(session).verify_email(request.token)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {"message": "Email address verified"}

@router.post("/password-reset/request", status_code=status.HTTP_202_ACCEPTED)
async def request_password_reset(
    request: EmailActionRequest,
    session: AsyncSession = Depends(get_session),
):
    if not is_email_delivery_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Email delivery is not configured",
        )
    try:
        await AuthService(session).request_password_reset(str(request.email))
    except EmailDeliveryError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Email delivery is unavailable",
        ) from exc
    return {"message": "If the account exists, a password reset email has been sent"}

@router.post("/password-reset/confirm")
async def confirm_password_reset(
    request: PasswordResetRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        await AuthService(session).reset_password(request.token, request.new_password)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {"message": "Password has been reset"}


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


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(request: RefreshTokenRequest, session: AsyncSession = Depends(get_session)):
    try:
        await AuthService(session).logout(request.refresh_token)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
