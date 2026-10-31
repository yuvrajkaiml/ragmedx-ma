import hashlib
from datetime import datetime, timezone

from sqlalchemy import delete, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.security import (
    create_access_token,
    create_email_verification_token,
    create_password_reset_token,
    create_refresh_token,
    decode_token_payload,
    hash_password,
    verify_password,
)
from app.db.models import ConsumedAuthToken, RevokedRefreshToken, User
from app.schemas.auth import TokenResponse, UserLoginRequest, UserRegisterRequest
from app.services.email_service import send_auth_email


class AuthService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def register(self, request: UserRegisterRequest) -> User:
        existing = await self.session.scalar(select(User).where(or_(User.email == str(request.email), User.username == request.username)))
        if existing:
            raise ValueError("An account with that email or username already exists")
        settings = get_settings()
        user = User(
            email=str(request.email),
            username=request.username,
            password_hash=hash_password(request.password),
            full_name=request.full_name,
            role=request.role.value,
            specialty=request.specialty,
            institution=request.institution,
            is_verified=not settings.require_email_verification,
        )
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user

    async def login(self, request: UserLoginRequest) -> User:
        user = await self.session.scalar(select(User).where(User.email == str(request.email)))
        if not user or not user.is_active or not verify_password(request.password, user.password_hash):
            raise ValueError("Incorrect email or password")
        if get_settings().require_email_verification and not user.is_verified:
            raise ValueError("Email address has not been verified")
        return user

    def tokens_for(self, user: User) -> TokenResponse:
        settings = get_settings()
        auth_version = getattr(user, "auth_version", 0)
        return TokenResponse(
            access_token=create_access_token(str(user.id), auth_version),
            refresh_token=create_refresh_token(str(user.id), auth_version),
            expires_in=settings.access_token_expire_minutes * 60,
            user=user,
        )

    async def refresh(self, refresh_token: str) -> TokenResponse:
        payload = decode_token_payload(refresh_token, expected_type="refresh")
        token_hash = self._token_hash(refresh_token)
        if await self.session.get(RevokedRefreshToken, token_hash):
            raise ValueError("Refresh token has been revoked")

        user_id = payload["sub"]
        user = await self.session.get(User, user_id)
        if not user or not user.is_active:
            raise ValueError("User account is unavailable")
        if payload.get("ver", 0) != getattr(user, "auth_version", 0):
            raise ValueError("Refresh token has been revoked")

        await self._revoke_refresh_token(token_hash, payload["exp"])
        return self.tokens_for(user)

    async def logout(self, refresh_token: str) -> None:
        payload = decode_token_payload(refresh_token, expected_type="refresh")
        await self._revoke_refresh_token(
            self._token_hash(refresh_token),
            payload["exp"],
        )

    async def send_verification_email(self, user: User) -> None:
        if user.is_verified:
            return
        token = create_email_verification_token(str(user.id))
        await send_auth_email(
            user.email,
            "Verify your RAGMedX account",
            f"Use this verification token within 24 hours:\n\n{token}\n",
        )

    async def request_email_verification(self, email: str) -> None:
        user = await self.session.scalar(select(User).where(User.email == email))
        if user and user.is_active and not user.is_verified:
            await self.send_verification_email(user)

    async def verify_email(self, token: str) -> None:
        payload = decode_token_payload(token, expected_type="email_verification")
        user = await self.session.get(User, payload["sub"])
        if not user or not user.is_active:
            raise ValueError("User account is unavailable")
        user.is_verified = True
        await self._consume_auth_token(token, payload["exp"])

    async def request_password_reset(self, email: str) -> None:
        user = await self.session.scalar(select(User).where(User.email == email))
        if user and user.is_active:
            token = create_password_reset_token(str(user.id))
            await send_auth_email(
                user.email,
                "Reset your RAGMedX password",
                f"Use this password reset token within 30 minutes:\n\n{token}\n",
            )

    async def reset_password(self, token: str, new_password: str) -> None:
        payload = decode_token_payload(token, expected_type="password_reset")
        if await self.session.get(ConsumedAuthToken, self._token_hash(token)):
            raise ValueError("Token has already been used")
        user = await self.session.get(User, payload["sub"])
        if not user or not user.is_active:
            raise ValueError("User account is unavailable")
        user.password_hash = hash_password(new_password)
        user.auth_version = getattr(user, "auth_version", 0) + 1
        await self._consume_auth_token(token, payload["exp"])

    async def _consume_auth_token(self, token: str, expires_at: int | float) -> None:
        token_hash = self._token_hash(token)
        if await self.session.get(ConsumedAuthToken, token_hash):
            raise ValueError("Token has already been used")

        now = datetime.now(timezone.utc)
        await self.session.execute(
            delete(ConsumedAuthToken).where(ConsumedAuthToken.expires_at <= now)
        )
        self.session.add(
            ConsumedAuthToken(
                token_hash=token_hash,
                expires_at=datetime.fromtimestamp(expires_at, tz=timezone.utc),
            )
        )
        try:
            await self.session.commit()
        except IntegrityError as exc:
            await self.session.rollback()
            raise ValueError("Token has already been used") from exc

    async def _revoke_refresh_token(self, token_hash: str, expires_at: int | float) -> None:
        existing = await self.session.get(RevokedRefreshToken, token_hash)
        if existing:
            return

        now = datetime.now(timezone.utc)
        await self.session.execute(
            delete(RevokedRefreshToken).where(RevokedRefreshToken.expires_at <= now)
        )
        self.session.add(
            RevokedRefreshToken(
                token_hash=token_hash,
                expires_at=datetime.fromtimestamp(expires_at, tz=timezone.utc),
            )
        )
        try:
            await self.session.commit()
        except IntegrityError as exc:
            await self.session.rollback()
            raise ValueError("Refresh token has been revoked") from exc

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()
