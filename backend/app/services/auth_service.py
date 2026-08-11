from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.security import create_access_token, create_refresh_token, decode_token, hash_password, verify_password
from app.db.models import User
from app.schemas.auth import TokenResponse, UserLoginRequest, UserRegisterRequest


class AuthService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def register(self, request: UserRegisterRequest) -> User:
        existing = await self.session.scalar(select(User).where(or_(User.email == str(request.email), User.username == request.username)))
        if existing:
            raise ValueError("An account with that email or username already exists")
        user = User(email=str(request.email), username=request.username, password_hash=hash_password(request.password), full_name=request.full_name, role=request.role.value, specialty=request.specialty, institution=request.institution)
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user

    async def login(self, request: UserLoginRequest) -> User:
        user = await self.session.scalar(select(User).where(User.email == str(request.email)))
        if not user or not user.is_active or not verify_password(request.password, user.password_hash):
            raise ValueError("Incorrect email or password")
        return user

    def tokens_for(self, user: User) -> TokenResponse:
        settings = get_settings()
        return TokenResponse(access_token=create_access_token(str(user.id)), refresh_token=create_refresh_token(str(user.id)), expires_in=settings.access_token_expire_minutes * 60, user=user)

    async def refresh(self, refresh_token: str) -> TokenResponse:
        user_id = decode_token(refresh_token, expected_type="refresh")
        user = await self.session.get(User, user_id)
        if not user or not user.is_active:
            raise ValueError("User account is unavailable")
        return self.tokens_for(user)
