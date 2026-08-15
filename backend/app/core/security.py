from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import get_settings

password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return password_context.hash(password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    return password_context.verify(plain_password, password_hash)


def create_token(subject: str, expires_delta: timedelta, token_type: str) -> str:
    settings = get_settings()
    payload: dict[str, Any] = {"sub": subject, "type": token_type, "exp": datetime.now(timezone.utc) + expires_delta}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def create_access_token(subject: str) -> str:
    return create_token(subject, timedelta(minutes=get_settings().access_token_expire_minutes), "access")


def create_refresh_token(subject: str) -> str:
    return create_token(subject, timedelta(days=get_settings().refresh_token_expire_days), "refresh")


def decode_token(token: str, expected_type: str = "access") -> str:
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise ValueError("Invalid or expired token") from exc
    if payload.get("type") != expected_type or not isinstance(payload.get("sub"), str):
        raise ValueError("Invalid token type")
    return payload["sub"]
