from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import get_settings

password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return password_context.hash(password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    return password_context.verify(plain_password, password_hash)


def create_token(
    subject: str,
    expires_delta: timedelta,
    token_type: str,
    auth_version: int = 0,
) -> str:
    settings = get_settings()
    payload: dict[str, Any] = {
        "sub": subject,
        "type": token_type,
        "exp": datetime.now(timezone.utc) + expires_delta,
        "ver": auth_version,
        "jti": uuid4().hex,
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def create_access_token(subject: str, auth_version: int = 0) -> str:
    return create_token(
        subject,
        timedelta(minutes=get_settings().access_token_expire_minutes),
        "access",
        auth_version,
    )


def create_refresh_token(subject: str, auth_version: int = 0) -> str:
    return create_token(
        subject,
        timedelta(days=get_settings().refresh_token_expire_days),
        "refresh",
        auth_version,
    )


def create_email_verification_token(subject: str) -> str:
    return create_token(subject, timedelta(hours=24), "email_verification")


def create_password_reset_token(subject: str) -> str:
    return create_token(subject, timedelta(minutes=30), "password_reset")


def decode_token(token: str, expected_type: str = "access") -> str:
    return decode_token_payload(token, expected_type)["sub"]


def decode_token_payload(token: str, expected_type: str = "access") -> dict[str, Any]:
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise ValueError("Invalid or expired token") from exc
    if payload.get("type") != expected_type or not isinstance(payload.get("sub"), str):
        raise ValueError("Invalid token type")
    if isinstance(payload.get("exp"), bool) or not isinstance(payload.get("exp"), (int, float)):
        raise ValueError("Invalid token expiration")
    return payload
