from types import SimpleNamespace
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.exc import IntegrityError

from app.core.security import (
    create_email_verification_token,
    create_password_reset_token,
    create_refresh_token,
    hash_password,
    verify_password,
)
from app.db import get_session
from app.db.models import ConsumedAuthToken, RevokedRefreshToken, User
from app.main import app
from app.services.auth_service import AuthService


class FakeAuthSession:
    def __init__(self, user=None):
        self.user = user
        self.revoked_tokens = {}
        self.consumed_tokens = {}
        self.commit_error = None
        self.rollback_called = False

    async def get(self, model, key):
        if model is RevokedRefreshToken:
            return self.revoked_tokens.get(key)
        if model is ConsumedAuthToken:
            return self.consumed_tokens.get(key)
        if model is User and self.user is not None and str(self.user.id) == str(key):
            return self.user
        return None

    async def execute(self, statement):
        return None

    def add(self, record):
        if isinstance(record, ConsumedAuthToken):
            self.consumed_tokens[record.token_hash] = record
        else:
            self.revoked_tokens[record.token_hash] = record

    async def commit(self):
        if self.commit_error is not None:
            raise self.commit_error
        return None

    async def rollback(self):
        self.rollback_called = True


async def test_logout_endpoint_revokes_refresh_token():
    user_id = uuid4()
    refresh_token = create_refresh_token(str(user_id))
    session = FakeAuthSession()

    async def fake_session_dependency():
        yield session

    app.dependency_overrides[get_session] = fake_session_dependency
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/auth/logout",
                json={"refresh_token": refresh_token},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 204
    assert len(session.revoked_tokens) == 1
    assert refresh_token not in session.revoked_tokens

    with pytest.raises(ValueError, match="revoked"):
        await AuthService(session).refresh(refresh_token)


async def test_refresh_token_is_single_use(monkeypatch):
    user = SimpleNamespace(id=uuid4(), is_active=True)
    session = FakeAuthSession(user=user)
    service = AuthService(session)
    monkeypatch.setattr(service, "tokens_for", lambda current_user: "rotated-token-pair")
    refresh_token = create_refresh_token(str(user.id))

    assert await service.refresh(refresh_token) == "rotated-token-pair"
    assert len(session.revoked_tokens) == 1

    with pytest.raises(ValueError, match="revoked"):
        await service.refresh(refresh_token)


async def test_duplicate_revocation_insert_is_treated_as_replay():
    session = FakeAuthSession()
    session.commit_error = IntegrityError("insert", {}, RuntimeError("duplicate key"))
    service = AuthService(session)
    refresh_token = create_refresh_token(str(uuid4()))

    with pytest.raises(ValueError, match="revoked"):
        await service.logout(refresh_token)

    assert session.rollback_called


async def test_email_verification_token_is_single_use():
    user = SimpleNamespace(
        id=uuid4(),
        is_active=True,
        is_verified=False,
        auth_version=0,
    )
    service = AuthService(FakeAuthSession(user=user))
    token = create_email_verification_token(str(user.id))

    await service.verify_email(token)

    assert user.is_verified
    with pytest.raises(ValueError, match="already been used"):
        await service.verify_email(token)


async def test_password_reset_invalidates_previous_refresh_tokens():
    user = SimpleNamespace(
        id=uuid4(),
        is_active=True,
        is_verified=True,
        auth_version=0,
        password_hash=hash_password("old-password-123"),
    )
    session = FakeAuthSession(user=user)
    service = AuthService(session)
    old_refresh_token = create_refresh_token(str(user.id), auth_version=0)
    reset_token = create_password_reset_token(str(user.id))

    await service.reset_password(reset_token, "new-password-456")

    assert verify_password("new-password-456", user.password_hash)
    assert user.auth_version == 1
    with pytest.raises(ValueError, match="already been used"):
        await service.reset_password(reset_token, "replayed-password-789")
    assert verify_password("new-password-456", user.password_hash)
    assert user.auth_version == 1
    with pytest.raises(ValueError, match="revoked"):
        await service.refresh(old_refresh_token)
