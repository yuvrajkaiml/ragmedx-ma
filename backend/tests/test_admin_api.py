from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from fastapi import Depends
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.main as main_module
from app.api.deps import get_current_admin, get_current_user
from app.db import Base, get_session
from app.db.models import QueryExecutionRecord, User
from app.main import app


@pytest.fixture
async def api_context(tmp_path, monkeypatch):
    database_path = tmp_path / "admin-api.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{database_path}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    admin = User(
        id=uuid4(),
        email="admin@example.org",
        username="admin-user",
        password_hash="not-used",
        role="system_admin",
        is_active=True,
        is_verified=True,
        auth_version=0,
    )
    owner = User(
        id=uuid4(),
        email="user@example.org",
        username="regular-user",
        password_hash="not-used",
        role="healthcare_professional",
        is_active=True,
        is_verified=True,
        auth_version=0,
    )
    other_user = User(
        id=uuid4(),
        email="other@example.org",
        username="other-user",
        password_hash="not-used",
        role="healthcare_professional",
        is_active=True,
        is_verified=True,
        auth_version=0,
    )
    query_id = uuid4()
    now = datetime.now(timezone.utc)
    async with session_factory() as session:
        session.add_all(
            [
                admin,
                owner,
                other_user,
                QueryExecutionRecord(
                    id=query_id,
                    user_id=owner.id,
                    question="A completed medical query",
                    include_agent_trace=True,
                    status="completed",
                    response_data={
                        "response_type": "verified_answer",
                        "processing_time_ms": 120,
                        "agent_trace": {
                            "steps": [
                                {"agent_type": "generator", "duration_ms": 70},
                                {"agent_type": "critic", "duration_ms": 30},
                            ]
                        },
                    },
                    created_at=now,
                    completed_at=now,
                ),
                QueryExecutionRecord(
                    id=uuid4(),
                    user_id=owner.id,
                    question="A failed medical query",
                    include_agent_trace=False,
                    status="failed",
                    created_at=now,
                    completed_at=now,
                ),
            ]
        )
        await session.commit()

    async def session_override():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_current_admin] = lambda: admin
    app.dependency_overrides[get_current_user] = lambda: owner
    app.dependency_overrides[get_session] = session_override

    index_directory = tmp_path / "faiss-index"
    index_directory.mkdir()
    (index_directory / "index.faiss").write_bytes(b"index")
    (index_directory / "chunks.pkl").write_bytes(b"chunks")
    monkeypatch.setattr(main_module.get_settings(), "faiss_index_dir", index_directory)
    previous_retrieval_service = getattr(app.state, "retrieval_service", None)
    app.state.retrieval_service = SimpleNamespace(is_ready=True)

    yield session_factory, admin, owner, other_user, query_id, index_directory

    app.dependency_overrides.clear()
    if previous_retrieval_service is None:
        del app.state.retrieval_service
    else:
        app.state.retrieval_service = previous_retrieval_service
    await engine.dispose()


async def test_feedback_is_owner_scoped_and_admin_listable(api_context):
    _, _, _, other_user, query_id, _ = api_context
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        accepted = await client.post(
            "/api/v1/feedback",
            json={"query_id": str(query_id), "rating": 5, "comment": "Helpful"},
        )
        duplicate = await client.post(
            "/api/v1/feedback",
            json={"query_id": str(query_id), "rating": 4},
        )
        listing = await client.get("/api/v1/admin/feedback")

    assert accepted.status_code == 201
    assert duplicate.status_code == 409
    assert listing.status_code == 200
    assert listing.json()[0]["comment"] == "Helpful"

    app.dependency_overrides[get_current_user] = lambda: other_user
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            denied = await client.post(
                "/api/v1/feedback",
                json={"query_id": str(query_id), "rating": 5},
            )
    finally:
        app.dependency_overrides.clear()

    assert denied.status_code == 404


async def test_admin_latency_and_knowledge_base_stats(api_context):
    _, _, _, _, _, index_directory = api_context
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        latency = await client.get("/api/v1/admin/metrics/latency")
        knowledge_base = await client.get("/api/v1/admin/knowledge-base/stats")

    assert latency.status_code == 200
    assert latency.json()["measured_queries"] == 1
    assert latency.json()["failed_queries"] == 1
    assert latency.json()["average_processing_time_ms"] == 120
    assert latency.json()["by_agent"][0]["agent_type"] == "critic"

    assert knowledge_base.status_code == 200
    payload = knowledge_base.json()
    assert payload["index_loaded"] is True
    assert payload["required_files_present"] == {"index.faiss": True, "chunks.pkl": True}
    assert payload["total_index_bytes"] == sum(
        path.stat().st_size for path in index_directory.iterdir()
    )


async def test_user_can_deactivate_own_account(api_context):
    session_factory, _, owner, _, _, _ = api_context

    async def owner_from_database(session=Depends(get_session)):
        return await session.get(User, owner.id)

    app.dependency_overrides[get_current_user] = owner_from_database
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.delete("/api/v1/users/me")
    finally:
        app.dependency_overrides.clear()

    async with session_factory() as session:
        updated_user = await session.get(User, owner.id)

    assert response.status_code == 204
    assert updated_user.is_active is False
    assert updated_user.auth_version == 1


async def test_admin_deactivation_invalidates_user_tokens(api_context):
    session_factory, _, _, other_user, _, _ = api_context
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.patch(
            f"/api/v1/admin/users/{other_user.id}/status",
            json={"is_active": False},
        )

    async with session_factory() as session:
        updated_user = await session.get(User, other_user.id)

    assert response.status_code == 200
    assert updated_user.is_active is False
    assert updated_user.auth_version == 1
