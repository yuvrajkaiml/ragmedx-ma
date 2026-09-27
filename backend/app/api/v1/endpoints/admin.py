# Owner: Pranav Sukale — not yet implemented, see Task_Assignment.md
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_admin
from app.config import BASE_DIR, get_settings
from app.db import get_session
from app.db.models import (
    Conversation,
    ConversationMessage,
    QueryExecutionRecord,
    QueryFeedback,
    User,
)
from app.schemas.admin import (
    AgentLatencySummary,
    AdminLatencyResponse,
    AdminMetricsResponse,
    AdminUserStatusUpdate,
    AdminUserSummary,
    KnowledgeBaseStatsResponse,
)
from app.schemas.feedback import FeedbackResponse

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


@router.get("/knowledge-base/stats", response_model=KnowledgeBaseStatsResponse)
async def get_knowledge_base_stats(
    request: Request,
    _: User = Depends(get_current_admin),
):
    settings = get_settings()
    index_directory = Path(settings.faiss_index_dir)
    if not index_directory.is_absolute():
        index_directory = BASE_DIR / index_directory

    index_files = {
        "index.faiss": index_directory / "index.faiss",
        "chunks.pkl": index_directory / "chunks.pkl",
    }
    file_status = {name: path.is_file() for name, path in index_files.items()}
    total_bytes = sum(
        path.stat().st_size for path in index_files.values() if path.is_file()
    )
    retrieval_service = getattr(request.app.state, "retrieval_service", None)

    return KnowledgeBaseStatsResponse(
        index_loaded=bool(retrieval_service and retrieval_service.is_ready),
        index_directory=str(index_directory),
        required_files_present=file_status,
        total_index_bytes=total_bytes,
    )


@router.get("/metrics/latency", response_model=AdminLatencyResponse)
async def get_latency_metrics(
    _: User = Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
    sample_limit: int = Query(default=1_000, ge=1, le=5_000),
):
    result = await session.execute(
        select(QueryExecutionRecord.response_data)
        .where(
            QueryExecutionRecord.status == "completed",
            QueryExecutionRecord.response_data.is_not(None),
        )
        .order_by(QueryExecutionRecord.completed_at.desc())
        .limit(sample_limit)
    )
    responses = result.scalars().all()
    latencies: list[int] = []
    agent_durations: dict[str, list[int]] = {}

    for response in responses:
        if not isinstance(response, dict):
            continue
        duration = response.get("processing_time_ms")
        if isinstance(duration, int) and duration >= 0:
            latencies.append(duration)

        trace = response.get("agent_trace")
        steps = trace.get("steps", []) if isinstance(trace, dict) else []
        for step in steps:
            if not isinstance(step, dict):
                continue
            agent_type = step.get("agent_type")
            step_duration = step.get("duration_ms")
            if isinstance(agent_type, str) and isinstance(step_duration, int) and step_duration >= 0:
                agent_durations.setdefault(agent_type, []).append(step_duration)

    ordered_latencies = sorted(latencies)
    failed_queries = await session.scalar(
        select(func.count())
        .select_from(QueryExecutionRecord)
        .where(QueryExecutionRecord.status == "failed")
    )

    def percentile(values: list[int], proportion: float) -> int | None:
        if not values:
            return None
        return values[max(0, ceil(proportion * len(values)) - 1)]

    return AdminLatencyResponse(
        sampled_queries=len(responses),
        measured_queries=len(ordered_latencies),
        failed_queries=failed_queries or 0,
        average_processing_time_ms=(
            sum(ordered_latencies) / len(ordered_latencies)
            if ordered_latencies
            else None
        ),
        p50_processing_time_ms=percentile(ordered_latencies, 0.50),
        p95_processing_time_ms=percentile(ordered_latencies, 0.95),
        by_agent=[
            AgentLatencySummary(
                agent_type=agent_type,
                sample_count=len(durations),
                average_duration_ms=sum(durations) / len(durations),
            )
            for agent_type, durations in sorted(agent_durations.items())
        ],
        generated_at=datetime.now(timezone.utc),
    )


@router.get("/feedback", response_model=list[FeedbackResponse])
async def list_feedback(
    _: User = Depends(get_current_admin),
    session: AsyncSession = Depends(get_session),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    result = await session.execute(
        select(QueryFeedback)
        .order_by(QueryFeedback.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return result.scalars().all()

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

    if user.is_active != request.is_active:
        user.auth_version += 1
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