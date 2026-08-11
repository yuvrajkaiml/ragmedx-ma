# Owner: Pranav Sukale — not yet implemented, see Task_Assignment.md
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.db import get_session
from app.db.models import User
from app.schemas.conversation import (
    ConversationCreate,
    ConversationDetail,
    ConversationList,
    ConversationSummary,
    ConversationUpdate,
)
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["Conversations"])


@router.post(
    "/conversations",
    response_model=ConversationSummary,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    request: ConversationCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    return await ChatService().create_conversation(
        request,
        current_user,
        session,
    )


@router.get("/conversations", response_model=ConversationList)
async def list_conversations(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    return await ChatService().list_conversations(current_user, session)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    conversation = await ChatService().get_conversation(
        conversation_id,
        current_user,
        session,
    )

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    return conversation

@router.get("/conversations/{conversation_id}/export")
async def export_conversation(
    conversation_id: UUID,
    format: Literal["json", "markdown"] = Query(default="json"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    conversation = await ChatService().get_conversation(
        conversation_id,
        current_user,
        session,
    )

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    if format == "json":
        return JSONResponse(
            content=conversation.model_dump(mode="json"),
            headers={
                "Content-Disposition": (
                    f'attachment; filename="conversation-{conversation_id}.json"'
                )
            },
        )

    lines = [
        f"# {conversation.title}",
        "",
        f"- Conversation ID: `{conversation.id}`",
        f"- Created: {conversation.created_at.isoformat()}",
        f"- Updated: {conversation.updated_at.isoformat()}",
        "",
    ]

    for message in conversation.messages:
        lines.extend(
            [
                f"## {message.role.value.title()}",
                "",
                message.content,
                "",
            ]
        )

    return PlainTextResponse(
        content="\n".join(lines),
        media_type="text/markdown",
        headers={
            "Content-Disposition": (
                f'attachment; filename="conversation-{conversation_id}.md"'
            )
        },
    )

@router.patch("/conversations/{conversation_id}", response_model=ConversationSummary)
async def update_conversation(
    conversation_id: UUID,
    request: ConversationUpdate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    conversation = await ChatService().update_conversation(
        conversation_id,
        request,
        current_user,
        session,
    )

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    return conversation


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_conversation(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    deleted = await ChatService().delete_conversation(
        conversation_id,
        current_user,
        session,
    )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)