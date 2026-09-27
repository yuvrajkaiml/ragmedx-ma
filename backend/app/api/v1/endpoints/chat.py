# Owner: Pranav Sukale — not yet implemented, see Task_Assignment.md
from io import BytesIO
from typing import Literal
from uuid import UUID
from xml.sax.saxutils import escape

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import JSONResponse, PlainTextResponse, Response as FastAPIResponse
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
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
    format: Literal["json", "markdown", "pdf"] = Query(default="json"),
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

    if format == "pdf":
        buffer = BytesIO()
        document = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.7 * inch,
            bottomMargin=0.7 * inch,
            title=conversation.title,
        )
        styles = getSampleStyleSheet()
        metadata_style = ParagraphStyle(
            "ConversationMetadata",
            parent=styles["BodyText"],
            textColor=colors.HexColor("#555555"),
            fontSize=9,
            leading=12,
        )
        message_style = ParagraphStyle(
            "ConversationMessage",
            parent=styles["BodyText"],
            alignment=TA_LEFT,
            leading=15,
            spaceAfter=10,
        )
        story = [
            Paragraph(escape(conversation.title), styles["Title"]),
            Spacer(1, 0.12 * inch),
            Paragraph(f"Conversation ID: {conversation.id}", metadata_style),
            Paragraph(f"Created: {conversation.created_at.isoformat()}", metadata_style),
            Paragraph(f"Updated: {conversation.updated_at.isoformat()}", metadata_style),
            Spacer(1, 0.25 * inch),
        ]

        for message in conversation.messages:
            story.extend(
                [
                    Paragraph(escape(message.role.value.title()), styles["Heading3"]),
                    Paragraph(escape(message.created_at.isoformat()), metadata_style),
                    Spacer(1, 0.06 * inch),
                    Paragraph(
                        escape(message.content).replace("\n", "<br/>") or "&nbsp;",
                        message_style,
                    ),
                ]
            )

        document.build(story)
        return FastAPIResponse(
            content=buffer.getvalue(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="conversation-{conversation_id}.pdf"'
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