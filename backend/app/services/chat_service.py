# Owner: Pranav Sukale — not yet implemented, see Task_Assignment.md
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Conversation, ConversationMessage, User
from app.schemas.conversation import (
    ConversationCreate,
    ConversationDetail,
    ConversationList,
    ConversationMessageResponse,
    ConversationSummary,
    ConversationUpdate,
)


class ChatService:
    async def create_conversation(
        self,
        request: ConversationCreate,
        user: User,
        session: AsyncSession,
    ) -> ConversationSummary:
        conversation = Conversation(
            user_id=user.id,
            title=request.title or "New conversation",
        )
        session.add(conversation)
        await session.commit()
        await session.refresh(conversation)
        return self._summary(conversation, message_count=0)

    async def list_conversations(
        self,
        user: User,
        session: AsyncSession,
    ) -> ConversationList:
        statement = (
            select(
                Conversation,
                func.count(ConversationMessage.id).label("message_count"),
            )
            .outerjoin(
                ConversationMessage,
                ConversationMessage.conversation_id == Conversation.id,
            )
            .where(Conversation.user_id == user.id)
            .group_by(Conversation.id)
            .order_by(Conversation.updated_at.desc())
        )
        rows = (await session.execute(statement)).all()

        conversations = [
            self._summary(conversation, message_count=message_count)
            for conversation, message_count in rows
        ]
        return ConversationList(conversations=conversations, total=len(conversations))

    async def get_conversation(
        self,
        conversation_id: UUID,
        user: User,
        session: AsyncSession,
    ) -> ConversationDetail | None:
        statement = (
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == user.id,
            )
        )
        conversation = (await session.execute(statement)).scalar_one_or_none()

        if conversation is None:
            return None

        messages = [
            ConversationMessageResponse.model_validate(message)
            for message in sorted(conversation.messages, key=lambda item: item.created_at)
        ]
        return ConversationDetail(
            **self._summary(
                conversation,
                message_count=len(messages),
            ).model_dump(),
            messages=messages,
        )

    async def update_conversation(
        self,
        conversation_id: UUID,
        request: ConversationUpdate,
        user: User,
        session: AsyncSession,
    ) -> ConversationSummary | None:
        conversation = await self._get_owned_conversation(
            conversation_id,
            user,
            session,
        )

        if conversation is None:
            return None

        for field, value in request.model_dump(exclude_unset=True).items():
            setattr(conversation, field, value)

        await session.commit()
        await session.refresh(conversation)
        return self._summary(conversation, message_count=0)

    async def delete_conversation(
        self,
        conversation_id: UUID,
        user: User,
        session: AsyncSession,
    ) -> bool:
        conversation = await self._get_owned_conversation(
            conversation_id,
            user,
            session,
        )

        if conversation is None:
            return False

        await session.delete(conversation)
        await session.commit()
        return True

    async def _get_owned_conversation(
        self,
        conversation_id: UUID,
        user: User,
        session: AsyncSession,
    ) -> Conversation | None:
        statement = select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
        )
        return (await session.execute(statement)).scalar_one_or_none()

    @staticmethod
    def _summary(
        conversation: Conversation,
        message_count: int,
    ) -> ConversationSummary:
        return ConversationSummary(
            id=conversation.id,
            title=conversation.title,
            is_archived=conversation.is_archived,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            message_count=message_count,
        )