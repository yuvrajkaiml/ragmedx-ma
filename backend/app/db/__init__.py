from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings


class Base(DeclarativeBase):
    pass


engine = create_async_engine(get_settings().database_url, echo=get_settings().debug)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_session():
    async with AsyncSessionLocal() as session:
        yield session


# Re-export the ORM models for convenience. Imported lazily at module end to
# avoid a circular import (models.py imports ``Base`` from this package).
from app.db.models import (  # noqa: E402
    Chunk,
    ConsumedAuthToken,
    Conversation,
    ConversationMessage,
    Document,
    QueryExecutionRecord,
    QueryFeedback,
    RetrievedEvidence,
    RevokedRefreshToken,
    User,
)

__all__ = [
    "Base",
    "engine",
    "AsyncSessionLocal",
    "get_session",
    # Auth / conversation / query models (yuvraj/llm)
    "User",
    "RevokedRefreshToken",
    "ConsumedAuthToken",
    "Conversation",
    "ConversationMessage",
    "QueryExecutionRecord",
    "QueryFeedback",
    # Retrieval models (sana/retrieval)
    "Document",
    "Chunk",
    "RetrievedEvidence",
]
