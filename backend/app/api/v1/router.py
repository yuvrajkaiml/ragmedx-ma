from uuid import UUID

from fastapi import APIRouter, WebSocket

from app.api.v1.endpoints import (
    admin_router,
    auth_router,
    chat_router,
    documents_router,
    feedback_router,
    query_router,
    users_router,
)
from app.api.v1.websockets import query_stream

api_router = APIRouter()
api_router.include_router(admin_router)
api_router.include_router(auth_router)
api_router.include_router(users_router)
api_router.include_router(chat_router)
api_router.include_router(documents_router)
api_router.include_router(feedback_router)
api_router.include_router(query_router)


@api_router.websocket("/ws/query-stream")
async def stream_query(websocket: WebSocket, query_id: UUID):
    await query_stream(websocket, query_id)