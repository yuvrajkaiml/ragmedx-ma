from uuid import UUID

from fastapi import WebSocket, WebSocketDisconnect

from app.core.security import decode_token
from app.db import AsyncSessionLocal
from app.db.models import User
from app.schemas.websocket import WebSocketEventType
from app.services.query_execution_service import query_execution_service


async def _get_websocket_user(token: str | None) -> User | None:
    if not token:
        return None

    try:
        user_id = UUID(decode_token(token))
    except (TypeError, ValueError):
        return None

    async with AsyncSessionLocal() as session:
        user = await session.get(User, user_id)
        if user is None or not user.is_active:
            return None
        return user


async def query_stream(websocket: WebSocket, query_id: UUID) -> None:
    user = await _get_websocket_user(websocket.query_params.get("token"))

    if user is None:
        await websocket.close(code=1008, reason="Authentication required")
        return

    await websocket.accept()
    subscription = query_execution_service.subscribe(query_id, user.id)
    if subscription is None:
        await websocket.close(code=1008, reason="Query not found or access denied")
        return

    backlog, event_queue = subscription

    try:
        for event in backlog:
            await websocket.send_json(event.model_dump(mode="json"))
            if event.event_type in {
                WebSocketEventType.FINAL_RESPONSE,
                WebSocketEventType.ERROR,
            }:
                return

        while True:
            event = await event_queue.get()
            await websocket.send_json(event.model_dump(mode="json"))
            if event.event_type in {
                WebSocketEventType.FINAL_RESPONSE,
                WebSocketEventType.ERROR,
            }:
                return
    except WebSocketDisconnect:
        return
    finally:
        query_execution_service.unsubscribe(query_id, event_queue)
