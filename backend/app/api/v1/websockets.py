from datetime import datetime, timezone
from uuid import UUID

from fastapi import WebSocket, WebSocketDisconnect

from app.core.security import decode_token
from app.db import AsyncSessionLocal
from app.db.models import User
from app.schemas.websocket import WebSocketEvent, WebSocketEventType


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

    connected_event = WebSocketEvent(
        query_id=query_id,
        event_type=WebSocketEventType.AGENT_STATUS,
        timestamp=datetime.now(timezone.utc),
        data={
            "agent": "orchestrator",
            "status": "connected",
        },
    )
    await websocket.send_json(connected_event.model_dump(mode="json"))

    try:
        while True:
            message = await websocket.receive_json()

            if message.get("event_type") == WebSocketEventType.PING.value:
                pong_event = WebSocketEvent(
                    query_id=query_id,
                    event_type=WebSocketEventType.PING,
                    timestamp=datetime.now(timezone.utc),
                    data={"status": "pong"},
                )
                await websocket.send_json(pong_event.model_dump(mode="json"))
    except WebSocketDisconnect:
        return