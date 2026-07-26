from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter()

chat_service = ChatService()


@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):

    answer = chat_service.generate_response(
        question=request.question,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    return ChatResponse(
        answer=answer,
        model="Qwen2.5-1.5B",
    )