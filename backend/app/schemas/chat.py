from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    question: str
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 512


class ChatResponse(BaseModel):
    answer: str
    model: str