from fastapi import APIRouter

from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router

router = APIRouter()

router.include_router(chat_router, tags=["Chat"])
router.include_router(health_router, tags=["Health"])