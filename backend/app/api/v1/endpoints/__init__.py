from app.api.v1.endpoints.admin import router as admin_router
from app.api.v1.endpoints.auth import router as auth_router
from app.api.v1.endpoints.chat import router as chat_router
from app.api.v1.endpoints.documents import router as documents_router
from app.api.v1.endpoints.query import router as query_router
from app.api.v1.endpoints.users import router as users_router

__all__ = [
    "admin_router",
    "auth_router",
    "chat_router",
    "documents_router",
    "query_router",
    "users_router",
]