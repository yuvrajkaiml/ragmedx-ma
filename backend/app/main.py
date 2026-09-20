from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.config import FAISS_INDEX_DIR, get_settings
from app.db import Base, engine
import app.db.models  # noqa: F401 - registers ORM models before create_all


@asynccontextmanager
async def lifespan(_: FastAPI):
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok", "service": settings.app_name, "version": settings.app_version}

    @app.get("/readiness", tags=["Health"])
    async def readiness_check():
        index_ready = (FAISS_INDEX_DIR / "index.faiss").is_file() and (
            FAISS_INDEX_DIR / "chunks.pkl"
        ).is_file()
        llm_ready = settings.llm_runtime_mode.lower() != "mock"
        checks = {
            "llm": {
                "ready": llm_ready,
                "mode": settings.llm_runtime_mode,
                "model": settings.llm_model_name,
                "fallback_allowed": settings.llm_allow_fallback,
            },
            "retrieval": {
                "ready": index_ready,
                "index_directory": str(FAISS_INDEX_DIR),
                "required_files": ["index.faiss", "chunks.pkl"],
            },
        }
        ready = all(check["ready"] for check in checks.values())
        payload = {"status": "ready" if ready else "not_ready", "checks": checks}
        return JSONResponse(status_code=200 if ready else 503, content=payload)

    return app


app = create_app()
