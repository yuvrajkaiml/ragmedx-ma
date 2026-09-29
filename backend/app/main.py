from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text

from app.api.v1.router import api_router
from app.config import FAISS_INDEX_DIR, get_settings
from app.core.exceptions import register_exception_handlers
from app.db import Base, engine
from app.llm.local_llm import LLMRuntimeMode, LocalLLMWrapper
from app.services.agent_service import AgentService
from app.services.retrieval_service import RetrievalService
from app.services.query_execution_service import query_execution_service
from app.utils.logger import logger
import app.db.models  # noqa: F401 - registers ORM models before create_all


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        user_columns = await connection.run_sync(
            lambda sync_connection: {
                column["name"] for column in inspect(sync_connection).get_columns("users")
            }
        )
        if "auth_version" not in user_columns:
            await connection.execute(
                text("ALTER TABLE users ADD COLUMN auth_version INTEGER NOT NULL DEFAULT 0")
            )

        query_columns = await connection.run_sync(
            lambda sync_connection: {
                column["name"]
                for column in inspect(sync_connection).get_columns("query_executions")
            }
        )
        if "document_id" not in query_columns:
            await connection.execute(
                text("ALTER TABLE query_executions ADD COLUMN document_id CHAR(32)")
            )

    settings = get_settings()

    # Build retrieval service and store the singleton on app.state so every
    # request reuses the same loaded FAISS index, BGE-M3 embedder, and
    # cross-encoder — not a freshly deserialised copy per request.
    retrieval_service = RetrievalService.build()
    app.state.retrieval_service = retrieval_service

    # In real mode, load the LLM eagerly at startup so a VRAM/weight failure
    # surfaces immediately in the startup log rather than mid-request.
    # In mock mode, skip the load — the LLM wrapper uses deterministic stubs.
    if settings.llm_mode == "real":
        logger.info(
            f"Loading LLM '{settings.llm_model_name}' eagerly at startup "
            f"(llm_mode=real) …"
        )
        llm_wrapper = LocalLLMWrapper(
            model_name_or_path=settings.llm_model_name,
            mode=LLMRuntimeMode.LOCAL,
            allow_fallback=False,
        )
        llm_wrapper.load_model()   # raises ModelLoadError on failure — intentional
        app.state.llm_wrapper = llm_wrapper
        logger.info("LLM loaded successfully.")
    else:
        app.state.llm_wrapper = None

    query_execution_service.agent_service = AgentService(
        llm_wrapper=app.state.llm_wrapper,
        retrieval_provider=retrieval_service.retrieve_evidence,
    )

    logger.info(
        f"RAGMedX-MA startup complete — llm_mode={settings.llm_mode}, "
        f"retrieval_ready={retrieval_service.is_ready}"
    )

    yield
    await engine.dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok", "service": settings.app_name, "version": settings.app_version}

    @app.get("/readiness", tags=["Health"])
    async def readiness_check():
        runtime_settings = get_settings()
        index_ready = (FAISS_INDEX_DIR / "index.faiss").is_file() and (
            FAISS_INDEX_DIR / "chunks.pkl"
        ).is_file()
        llm_mode = (runtime_settings.llm_runtime_mode or runtime_settings.llm_mode).lower()
        llm_ready = llm_mode != "mock"
        checks = {
            "llm": {
                "ready": llm_ready,
                "mode": llm_mode,
                "model": runtime_settings.llm_model_name,
                "fallback_allowed": runtime_settings.llm_allow_fallback,
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
