from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.config import get_settings
from app.db import Base, engine
from app.llm.local_llm import LLMRuntimeMode, LocalLLMWrapper
from app.services.retrieval_service import RetrievalService
from app.utils.logger import logger
import app.db.models  # noqa: F401 - registers ORM models before create_all


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

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
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok", "service": settings.app_name, "version": settings.app_version}

    return app


app = create_app()
