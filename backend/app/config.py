from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================
# Project Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

DATASETS_DIR = DATA_DIR / "datasets"
DOCUMENTS_DIR = DATA_DIR / "documents"

EMBEDDINGS_DIR = DATA_DIR / "embeddings"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

CACHE_DIR = DATA_DIR / "cache"

RESULTS_DIR = DATA_DIR / "results"
EXPERIMENTS_DIR = DATA_DIR / "experiments"


# ============================================================
# Retrieval Parameters
# ============================================================

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400

TOP_K_INITIAL = 50
TOP_K_FINAL = 5

# MAX_ITERATIONS = 3


# ============================================================
# Application Settings
# ============================================================

class Settings(BaseSettings):
    """Application settings loaded from environment variables or backend/.env."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "RAGMedX-MA"
    app_version: str = "0.1.0"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    # Security
    secret_key: str = Field(
        "change-me-before-production",
        min_length=16,
    )
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # Database
    database_url: str = "sqlite+aiosqlite:///./ragmedx.db"

    # CORS
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
    ]

    # Uploads
    upload_data_dir: Path = Path("data/uploads")
    max_upload_size_mb: int = Field(
        default=50,
        ge=1,
        le=100,
    )

    # Retrieval
    retrieval_data_dir: Path = Path("data/retrieval")
    retrieval_collection_name: str = "medical_evidence"

    retrieval_chunk_size: int = Field(
        default=1_000,
        ge=200,
        le=4_000,
    )

    retrieval_chunk_overlap: int = Field(
        default=150,
        ge=0,
        le=1_000,
    )

    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    # Embeddings
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"


@lru_cache
def get_settings() -> Settings:
    return Settings()