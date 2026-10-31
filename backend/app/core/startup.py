# Owner: Pranav Sukale
"""Application startup utilities.

The ``lifespan`` context manager in ``app.main`` is the primary startup
orchestrator.  This module provides the helper functions it calls so that
startup logic is testable in isolation and ``main.py`` stays concise.
"""

from __future__ import annotations

from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


def initialise_logging(level: str | None = None) -> None:
    """Apply loguru configuration early in the startup sequence.

    Call this before any other import that might emit a log message so that
    all output goes through the configured sinks from the very first line.

    Parameters
    ----------
    level:
        Console log level.  Defaults to the ``LOG_LEVEL`` environment variable
        (``"INFO"`` when not set).
    """
    configure_logging(level=level)
    logger.info("Logging initialised.")


def log_startup_banner(app_name: str, version: str, llm_mode: str, retrieval_ready: bool) -> None:
    """Emit a structured startup summary to the log.

    Parameters
    ----------
    app_name:
        Application name from settings.
    version:
        Application version from settings.
    llm_mode:
        ``"real"`` or ``"mock"``.
    retrieval_ready:
        Whether the FAISS index loaded successfully.
    """
    logger.info("=" * 60)
    logger.info(f"  {app_name} v{version}")
    logger.info(f"  LLM mode       : {llm_mode}")
    logger.info(f"  Retrieval ready: {retrieval_ready}")
    logger.info("=" * 60)
