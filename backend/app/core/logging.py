# Owner: Pranav Sukale
"""Centralised logging configuration for RAGMedX-MA.

All application code should import the logger from this module or from the
convenience re-export in ``app.utils.logger``:

    from app.utils.logger import logger          # convenience (existing)
    from app.core.logging import get_logger      # canonical (this module)

The configuration is applied once at import time via :func:`configure_logging`.
Calling it again is a no-op unless ``force=True`` is passed — this keeps test
isolation clean.

Log sinks
---------
stdout
    Human-readable coloured output.  Level controlled by the ``LOG_LEVEL``
    environment variable (default ``INFO``).

logs/app.log
    Structured file sink with rotation/retention.  Always at DEBUG level so
    diagnostic detail is captured without flooding the console.

logs/retrieval.log
    Dedicated sink for retrieval-pipeline traces (existing; kept for
    backward compatibility with the ``utils/logger.py`` setup).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

_CONFIGURED = False

# Resolve the logs directory relative to backend/ regardless of cwd.
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_LOGS_DIR = _BACKEND_DIR / "logs"


def configure_logging(
    level: str | None = None,
    force: bool = False,
) -> None:
    """Apply the application-wide loguru configuration.

    Parameters
    ----------
    level:
        Console log level override.  Defaults to the ``LOG_LEVEL`` environment
        variable, falling back to ``"INFO"``.
    force:
        Re-apply the configuration even if it has been applied before.
        Useful in tests that need a fresh logging state.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    _LOGS_DIR.mkdir(parents=True, exist_ok=True)

    console_level = (
        level
        or os.environ.get("LOG_LEVEL", "INFO").upper()
    )

    # Remove all existing handlers so callers that import the module
    # multiple times don't accumulate duplicate sinks.
    logger.remove()

    # ── Sink 1: stdout (coloured, human-readable) ───────────────────────
    logger.add(
        sys.stdout,
        level=console_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
            "{message}"
        ),
        colorize=True,
        enqueue=True,       # thread-safe for async workers
    )

    # ── Sink 2: app.log (structured, full debug) ────────────────────────
    logger.add(
        str(_LOGS_DIR / "app.log"),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{line} | {message}",
        rotation="10 MB",
        retention="14 days",
        compression="gz",
        enqueue=True,
    )

    # ── Sink 3: retrieval.log (backward-compatible with utils/logger.py) ─
    logger.add(
        str(_LOGS_DIR / "retrieval.log"),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="5 MB",
        retention="7 days",
        enqueue=True,
    )

    _CONFIGURED = True


def get_logger(name: str | None = None):
    """Return the configured loguru logger, applying configuration if needed.

    Parameters
    ----------
    name:
        Optional module name for ``logger.bind(name=name)`` context.
        When ``None`` the root loguru logger is returned.

    Returns
    -------
    loguru.Logger
        The configured logger instance.
    """
    configure_logging()
    if name:
        return logger.bind(name=name)
    return logger


# Apply configuration immediately on import so the module is self-contained.
configure_logging()
