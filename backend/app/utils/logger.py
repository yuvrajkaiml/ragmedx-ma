# Owner: Pranav Sukale
"""Convenience re-export of the application logger.

All existing code that imports from here continues to work unchanged:

    from app.utils.logger import logger

The actual logging configuration lives in ``app.core.logging``.
Importing this module triggers ``configure_logging()`` exactly once.
"""

from app.core.logging import get_logger  # noqa: F401 — triggers configure_logging

logger = get_logger()

__all__ = ["logger"]
