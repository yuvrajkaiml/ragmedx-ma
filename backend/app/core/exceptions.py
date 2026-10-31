# Owner: Pranav Sukale
"""Application exception hierarchy and FastAPI exception handlers.

All handled errors are rendered with the canonical ``ErrorResponse`` shape
(``schemas/response.py``) so the frontend and every client see one consistent
error contract:

    {"response_type": "error", "message": "...", "error_code": "..."}

Register the handlers once from ``main.create_app`` via
:func:`register_exception_handlers`.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.schemas.response import ErrorResponse
from app.utils.logger import logger


class RAGMedXError(Exception):
    """Base class for domain errors that map to an HTTP error response.

    Parameters
    ----------
    message:
        Human-readable message safe to return to the client.
    error_code:
        Stable machine-readable code (e.g. ``CONVERSATION_NOT_FOUND``).
    status_code:
        HTTP status to return. Defaults to 400 Bad Request.
    """

    status_code: int = status.HTTP_400_BAD_REQUEST
    error_code: str = "RAGMEDX_ERROR"

    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if error_code is not None:
            self.error_code = error_code
        if status_code is not None:
            self.status_code = status_code


class NotFoundError(RAGMedXError):
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "NOT_FOUND"


class AuthenticationError(RAGMedXError):
    status_code = status.HTTP_401_UNAUTHORIZED
    error_code = "AUTHENTICATION_FAILED"


class AuthorizationError(RAGMedXError):
    status_code = status.HTTP_403_FORBIDDEN
    error_code = "AUTHORIZATION_FAILED"


class ConflictError(RAGMedXError):
    status_code = status.HTTP_409_CONFLICT
    error_code = "CONFLICT"


class ValidationError(RAGMedXError):
    status_code = 422  # Unprocessable Content
    error_code = "VALIDATION_ERROR"


class ServiceUnavailableError(RAGMedXError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "SERVICE_UNAVAILABLE"


class DocumentProcessingError(RAGMedXError):
    """Raised when an uploaded document cannot be parsed, chunked, or indexed."""

    status_code = 422  # Unprocessable Content
    error_code = "DOCUMENT_PROCESSING_FAILED"


def _error_payload(message: str, error_code: str | None) -> dict:
    """Build the canonical ErrorResponse JSON body."""
    return ErrorResponse(message=message, error_code=error_code).model_dump(mode="json")


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers that render every error as an ``ErrorResponse``."""

    @app.exception_handler(RAGMedXError)
    async def _handle_domain_error(_: Request, exc: RAGMedXError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(exc.message, exc.error_code),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(
        _: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        # ``detail`` is usually a string; fall back to a generic message otherwise.
        message = exc.detail if isinstance(exc.detail, str) else "Request failed"
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(message, f"HTTP_{exc.status_code}"),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Summarise the first error for a readable message; keep the code stable.
        errors = exc.errors()
        first = errors[0] if errors else None
        if first is not None:
            location = ".".join(str(part) for part in first.get("loc", []) if part != "body")
            detail = first.get("msg", "Invalid request")
            message = f"{location}: {detail}" if location else detail
        else:
            message = "Request validation failed"
        return JSONResponse(
            status_code=422,  # Unprocessable Content
            content=_error_payload(message, "VALIDATION_ERROR"),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        # Never leak internal details to the client; log the full trace server-side.
        logger.exception(f"Unhandled server error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_payload(
                "An unexpected server error occurred.", "INTERNAL_SERVER_ERROR"
            ),
        )
