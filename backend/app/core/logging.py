"""Reusable application logging configuration."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a named logger; centralized formatting is a future concern."""
    # TODO: Configure structured logging from application settings.
    return logging.getLogger(name)
