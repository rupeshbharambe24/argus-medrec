"""Structured logging via structlog — one line of JSON per event in prod,
human-friendly pretty output in dev.

Every tool call emits a structured event with request_id, tool, latency, outcome.
"""

from __future__ import annotations

import logging
import sys

import structlog

from argus.config import get_settings


def configure_logging() -> None:
    """Configure structlog. Call once at server startup."""
    settings = get_settings()

    level = getattr(logging, settings.log_level, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.is_prod:
        # Production: JSON lines — easy to ingest into any log aggregator
        processors.append(structlog.processors.JSONRenderer(serializer=_orjson_dumps))
    else:
        # Dev: colored, readable
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _orjson_dumps(obj, default=None) -> str:
    """structlog wants a str; orjson returns bytes."""
    import orjson

    return orjson.dumps(obj, default=default).decode()


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a namespaced structlog logger. Usage: `log = get_logger(__name__)`."""
    return structlog.get_logger(name)
