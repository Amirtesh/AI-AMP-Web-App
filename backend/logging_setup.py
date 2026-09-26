"""Structured request logging (stdlib only, no third-party dependency)."""

import json
import logging
import sys

_configured = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    global _configured
    logger = logging.getLogger("ai_amp_backend")
    if not _configured:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.propagate = False
        _configured = True
    return logger


def log_request(
    logger: logging.Logger,
    *,
    endpoint: str,
    client: str,
    n_sequences: int,
    models: list,
    latency_ms: float,
    rejection: str | None = None,
) -> None:
    """One JSON line per request: input size, model(s), latency, rejection."""
    logger.info(
        json.dumps(
            {
                "event": "request",
                "endpoint": endpoint,
                "client": client,
                "n_sequences": n_sequences,
                "models": models,
                "latency_ms": round(latency_ms, 1),
                "rejection": rejection,
            }
        )
    )
