"""
Structured logging with JSON and plain-text formatters.

Usage:
    from yiagent.common.log import logger
    logger.info("message", extra={"session_id": "x"})
"""

from __future__ import annotations

import logging
import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            payload["exc"] = str(record.exc_info[1])
        extra = getattr(record, "_extra", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False, default=str)


class _PlainFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        base = f"[{ts}] [{record.levelname}] [{record.name}] {record.getMessage()}"
        if record.exc_info and record.exc_info[1]:
            base += f"\n  {record.exc_info[1]}"
        return base


def _make_logger() -> logging.Logger:
    from yiagent.common.config import conf
    cfg = conf()
    level = getattr(logging, cfg.get("log_level", "INFO").upper(), logging.INFO)
    fmt = cfg.get("log_format", "json")

    logger = logging.getLogger("yiagent")
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setLevel(level)
        h.setFormatter(_JsonFormatter() if fmt == "json" else _PlainFormatter())
        logger.addHandler(h)

    return logger


logger = _make_logger()
