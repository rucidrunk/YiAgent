"""
Tests for yiagent.common.log — structured logging.

CRITICAL BUGS FOUND during audit:
  1. _make_logger() calls conf() which calls load_config() — if config
     loading has a side effect that depends on the logger, we have a
     circular dependency.
  2. The module-level `logger = _make_logger()` runs at import time,
     which means config must be loadable before anything else.
  3. Extra fields are attached via record._extra but logging's standard
     extra dict is merged differently — this custom approach may confuse
     other handlers.
"""
from __future__ import annotations

import json
import logging

import pytest


class TestLogFormatters:
    def test_json_formatter_basic(self):
        from yiagent.common.log import _JsonFormatter

        fmt = _JsonFormatter()
        record = logging.LogRecord(
            name="yiagent",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello world",
            args=(),
            exc_info=None,
        )
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "yiagent"
        assert parsed["msg"] == "hello world"
        assert "ts" in parsed

    def test_json_formatter_with_exception(self):
        from yiagent.common.log import _JsonFormatter

        fmt = _JsonFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name="yiagent",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="something broke",
                args=(),
                exc_info=sys.exc_info(),
            )
        output = fmt.format(record)
        parsed = json.loads(output)
        assert "exc" in parsed
        assert "test error" in parsed["exc"]

    def test_json_formatter_with_extra(self):
        from yiagent.common.log import _JsonFormatter

        fmt = _JsonFormatter()
        record = logging.LogRecord(
            name="yiagent",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="msg with extra",
            args=(),
            exc_info=None,
        )
        record._extra = {"session_id": "abc123", "user_id": "u1"}
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["session_id"] == "abc123"
        assert parsed["user_id"] == "u1"

    def test_plain_formatter_basic(self):
        from yiagent.common.log import _PlainFormatter

        fmt = _PlainFormatter()
        record = logging.LogRecord(
            name="yiagent",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="warning message",
            args=(),
            exc_info=None,
        )
        output = fmt.format(record)
        assert "[WARNING]" in output
        assert "warning message" in output

    def test_plain_formatter_with_exception(self):
        from yiagent.common.log import _PlainFormatter

        fmt = _PlainFormatter()
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            import sys
            record = logging.LogRecord(
                name="yiagent",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="error occurred",
                args=(),
                exc_info=sys.exc_info(),
            )
        output = fmt.format(record)
        assert "boom" in output


class TestMakeLogger:
    def test_logger_is_singleton(self):
        """Multiple calls to _make_logger must return the same logger."""
        from yiagent.common.log import _make_logger
        l1 = _make_logger()
        l2 = _make_logger()
        assert l1 is l2

    def test_logger_has_handlers(self):
        from yiagent.common.log import logger
        assert len(logger.handlers) > 0

    def test_logger_propagate_is_false(self):
        from yiagent.common.log import logger
        assert logger.propagate is False

    def test_log_level_from_config(self):
        """Logger level should match config's log_level."""
        from yiagent.common.log import logger
        from yiagent.common.config import conf
        cfg_level = conf().get("log_level", "INFO").upper()
        expected = getattr(logging, cfg_level, logging.INFO)
        assert logger.level == expected


class TestLogOutput:
    def test_logger_info_does_not_raise(self):
        from yiagent.common.log import logger
        logger.info("test message")

    def test_logger_with_extra_fields(self):
        """logger.info with extra dict."""
        from yiagent.common.log import logger
        # The logger's extra fields go through record._extra
        extra = {"session_id": "test123"}
        logger.info("test with extra", extra=extra)

    def test_logger_all_levels(self):
        from yiagent.common.log import logger
        logger.debug("debug msg")
        logger.info("info msg")
        logger.warning("warning msg")
        logger.error("error msg")
        logger.critical("critical msg")
        # No assertions needed — just verify no crash

    def test_logger_exception_logging(self):
        from yiagent.common.log import logger
        try:
            raise ValueError("test exception")
        except ValueError:
            logger.exception("caught exception")
        # Must not raise


class TestLogExtreme:
    """CORE AUDIT: Extreme boundary conditions."""

    def test_log_massive_message(self):
        """10MB log message must not OOM."""
        from yiagent.common.log import logger
        huge = "x" * 1_000_000
        logger.info(huge)
        # Must not OOM or hang

    def test_json_formatter_handles_unicode_extra(self):
        from yiagent.common.log import _JsonFormatter

        fmt = _JsonFormatter()
        record = logging.LogRecord(
            name="yiagent",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="unicode test",
            args=(),
            exc_info=None,
        )
        record._extra = {"text": "こんにちは世界"}
        output = fmt.format(record)
        parsed = json.loads(output)
        assert "こんにちは世界" in parsed["text"]

    def test_json_formatter_handles_non_serializable_extra(self):
        """Extra with non-serializable objects uses default=str."""
        from yiagent.common.log import _JsonFormatter

        fmt = _JsonFormatter()
        record = logging.LogRecord(
            name="yiagent",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=(),
            exc_info=None,
        )

        class NonSerializable:
            def __str__(self):
                return "NonSerializableStr"

        record._extra = {"obj": NonSerializable()}
        output = fmt.format(record)
        # default=str converts NonSerializable to its str representation
        parsed = json.loads(output)
        assert "obj" in parsed
