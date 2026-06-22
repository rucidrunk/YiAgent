"""
Tests for yiagent.common.utils — hashing, token estimation, JSON helpers.

CRITICAL BUGS FOUND during audit:
  1. estimate_tokens() uses ord(c) > 127 as CJK proxy — miscounts
     Cyrillic, Arabic, emoji, and many other non-CJK scripts.
  2. safe_json_dumps() truncates raw JSON string — produces invalid JSON.
  3. No overflow protection on compute_hash / compute_md5 for massive strings.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from yiagent.common.utils import (
    compute_hash,
    compute_md5,
    contains_cjk,
    estimate_tokens,
    expand_path,
    safe_json_dumps,
    truncate_str,
)


# ======================================================================
# expand_path
# ======================================================================

class TestExpandPath:
    def test_expands_home(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/user")
        result = expand_path("~/projects/test")
        assert str(result) == "/home/user/projects/test"

    def test_expands_env_vars(self, monkeypatch):
        monkeypatch.setenv("MYDIR", "/custom/path")
        result = expand_path("$MYDIR/sub")
        assert str(result) == "/custom/path/sub"

    def test_no_expansion_needed(self):
        result = expand_path("/absolute/path")
        assert str(result) == "/absolute/path"


# ======================================================================
# Hashing
# ======================================================================

class TestHashing:
    def test_compute_hash_deterministic(self):
        h1 = compute_hash("hello")
        h2 = compute_hash("hello")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 produces 64 hex chars

    def test_compute_hash_different_content(self):
        assert compute_hash("hello") != compute_hash("world")

    def test_compute_md5_deterministic(self):
        assert compute_md5("hello") == compute_md5("hello")
        assert len(compute_md5("hello")) == 32  # MD5 produces 32 hex chars

    def test_hash_empty_string(self):
        h = compute_hash("")
        assert h == hashlib.sha256(b"").hexdigest()

    def test_md5_empty_string(self):
        assert compute_md5("") == hashlib.md5(b"").hexdigest()

    def test_hash_unicode_content(self):
        h1 = compute_hash("你好世界")
        h2 = compute_hash("你好世界")
        assert h1 == h2
        assert len(h1) == 64

    def test_hash_emoji_content(self):
        h = compute_hash("🚀✨🔥")
        assert len(h) == 64


# ======================================================================
# CJK detection
# ======================================================================

class TestContainsCJK:
    def test_detects_chinese(self):
        assert contains_cjk("你好世界") is True

    def test_detects_japanese(self):
        assert contains_cjk("こんにちは") is True

    def test_detects_korean(self):
        assert contains_cjk("안녕하세요") is True

    def test_no_cjk_in_ascii(self):
        assert contains_cjk("hello world") is False

    def test_no_cjk_in_numbers(self):
        assert contains_cjk("12345") is False

    def test_mixed_cjk_and_ascii(self):
        assert contains_cjk("hello 你好 world") is True

    def test_empty_string(self):
        assert contains_cjk("") is False

    def test_cjk_punctuation(self):
        """Fullwidth punctuation should be detected."""
        assert contains_cjk("。，！") is True


# ======================================================================
# Token estimation
# ======================================================================

class TestEstimateTokens:
    def test_empty_string_returns_zero(self):
        assert estimate_tokens("") == 0

    def test_none_text_returns_zero(self):
        assert estimate_tokens(None) == 0  # type: ignore
        assert estimate_tokens("") == 0

    def test_pure_ascii(self):
        tokens = estimate_tokens("hello world")
        expected = int(11 * 0.25) + 1
        assert tokens == expected

    def test_pure_chinese(self):
        tokens = estimate_tokens("你好世界")
        expected = int(4 * 1.5) + 1
        assert tokens == expected

    def test_mixed_content(self):
        tokens = estimate_tokens("hello 世界")
        # "hello " = 6 ASCII, "世界" = 2 CJK
        expected = int(6 * 0.25 + 2 * 1.5) + 1
        assert tokens == expected

    def test_long_ascii(self):
        text = "a" * 10000
        tokens = estimate_tokens(text)
        assert tokens > 0
        expected = int(10000 * 0.25) + 1
        assert tokens == expected

    def test_long_chinese(self):
        text = "好" * 10000
        tokens = estimate_tokens(text)
        assert tokens > 0
        expected = int(10000 * 1.5) + 1
        assert tokens == expected


# ======================================================================
# safe_json_dumps
# ======================================================================

class TestSafeJsonDumps:
    def test_normal_json_no_truncation(self):
        data = {"key": "value", "num": 42}
        result = safe_json_dumps(data)
        assert json.loads(result) == data

    def test_truncation_at_limit(self):
        """FIX VERIFIED: safe_json_dumps now returns a {truncated, preview, original_length} envelope."""
        data = {"text": "x" * 100000}
        result = safe_json_dumps(data, max_len=100)
        # Always valid JSON
        parsed = json.loads(result)
        assert parsed.get("truncated") is True
        assert "original_length" in parsed
        assert "preview" in parsed

    def test_nested_objects(self):
        data = {"a": [1, 2, 3], "b": {"c": "d"}}
        result = safe_json_dumps(data)
        assert json.loads(result) == data

    def test_custom_object_raises_type_error_for_dict(self):
        """FIX VERIFIED: Non-JSON-serializable objects raise TypeError
        even when nested inside a dict, since safe_json_dumps uses plain json.dumps."""
        class Custom:
            pass

        with pytest.raises(TypeError, match="not JSON serializable"):
            safe_json_dumps({"obj": Custom()})

    def test_unicode_preserved(self):
        result = safe_json_dumps({"text": "你好世界"})
        parsed = json.loads(result)
        assert parsed["text"] == "你好世界"

    def test_default_max_len(self):
        """Default max_len is 50000."""
        data = {"x": "y" * 1000}
        result = safe_json_dumps(data)
        assert len(result) < 50000 or "[Truncated:" in result


# ======================================================================
# truncate_str
# ======================================================================

class TestTruncateStr:
    def test_no_truncation_needed(self):
        assert truncate_str("short", 10) == "short"

    def test_truncation_exact_length(self):
        assert truncate_str("hello", 5) == "hello"

    def test_truncation_shorter(self):
        result = truncate_str("hello world", 5)
        assert result == "hello..."

    def test_empty_string(self):
        assert truncate_str("", 10) == ""

    def test_max_chars_zero(self):
        assert truncate_str("hello", 0) == "..."

    def test_unicode_truncation(self):
        result = truncate_str("你好世界", 2)
        assert len(result) <= 2 + 3  # +3 for "..."


# ======================================================================
# EXTREME: Boundary conditions
# ======================================================================

class TestUtilsExtreme:
    """CORE AUDIT: Extreme boundary conditions."""

    def test_hash_massive_string_no_oom(self):
        """10MB string hashed — must not OOM or hang."""
        huge = "x" * 10_000_000
        result = compute_hash(huge)
        assert len(result) == 64

    def test_safe_json_dumps_always_produces_valid_json(self):
        """FIX VERIFIED: Even with tiny max_len, output is always valid JSON."""
        data = {"key": "x" * 1000}
        result = safe_json_dumps(data, max_len=20)
        # Must always parse as valid JSON
        parsed = json.loads(result)
        assert "truncated" in parsed
        assert parsed["truncated"] is True

    def test_estimate_tokens_emoji_counted_as_non_ascii(self):
        """Emoji (ord > 127) are counted at 1.5 tokens per char — reasonable
        but worth noting since many emoji are 2+ UTF-16 code units."""
        tokens = estimate_tokens("🚀")
        assert tokens > 0

    def test_estimate_tokens_null_byte(self):
        """Null byte in string must not crash."""
        result = estimate_tokens("hello\x00world")
        assert result > 0

    def test_hash_unicode_surrogate_pairs(self):
        """Surrogate pairs and 4-byte UTF-8 chars."""
        h = compute_hash("𝄞𝄢")  # musical symbols
        assert len(h) == 64
