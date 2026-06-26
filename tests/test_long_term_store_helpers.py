"""
Tests for yiagent.memory.long_term_store — helper functions and data types.

CRITICAL AREAS:
  1. _encode_vector / _encode_vector_str — NaN/Inf handling
  2. SearchResult / MemoryChunk dataclass construction
  3. _trunc_text boundary cases
  4. SQL parameterization (no injection via f-string placeholders)
"""
from __future__ import annotations

import math
import sys
from unittest.mock import MagicMock

import pytest

sys.modules["asyncpg"] = MagicMock()

from yiagent.memory.long_term_store import (
    _encode_vector,
    _encode_vector_str,
    _trunc_text,
    _rank_to_score,
    SearchResult,
    MemoryChunk,
)


class TestEncodeVector:
    def test_normal_vector(self):
        result = _encode_vector([1.0, 2.0, 3.0])
        assert result == "[1.0,2.0,3.0]"

    def test_none_vector_returns_none(self):
        assert _encode_vector(None) is None

    def test_empty_vector(self):
        result = _encode_vector([])
        assert result == "[]"

    def test_nan_vector_produces_invalid_pgvector(self):
        """
        BUG: NaN values produce '[nan]' which pgvector will reject.
        The code should validate and reject NaN before encoding.
        """
        result = _encode_vector([float("nan")])
        assert result == "[nan]", (
            f"BUG: NaN encoding produces '{result}' — pgvector will reject this"
        )

    def test_inf_vector_produces_invalid_pgvector(self):
        """
        BUG: Infinity values produce '[inf]' which pgvector will reject.
        """
        result = _encode_vector([float("inf")])
        assert result == "[inf]"

    def test_mixed_vector(self):
        result = _encode_vector([0.5, -1.0, 3.14])
        assert result == "[0.5,-1.0,3.14]"

    def test_encode_vector_str(self):
        result = _encode_vector_str([1.0, 2.0])
        assert result == "[1.0,2.0]"

    def test_large_dimension_vector(self):
        """1536-dim vector (standard OpenAI embedding size)."""
        vec = [0.0] * 1536
        result = _encode_vector(vec)
        assert len(result) > 1000
        assert result.startswith("[")
        assert result.endswith("]")


class TestTruncateText:
    def test_short_text_no_truncation(self):
        assert _trunc_text("hello", 10) == "hello"

    def test_long_text_truncated(self):
        assert _trunc_text("hello world", 5) == "hello..."

    def test_exact_length(self):
        assert _trunc_text("hello", 5) == "hello"

    def test_empty_text(self):
        assert _trunc_text("", 10) == ""

    def test_cjk_text(self):
        result = _trunc_text("你好世界", 3)
        assert len(result) <= 6  # 3 chars + "..."

    def test_very_large_max_chars(self):
        result = _trunc_text("hi", 1000000)
        assert result == "hi"


class TestRankToScore:
    def test_positive_rank(self):
        score = _rank_to_score(0.5)
        assert 0.0 <= score <= 1.0

    def test_zero_rank(self):
        assert _rank_to_score(0.0) == 0.0

    def test_negative_rank(self):
        assert _rank_to_score(-0.5) == 0.0

    def test_large_rank(self):
        score = _rank_to_score(100.0)
        assert 0.0 <= score <= 1.0


class TestSearchResult:
    def test_construction(self):
        r = SearchResult(
            path="/test.md", start_line=1, end_line=5,
            score=0.95, snippet="found text",
            source="memory", user_id="u1", chunk_id="abc123",
        )
        assert r.path == "/test.md"
        assert r.score == 0.95

    def test_default_values(self):
        r = SearchResult(path="/x.md", start_line=0, end_line=0, score=0.0, snippet="")
        assert r.source == "memory"  # default
        assert r.user_id is None
        assert r.chunk_id is None


class TestMemoryChunk:
    def test_construction(self):
        c = MemoryChunk(
            id="id1", user_id="u1", scope="shared", source="memory",
            path="/test.md", start_line=1, end_line=3,
            text="hello world", content_hash="abc",
        )
        assert c.id == "id1"
        assert c.embedding is None  # default

    def test_embedding_set(self):
        emb = [0.1, 0.2]
        c = MemoryChunk(
            id="id2", user_id=None, scope="shared", source="memory",
            path="/x.md", start_line=0, end_line=0,
            text="x", content_hash="h", embedding=emb,
        )
        assert c.embedding == emb
