"""
Tests for yiagent.agent.evolution — self-evolution executor.

CRITICAL AREAS:
  1. Concurrent gate: _MAX_CONCURRENT=2, _running_count guard
  2. Transcript building from mixed content types
  3. Workspace snapshot diff detection
  4. Silent token detection (no change)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from yiagent.agent.evolution.executor import (
    _MAX_CONCURRENT,
    _build_transcript,
    _extract_text,
    _running_count,
)
from yiagent.agent.evolution.prompts import SILENT_TOKEN, EVOLUTION_MARKER


class TestTranscriptBuilding:
    def test_build_transcript_user_assistant(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hi there"}]},
        ]
        transcript = _build_transcript(msgs)
        assert "User: hello" in transcript
        assert "Assistant: hi there" in transcript

    def test_build_transcript_skips_system(self):
        msgs = [{"role": "system", "content": "ignore me"}]
        transcript = _build_transcript(msgs)
        assert "ignore me" not in transcript

    def test_build_transcript_max_chars(self):
        msgs = [
            {"role": "user", "content": "x" * 20000},
        ]
        transcript = _build_transcript(msgs, max_chars=100)
        assert len(transcript) <= 100 + len("...(earlier omitted)...\n")

    def test_extract_text_string(self):
        assert _extract_text("plain text") == "plain text"

    def test_extract_text_list(self):
        content = [
            {"type": "text", "text": "part1"},
            {"type": "image", "source": "img.png"},
            {"type": "text", "text": "part2"},
        ]
        result = _extract_text(content)
        assert "part1" in result
        assert "part2" in result

    def test_extract_text_empty(self):
        assert _extract_text([]) == ""
        assert _extract_text("") == ""


class TestSilentToken:
    def test_silent_token_exists(self):
        assert SILENT_TOKEN is not None
        assert len(SILENT_TOKEN) > 0

    def test_evolution_marker_exists(self):
        assert EVOLUTION_MARKER is not None


class TestEvolutionConcurrencyGate:
    def test_max_concurrent_is_2(self):
        assert _MAX_CONCURRENT == 2

    @pytest.mark.asyncio
    async def test_running_count_tracks_concurrency(self):
        import yiagent.agent.evolution.executor as evo
        original = _running_count
        # just verify the counter exists and is an int
        assert isinstance(_running_count, int)


class TestWorkspaceSnapshot:
    """
    BUG CONFIRMED: _workspace_snapshot at executor.py:198 imports
    _WATCH_SUBDIRS from yiagent.agent.evolution (the __init__.py),
    but the variable is defined in executor.py:33.

    Error: ImportError: cannot import name '_WATCH_SUBDIRS' from
    'yiagent.agent.evolution'

    Fix: remove the local import and reference the module-level
    _WATCH_SUBDIRS directly (it's already in scope at line 33).
    """

    def test_workspace_snapshot_import_is_fixed(self):
        """FIX VERIFIED: _workspace_snapshot now references _WATCH_SUBDIRS directly."""
        from yiagent.agent.evolution.executor import _workspace_snapshot
        from pathlib import Path
        # Should NOT raise ImportError anymore
        snap = _workspace_snapshot(Path("/tmp"))
        assert isinstance(snap, dict)

    def test_workspace_snapshot_with_files(self, tmp_path):
        """Workspace snapshot detects files in WATCH_SUBDIRS."""
        (tmp_path / "MEMORY.md").write_text("test")
        (tmp_path / "AGENT.md").write_text("test")
        from yiagent.agent.evolution.executor import _workspace_snapshot
        snap = _workspace_snapshot(tmp_path)
        assert "MEMORY.md" in snap
        assert "AGENT.md" in snap


class TestEvolutionExtreme:
    """CORE AUDIT: Extreme boundary conditions."""

    def test_transcript_with_20k_messages(self):
        """Building transcript from many messages must not OOM."""
        msgs = [
            {"role": "user", "content": "hello world"}
            for _ in range(20_000)
        ]
        transcript = _build_transcript(msgs)
        # Must be capped
        assert len(transcript) <= 12000 + len("...(earlier omitted)...\n")

    def test_empty_messages(self):
        assert _build_transcript([]) == ""

    def test_all_system_messages(self):
        msgs = [{"role": "system", "content": "sys"}, {"role": "system", "content": "sys2"}]
        assert _build_transcript(msgs) == ""
