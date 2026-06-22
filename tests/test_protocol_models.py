"""
Tests for yiagent.protocol.models — the core data models.

CRITICAL BUGS FOUND during audit:
  1. ContentBlock.to_dict() ignores the "name" field in tool_use — writes
     to key "name" but from_dict reads from "name" too. Wait, let me check:
     to_dict writes: d["name"] = self.tool_name
     from_dict reads: tool_name=d.get("name")
     OK this is consistent.

  2. Message.to_dict() does NOT include session_id, user_id, channel_type,
     receiver, timestamp — these are LOST on serialization. This means
     rehydrated messages from Redis/PG lose critical routing metadata.

  3. LLMRequest model has no validation. max_tokens can be -1, temperature
     can be 100.0, messages can be empty.
"""
from __future__ import annotations

import json

import pytest

from yiagent.protocol.models import (
    AgentAction,
    AgentActionType,
    AgentEvent,
    ContentBlock,
    ContentType,
    LLMModel,
    LLMRequest,
    Message,
    ToolResult,
)


# ======================================================================
# ContentBlock
# ======================================================================

class TestContentBlock:
    def test_text_block(self):
        block = ContentBlock.text_block("hello world")
        assert block.type == "text"
        assert block.text == "hello world"

    def test_image_block(self):
        block = ContentBlock.image_block(url="http://img.com/1.png", data="base64...")
        assert block.type == "image"
        assert block.source == {"url": "http://img.com/1.png", "data": "base64..."}

    def test_image_block_no_data(self):
        block = ContentBlock.image_block(url="http://img.com/1.png")
        assert block.source == {"url": "http://img.com/1.png"}

    def test_image_block_no_url(self):
        block = ContentBlock.image_block(data="base64...")
        assert block.source == {"data": "base64..."}

    def test_tool_use_block(self):
        block = ContentBlock.tool_use_block("id1", "search", {"q": "test"})
        assert block.type == "tool_use"
        assert block.tool_use_id == "id1"
        assert block.tool_name == "search"
        assert block.tool_input == {"q": "test"}

    def test_tool_result_block(self):
        block = ContentBlock.tool_result_block("id1", "results here", is_error=False)
        assert block.type == "tool_result"
        assert block.text == "results here"
        assert block.is_error is False

    def test_tool_result_block_error(self):
        block = ContentBlock.tool_result_block("id1", "crash", is_error=True)
        assert block.is_error is True

    def test_to_dict_from_dict_roundtrip(self):
        block = ContentBlock(
            type="tool_use",
            tool_use_id="t1",
            tool_name="search",
            tool_input={"q": "hello"},
        )
        d = block.to_dict()
        restored = ContentBlock.from_dict(d)
        assert restored.type == "tool_use"
        assert restored.tool_use_id == "t1"
        assert restored.tool_name == "search"
        assert restored.tool_input == {"q": "hello"}

    def test_to_dict_excludes_none_fields(self):
        block = ContentBlock.text_block("hello")
        d = block.to_dict()
        assert "type" in d
        assert "text" in d
        assert "source" not in d
        assert "tool_use_id" not in d
        assert "is_error" not in d

    def test_from_dict_with_minimal_data(self):
        block = ContentBlock.from_dict({"type": "text"})
        assert block.type == "text"
        assert block.text is None

    def test_from_dict_with_extra_keys(self):
        """Extra keys in dict must not crash from_dict."""
        block = ContentBlock.from_dict({"type": "text", "text": "hi", "unknown": 123})
        assert block.text == "hi"


# ======================================================================
# Message
# ======================================================================

class TestMessage:
    def test_text_content_extraction(self):
        msg = Message(
            role="user",
            content=[
                ContentBlock.text_block("part1"),
                ContentBlock.text_block("part2"),
            ],
        )
        assert msg.text_content == "part1\npart2"

    def test_text_content_skips_non_text(self):
        msg = Message(
            role="user",
            content=[
                ContentBlock.text_block("text"),
                ContentBlock.tool_use_block("id", "search", {}),
            ],
        )
        assert msg.text_content == "text"

    def test_is_visible_user_message_true(self):
        msg = Message(
            role="user",
            content=[ContentBlock.text_block("hello")],
        )
        assert msg.is_visible_user_message() is True

    def test_is_visible_user_message_false_for_assistant(self):
        msg = Message(
            role="assistant",
            content=[ContentBlock.text_block("hello")],
        )
        assert msg.is_visible_user_message() is False

    def test_is_visible_user_message_false_for_tool_result(self):
        msg = Message(
            role="user",
            content=[ContentBlock.tool_result_block("id", "result")],
        )
        assert msg.is_visible_user_message() is False

    def test_is_visible_user_message_false_mixed(self):
        """User message with both text AND tool_result is NOT visible."""
        msg = Message(
            role="user",
            content=[
                ContentBlock.text_block("question"),
                ContentBlock.tool_result_block("id", "result"),
            ],
        )
        assert msg.is_visible_user_message() is False

    def test_to_dict_from_dict_roundtrip(self):
        msg = Message(
            role="user",
            content=[ContentBlock.text_block("hello")],
            extras={"key": "value"},
        )
        d = msg.to_dict()
        restored = Message.from_dict(d)
        assert restored.role == "user"
        assert restored.extras == {"key": "value"}
        assert restored.content[0].text == "hello"

    def test_to_dict_preserves_routing_metadata(self):
        """FIX VERIFIED: to_dict() now serializes session_id, user_id, etc."""
        msg = Message(
            role="user",
            content=[ContentBlock.text_block("hi")],
            session_id="s1",
            user_id="u1",
            channel_type="web",
            receiver="channel_xyz",
            timestamp=1234567890.0,
        )
        d = msg.to_dict()
        assert d["session_id"] == "s1"
        assert d["user_id"] == "u1"
        assert d["channel_type"] == "web"
        assert d["receiver"] == "channel_xyz"
        assert d["timestamp"] == 1234567890.0

    def test_to_dict_roundtrip_preserves_routing_metadata(self):
        """Full round-trip must restore all routing fields."""
        msg = Message(
            role="user",
            content=[ContentBlock.text_block("hi")],
            session_id="s1",
            user_id="u1",
            channel_type="web",
            receiver="channel_xyz",
            timestamp=1234567890.0,
        )
        restored = Message.from_dict(msg.to_dict())
        assert restored.session_id == "s1"
        assert restored.user_id == "u1"
        assert restored.channel_type == "web"
        assert restored.receiver == "channel_xyz"
        assert restored.timestamp == 1234567890.0

    def test_to_dict_excludes_empty_extras(self):
        msg = Message(
            role="user",
            content=[ContentBlock.text_block("hi")],
            extras={},
        )
        d = msg.to_dict()
        assert "extras" not in d

    def test_from_dict_defaults(self):
        msg = Message.from_dict({"role": "system", "content": []})
        assert msg.role == "system"
        assert msg.extras == {}


# ======================================================================
# LLMRequest
# ======================================================================

class TestLLMRequest:
    def test_default_values(self):
        req = LLMRequest(messages=[{"role": "user", "content": "hello"}])
        assert req.temperature == 0.0
        assert req.stream is True
        assert req.model is None
        assert req.tools is None
        assert req.system is None
        assert req.extra == {}

    def test_custom_values(self):
        req = LLMRequest(
            messages=[],
            model="gpt-4o",
            temperature=0.7,
            max_tokens=4096,
            stream=False,
        )
        assert req.model == "gpt-4o"
        assert req.temperature == 0.7
        assert req.max_tokens == 4096
        assert req.stream is False

    def test_no_validation_on_temperature(self):
        """BUG: temperature=100.0 is accepted without validation."""
        req = LLMRequest(messages=[], temperature=100.0)
        assert req.temperature == 100.0  # Should ideally be 0.0-2.0

    def test_no_validation_on_negative_max_tokens(self):
        """BUG: max_tokens=-1 is accepted without validation."""
        req = LLMRequest(messages=[], max_tokens=-1)
        assert req.max_tokens == -1


# ======================================================================
# LLMModel
# ======================================================================

class TestLLMModel:
    def test_default_context_window(self):
        class DummyModel(LLMModel):
            async def call(self, request):
                return {}
            async def call_stream(self, request):
                yield ""

        model = DummyModel("test-model")
        assert model.estimate_context_window() == 128000

    def test_model_attributes(self):
        class DummyModel(LLMModel):
            async def call(self, request):
                return {}
            async def call_stream(self, request):
                yield ""

        model = DummyModel("my-model", api_key="sk-123")
        assert model.model == "my-model"
        assert model.config == {"api_key": "sk-123"}


# ======================================================================
# ToolResult
# ======================================================================

class TestToolResult:
    def test_success_result(self):
        tr = ToolResult(
            tool_name="search",
            status="success",
            result=["result1", "result2"],
            input_params={"q": "test"},
            execution_time=0.5,
        )
        assert tr.tool_name == "search"
        assert tr.status == "success"
        assert tr.result == ["result1", "result2"]

    def test_error_result(self):
        tr = ToolResult(
            tool_name="search",
            status="error",
            error_message="timeout",
        )
        assert tr.error_message == "timeout"


# ======================================================================
# AgentAction / AgentEvent
# ======================================================================

class TestAgentAction:
    def test_basic_action(self):
        action = AgentAction(
            agent_id="a1",
            agent_name="main",
            action_type=AgentActionType.TEXT_REPLY,
        )
        assert action.agent_id == "a1"
        assert action.action_type == AgentActionType.TEXT_REPLY

    def test_tool_use_action(self):
        tr = ToolResult(tool_name="search", status="success")
        action = AgentAction(
            agent_id="a1",
            agent_name="main",
            action_type=AgentActionType.TOOL_USE,
            tool_result=tr,
            thought="let me search...",
        )
        assert action.tool_result is tr
        assert action.thought == "let me search..."


class TestAgentEvent:
    def test_basic_event(self):
        event = AgentEvent(
            type="message_start",
            data={"session_id": "s1"},
            timestamp=1234567890.0,
        )
        assert event.type == "message_start"
        assert event.data == {"session_id": "s1"}

    def test_error_event(self):
        event = AgentEvent(
            type="error",
            data={"message": "connection lost", "code": 500},
        )
        assert event.type == "error"
        assert event.data["code"] == 500


# ======================================================================
# ContentType enum
# ======================================================================

class TestContentType:
    def test_all_values(self):
        assert ContentType.TEXT.value == "text"
        assert ContentType.IMAGE.value == "image"
        assert ContentType.AUDIO.value == "audio"
        assert ContentType.VIDEO.value == "video"
        assert ContentType.FILE.value == "file"


# ======================================================================
# AgentActionType enum
# ======================================================================

class TestAgentActionType:
    def test_all_values(self):
        assert AgentActionType.TOOL_USE.value == "tool_use"
        assert AgentActionType.TOOL_RESULT.value == "tool_result"
        assert AgentActionType.TEXT_REPLY.value == "text_reply"
        assert AgentActionType.ERROR.value == "error"


# ======================================================================
# EXTREME: Boundary conditions
# ======================================================================

class TestProtocolExtreme:
    """CORE AUDIT: Extreme boundary conditions."""

    def test_massive_message_serialization(self):
        """Message with 100K content blocks — must not OOM."""
        msg = Message(
            role="user",
            content=[ContentBlock.text_block(f"block{i}") for i in range(100_000)],
        )
        d = msg.to_dict()
        assert len(d["content"]) == 100_000

    def test_deeply_nested_tool_input(self):
        """Tool input with deeply nested dict — to_dict must handle it."""
        nested = {"level0": {"level1": {"level2": {"level3": {"level4": "deep"}}}}}
        block = ContentBlock.tool_use_block("id", "deep", nested)
        d = block.to_dict()
        restored = ContentBlock.from_dict(d)
        assert restored.tool_input["level0"]["level1"]["level2"]["level3"]["level4"] == "deep"

    def test_message_with_all_block_types(self):
        """Message containing every block type simultaneously."""
        msg = Message(
            role="user",
            content=[
                ContentBlock.text_block("text"),
                ContentBlock.image_block(url="http://x.com/img.png"),
                ContentBlock(type="audio", source={"url": "http://x.com/audio.mp3"}),
                ContentBlock(type="video", source={"url": "http://x.com/video.mp4"}),
                ContentBlock(type="file", source={"url": "http://x.com/file.pdf"}),
                ContentBlock.tool_use_block("t1", "search", {"q": "x"}),
                ContentBlock.tool_result_block("t1", "result"),
            ],
        )
        d = msg.to_dict()
        restored = Message.from_dict(d)
        assert len(restored.content) == 7

    def test_content_block_with_all_fields_set(self):
        """Block with every optional field set — roundtrip must preserve all."""
        block = ContentBlock(
            type="text",
            text="hello",
            source={"url": "http://x.com"},
            tool_use_id="t1",
            tool_name="search",
            tool_input={"q": "x"},
            is_error=False,
            thinking="hmm...",
        )
        d = block.to_dict()
        restored = ContentBlock.from_dict(d)
        assert restored.text == "hello"
        assert restored.source == {"url": "http://x.com"}
        assert restored.tool_name == "search"
        assert restored.thinking == "hmm..."

    def test_empty_content_block_list(self):
        msg = Message(role="system", content=[])
        assert msg.text_content == ""

    def test_text_content_with_none_text_block(self):
        """Block with type=text but text=None."""
        msg = Message(
            role="user",
            content=[ContentBlock(type="text", text=None)],
        )
        assert msg.text_content == ""

    def test_message_with_only_thinking_block(self):
        """Thinking block without text is not visible."""
        block = ContentBlock(type="thinking", thinking="hmm...")
        msg = Message(role="user", content=[block])
        # is_visible checks for type="text" blocks, not thinking
        assert msg.is_visible_user_message() is False
