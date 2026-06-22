"""
Core data models for the agent protocol.

Defines the unified Message format, LLM request/response types,
tool results, and agent action tracking — all as plain dataclasses
for zero-overhead serialisation across the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Unified Message Protocol (multi-modal ready)
# ---------------------------------------------------------------------------

class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"


@dataclass
class ContentBlock:
    """A single content block inside a message."""
    type: str  # "text" | "image" | "audio" | "video" | "file" | "tool_use" | "tool_result"
    text: Optional[str] = None
    source: Optional[Dict[str, str]] = None  # {"url": ..., "data": ...}
    tool_use_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    is_error: Optional[bool] = None
    thinking: Optional[str] = None  # for reasoning/thinking blocks

    @classmethod
    def text_block(cls, text: str) -> "ContentBlock":
        return cls(type="text", text=text)

    @classmethod
    def image_block(cls, url: Optional[str] = None, data: Optional[str] = None) -> "ContentBlock":
        source = {}
        if url:
            source["url"] = url
        if data:
            source["data"] = data
        return cls(type="image", source=source)

    @classmethod
    def tool_use_block(cls, tool_id: str, name: str, input: dict) -> "ContentBlock":
        return cls(type="tool_use", tool_use_id=tool_id, tool_name=name, tool_input=input)

    @classmethod
    def tool_result_block(cls, tool_id: str, content: str, is_error: bool = False) -> "ContentBlock":
        return cls(type="tool_result", tool_use_id=tool_id, text=content, is_error=is_error)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"type": self.type}
        if self.text is not None:
            d["text"] = self.text
        if self.source is not None:
            d["source"] = self.source
        if self.tool_use_id is not None:
            d["tool_use_id"] = self.tool_use_id
        if self.tool_name is not None:
            d["name"] = self.tool_name
        if self.tool_input is not None:
            d["input"] = self.tool_input
        if self.is_error is not None:
            d["is_error"] = self.is_error
        if self.thinking is not None:
            d["thinking"] = self.thinking
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ContentBlock":
        return cls(
            type=d.get("type", "text"),
            text=d.get("text"),
            source=d.get("source"),
            tool_use_id=d.get("tool_use_id"),
            tool_name=d.get("name"),
            tool_input=d.get("input"),
            is_error=d.get("is_error"),
            thinking=d.get("thinking"),
        )


@dataclass
class Message:
    """Unified message envelope. All channels normalise to this format."""
    role: str  # "user" | "assistant" | "system"
    content: List[ContentBlock] = field(default_factory=list)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    channel_type: Optional[str] = None  # "web" | "feishu" | "slack" | "rest"
    receiver: Optional[str] = None
    timestamp: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def text_content(self) -> str:
        """Extract plain text from content blocks."""
        parts = []
        for b in self.content:
            if b.type == "text" and b.text:
                parts.append(b.text)
        return "\n".join(parts)

    def is_visible_user_message(self) -> bool:
        """True if this is real user input (not tool_result injection)."""
        if self.role != "user":
            return False
        has_text = any(b.type == "text" and b.text for b in self.content)
        has_tool = any(b.type == "tool_result" for b in self.content)
        return has_text and not has_tool

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "role": self.role,
            "content": [b.to_dict() for b in self.content],
        }
        # Preserve routing metadata so round-trip (Redis/PG) never loses state
        if self.session_id is not None:
            d["session_id"] = self.session_id
        if self.user_id is not None:
            d["user_id"] = self.user_id
        if self.channel_type is not None:
            d["channel_type"] = self.channel_type
        if self.receiver is not None:
            d["receiver"] = self.receiver
        if self.timestamp is not None:
            d["timestamp"] = self.timestamp
        if self.extras:
            d["extras"] = self.extras
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        content = [ContentBlock.from_dict(b) for b in d.get("content", [])]
        return cls(
            role=d["role"],
            content=content,
            session_id=d.get("session_id"),
            user_id=d.get("user_id"),
            channel_type=d.get("channel_type"),
            receiver=d.get("receiver"),
            timestamp=d.get("timestamp"),
            extras=d.get("extras", {}),
        )


# ---------------------------------------------------------------------------
# LLM types
# ---------------------------------------------------------------------------

@dataclass
class LLMRequest:
    messages: List[Dict[str, Any]]
    model: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    stream: bool = True
    tools: Optional[List[Dict[str, Any]]] = None
    system: Optional[str] = None

    # Extra provider-specific kwargs
    extra: Dict[str, Any] = field(default_factory=dict)


class LLMModel:
    """Abstract base for model providers. Subclass to add a vendor."""

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.config = kwargs
        self.channel_type: str = ""
        self.session_id: str = ""

    async def call(self, request: LLMRequest) -> Dict[str, Any]:
        raise NotImplementedError

    async def call_stream(self, request: LLMRequest):
        """Async generator yielding chunks."""
        raise NotImplementedError

    def estimate_context_window(self) -> int:
        """Return the model's context window in tokens."""
        return 128000


# ---------------------------------------------------------------------------
# Tool / Action types
# ---------------------------------------------------------------------------

class AgentActionType(str, Enum):
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    TEXT_REPLY = "text_reply"
    ERROR = "error"


@dataclass
class ToolResult:
    tool_name: str
    status: str  # "success" | "error" | "critical_error"
    result: Any = None
    input_params: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None


@dataclass
class AgentAction:
    agent_id: str
    agent_name: str
    action_type: AgentActionType
    tool_result: Optional[ToolResult] = None
    thought: Optional[str] = None
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Event system (for streaming callbacks)
# ---------------------------------------------------------------------------

@dataclass
class AgentEvent:
    type: str  # "message_start" | "message_update" | "message_end" | "tool_execution_start" | "tool_execution_end" | "turn_start" | "turn_end" | "agent_start" | "agent_end" | "error"
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
