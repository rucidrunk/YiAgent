"""
Base tool class — all tools inherit from this.

Design:
  - Tools can be PRE_PROCESS (agent-explicit call) or POST_PROCESS (auto-run)
  - execute_tool() is the standard entry; wraps execute() with error handling
  - get_json_schema() returns OpenAI-compatible function schema
  - Pluggable: register new tools by subclassing and adding to ToolManager
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class ToolStage(str, Enum):
    PRE_PROCESS = "pre_process"
    POST_PROCESS = "post_process"


@dataclass
class ToolResult:
    status: str  # "success" | "error" | "critical_error"
    result: Any = None

    @classmethod
    def success(cls, result: Any) -> "ToolResult":
        return cls(status="success", result=result)

    @classmethod
    def fail(cls, result: Any) -> "ToolResult":
        return cls(status="error", result=result)

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "result": self.result}


class BaseTool:
    """Abstract base for all tools."""

    stage: ToolStage = ToolStage.PRE_PROCESS

    # Must be overridden by subclasses
    name: str = "base_tool"
    description: str = "Base tool (override this)"

    # JSON Schema for the tool's parameters (OpenAI function-calling format)
    params: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self):
        self.model: Any = None       # LLM model reference (set by agent)
        self.context: Any = None     # Agent context reference
        self.config: Optional[Dict[str, Any]] = None
        self.progress_callback = None

    def get_json_schema(self) -> Dict[str, Any]:
        """Return OpenAI-compatible function schema. Override for dynamic schemas."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.params,
            },
        }

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Claude-compatible tool definition."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.params,
        }

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Override in subclasses with actual tool logic."""
        raise NotImplementedError(f"Tool '{self.name}' has no execute() implementation")

    async def execute_tool(self, params: Dict[str, Any]) -> ToolResult:
        """Standard entry point with error handling. Called by agent runtime."""
        try:
            return await self.execute(params)
        except Exception as e:
            from yiagent.common.log import logger
            logger.error(f"[{self.name}] Tool execution error: {e}")
            return ToolResult.fail(str(e))

    def should_auto_execute(self, context) -> bool:
        """Return True for POST_PROCESS tools that auto-run after agent loop."""
        return self.stage == ToolStage.POST_PROCESS

    async def close(self) -> None:
        """Release resources. Override in subclasses if needed."""
        pass
