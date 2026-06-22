"""
McpTool — wraps a remote MCP tool as a local BaseTool subclass.

The agent runtime sees McpTool exactly like any built-in tool;
dispatch goes through the MCP client's JSON-RPC transport.
"""

from __future__ import annotations

from typing import Any, Dict

from yiagent.agent.tools.base_tool import BaseTool, ToolResult
from yiagent.common.log import logger


class McpTool(BaseTool):
    """Wraps an MCP server tool so it looks native to the agent runtime."""

    def __init__(self, client, schema: Dict[str, Any], server_name: str):
        super().__init__()
        self.name = schema.get("name", "")
        self.description = schema.get("description", "")
        self.params = schema.get("inputSchema") or schema.get("input_schema") or {
            "type": "object",
            "properties": {},
        }
        self._client = client
        self.server_name = server_name

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Dispatch via MCP client JSON-RPC."""
        import asyncio as _asyncio
        loop = _asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None, self._client.call_tool, self.name, params
            )
            if result.get("error"):
                return ToolResult.fail(result["error"])
            content = result.get("content", [])
            # Extract text content from MCP response
            text_parts = []
            for item in content if isinstance(content, list) else [content]:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            return ToolResult.success("\n".join(text_parts) if text_parts else str(result))
        except Exception as e:
            logger.error(f"[McpTool] {self.name} call failed: {e}")
            return ToolResult.fail(str(e))
