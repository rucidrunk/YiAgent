"""
MCP Client — stdio/SSE JSON-RPC 2.0 client for Model Context Protocol.

Connects to MCP servers (local subprocess via stdio, or remote via SSE),
enumerates their tools, and dispatches tool calls with JSON-RPC 2.0 framing.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

import httpx

from yiagent.common.log import logger


class McpClient:
    """
    MCP client wrapping one server connection.

    Transport:
      - stdio: spawns the configured command as a subprocess
      - SSE: connects to a remote HTTP endpoint
    Protocol: JSON-RPC 2.0
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._name: str = config.get("name", "<unnamed>")
        self._transport: str = config.get("type", "stdio")
        self._process: Optional[subprocess.Popen] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._url: str = config.get("url", "")
        self._tools: List[Dict[str, Any]] = []
        self._next_id: int = 1
        self._lock = threading.Lock()

    def initialize(self) -> bool:
        """Connect and handshake. Returns True on success."""
        try:
            if self._transport == "sse":
                return self._init_sse()
            else:
                return self._init_stdio()
        except Exception as e:
            logger.error(f"[MCP] {self._name}: init failed: {e}")
            return False

    def _init_stdio(self) -> bool:
        command = self._config.get("command", "")
        args = self._config.get("args", [])
        env = self._config.get("env", {})
        env_vars = {**__import__("os").environ, **env}

        self._process = subprocess.Popen(
            [command] + args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env_vars,
            text=True,
        )

        # Handshake: send initialize
        self._send_stdio(json.dumps({
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "YiAgent", "version": "0.1.0"},
            },
            "id": self._next_id,
        }))
        self._next_id += 1

        # Read response (simple line-based)
        line = self._process.stdout.readline()
        if not line:
            logger.warning(f"[MCP] {self._name}: no response from initialize")
            return False

        try:
            resp = json.loads(line)
            if resp.get("error"):
                logger.warning(f"[MCP] {self._name}: init error: {resp['error']}")
                return False
        except json.JSONDecodeError:
            logger.warning(f"[MCP] {self._name}: invalid JSON from initialize")
            return False

        logger.info(f"[MCP] {self._name}: connected via stdio")
        return True

    def _init_sse(self) -> bool:
        # Note: SSE transport needs async event loop; simplified here for sync init check.
        logger.info(f"[MCP] {self._name}: SSE endpoint at {self._url}")
        return True

    def list_tools(self) -> List[Dict[str, Any]]:
        """Enumerate tools from the server."""
        if self._transport == "sse":
            return self._list_tools_sse()

        self._send_stdio(json.dumps({
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": self._next_id,
        }))
        self._next_id += 1

        line = self._process.stdout.readline()
        if not line:
            return []
        try:
            resp = json.loads(line)
            return resp.get("result", {}).get("tools", [])
        except (json.JSONDecodeError, KeyError):
            return []

    def _list_tools_sse(self) -> List[Dict[str, Any]]:
        # Simplified: return cached
        return self._tools

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous tool call. Returns {result/error}."""
        if self._transport == "sse":
            return {"error": "SSE tool calls not yet implemented"}

        with self._lock:
            self._send_stdio(json.dumps({
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
                "id": self._next_id,
            }))
            req_id = self._next_id
            self._next_id += 1

        line = self._process.stdout.readline()
        if not line:
            return {"error": "No response from MCP server"}

        try:
            resp = json.loads(line)
            if resp.get("error"):
                return {"error": resp["error"].get("message", str(resp["error"]))}
            return resp.get("result", {})
        except json.JSONDecodeError as e:
            return {"error": str(e)}

    def _send_stdio(self, payload: str) -> None:
        if self._process and self._process.stdin:
            self._process.stdin.write(payload + "\n")
            self._process.stdin.flush()

    def shutdown(self) -> None:
        """Terminate the server process."""
        if self._process:
            try:
                self._process.stdin.close()
                self._process.stdout.close()
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

    @property
    def name(self) -> str:
        return self._name


class McpClientRegistry:
    """Thread-safe registry of MCP clients."""

    def __init__(self):
        self._clients: Dict[str, McpClient] = {}
        self._registry_lock = threading.Lock()

    def shutdown_all(self) -> None:
        with self._registry_lock:
            for name, client in list(self._clients.items()):
                try:
                    client.shutdown()
                except Exception:
                    pass
            self._clients.clear()
