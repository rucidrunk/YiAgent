"""
ToolManager — singleton registry for all available tools.

Design:
  - Built-in tools: registered via register_tool()
  - MCP tools: dynamically loaded from mcp.json, refreshed on change
  - sync_mcp_into_agent: reconciles agent's tool collection with live MCP registry
  - Hot-reload: detects mcp.json changes and restarts only affected servers
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from yiagent.common.config import conf
from yiagent.common.log import logger
from yiagent.agent.tools.base_tool import BaseTool


class ToolManager:
    """Singleton tool registry."""
    _instance: Optional["ToolManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ToolManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._tool_classes: Dict[str, Type[BaseTool]] = {}
                    cls._instance._tool_configs: Dict[str, Any] = {}
                    cls._instance._mcp_tool_instances: Dict[str, BaseTool] = {}
                    cls._instance._mcp_registry: Any = None
                    cls._instance._mcp_loaded = False
                    cls._instance._mcp_status: Dict[str, str] = {}
                    cls._instance._mcp_signature: Tuple[Optional[float], Optional[str]] = (None, None)
                    cls._instance._mcp_active_configs: Dict[str, dict] = {}
                    cls._instance._mcp_lock = threading.Lock()
        return cls._instance

    def register_tool(self, tool_cls: Type[BaseTool]) -> None:
        """Register a built-in tool class."""
        try:
            temp = tool_cls()
            self._tool_classes[temp.name] = tool_cls
            logger.debug(f"[ToolManager] Registered: {temp.name}")
        except Exception as e:
            logger.warning(f"[ToolManager] Failed to register {tool_cls.__name__}: {e}")

    def create_tool_instance(self, name: str) -> Optional[BaseTool]:
        """Create a fresh tool instance by name."""
        cls = self._tool_classes.get(name)
        if cls:
            tool = cls()
            if name in self._tool_configs:
                tool.config = self._tool_configs[name]
            return tool
        # Fall back to MCP
        return self._mcp_tool_instances.get(name)

    def create_all_instances(self) -> List[BaseTool]:
        """Create instances of all registered built-in tools."""
        instances = []
        for name, cls in self._tool_classes.items():
            try:
                tool = cls()
                if name in self._tool_configs:
                    tool.config = self._tool_configs[name]
                instances.append(tool)
            except Exception as e:
                logger.warning(f"[ToolManager] Failed to instantiate {name}: {e}")
        return instances

    def list_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas for all available tools (built-in + MCP)."""
        schemas = []
        for name, cls in self._tool_classes.items():
            try:
                schemas.append(cls().get_tool_definition())
            except Exception:
                pass
        for tool in self._mcp_tool_instances.values():
            schemas.append(tool.get_tool_definition())
        return schemas

    # ------------------------------------------------------------------
    # MCP integration
    # ------------------------------------------------------------------

    def _mcp_json_path(self) -> str:
        ws = conf().get("workspace_root", "~/yiagent")
        return os.path.join(os.path.expanduser(ws), "mcp.json")

    def _load_mcp_configs(self) -> list:
        """Load MCP server configs with mcp.json priority, fallback to config."""
        mcp_path = self._mcp_json_path()
        if os.path.exists(mcp_path):
            try:
                with open(mcp_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                raw = data.get("mcpServers") or data.get("mcp_servers") or []
                logger.info(f"[ToolManager] Loading MCP from {mcp_path}")
                return self._normalize_mcp(raw)
            except Exception as e:
                logger.warning(f"[ToolManager] mcp.json error: {e}")

        raw = conf().get("mcp_servers", [])
        return self._normalize_mcp(raw)

    @staticmethod
    def _normalize_mcp(raw) -> list:
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            result = []
            for name, cfg in raw.items():
                entry = {"name": name, **cfg}
                if "type" not in entry:
                    entry["type"] = "sse" if "url" in entry else "stdio"
                result.append(entry)
            return result
        return []

    def _read_mcp_signature(self) -> Tuple[Optional[float], Optional[str]]:
        path = self._mcp_json_path()
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return (None, None)
        try:
            with open(path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
        except OSError:
            return (mtime, None)
        return (mtime, digest)

    def load_mcp_tools(self) -> None:
        """Trigger MCP loading in a background thread. Idempotent, non-blocking."""
        with self._mcp_lock:
            if self._mcp_loaded:
                return
            configs = self._load_mcp_configs()
            self._mcp_signature = self._read_mcp_signature()
            self._mcp_active_configs = {
                cfg.get("name", "<unnamed>"): cfg for cfg in configs
            }
            if not configs:
                self._mcp_loaded = True
                return

            for cfg in configs:
                self._mcp_status[cfg.get("name", "<unnamed>")] = "pending"

            self._mcp_loaded = True
            threading.Thread(
                target=self._mcp_loader_thread, args=(configs,),
                daemon=True, name="mcp-loader",
            ).start()
            logger.info(f"[ToolManager] MCP loading started ({len(configs)} servers)")

    def refresh_mcp_if_changed(self) -> None:
        """Cheap check: re-read mcp.json; restart only changed servers."""
        with self._mcp_lock:
            new_sig = self._read_mcp_signature()
            if new_sig == self._mcp_signature:
                return
            try:
                new_configs = self._load_mcp_configs()
            except Exception as e:
                logger.warning(f"[ToolManager] MCP reload parse failed: {e}")
                return

            new_by_name = {c.get("name", "<unnamed>"): c for c in new_configs}
            old_by_name = self._mcp_active_configs

            added = [n for n in new_by_name if n not in old_by_name]
            removed = [n for n in old_by_name if n not in new_by_name]
            changed = [n for n in new_by_name if n in old_by_name and new_by_name[n] != old_by_name[n]]

            if not (added or removed or changed):
                self._mcp_signature = new_sig
                return

            logger.info(f"[ToolManager] mcp.json changed: +{added} -{removed} ~{changed}")

            for name in removed + changed:
                self._teardown_mcp_server(name)

            to_start = [new_by_name[n] for n in added + changed]
            if to_start:
                for cfg in to_start:
                    self._mcp_status[cfg.get("name", "<unnamed>")] = "pending"
                threading.Thread(
                    target=self._mcp_loader_thread, args=(to_start,),
                    daemon=True, name="mcp-loader-reload",
                ).start()

            self._mcp_active_configs = new_by_name
            self._mcp_signature = new_sig

    def _mcp_loader_thread(self, configs: list) -> None:
        """Background: bring up MCP servers one-by-one."""
        try:
            from yiagent.agent.tools.mcp.client import McpClient, McpClientRegistry
            from yiagent.agent.tools.mcp.tool import McpTool

            registry = McpClientRegistry()
            self._mcp_registry = registry

            for cfg in configs:
                name = cfg.get("name", "<unnamed>")
                try:
                    client = McpClient(cfg)
                    if not client.initialize():
                        self._mcp_status[name] = "failed"
                        logger.warning(f"[MCP] '{name}' init failed")
                        continue

                    schemas = client.list_tools()
                    added = []
                    for schema in schemas:
                        tool_name = schema.get("name", "")
                        if not tool_name:
                            continue
                        self._mcp_tool_instances[tool_name] = McpTool(client, schema, name)
                        added.append(tool_name)

                    with registry._registry_lock:
                        registry._clients[name] = client
                    self._mcp_status[name] = "ready"
                    logger.info(f"[MCP] '{name}' ready: {added}")
                except Exception as e:
                    self._mcp_status[name] = "failed"
                    logger.warning(f"[MCP] '{name}' failed: {e}")

            ready = sum(1 for s in self._mcp_status.values() if s == "ready")
            logger.info(f"[ToolManager] MCP: {ready}/{len(self._mcp_status)} ready, "
                        f"{len(self._mcp_tool_instances)} tools")
        except BaseException as e:
            # BaseException covers CancelledError (3.9+) and other non-Exception
            # throwables that would otherwise kill the daemon thread silently.
            logger.error(f"[ToolManager] MCP loader crashed: {e}", exc_info=True)

    def _teardown_mcp_server(self, name: str) -> None:
        if self._mcp_registry is None:
            return
        client = None
        with self._mcp_registry._registry_lock:
            client = self._mcp_registry._clients.pop(name, None)
        if client:
            client.shutdown()
        for tool_name in list(self._mcp_tool_instances):
            tool = self._mcp_tool_instances.get(tool_name)
            if tool is not None and getattr(tool, "server_name", None) == name:
                self._mcp_tool_instances.pop(tool_name, None)
        self._mcp_status.pop(name, None)

    def sync_mcp_into_agent(self, agent) -> Tuple[List[str], List[str]]:
        """Reconcile agent's tools with live MCP registry."""
        if agent is None or not hasattr(agent, "tools"):
            return ([], [])

        from yiagent.agent.tools.mcp.tool import McpTool
        current = self._mcp_tool_instances
        registry_names = set(current.keys())

        agent_tools = agent.tools
        if isinstance(agent_tools, dict):
            agent_mcp = {n for n, t in agent_tools.items() if isinstance(t, McpTool)}
            added = registry_names - agent_mcp
            removed = agent_mcp - registry_names
            for name in added:
                agent_tools[name] = current[name]
            for name in removed:
                agent_tools.pop(name, None)
        elif isinstance(agent_tools, list):
            agent_mcp = {t.name for t in agent_tools if isinstance(t, McpTool)}
            added = registry_names - agent_mcp
            removed = agent_mcp - registry_names
            if removed:
                agent.tools = [t for t in agent_tools if not (isinstance(t, McpTool) and t.name in removed)]
            for name in added:
                agent.tools.append(current[name])
        else:
            return ([], [])

        return (sorted(added), sorted(removed))

    def list_mcp_status(self) -> Dict[str, str]:
        return dict(self._mcp_status)

    def shutdown_mcp(self) -> None:
        if self._mcp_registry:
            self._mcp_registry.shutdown_all()
