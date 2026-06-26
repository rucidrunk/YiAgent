"""
AgentRuntime — stateless agent executor with distributed lock.

Design:
  - Any pod can serve any session; all state is in Redis + PostgreSQL
  - Distributed lock via Redis SET NX EX prevents concurrent runs per session
  - Tool-call loop with streaming, max_steps guard, cancel support
  - Event system for real-time UI updates
  - Context trimming + memory flush on overflow
  - Post-process tools auto-execute after loop completion

Patterns borrowed from CowAgent's Agent + AgentStreamExecutor:
  - Visible-turn counting (only user text, not tool_result)
  - Consecutive failure detection to break tool loops
  - Context overflow recovery with aggressive trimming
  - Message history sync between executor ↔ agent
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import deque
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set, Tuple

from yiagent.common.config import conf
from yiagent.common.log import logger
from yiagent.protocol.models import (
    AgentAction, AgentActionType, AgentEvent, ContentBlock,
    LLMRequest, LLMModel, Message, ToolResult as ProtoToolResult,
)
from yiagent.agent.tools.base_tool import BaseTool, ToolResult
from yiagent.agent.tools.tool_manager import ToolManager
from yiagent.agent.tools.dry_run import DryRunInterceptor, DryRunResult
from yiagent.agent.prompt.builder import build_system_prompt
from yiagent.memory.manager import MemoryManager
from yiagent.memory.conversation_store import get_conversation_store


# Write tools that must go through DryRunInterceptor
_WRITE_TOOLS: Set[str] = {"write", "edit", "bash"}


class SessionBusyError(Exception):
    """Raised when a session's distributed lock is already held."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Session {session_id} is already running")


class AgentRuntime:
    """
    Stateless agent. Created per-session, loaded from Redis.

    Usage:
        runtime = AgentRuntime(session_id="user_x:web", model=model, tools=tools)
        async for event in runtime.handle_message("Hello"):
            # SSE stream to client
            pass
    """

    def __init__(
        self,
        session_id: str,
        model: LLMModel,
        tools: Optional[List[BaseTool]] = None,
        description: str = "YiAgent",
        max_steps: int = 50,
        max_context_turns: int = 30,
        max_context_tokens: Optional[int] = None,
        workspace_dir: Optional[str] = None,
        memory_manager: Optional[MemoryManager] = None,
        enable_skills: bool = True,
        skill_manager: Any = None,
        runtime_info: Optional[Dict[str, Any]] = None,
    ):
        self.session_id = session_id
        self.model = model
        self.description = description
        self.max_steps = max_steps
        self.max_context_turns = max_context_turns
        self.max_context_tokens = max_context_tokens
        self.workspace_dir = workspace_dir or conf().get("workspace_root", "~/yiagent")
        self.memory_manager = memory_manager
        self.runtime_info = runtime_info or {}
        self.enable_skills = enable_skills
        self.skill_manager = skill_manager

        # Tools
        self.tools: List[BaseTool] = tools or []
        self._tool_map: Dict[str, BaseTool] = {t.name: t for t in self.tools}

        # Message history (in-memory copy, synced to Redis)
        self.messages: List[Dict[str, Any]] = []
        self._messages_lock = asyncio.Lock()
        self.captured_actions: List[AgentAction] = []

        # Evolution state
        self._evo_last_active: float = 0.0
        self._evo_turns: int = 0
        self._evo_done_msg_count: int = 0
        self._evo_run_active: bool = False
        self._evo_channel_type: str = ""
        self._evo_receiver: str = ""

        # Dry-run interceptor
        self._dry_run_interceptor = DryRunInterceptor(workspace_root=self.workspace_dir)

        # System prompt cache (rebuilt on demand)
        self._system_prompt: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle_message(
        self,
        message: Message,
        on_event: Optional[Callable[[AgentEvent], Any]] = None,
        skill_filter: Optional[List[str]] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> str:
        """
        Handle an incoming message. Returns the final assistant response text.

        This is the main entry point for the agent run loop.
        """
        cfg = conf()

        # 1. Acquire distributed lock (degrade gracefully if Redis is down)
        store = await get_conversation_store()
        try:
            acquired = await store.acquire_lock(self.session_id)
        except Exception as redis_err:
            logger.warning(f"[AgentRuntime] Redis lock unavailable, proceeding without: {redis_err}")
            acquired = True  # degraded mode — proceed without lock
        if not acquired:
            raise SessionBusyError(self.session_id)

        try:
            # 2. Load context from Redis — keep in-memory messages if Redis is down
            async with self._messages_lock:
                loaded = await store.load_context(
                    self.session_id,
                    max_turns=self.max_context_turns,
                )
                # Only replace if Redis actually returned data; otherwise preserve
                # in-memory context so the agent remembers this session's history.
                if loaded:
                    self.messages = loaded
                original_length = len(self.messages)

            # 3. Append user message
            user_msg_dict = message.to_dict()
            async with self._messages_lock:
                self.messages.append(user_msg_dict)

            # 4. Build system prompt (lazy, from disk)
            system_prompt = await self._build_system_prompt(skill_filter)

            # 5. Run tool-call loop
            executor = AgentStreamExecutor(
                runtime=self,
                model=self.model,
                system_prompt=system_prompt,
                tools=self.tools,
                max_turns=self.max_steps,
                on_event=on_event,
                messages=list(self.messages),
                max_context_turns=self.max_context_turns,
                cancel_event=cancel_event,
            )

            try:
                response = await executor.run_stream(message.text_content)
            except Exception:
                # Sync cleared messages back
                if len(executor.messages) == 0:
                    async with self._messages_lock:
                        self.messages.clear()
                        logger.info(f"[AgentRuntime] Cleared messages after executor recovery")
                raise

            # 6. Sync executor messages back
            async with self._messages_lock:
                self.messages = list(executor.messages)

            # 7. Persist to PG (async, non-blocking, tracked for error logging)
            new_msgs = self.messages[original_length:]
            if new_msgs:
                from yiagent.memory.conversation_store import _bg_tracker
                _bg_tracker.spawn(
                    store.append_messages_batch(
                        self.session_id, new_msgs, message.channel_type or "",
                        message.user_id or "",
                    )
                )

            # 8. Context pressure check (tracked background task)
            est_tokens = sum(self._estimate_message_tokens(m) for m in self.messages)
            if self.memory_manager:
                budget = self.max_context_tokens or self._get_model_context_window()
                from yiagent.memory.conversation_store import _bg_tracker
                _bg_tracker.spawn(
                    self.memory_manager.monitor_and_flush(
                        session_id=self.session_id,
                        token_estimate=est_tokens,
                        max_tokens=budget,
                        messages=list(self.messages),
                        user_id=message.user_id,
                    )
                )

            # 9. Run post-process tools
            await self._execute_post_process_tools()

            # 10. Note evolution turn
            self._evo_last_active = time.time()
            self._evo_turns += 1

            return response

        finally:
            await store.release_lock(self.session_id)

    async def _execute_post_process_tools(self) -> None:
        """Auto-execute POST_PROCESS stage tools."""
        for tool in list(self._tool_map.values()):
            if tool.stage.name == "POST_PROCESS":
                try:
                    tool.context = self
                    await tool.execute_tool({})
                except Exception as e:
                    logger.warning(f"[AgentRuntime] Post-process tool {tool.name} failed: {e}")

    async def _build_system_prompt(self, skill_filter: Optional[List[str]] = None) -> str:
        """Build system prompt from disk (always fresh)."""
        try:
            return build_system_prompt(
                workspace_dir=self.workspace_dir,
                tools=self.tools,
                skill_manager=self.skill_manager,
                memory_manager=self.memory_manager,
                runtime_info=self.runtime_info,
                skill_filter=skill_filter,
            )
        except Exception as e:
            logger.warning(f"[AgentRuntime] Prompt build failed, using cached: {e}")
            return self._system_prompt or "You are a helpful AI assistant."

    def _get_model_context_window(self) -> int:
        return getattr(self.model, "estimate_context_window", lambda: 128000)()

    def _estimate_message_tokens(self, message: Dict[str, Any]) -> int:
        """Estimate token count for a message dict."""
        from yiagent.common.utils import estimate_tokens
        content = message.get("content", "")
        if isinstance(content, str):
            return max(1, estimate_tokens(content))
        if isinstance(content, list):
            total = 0
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        total += estimate_tokens(block.get("text", ""))
                    elif block.get("type") == "image":
                        total += 1200
                    elif block.get("type") == "tool_use":
                        total += 50 + estimate_tokens(json.dumps(block.get("input", {}), ensure_ascii=False))
                    elif block.get("type") == "tool_result":
                        total += 30 + estimate_tokens(str(block.get("content", "")))
            return max(1, total)
        return 1

    # ------------------------------------------------------------------
    # Tool management
    # ------------------------------------------------------------------

    def add_tool(self, tool: BaseTool) -> None:
        tool.model = self.model
        self.tools.append(tool)
        self._tool_map[tool.name] = tool

    def find_tool(self, name: str) -> Optional[BaseTool]:
        return self._tool_map.get(name)

    async def refresh_skills(self) -> None:
        if self.skill_manager:
            self.skill_manager.refresh_skills()


# ---------------------------------------------------------------------------
# AgentStreamExecutor — tool-call loop
# ---------------------------------------------------------------------------

class AgentStreamExecutor:
    """Handles the multi-turn tool-call loop with streaming and event emission."""

    def __init__(
        self,
        runtime: AgentRuntime,
        model: LLMModel,
        system_prompt: str,
        tools: List[BaseTool],
        max_turns: int = 50,
        on_event: Optional[Callable[[AgentEvent], Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        max_context_turns: int = 30,
        cancel_event: Optional[asyncio.Event] = None,
    ):
        self.runtime = runtime
        self.model = model
        self.system_prompt = system_prompt
        self.tools: Dict[str, BaseTool] = {t.name: t for t in tools}
        self.max_turns = max_turns
        self.on_event = on_event
        self.max_context_turns = max_context_turns
        self.cancel_event = cancel_event
        self.messages: List[Dict[str, Any]] = messages or []
        self.tool_failure_history: deque = deque(maxlen=50)
        self.files_to_send: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run_stream(self, user_message: str) -> str:
        """Execute the tool-call loop. Returns final assistant text."""
        # Append user message
        self.messages.append({
            "role": "user",
            "content": [{"type": "text", "text": user_message}],
        })

        # Trim context before loop
        self._trim_messages()

        await self._emit("agent_start", {})

        final_response = ""
        turn = 0

        try:
            while turn < self.max_turns:
                await self._check_cancelled()

                turn += 1
                logger.info(f"[Agent] Turn {turn}")
                await self._emit("turn_start", {"turn": turn})

                assistant_text, tool_calls = await self._call_llm_stream(retry_on_empty=True)
                final_response = assistant_text

                if not tool_calls:
                    if not assistant_text:
                        final_response = await self._handle_empty_response(turn)
                    await self._emit("turn_end", {"turn": turn, "has_tool_calls": False})
                    break

                # Execute tools
                tool_result_blocks = []
                for tool_call in tool_calls:
                    await self._check_cancelled()
                    result = await self._execute_tool(tool_call)
                    if result.get("status") == "critical_error":
                        final_response = result.get("result", "Task execution failed")
                        return final_response

                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call["id"],
                        "content": self._format_tool_result(result),
                        "is_error": result.get("status") == "error",
                    })

                if tool_result_blocks:
                    self.messages.append({"role": "user", "content": tool_result_blocks})

                await self._emit("turn_end", {
                    "turn": turn,
                    "has_tool_calls": True,
                    "tool_count": len(tool_calls),
                })

            if turn >= self.max_turns:
                logger.warning(f"[Agent] Max steps reached: {self.max_turns}")
                final_response = await self._force_summary(turn)

        except asyncio.CancelledError:
            final_response = "(Cancelled)"
            await self._emit("error", {"error": "cancelled"})

        except Exception as e:
            logger.error(f"[Agent] Execution error: {e!r}", exc_info=True)
            await self._emit("error", {"error": f"{type(e).__name__}: {e}"})
            raise

        finally:
            await self._emit("agent_end", {"final_response": final_response.strip()})

        return final_response.strip()

    # ------------------------------------------------------------------
    # LLM Call
    # ------------------------------------------------------------------

    async def _call_llm_stream(
        self, retry_on_empty: bool = True, retry_count: int = 0, max_retries: int = 3,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Stream LLM response and parse tool calls."""
        # Build tool schemas
        tools_schema = None
        if self.tools:
            tools_schema = [t.get_tool_definition() for t in self.tools.values()]

        request = LLMRequest(
            messages=list(self.messages),
            temperature=conf().get("model_temperature", 0.0),
            stream=True,
            tools=tools_schema,
            system=self.system_prompt,
        )

        await self._emit("message_start", {"role": "assistant"})

        full_content = ""
        full_reasoning = ""
        tool_calls_buffer: Dict[int, Dict[str, Any]] = {}

        try:
            async for chunk in self.model.call_stream(request):
                await self._check_cancelled()

                if isinstance(chunk, dict):
                    if chunk.get("error"):
                        raise RuntimeError(chunk["error"].get("message", str(chunk["error"])))

                    if chunk.get("choices"):
                        delta = chunk["choices"][0].get("delta", {})

                        # Reasoning
                        reasoning = delta.get("reasoning_content", "")
                        if reasoning:
                            full_reasoning += reasoning
                            await self._emit("reasoning_update", {"delta": reasoning})

                        # Text content
                        content_delta = delta.get("content", "")
                        if content_delta:
                            full_content += content_delta
                            await self._emit("message_update", {"delta": content_delta})

                        # Tool calls
                        for tc in delta.get("tool_calls", []):
                            idx = tc.get("index", 0)
                            if idx not in tool_calls_buffer:
                                tool_calls_buffer[idx] = {"id": "", "name": "", "arguments": ""}
                            if tc.get("id"):
                                tool_calls_buffer[idx]["id"] = tc["id"]
                            func = tc.get("function", {})
                            if func.get("name"):
                                tool_calls_buffer[idx]["name"] = func["name"]
                            if func.get("arguments"):
                                tool_calls_buffer[idx]["arguments"] += func["arguments"]

        except asyncio.CancelledError:
            raise
        except Exception as e:
            error_str = str(e).lower()
            error_type = type(e).__name__.lower()
            is_retryable = any(kw in error_str for kw in [
                "timeout", "connection", "connect", "rate limit", "overloaded", "429", "500", "502", "503",
            ]) or any(kw in error_type for kw in [
                "timeout", "connect", "read", "write", "remote",
            ])
            if is_retryable and retry_count < max_retries:
                wait = (retry_count + 1) * 2
                logger.warning(f"[Agent] LLM error, retrying in {wait}s: {e}")
                await asyncio.sleep(wait)
                return await self._call_llm_stream(retry_on_empty, retry_count + 1, max_retries)
            raise

        # Parse tool calls
        tool_calls = []
        for idx in sorted(tool_calls_buffer.keys()):
            tc = tool_calls_buffer[idx]
            args_str = tc.get("arguments", "")
            try:
                args = json.loads(args_str) if args_str.strip() else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({"id": tc["id"] or f"call_{hashlib.md5(args_str.encode()).hexdigest()[:24]}",
                               "name": tc["name"], "arguments": args})

        # Build assistant message
        assistant_msg = {"role": "assistant", "content": []}
        if full_reasoning:
            assistant_msg["content"].append({"type": "thinking", "thinking": full_reasoning[:4096]})
        if full_content:
            assistant_msg["content"].append({"type": "text", "text": full_content})
        for tc in tool_calls:
            assistant_msg["content"].append({"type": "tool_use", "id": tc["id"],
                                              "name": tc["name"], "input": tc["arguments"]})

        if assistant_msg["content"]:
            self.messages.append(assistant_msg)

        await self._emit("message_end", {"content": full_content, "tool_calls": tool_calls})

        return full_content, tool_calls

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        name = tool_call["name"]
        args = tool_call["arguments"]
        tool_id = tool_call["id"]

        # Check for tool loops
        should_stop, reason, is_critical = self._check_consecutive_failures(name, args)
        if should_stop:
            self._record_tool(name, args, False)
            if is_critical:
                return {"status": "critical_error", "result": reason}
            return {"status": "error", "result": reason}

        tool = self.tools.get(name)
        if not tool:
            return {"status": "error", "result": f"Tool '{name}' not found. Available: {list(self.tools)}"}

        # Dry-run for write tools
        if name in _WRITE_TOOLS:
            interceptor = self.runtime._dry_run_interceptor
            dry_result = await interceptor.intercept(name, args)
            if not dry_result.allowed:
                logger.warning(f"[Agent] DryRun blocked {name}: {dry_result.stage_failed}")
                self._record_tool(name, args, False)
                return {"status": "error", "result": f"Safety blocked: {dry_result.stage_failed}"}

        # Execute
        await self._emit("tool_execution_start", {"tool_call_id": tool_id, "tool_name": name, "arguments": args})

        tool.model = self.model
        tool.context = self.runtime
        start = time.time()
        result: ToolResult = await tool.execute_tool(args)
        elapsed = time.time() - start

        self._record_tool(name, args, result.status == "success")

        await self._emit("tool_execution_end", {
            "tool_call_id": tool_id, "tool_name": name,
            "status": result.status, "result": result.result,
            "execution_time": elapsed,
        })

        return {"status": result.status, "result": result.result, "execution_time": elapsed}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        if self.on_event:
            try:
                event = AgentEvent(type=event_type, data=data, timestamp=time.time())
                result = self.on_event(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.debug(f"[Agent] Event callback error: {e}")

    async def _check_cancelled(self) -> None:
        if self.cancel_event and self.cancel_event.is_set():
            raise asyncio.CancelledError("Agent cancelled")

    def _check_consecutive_failures(self, tool_name: str, args: Dict) -> Tuple[bool, str, bool]:
        args_hash = hashlib.md5(json.dumps(args, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:8]
        # deque doesn't support slicing — convert to list for the last 10
        recent = list(self.tool_failure_history)[-10:]
        same = sum(1 for n, h, _ in reversed(recent) if n == tool_name and h == args_hash)
        if same >= 5:
            return (True, f"Tool '{tool_name}' called {same} times with same args — stopping loop", False)
        fails = sum(1 for n, h, s in reversed(recent) if n == tool_name and not s)
        if fails >= 8:
            return (True, "Too many consecutive failures — aborting", True)
        if fails >= 3:
            return (True, f"Tool '{tool_name}' failed {fails} times consecutively", False)
        return (False, "", False)

    def _record_tool(self, name: str, args: Dict, success: bool) -> None:
        args_hash = hashlib.md5(json.dumps(args, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:8]
        self.tool_failure_history.append((name, args_hash, success))

    @staticmethod
    def _format_tool_result(result: Dict[str, Any]) -> str:
        from yiagent.common.utils import safe_json_dumps
        data = result.get("result", "")
        if isinstance(data, (dict, list)):
            return safe_json_dumps(data, max_len=50000)
        if isinstance(data, str) and len(data) > 50000:
            return safe_json_dumps(data, max_len=50000)
        return str(data)

    async def _handle_empty_response(self, turn: int) -> str:
        """Request explicit response from LLM when it returns nothing."""
        if turn > 1:
            self.messages.append({"role": "user", "content": [{
                "type": "text",
                "text": "Please respond to the user based on the tool results above."
            }]})
            text, tools = await self._call_llm_stream(retry_on_empty=False)
            return text or "I cannot generate a response right now. Please try again."
        return "I cannot generate a response right now. Please try again."

    async def _force_summary(self, turn: int) -> str:
        """Force a text summary after max steps."""
        self.messages.append({"role": "user", "content": [{
            "type": "text",
            "text": f"You have taken {turn} steps (limit reached). Summarize your progress and results. Do NOT call any tools."
        }]})
        try:
            text, _ = await self._call_llm_stream(retry_on_empty=False)
            return text or f"Task reached step limit ({turn} steps). Please try a simpler request."
        except Exception:
            return f"Task reached step limit ({turn} steps). Please try a simpler request."

    # ------------------------------------------------------------------
    # Context trimming
    # ------------------------------------------------------------------

    def _trim_messages(self) -> None:
        """Trim context to max_context_turns visible turns."""
        if not self.messages:
            return

        turns = self._identify_turns()
        if len(turns) <= self.max_context_turns:
            return

        kept = turns[-self.max_context_turns:]
        new_messages = []
        for t in kept:
            new_messages.extend(t)
        logger.info(f"[Agent] Trimmed context: {len(self.messages)} → {len(new_messages)} messages")
        self.messages = new_messages

    def _identify_turns(self) -> List[List[Dict[str, Any]]]:
        turns = []
        current = []
        for msg in self.messages:
            role = msg.get("role")
            if role == "user":
                content = msg.get("content", [])
                if isinstance(content, list):
                    has_text = any(
                        isinstance(b, dict) and b.get("type") == "text" for b in content
                    )
                    has_tool = any(
                        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
                    )
                    if has_text and not has_tool and current:
                        turns.append(current)
                        current = [msg]
                    else:
                        current.append(msg)
                elif isinstance(content, str) and current:
                    turns.append(current)
                    current = [msg]
                else:
                    current.append(msg)
            else:
                current.append(msg)
        if current:
            turns.append(current)
        return turns
