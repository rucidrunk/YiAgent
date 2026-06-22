"""
Web channel adapter — HTTP/WebSocket with SSE streaming.

Normalises web input (JSON body or WebSocket frame) into the unified
Message protocol, and streams agent events back via SSE or WS frames.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from yiagent.common.log import logger
from yiagent.channel.base import ChannelAdapter
from yiagent.protocol.models import AgentEvent, ContentBlock, Message


class WebChannelAdapter(ChannelAdapter):
    """Channel adapter for web (HTTP REST + WebSocket) clients."""

    def __init__(self):
        super().__init__(channel_type="web")
        # request_id → session_id mapping for SSE polling
        self.request_to_session: Dict[str, str] = {}
        # session_id → queue for SSE event delivery
        self._sse_queues: Dict[str, asyncio.Queue] = {}

    # ------------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------------

    async def normalize(self, raw_payload: Any) -> Message:
        """Convert a web JSON payload to unified Message.

        Expected format:
          {
            "session_id": "user_xxx",
            "user_id": "user_xxx",
            "content": "hello" | [{"type": "text", "text": "hello"}, ...],
            "receiver": "...",
            "channel_type": "web"
          }
        """
        if isinstance(raw_payload, str):
            try:
                payload = json.loads(raw_payload)
            except json.JSONDecodeError:
                payload = {"content": raw_payload}
        elif isinstance(raw_payload, dict):
            payload = raw_payload
        else:
            payload = {"content": str(raw_payload)}

        # Parse content
        raw_content = payload.get("content", "")
        if isinstance(raw_content, str):
            blocks = [ContentBlock.text_block(raw_content)]
        elif isinstance(raw_content, list):
            blocks = [ContentBlock.from_dict(b) if isinstance(b, dict) else ContentBlock.text_block(str(b))
                      for b in raw_content]
        else:
            blocks = [ContentBlock.text_block(str(raw_content))]

        return Message(
            role="user",
            content=blocks,
            session_id=payload.get("session_id", ""),
            user_id=payload.get("user_id", ""),
            channel_type="web",
            receiver=payload.get("receiver", ""),
            timestamp=time.time(),
        )

    # ------------------------------------------------------------------
    # Send / SSE
    # ------------------------------------------------------------------

    async def send_event(self, event: AgentEvent, context: Dict[str, Any]) -> None:
        """Push an event to the SSE queue for the session."""
        session_id = context.get("session_id", "")
        if not session_id:
            return

        queue = self._sse_queues.get(session_id)
        if queue is None:
            return

        try:
            sse_data = self._event_to_sse(event)
            queue.put_nowait(sse_data)
        except asyncio.QueueFull:
            logger.debug(f"[Web] SSE queue full for {session_id}, dropping event")

    async def on_connect(self, session_id: str) -> None:
        if session_id not in self._sse_queues:
            self._sse_queues[session_id] = asyncio.Queue(maxsize=256)
            logger.debug(f"[Web] SSE queue created for {session_id}")

    async def on_disconnect(self, session_id: str) -> None:
        self._sse_queues.pop(session_id, None)
        logger.debug(f"[Web] SSE queue removed for {session_id}")

    # ------------------------------------------------------------------
    # SSE stream generator
    # ------------------------------------------------------------------

    async def sse_stream(self, session_id: str) -> AsyncIterator[str]:
        """Async generator yielding SSE events for a session."""
        queue = self._sse_queues.get(session_id)
        if queue is None:
            queue = asyncio.Queue(maxsize=256)
            self._sse_queues[session_id] = queue

        try:
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield data
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            self._sse_queues.pop(session_id, None)

    @staticmethod
    def _event_to_sse(event: AgentEvent) -> str:
        """Convert AgentEvent to SSE text/event-stream format."""
        payload = json.dumps({"type": event.type, "data": event.data}, ensure_ascii=False)
        return f"event: {event.type}\ndata: {payload}\n\n"
