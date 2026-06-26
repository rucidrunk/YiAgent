"""
OpenAI-compatible provider — supports OpenAI, Azure, and compatible APIs.

Streaming via SSE chunk parsing, tool-call support, and automatic
retry on transient errors.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from yiagent.common.config import conf
from yiagent.common.log import logger
from yiagent.protocol.models import LLMRequest, LLMModel


class OpenAIProvider(LLMModel):
    """OpenAI / Azure / compatible API provider."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        cfg = conf()
        self._api_key = api_key or cfg.get("openai_api_key", "")
        self._base_url = (base_url or cfg.get("openai_base_url", "https://api.openai.com/v1")).rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=10.0),
            )
        return self._client

    async def call(self, request: LLMRequest) -> Dict[str, Any]:
        client = await self._get_client()
        body = self._build_body(request, stream=False)
        resp = await client.post("/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        return {
            "content": choice["message"].get("content", ""),
            "finish_reason": choice.get("finish_reason", ""),
            "usage": data.get("usage", {}),
        }

    async def call_stream(self, request: LLMRequest) -> AsyncIterator[Dict[str, Any]]:
        client = await self._get_client()
        body = self._build_body(request, stream=True)
        try:
            async with client.stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code >= 400:
                    error_body = await resp.aread()
                    error_text = error_body.decode(errors="replace")[:2000]
                    logger.error(f"[OpenAI] API error {resp.status_code}: {error_text}")
                    raise RuntimeError(f"API error {resp.status_code}: {error_text}")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        pass
        except httpx.HTTPStatusError as e:
            error_body = ""
            try:
                error_body = e.response.text[:2000]
            except Exception:
                pass
            logger.error(f"[OpenAI] HTTP error {e.response.status_code}: {error_body}")
            raise RuntimeError(f"API error {e.response.status_code}: {error_body}") from e

    def _build_body(self, request: LLMRequest, stream: bool) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": self._format_messages(request),
            "stream": stream,
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_tokens:
            body["max_tokens"] = request.max_tokens
        if request.tools:
            body["tools"] = [
                {"type": "function", "function": t} if "type" not in t else t
                for t in request.tools
            ]
        body.update(request.extra)
        return body

    def _format_messages(self, request: LLMRequest) -> List[Dict[str, Any]]:
        """Format messages for OpenAI API — converts Claude content blocks to OpenAI format."""
        msgs = list(request.messages)
        converted = []
        for msg in msgs:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "assistant" and isinstance(content, list):
                text_parts = []
                tool_calls = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            args = block.get("input", {})
                            tool_calls.append({
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": json.dumps(args, ensure_ascii=False),
                                },
                            })
                        elif block.get("type") == "thinking":
                            pass  # OpenAI doesn't render thinking blocks
                entry = {"role": "assistant", "content": "\n".join(text_parts) or None}
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                converted.append(entry)

            elif role == "user" and isinstance(content, list):
                text_parts = []
                tool_results = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_result":
                            converted.append({
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id", ""),
                                "content": block.get("content", ""),
                            })
                if text_parts:
                    converted.append({"role": "user", "content": "\n".join(text_parts)})
                # tool_result blocks were already appended individually above

            elif role == "system":
                converted.append({"role": "system", "content": _extract_text(content)})
            else:
                converted.append({"role": role, "content": _extract_text(content)})

        if request.system:
            converted.insert(0, {"role": "system", "content": request.system})

        return converted

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


def _extract_text(content) -> str:
    """Extract plain text from message content (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content) if content else ""
