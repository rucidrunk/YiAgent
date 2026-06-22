"""
Channel Adapter — abstract base for all input/output channels.

Every channel (Web, Feishu, Slack, REST) normalises its native message
format into the unified protocol.models.Message, and converts agent
events back into channel-specific responses.

Extension point: subclass ChannelAdapter and register via
register_adapter().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from yiagent.protocol.models import AgentEvent, Message


class ChannelAdapter(ABC):
    """Abstract base for channel adapters.

    Subclasses must implement:
      - normalize(): convert channel-native payload → Message
      - send_event(): deliver an agent event to the channel
    """

    def __init__(self, channel_type: str):
        self.channel_type = channel_type

    @abstractmethod
    async def normalize(self, raw_payload: Any) -> Message:
        """Convert a channel-native incoming payload to a unified Message."""
        ...

    @abstractmethod
    async def send_event(self, event: AgentEvent, context: Dict[str, Any]) -> None:
        """Deliver an agent event back through the channel."""
        ...

    async def on_connect(self, session_id: str) -> None:
        """Called when a client connects (e.g. WebSocket upgrade)."""

    async def on_disconnect(self, session_id: str) -> None:
        """Called when a client disconnects."""

    async def close(self) -> None:
        """Release channel resources."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_adapters: Dict[str, ChannelAdapter] = {}


def register_adapter(channel_type: str, adapter: ChannelAdapter) -> None:
    _adapters[channel_type] = adapter


def get_adapter(channel_type: str) -> Optional[ChannelAdapter]:
    return _adapters.get(channel_type)


def list_adapters() -> List[str]:
    return list(_adapters.keys())
