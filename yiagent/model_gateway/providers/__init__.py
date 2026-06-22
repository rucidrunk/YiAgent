"""Model provider implementations."""

from yiagent.model_gateway.providers.openai import OpenAIProvider
from yiagent.model_gateway.router import register_provider

register_provider("openai", OpenAIProvider)
