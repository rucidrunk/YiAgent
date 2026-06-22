"""Additional embedding vendor implementations.

Each provider follows the EmbeddingProvider ABC and is auto-registered
via register_embedding_provider().
"""

from yiagent.memory.embedding.vendors.dashscope import DashScopeEmbeddingProvider
from yiagent.memory.embedding.vendors.openai_compat import OpenAICompatEmbeddingProvider

from yiagent.memory.embedding.provider import register_embedding_provider

register_embedding_provider("dashscope", DashScopeEmbeddingProvider)
register_embedding_provider("doubao", OpenAICompatEmbeddingProvider)
register_embedding_provider("zhipu", OpenAICompatEmbeddingProvider)
