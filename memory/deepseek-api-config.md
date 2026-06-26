---
name: deepseek-api-config
description: 模型提供商使用 DeepSeek API，通过 OpenAI 兼容接口接入
metadata:
  type: project
---

当前 YiAgent 的模型提供商配置（见 `.env`）：

- `YIAGENT_OPENAI_API_KEY`: 使用 DeepSeek API Key
- `YIAGENT_OPENAI_BASE_URL`: `https://api.deepseek.com`
- `YIAGENT_MODEL_NAME`: `deepseek-chat`

通过 `model_gateway.providers.openai.OpenAIProvider` 注册，在 `app.py` 启动时通过 `register_provider("openai", OpenAIProvider)` 完成注册。

**Why:** DeepSeek 提供高性价比的 API，OpenAI 兼容接口让切换模型零成本。

**How to apply:** 如需切换模型提供商，修改 `.env` 中的三个环境变量，并在 `app.py` 中注册对应的 Provider。
关联记忆：[[yiagent-architecture]]
