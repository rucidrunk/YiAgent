---
name: yiagent-architecture
description: YiAgent 采用分层架构：protocol → agent → memory，使用 pgvector + Redis 双存储
metadata:
  type: project
---

YiAgent 是一个企业级 AI Agent 运行时，架构分为以下几层：

- **protocol 层**：定义 AgentEvent、ContentBlock、Message 等核心协议模型
- **agent 层**：Agent 主循环、tool 调用编排
- **memory 层**：基于 pgvector 的长短期记忆存储，Redis 做会话缓存和消息队列
- **model_gateway 层**：模型提供商路由，支持 OpenAI 兼容接口
- **channel 层**：多渠道接入（HTTP/SSE/WebSocket）

**Why:** 分层设计让各模块独立演化和测试，memory 层的双存储让向量搜索和实时缓存各司其职。

**How to apply:** 新增功能时遵循现有分层，不要跨层直接访问底层存储。
关联记忆：[[pgvector-config]] [[redis-connection-pool]]
