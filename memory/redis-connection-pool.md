---
name: redis-connection-pool
description: Redis 连接池配置，用于会话缓存和消息队列，启动时支持优雅降级
metadata:
  type: project
---

Redis 在 YiAgent 中承担两个角色：

1. **会话缓存**：存储活跃会话状态，TTL 默认 3600 秒
2. **消息队列**：Agent 事件的异步发布/订阅

启动时的优雅降级逻辑（`app.py` 中的 BUG-2/3 fix）：
- 如果 Redis 不可用，`app.state.redis_ok = False`，服务仍可启动
- 使用 `hiredis` 解析器提升性能

**Why:** Agent 运行时需要低延迟的会话状态访问，Redis 是自然选择。优雅降级保证开发环境（无 Redis）也能跑。

**How to apply:** 通过 `yiagent.common.redis_pool.get_redis()` 获取连接，不要直接创建 Redis 客户端。
关联记忆：[[yiagent-architecture]] [[pgvector-config]]
