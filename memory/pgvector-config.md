---
name: pgvector-config
description: pgvector 用于长短期记忆的向量存储，默认维度 1536，索引使用 IVFFlat
metadata:
  type: reference
---

YiAgent 使用 PostgreSQL + pgvector 扩展存储向量化的对话记忆：

- **嵌入维度**: 1536（对应 text-embedding-ada-002 输出维度）
- **索引类型**: IVFFlat（平衡查询速度和构建成本）
- **距离度量**: cosine distance
- **连接池**: 通过 `yiagent.common.redis_pool` 类似模式管理（asyncpg 连接池）

**Why:** pgvector 让向量搜索和业务数据在同一数据库中，简化运维。1536 维度是 OpenAI embedding 的标准输出。

**How to apply:** `yiagent/memory/long_term_store.py` 中的 `get_long_term_store()` 是唯一入口，不要在业务代码中直接操作 pgvector。
关联记忆：[[yiagent-architecture]] [[redis-connection-pool]]
