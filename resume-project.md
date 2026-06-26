## YiAgent — 企业级 AI Agent 运行时

**个人角色：** 后端架构设计与核心模块开发 | Python 3.11 + FastAPI + Redis + PostgreSQL/pgvector

---

### 项目概述

从零设计并实现了一个生产级 AI Agent 运行时，支持多轮工具调用、流式响应、长短期记忆管理和自主进化。
核心架构采用 **无状态 Agent + 外部存储** 设计，任意 Pod 可服务任意会话，状态完全下沉到 Redis（热路径）和
PostgreSQL/pgvector（冷路径），实现水平扩展。

---

### 技术亮点

**1. 双层会话记忆引擎（Redis List + PG/pgvector）**

自研 conversation_store（~556 行），解决高并发写入下的缓存一致性难题：
- Redis List 热路径维护最近 200 条消息的滑动窗口，每次写入仅 1 次 RPUSH + 1 次 LTRIM，
  延迟控制在单个 Redis 往返内
- **singleflight 缓存惊群防护**：100 并发请求打同一个空 session，仅 1 个穿透到 PG，
  其余 99 个在 asyncio.Lock 上等待缓存回填——缓存击穿在设计层面消灭
- **base_seq 偏移量补偿**：LTRIM 截头导致序列号漂移，通过 base_seq + list_index 补偿
  保证 seq 永远单调递增，消除截断后 seq 跳回 0 的倒退 bug
- **HSETNX 原子写入** created_at：替代 hexists → hset 的 TOCTOU 模式，并发 upsert
  首个写入者胜出

**2. 连接池健康网关（Warm-up Gate 模式）**

Redis 连接池启动时先建池后 PING 探活，通过才设置 `_pool_ready=True`；
失败时 `candidate.disconnect()` 清理后重抛异常，后续调用者从头建池。
DNS 延迟传播、端口未就绪等瞬态故障不会转化为永久性池损坏。

**3. Agent 自进化系统（Self-Evolution）**

Agent 空闲时自动触发独立审查 Agent 分析会话记录，对 MEMORY.md / skills 等文件提出改进：
- 保守策略：大部分运行返回 [SILENT]（无变化），仅真正有产出时才落地
- 变更前创建 backup_id 快照，支持回滚
- 全局并发限制（max 2），避免审查 Agent 影响在线服务

**4. 工具安全拦截器（Dry-Run Interceptor）**

Write / Edit / Bash 等高风险工具在真实执行前经过三级审查流水线，
不合规操作直接拦截，不依赖模型自律。

**5. 分布式会话锁**

基于 Redis SET NX EX 实现会话级分布式锁，防止同一会话的并发执行。
长任务通过 EXPIRE 续期（心跳），锁获取失败返回 False（非抛异常），
调用者自行决定排队或拒绝。Redis 不可用时自动降级为无锁模式。

**6. 深度防御性设计**

- **三层配置优先级**：hardcoded defaults → JSON file → `YIAGENT_` 环境变量，
  `json.loads` 自动类型推断（int/float/list/bool），非法 JSON 回退 raw string
- **Double-checked locking**：config / redis_pool / conversation_store 均采用
  双重检查锁，100 线程并发压测通过
- **严格 JSON 编码**：`_JSONEncoder` 对 `datetime/date` 转 ISO-8601（可逆），
  其余不可序列化类型直接 `TypeError`，彻底杜绝 `default=str` 静默损坏
- **后台任务异常追踪**：`_BgTracker` 持有 `create_task` 强引用 + `add_done_callback`
  日志，消灭 asyncio "Task was never retrieved" 警告
- **安全 JSON 截断**：`safe_json_dumps` 超长时输出 `{truncated, preview, original_length}`
  合法 JSON 信封，永不出产破损 JSON
- **全链路优雅降级**：Redis / PG 任一不可用时服务正常启动不崩溃，热路径挂掉自动走冷路径回源

---

### 技术栈

`Python 3.11` `FastAPI` `asyncio` `Redis (hiredis)` `PostgreSQL/pgvector` `SSE 流式`
`MCP 协议` `分布式锁` `连接池管理` `向量嵌入` `混合搜索 (RRF)` `Docker`

---

### 量化成果

| 指标 | 数据 |
|---|---|
| 核心代码规模 | ~4,000 行 Python（不含测试） |
| 模块数量 | 6 个核心包（agent / memory / model_gateway / protocol / channel / common） |
| 测试覆盖 | 15 个测试文件，覆盖配置、存储、协议、工具、路由等 |
| 缓存穿透防护 | singleflight：100 并发 → 1 次 PG 查询 |
| 热路径延迟 | 消息追加 1 次 Redis 往返（RPUSH + LTRIM） |
| 防御层 | 15+ 项防御性设计（见 conclusion.md） |
