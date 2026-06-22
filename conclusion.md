 【1. 核心职责】
  
  common/config（150 行）

  进程级惰性单例配置引擎，三层优先级合并（hardcoded defaults → JSON file →
  YIAGENT_ 前缀环境变量），json.loads 自动类型推断（int/float/list），非法 JSON
  回退为 raw string。conf() 返回不可变快照引用，double-checked locking 保证 100
  线程并发首次调用只初始化一次。
  
  common/redis_pool（150 行）

  带健康网关的进程级 Redis 连接池。_pool_ready 布尔门控制发布：建池后先 PING
  探活，通过才设置 _pool_ready=True；失败则 await candidate.disconnect()
  清理后重抛异常，不留残损状态。close_redis() 持有 _pool_lock
  异步锁防止并发撕裂。_JSONEncoder 严格编码，datetime/date →
  ISO-8601，其他不可序列化类型直接 TypeError 拒绝 default=str 静默损坏。

  common/singleton（80 行）

  双重锁检查的单例基础设施。SingletonMeta 元类提供线程安全同步单例，LazyAsync
  提供惰性异步资源初始化。LazyAsync.close() 支持资源释放后重新惰性初始化。

  common/utils（86 行）

  零依赖工具函数集：safe_json_dumps 超长截断时输出 {truncated, preview, 
  original_length} 合法 JSON 信封，永不出产破损 JSON。estimate_tokens CJK/ASCII
  区分估算。CJK 检测 regex 覆盖中日韩越统一表意文字。
  
  common/log（62 行）

  结构化日志层，支持 JSON/Plain 双格式。_JsonFormatter 从 record._extra
  注入业务维度字段（session_id 等），default=str
  兜底不可序列化对象。_PlainFormatter 人类可读格式含异常堆栈。
  
  protocol/models（235 行）

  统一多模态消息协议：ContentBlock
  六种类型（text/image/audio/video/file/tool_use/tool_result）+ Message 信封 +
  LLMRequest 透传模型请求 + AgentAction/AgentEvent 事件流。to_dict()/from_dict()
   全字段往返序列化，路由元数据（session_id/user_id/channel_type/receiver/timest
  amp）完整保留。

  memory/conversation_store（556 行）

  双层会话记忆引擎：Redis List 热路径（最近 200 条消息）+ PG/pgvector
  冷路径异步持久化。核心机制链：
  - visible-turn 计数：仅统计真实用户文本轮次，排除 tool_result 注入
  - base_seq 偏移量补偿 ltrim 截头后的序列号漂移 → seq 永远单调递增
  - HSETNX 原子写 created_at → 消除 hexists→hset 的 TOCTOU 窗口
  - singleflight 合并并发冷缓存穿透 → 同一 session 只打一次 PG
  - _BgTracker 持有 create_task 强引用 + add_done_callback 日志 →
  后台持久化异常不再被 asyncio 吞掉

  ---
  【2. 防御性设计】
  
  防御层: Double-checked locking
  实现细节: if _x is not None: return _x → with _lock: → 再次检查 → 初始化。100
    线程压测通过
  涉及模块: config, redis_pool, conversation_store, singleton
  ────────────────────────────────────────
  防御层: 连接池失效保护
  实现细节: _pool_ready 布尔门：PING 失败时 _pool 不发布，candidate.disconnect()

    清理后重抛。后续调用者从头重试而非拿到死池
  涉及模块: redis_pool
  ────────────────────────────────────────
  防御层: 分布式锁 + 续租
  实现细节: SET NX EX 获取锁 → EXPIRE 续期 → DELETE 释放。锁获取失败返回 False
    而非崩溃
  涉及模块: conversation_store
  ────────────────────────────────────────
  防御层: 缓存惊群 singleflight
  实现细节: _load_from_pg_singleflight()：同 session 并发冷加载争抢
    asyncio.Lock，只有获取锁的请求打 PG，其余等待缓存回填
  涉及模块: conversation_store
  ────────────────────────────────────────
  防御层: ltrim 序列号补偿
  实现细节: base_seq 偏移量 + msg["seq"] = base_seq + i：列表头截断后 seq
    仍单调递增，不会跳回 0
  涉及模块: conversation_store
  ────────────────────────────────────────
  防御层: created_at 原子写入
  实现细节: HSETNX 替代 hexists → hset：并发 upsert
    不会覆盖初始时间戳，首个写入者胜出
  涉及模块: conversation_store
  ────────────────────────────────────────
  防御层: 有界队列反压
  实现细节: ltrim 硬性截断到 200 条上限。flush_queue_maxsize=256
  限制后台刷新队列
  涉及模块: conversation_store, config
  ────────────────────────────────────────
  防御层: 后台任务异常追踪
  实现细节: _BgTracker.spawn() 持有 task 强引用 + add_done_callback
    日志记录。asyncio "Task was never retrieved" 警告消除
  涉及模块: conversation_store
  ────────────────────────────────────────
  防御层: JSON 安全截断
  实现细节: safe_json_dumps：超长时构建 {truncated, preview, original_length}
    合法 JSON 信封，绝不在序列化后切割字符串
  涉及模块: utils
  ────────────────────────────────────────
  防御层: 严格 JSON 编码器
  实现细节: _JSONEncoder：datetime/date → ISO-8601，其他不可序列化类型直接
    TypeError。杜绝 default=str 导致的类型静默损坏
  涉及模块: redis_pool
  ────────────────────────────────────────
  防御层: Redis 全链路异常吞噬
  实现细节: 所有 Redis 操作 try/except 包裹，锁获取失败返回
  False、元数据为空返回
    None、写入失败记录 WARNING。不向上传播 Redis 故障
  涉及模块: conversation_store
  ────────────────────────────────────────
  防御层: TTL 双层续期
  实现细节: touch_session() 同时刷新 meta key 和 context key 的
    expire，不会出现元数据存活但上下文过期的不一致
  涉及模块: conversation_store
  ────────────────────────────────────────
  防御层: env var 类型推断
  实现细节: json.loads(env_val) 自动推断 int/float/list/bool，非法 JSON 回退为
    raw string
  涉及模块: config
  ────────────────────────────────────────
  防御层: 消息协议往返保真
  实现细节: to_dict()/from_dict()
    全字段序列化，路由元数据（session_id/user_id/channel_type/receiver/timestamp
  ）不丢失
  涉及模块: protocol

  ---
  【3. 面试对线卖点】
  
  ▎ Q：Redis 挂了怎么办？消息会丢吗？
  ▎
  ▎ "Redis 所有操作全 try/except 包裹，热路径挂掉自动走 PG 
  ▎ 冷路径回源。冷路径加了 singleflight 锁——100 个并发请求打同一个空 
  ▎ session，只有 1 个穿透到 PG，剩下 99 个等在 asyncio.Lock 
  ▎ 上吃回填结果。缓存击穿在设计层面就被消灭了。消息落地是 rpush → 后台 
  ▎ _BgTracker.spawn() 异步写 PG，异常有 done_callback 日志兜底，不会静默丢失。"

  ▎ Q：每秒数千条消息追加到同一个 session，Redis List 不会爆吗？
  ▎
  ▎ "有界队列硬截断到 200 条。关键是 ltrim 砍头之后用 base_seq 
  ▎ 偏移量补偿序列号——seq = base_seq + list_index 永远单调递增，不会出现 trim 
  ▎ 之后 seq 跳回 0 的倒退 bug。每次追加只做 1 次 rpush + 1 次 
  ▎ hsetnx，热路径延迟控制在一个 Redis 往返以内。HSETNX 保证并发 upsert 的 
  ▎ created_at 是原子 '先到先得'，不存在 TOCTOU 竞态。"

  ▎ Q：连接池初始化失败了会怎样？后续请求全挂？
  ▎ hsetnx，热路径延迟控制在一个 Redis 往返以内。HSETNX 保证并发 upsert 的
  ▎ created_at 是原子 '先到先得'，不存在 TOCTOU 竞态。"

  ▎ Q：连接池初始化失败了会怎样？后续请求全挂？
  ▎
  ▎ "不会。_pool_ready 是一个布尔门——PING 通过才翻牌。PING 失败时 _pool 
  ▎ 不发布，candidate.disconnect() 清理后重抛异常。下一次 get_redis() 
  ▎ 调用从头建池，DNS 延迟传播、端口未就绪等瞬态故障不会转化为永久性池损坏。这是
  ▎  warm-up gate 模式，不是 try-catch 赌命。"

  ▎ Q：分布式锁长任务不释放怎么办？
  ▎
  ▎ "锁用 SET NX EX 带 TTL 自动过期，默认 120 秒。长任务调用 extend_lock() 
  ▎ 续期，相当于心跳。锁获取失败返回 False，调用者决定是排队重试还是返回
  ▎ busy。不会死锁。"


【1. 核心职责】
  
  common/config：三层优先级配置引擎（hardcoded defaults → JSON file → YIAGENT_
  env vars），json.loads 自动类型推断（int/float/list/bool），非法 JSON 回退 raw
   string。double-checked locking 保证进程级惰性单例，conf()
  返回不可变快照引用——运行时永不发生配置漂移。
  
  common/redis_pool：带健康网关的进程级连接池。_pool_ready 布尔门：建池 → PING
  探活 → 成功才发布 _pool_ready=True，失败时 await candidate.disconnect() 清理 +
   重抛异常，不留残损状态。_JSONEncoder 严格编码，datetime/date →
  ISO-8601，其余不可序列化类型直接 TypeError，彻底消灭 default=str
  静默损坏。close_redis() 持有 _pool_lock 异步锁防止并发撕裂。

  common/singleton：SingletonMeta（同步线程锁）提供进程级单例元类；LazyAsync（as
  yncio.Lock）提供惰性异步资源初始化，close() 持锁释放资源后支持重新惰性初始化。

  common/utils：safe_json_dumps 超长截断时输出 {truncated, preview, 
  original_length} 合法 JSON 信封，永不出产破损 JSON。CJK regex
  覆盖中日韩越统一表意文字，estimate_tokens 按 CJK/ASCII 区分估算。
  
  common/log：JSON/Plain 双格式结构化日志，record._extra 注入 session
  维度上下文，default=str 兜底不可序列化对象。

  protocol/models：统一多模态消息协议。ContentBlock 六种类型 + Message 信封 +
  LLMRequest + AgentAction/AgentEvent。to_dict()/from_dict()
  全字段往返序列化——路由元数据（session_id/user_id/channel_type/receiver/timesta
  mp）完整保留，Redis/PG 往返零丢失。
  
  memory/conversation_store：双层会话记忆引擎——Redis List 热路径（200
  条有界窗口）+ PG 冷路径异步持久化。核心机制链：
  - visible-turn 计数：仅统计真实用户文本轮次，tool_result 注入不计
  - base_seq 偏移量补偿 ltrim 截头后的序列号 → seq 永远单调递增
  - HSETNX 原子写 created_at → 消除 hexists→hset TOCTOU
  - singleflight：async with lock 内双检 Redis → 第一个调用者回源 PG
  并写回缓存，后续调用者直接命中
  - _BgTracker：get_running_loop() 检测 + 强引用 + add_done_callback 日志 →
  后台异常零静默
  - 严格 json.dumps（无 default=str）→ datetime 等非 JSON
  类型直接拒绝，不静默损坏
  
  app.py：FastAPI 生命周期管理。startup 惰性预热 Redis/PG 池（get_redis() 自带
  warm-up PING，无冗余二次探测），任一后端不可用时优雅降级启动不崩溃。shutdown
  调用 close_redis() 清理连接池。

  ---
  【2. 防御性设计】
  
  防御层: Double-checked locking
  实现细节: if not None: return → with lock: → 再检 → 初始化。100 线程压测通过
  ────────────────────────────────────────
  防御层: 连接池健康网关
  实现细节: _pool_ready 布尔门：PING 失败池不发布，candidate.disconnect()
    清理后重抛，后续调用者从头重试
  ────────────────────────────────────────
  防御层: 分布式锁 + 续租
  实现细节: SET NX EX 获取 → EXPIRE 续期 → DELETE 释放。锁获取失败返回 False
    不崩溃
  ────────────────────────────────────────
  防御层: singleflight 缓存惊群
  实现细节: async with lock 内 Redis 双检：首个调用者回源 PG
    写回缓存，后续串行化调用者命中缓存直接返回。10 并发 → 1 次 PG
  ────────────────────────────────────────
  防御层: ltrim 序列号补偿
  实现细节: base_seq 偏移 + msg["seq"] = base_seq + i：列表截头后 seq
    仍单调递增不跳回 0
  ────────────────────────────────────────
  防御层: created_at 原子写入
  实现细节: HSETNX 替代 hexists→hset：并发 upsert 首个写入者胜出，无 TOCTOU 窗口
  ────────────────────────────────────────
  防御层: 有界队列反压
  实现细节: ltrim 硬截断至 200 条；flush_queue_maxsize=256 限制后台刷新
  ────────────────────────────────────────
  防御层: 后台任务异常追踪
  实现细节: _BgTracker：get_running_loop() 检测运行环境 + task 强引用 +
    add_done_callback 日志，消除 "Task was never retrieved"
  ────────────────────────────────────────
  防御层: JSON 安全截断
  实现细节: safe_json_dumps 超长时输出 {truncated, preview, original_length}
  合法
    JSON 信封
  ────────────────────────────────────────
  防御层: 严格 JSON 编码
  实现细节: redis 层
    _JSONEncoder（datetime→ISO-8601，其余拒绝）；conversation_store 层移除
    default=str
  ────────────────────────────────────────
  防御层: Redis 全链路异常吞噬
  实现细节: 所有操作 try/except
    包裹，锁失败→False，元数据空→None，写入失败→WARNING，不向上传播
  ────────────────────────────────────────
  防御层: TTL 双层续期
  实现细节: touch_session() 同时刷新 meta key 和 context
    key，不会元数据存活而上下文过期不一致
  ────────────────────────────────────────
  防御层: env var 类型推断
  实现细节: json.loads(env_val) 自动推断 int/float/list/bool，非法 JSON 回退 raw

    string
  ────────────────────────────────────────
  防御层: 消息往返保真
  实现细节: to_dict()/from_dict() 全字段序列化，路由元数据往返零丢失
  ────────────────────────────────────────
  防御层: 优雅降级启动
  实现细节: Redis/PG 任一不可用时 app 正常启动不崩溃，仅 WARNING 日志
  ────────────────────────────────────────
  防御层: shutdown 池清理
  实现细节: close_redis() 持 _pool_lock 清理，_pool_ready=False
    确保重启后重新建池
  ────────────────────────────────────────
  防御层: _BgTracker 环境检测
  实现细节: get_running_loop() 检测 → 无事件循环时 WARNING +
    no-op，不会在同步上下文崩溃
  
  ---
  【3. 面试对线卖点】
  
  ▎ Q：Redis 挂了，缓存全空，100 个并发请求同时打到 PG 怎么办？
  ▎
  ▎ "singleflight 锁内双检。async with asyncio.Lock 
  ▎ 串行化所有并发调用者，只有第一个获取锁的检查 Redis 发现为空，回源 PG 
  ▎ 并写回缓存；后续 99 个在锁队列里等待，获取锁后二次检查 
  ▎ Redis——此时缓存已被第一个调用者填满，直接命中返回。10 并发 → 1 次 
  ▎ PG，实验验证通过。这不是简单的 if cache miss → query PG，防止了 thundering 
  ▎ herd。"

  ▎ Q：每秒数千条消息 rpush 到同一个 session，Redis 的 List 
  ▎ 不会炸吗？序列号不会乱吗？
  ▎
  ▎ "有界队列硬截断至 200 条，每次 rpush 后 ltrim 砍头。砍头产生的序列号漂移用 
  ▎ base_seq 偏移量补偿——seq = base_seq + list_index 永远单调递增，不会出现 trim
  ▎  后 seq 跳回 0 的倒退。created_at 用 HSETNX 原子写入，并发 upsert 
  ▎ 第一个写入者胜出，不存在 TOCTOU 窗口。消息落地路径是 rpush → 
  ▎ _BgTracker.spawn() 异步刷 PG，后台任务异常有 done_callback 日志兜底，不会 
  ▎ asyncio 静默吞异常。每次追加热路径延迟控制在一个 Redis 往返以内。"

  ▎ Q：连接池初始化失败了，后续请求全拿死池吗？
  ▎
  ▎ "不会。_pool_ready 是布尔门——PING 通过才翻牌为 True。PING 失败时 _pool 
  ▎ 根本没发布，candidate.disconnect() 做 best-effort 清理后重抛原始异常。下一次
  ▎  get_redis() 调用从头建池，网络瞬态故障（DNS 
  ▎ 传播延迟、端口未就绪）不会转化为永久性池损坏。这是 warm-up gate 模式，不是 
  ▎ try-catch 赌命。"

  ▎ Q：datetime 塞进消息 extras，反序列化回来还是 datetime 吗？
  ▎
  ▎ "严格拒绝，不是静默损坏。Redis 层用 
  ▎ _JSONEncoder——datetime→ISO-8601（可逆），其他不可序列化类型直接 
  ▎ TypeError。conversation_store 的 append_message 同样去掉了 default=str——非 
  ▎ JSON 类型写入直接返回 0 并打 ERROR 
  ▎ 日志。宁可丢一条消息也不允许类型信息被悄悄吃掉。调用者必须显式序列化 
  ▎ datetime 再传入。"
