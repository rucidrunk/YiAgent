---
name: async-test-patterns
description: pytest + pytest-asyncio 异步测试模式，使用 fixture 管理资源生命周期
metadata:
  type: feedback
---

YiAgent 的测试模式（来自代码 review 反馈）：

1. **使用 `pytest-asyncio`**：所有涉及 async 的测试用例标记 `@pytest.mark.asyncio`
2. **Fixture 优于 setUp/tearDown**：用 `conftest.py` 中的 fixture 管理 Redis/PostgreSQL 测试实例
3. **Mock 外部依赖**：model_gateway 的测试应 mock HTTP 调用，避免依赖真实 API
4. **测试隔离**：每个测试函数独立创建和清理数据

**Why:** 异步代码的特殊性——事件循环管理、资源生命周期、超时处理都不一样，错误模式会误导排查方向。

**How to apply:** 写测试时先检查 `tests/` 目录中现有 fixture，优先复用。新增外部依赖 mock 放在 `conftest.py`。
关联记忆：[[haitao-dev-preferences]]
