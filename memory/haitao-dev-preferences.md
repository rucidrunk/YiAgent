---
name: haitao-dev-preferences
description: Haitao 的开发偏好：中文沟通、Python 3.11+、异步优先
metadata:
  type: user
---

Haitao 是 YiAgent 项目的主要开发者。

偏好：
- 使用中文进行沟通和代码注释
- Python 版本要求 >= 3.11
- 偏好异步编程（asyncio + FastAPI）
- 使用 pytest 进行测试，配置见 `pytest.ini`
- 代码风格：类型注解完整、docstring 简洁

**Why:** 统一风格减少团队协作成本，async 是因为 AI Agent 场景天然适合异步 IO。

**How to apply:** 所有新增代码需包含完整类型注解，关键函数添加 docstring，IO 密集型操作优先使用 async/await。
