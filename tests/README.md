# 正确性测试

从仓库根目录运行 `python -m pytest -q`。只收集此目录，不扫实验入口或第三方源码。

- `unit/`：纯编码、模型张量、算法与工具契约；个别历史用例包含多模块组合，尚未声称完全隔离。
- `integration/`：原生环境/搜索、checkpoint、训练、子进程预算。
- `conftest.py`：确定性 native fixture 与 Torch 线程预算。当前全套入口要求显式原生构建，缺少依赖是错误。
- `python -m tools.validation.bitwise`：跨版本 CPU 数值回归；固定步数，记录 checkpoint/Adam/rollout/replay/search，不是长程训练效果验收。

历史测试迁移保留断言。已知缺陷使用绑定 TODO ID 的 strict xfail。训练用 `training` marker；完整原生验收见 ../docs/development-checks.md。
