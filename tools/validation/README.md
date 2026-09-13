# 重构验证工具

- `bitwise capture/compare`：跨 revision 固定预算 PPO/分层/NOSL/replay/恢复推理快照。
- `eval_revisions`：真实 checkpoint 在两个 checkout 上各跑同参数的严格 Full Gumbel 小场景，逐字节比较 JSON。
- `locate_legacy OLD_PATH`：查历史文件/模块当前位置；`--verify` 核对归档原始哈希。

从仓库根目录用 `python -m tools.validation.<command> --help`。同机、同解释器、同原生构建是 bitwise 比较前提；超时和缺少依赖是失败，不能当通过。详见 tests/README.md。
