# 重构验证工具

- `bitwise capture/compare`：跨 revision 固定预算 PPO/分层/NOSL/replay/恢复推理快照。
- `eval_revisions`：真实 checkpoint 在两个 checkout 上各跑同参数的严格 Full Gumbel 小场景，逐字节比较 JSON。
- `locate_legacy OLD_PATH`：查历史文件/模块当前位置；`--verify` 核对归档原始哈希。

从仓库根目录用 `python -m tools.validation.<command> --help`。同机、同解释器、同原生构建是 bitwise 比较前提；超时和缺少依赖是失败，不能当通过。详见 tests/README.md。

`policy_scores.compare_policy_scores` 是无 CLI 的数值验证函数：对同序、仅含合法动作的
有限 logits，同时检查逐项绝对/相对误差、温度为 1 的 softmax 概率差和
KL(reference || candidate)。调用者显式提供四项容差，并另验输入、动作映射和
采样器；近同分 top1 变化只记录，不代替分布误差检查。容差需在实验协议中说明依据，
不能将原版浮点实现视作精确 oracle，也不能以此声称 bitwise 等价。
