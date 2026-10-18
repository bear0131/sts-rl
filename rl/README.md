# 学习运行时

数据流：`env` 产生观测/合法动作 → `policies` 前推 → `search` 可选搜索 → `algorithms` 收集与更新 → `training` 编排；`evaluation` 消费 checkpoint，`apps` 负责展示。

| 包 | 职责 |
|---|---|
| algorithms | PPO/分层回报/NOSL learner 与 replay |
| policies | tokenizer、共享编码器、combat/overworld actor-critic、checkpoint 观测适配 |
| search | Python NOSL、native adapter 消费者、随机流和战斗价值搜索 |
| training | CLI、向量环境、多 rank 通信、reward wrapper |
| tasks | combat/whole-run 环境与模型组合 |
| evaluation | 整局/战斗/搜索/规则评估协议 |
| rules | 规则表示、teacher 数据、模型及其专用训练 |
| benchmarks | 固定预算训练及资源采样（旧低层探针在 tools/benchmarks） |

新代码使用 canonical 包路径。`_compat/` 仅保留历史 checkpoint 所需的 20 个库别名；旧 CLI 包装已删除。不会导入两套类，不在这里实现新功能。映射/退役说明见 ../tools/validation/migrations.json。
训练/评估参数放 ../configs；启动环境放 ../experiments/launchers；输出放 ../runs/local。不要在 rl 根目录增加实验脚本或测试。

二代训练使用 `algorithms/sts2_ppo.py`、`policies/sts2.py`、`training/sts2.py`；
评估使用 `evaluation/sts2.py` 和共享 runner/DecisionStrategy。
构建、版本契约及命令见 [二代环境](../env/sts2/README.md)。一代 checkpoint 不兼容二代候选动作模型。
