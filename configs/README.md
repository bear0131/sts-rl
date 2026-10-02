# 参数预设

`.args` 是 argparse 原生 response file：每行一个原样参数，不写注释、不写 shell 引号、不展开环境变量。入口添加 `@文件` 即可；后面的同名选项可覆盖前面的值。

`train/entity_combat_mature_replay.args` 是已训练模型的独立NOSL学习实验：继承groups runtime的全部搜索配置和96环境×128步；学习率1e-4、每轮16次更新、每rank逻辑batch32768/微批8192、524288容量replay中当前轮至少25%、group CE与value分布损失各权重1、梯度裁剪1。四rank合计每次更新131072个样本；每轮训练样本访问2097152次，不是新增环境transition数。此为待比较的组合实验，不是已验证最优值。用完整bundle的`--fork-state`和新run/W&B身份启动，保留Adam/replay/RNG/环境，不重启预热。分布式入口的`--nosl-is-az-policy-coef`、`--nosl-is-az-value-coef`分别控制这两个损失，省略时保留checkpoint值；PPO的`--vf-coef`和`--ent-coef`不控制NOSL监督损失。

- `train/`：三个已存在 NOSL 实验的训练预算/搜索/replay/optimizer 参数，原 launcher 逐项迁出。不是建议现在重启实验。
- `eval/`：policy 与严格 Full Gumbel 固定小场景冒烟；训练 128 simulations 和烟测 8 simulations 是不同用途。
- 地址、rank、ring、checkpoint、W&B run 身份仍在 `experiments/launchers/` 部署脚本。其余历史脚本是尚未规范化的可执行部署预设，不能当作算法实现。

例：`python -m rl.evaluation.eval CHECKPOINT @configs/eval/policy_smoke.args`。
严格场景：`python -m rl.evaluation.nosl.eval_full_gumbel_bash_clash CHECKPOINT @configs/eval/strict_full_gumbel_smoke.args --output runs/local/strict-eval.json`。
这些评估自身未全部实现统一 wall timeout，自动化运行需父进程 timeout（见 tools.validation 和 [tests/README.md](../tests/README.md)）。

当前实验对应 `train/nosl_paper_replay_fresh10_20260913.args`：strict full_gumbel、depth=0、leaf batch=1、replay 131072、old fraction<=0.90、batch 24576、40 updates、microbatch2048、24 env/rank。冻结范围和模型结构由 checkpoint/训练代码共同决定，不能只看 .args 推断。

`eval/nosl_scenarios.py` 是纯配置：固定卡对、填充牌和 deck profile；从两个旧 eval 脚本原样提取。运行状态构造/搜索/计分仍在 evaluation，不能把含执行算法的 eval_*.py 整体伪装成配置。
