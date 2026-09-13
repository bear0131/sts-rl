# 当前 NOSL-IS-AZ 训练记录

更新时间：2026-09-13（中国标准时间）

本文记录当前正在运行的四机战斗内层训练。训练正在继续运行，本文中的参数以实际启动脚本和 W&B run 配置为准；不要据此停止或重启现有进程。

## 实验身份与状态

| 项目 | 当前值 |
| --- | --- |
| 方法 | NOSL-IS-AZ，C++ simulator，严格论文式 Full Gumbel |
| run name | `nosl-paper-full-gumbel-4way-s128-randomstart-boundary-lr1e4-replay-u40-no-end-turn-leaf-fresh10-20260913` |
| W&B run id | `fgs128u40replayfresh10` |
| W&B | [9njcxc7g](https://wandb.ai/1283802076-tsinghua-university/sts-rl/runs/9njcxc7g) |
| 总目标步数 | `10,000,000` 全局环境步 |
| 当前状态 | 四个 rank 正在运行；每台机器 25 个 Python worker 进程；没有发现训练错误 |
| 启动脚本 | `rl/launch_nosl_paper_replay_fresh10_20260913.sh` |

训练使用四个 rank 的 SSH ring 做梯度同步。每个 rank 使用一台 25 CPU 机器；rank 之间不是把一个环境拆成四份，而是各自产生 rollout，随后同步内层 combat specialist 的梯度。

## 远端拓扑

| rank | SSH 端点 | ring 端口 |
| ---: | --- | ---: |
| 0 | `connect.bjb2.seetacloud.com:12009` | `29788` |
| 1 | `connect.bjb2.seetacloud.com:17944` | `29788` |
| 2 | `connect.bjb2.seetacloud.com:13500` | `29788` |
| 3 | `connect.bjb2.seetacloud.com:44892` | `29788` |

远端仓库目录为 `/root/autodl-tmp/sts-rl`。每个 rank 使用 `/root/autodl-tmp/sts-rl/sim/sts_lightspeed/build-gumbel-current` 中的 C++ simulator，并通过 `PYTHONPATH` 加载该构建。

## 训练目标与冻结范围

- 外层 WholeRun policy 保持冻结，只训练 combat specialist 的 policy/value 参数。
- 训练从带 boundary observation feature 的 step-0 checkpoint 开始：
  `/root/autodl-tmp/sts-rl/migration/nosl-az/nosl_az_random_start_s512_boundary_step0.zip`。
- combat specialist 是随机初始化的；step-0 checkpoint 只提供已训练好的外层/共享模型结构和初始化所需的兼容字段。
- shared tokenizer、overworld specialist 以及外层 actor/critic 参数不参与优化。
- 战斗 value 的监督目标是战斗结束时玩家最终生命值的归一化值 `final_player_hp / 200`，不是搜索根节点的 bootstrap value。
- policy 监督目标是 Full Gumbel 搜索得到的 root visit policy。
- 当前 `end_turn_as_leaf=false`，因此不会在敌方回合结束的边界提前截断；搜索会继续进入后续状态。

## Rollout 与并行规模

| 参数 | 值 |
| --- | ---: |
| `world_size` | `4` |
| `n_envs` / rank | `24` |
| `envs_per_worker` | `1` |
| worker 数 / rank | `24` |
| `n_steps` / env | `128` |
| 每 rank rollout | `24 x 128 = 3,072` env steps |
| 全局 rollout | `4 x 24 x 128 = 12,288` env steps |
| `total_steps` | `10,000,000` |

这里的 `n_steps` 是外层环境 rollout 长度，不是 MCTS/Full Gumbel 的模拟次数。每次战斗决策会另外进行指定数量的搜索 simulations。

## NOSL 与 Full Gumbel 搜索

### 搜索参数

```text
nosl_is_az                 = true
nosl_is_az_cpp             = true
nosl_is_az_search_mode     = full_gumbel
nosl_is_az_paper_full_gumbel = true
nosl_is_az_simulations     = 128
nosl_is_az_depth           = 0          # 不设置深度截断，按节点展开
nosl_is_az_leaf_batch_size = 1          # 顺序参考实现
nosl_is_az_gumbel_candidates       = 16
nosl_is_az_gumbel_scale            = 1.0
nosl_is_az_gumbel_value_scale      = 0.1
nosl_is_az_gumbel_max_visit_init   = 50
nosl_is_az_gumbel_rescale_q        = true
nosl_is_az_gumbel_use_mixed_value  = true
nosl_is_az_prior_temperature       = 1.0
```

严格论文模式使用根部的 Gumbel-plus-prior 候选筛选和 considered-visit schedule。最多保留 `K=16` 个合法候选；根部的最终 policy target 是搜索访问分布，训练 target 不把采样出的 Gumbel 噪声本身加入 softmax。

非根决策节点使用论文式 completed-Q 的内部选择，不使用此前实验中的额外 interior coverage、minimum visits 或 progressive mean-to-max 保护项。未访问边的 completed-Q 会由 raw value 与已访问边的 prior 加权统计得到，随后按当前 Full Gumbel 规则进行重标度。

### NOSL 语义

- 每个 simulation 对隐藏 RNG 重新 redeterminize；未来观测不是沿用当前世界线的固定 RNG。
- 状态 key 只按可观察信息区分；不同手牌、抽牌堆、弃牌堆或其他可观察字段不同，视作不同信息状态。
- 随机事件产生的不同隐藏世界线在树中分裂，并对相应结果做 chance averaging。
- C++ 负责 simulator 展开、合法动作 mask、NOSL 搜索树和统计量；模型前推使用 combat actor/critic，并在 GPU 上批量处理叶节点。

### 噪声

```text
nosl_is_az_dirichlet_alpha   = 0
nosl_is_az_dirichlet_epsilon = 0
```

本实验没有额外 Dirichlet root noise。Full Gumbel 根部自身仍有 Gumbel 随机扰动，这是搜索算法的一部分，不等同于 Dirichlet root noise。

## Replay buffer 与最新数据保护

```text
nosl_is_az_replay_enabled        = true
nosl_is_az_replay_capacity       = 131072     # 每 rank 的 root 样本数
nosl_is_az_replay_min_size       = 1
nosl_is_az_replay_max_old_fraction = 0.90
nosl_is_az_batch_size             = 24576      # 每 rank、每次 collection
nosl_is_az_micro_batch_size       = 2048
nosl_is_az_updates                = 40         # 每次 collection 的 optimizer update
```

Replay 是 FIFO 环形缓冲区，相当于按样本数维护的滑动窗口。每次 collection 的新 root 先加入窗口，再按以下规则采样：

1. 一个训练 batch 中来自旧 collection 的样本最多占 90%。
2. 当前 collection 的样本至少占 10%；新样本不足 quota 时先全部覆盖一次，再有放回补齐。
3. 没有可用 fresh root 时跳过该轮训练，不退化为全旧 replay batch。

因此，缓冲区填满后仍然会保留至少约 10% 的最新数据。`batch_size=24576` 是内层 AlphaZero loss 的有效 batch；`micro_batch_size=2048` 只是显存受限时的单次前推/反向上限，不改变有效 batch 大小。

已验证的一次后期状态：

```text
global_timesteps       = 1,880,064
replay_size            = 131,072
replay_fill_ratio      = 1.0
replay_old_fraction    = 0.8999837
replay_current_fraction= 0.1000163
updates                = 40
optimizer_steps        = 40
batch_size_used        = 24,576
value_target_std       = 0.13569
finite_fraction        = 1.0
```

## 优化器与模型更新

```text
learning_rate              = 1e-4
n_epochs                   = 1          # 外层 PPO rollout 兼容配置
gradient_accumulation_steps= 1
max_grad_norm              = 0.5
amp_mode                   = bf16
float32_matmul_precision   = high
```

NOSL loss 系数：

```text
nosl_is_az_policy_coef = 1.0
nosl_is_az_value_coef  = 0.5
```

每个 collection 先将新战斗 root 放入 replay，再执行 40 次内层更新。每次更新从 replay 取一个最多 24,576 root 的有效 batch，拆成最多 2,048 root 的 micro-batch；policy 使用搜索 visit target 的交叉熵，value 使用最终 HP target 的 MSE。外层环境本身仍按兼容的 rollout 配置运行：

```text
batch_size = 1536
gamma      = 0.999
gae_lambda = 0.95
clip_range = 0.2
ent_coef   = 0.01
vf_coef    = 0.5
```

这些 PPO 字段主要是 checkpoint/rollout 兼容配置，不代表 combat specialist 每次只用 1,536 个样本或只更新一次；真正的内层更新口径是上一节的 `24,576 x 40`。

## 模型结构与推理

```text
policy class = SplitStSPointerPolicy
d_model      = 256
n_layers     = 8
n_heads      = 8
map_dag      = true
card_language  = false
combat_language= false
```

combat 搜索叶节点优先调用 `combat_actor_critic`。训练中的 C++ search 输出连续的 leaf observation buffer，再进行批量模型推理，避免逐字段 Python 字典在 simulator、CPU 和 GPU 之间反复传输。

## W&B 监控重点

重点查看以下指标：

- `global_timesteps`、`optimizer_steps`、`env_steps_per_second`；
- `nosl_is_az/replay_old_fraction`、`replay_current_fraction`、`replay_fill_ratio`；
- `nosl_is_az/batch_size_used`、`nosl_is_az/updates`；
- `value_target_mean/std`、`pred_value_mean/std`、`value_mae`、`value_bias`；
- `target_entropy`、`target_policy_kl`、`target_top1_agreement`；
- `finite_fraction`、CUDA OOM 或 rank barrier 错误。

训练曲线中 `pred_value_std` 变大不等于训练停止；它可能表示 combat specialist 开始区分不同最终 HP 的状态。应结合 `value_mae`、`value_bias`、最终战斗表现以及 fresh/replay 比例共同判断。

## Checkpoint 与本地副本

当前训练已下载的一份完整 checkpoint：

```text
D:\sts-rl\artifacts\nosl-paper-full-gumbel-replay-fresh10-20260913\nosl_az_replay_fresh10_1843200_steps.zip
```

SHA-256：

```text
cf8336a7a01a736d6ce36bff9726cc42b305f2235a0c3f23ed1d1b8e25470dd0
```

该文件是 `1,843,200` global env steps 的已保存快照，不代表当前远端已经停止或当前最新步数。训练脚本设置 `save_every_updates=25`、`max_checkpoints=2`，远端 checkpoint 目录为：

```text
/root/autodl-tmp/sts-rl/checkpoints/nosl-paper-full-gumbel-4way-s128-randomstart-boundary-lr1e4-replay-u40-no-end-turn-leaf-fresh10-20260913/
```

## 复现命令骨架

单个 rank 的启动入口是：

```text
bash rl/launch_nosl_paper_replay_fresh10_20260913.sh <rank>
```

完整命令必须保持四个 rank、同一个 `run_id=fgs128u40replayfresh10`、同一个 ring port、相同 checkpoint 和相同 replay 参数。重新实验前应先确认旧进程状态和 W&B run，避免两个 rank 集合使用同一个 ring/run id。

