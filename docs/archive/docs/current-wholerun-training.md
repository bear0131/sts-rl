# 当前 WholeRun 训练方法与卡牌向量初始化记录

更新时间：2026-09-05（中国标准时间）

本文记录上一组远端四机 WholeRun 训练的实际方法、恢复日志和卡牌向量初始化位置。该组训练已暂停并归档；当前进行的实验见 `docs/nosl-is-az-training.md` 中的四机 NOSL-IS-AZ 记录。这里的“参数”指运行命令传入的参数；checkpoint 内保存的旧超参数可能不同，见下文说明。

## 1. 已暂停并归档的运行

| 项目 | 当前值 |
| --- | --- |
| 任务 | WholeRun PPO，战斗内/战斗外分层回报 |
| policy | `SplitStSPointerPolicy` |
| run id | `wrcv10h` |
| W&B run | `67vmdrx8` |
| W&B 地址 | <https://wandb.ai/1283802076-tsinghua-university/sts-rl/runs/67vmdrx8> |
| 远端仓库 | `/root/autodl-tmp/sts-rl` |
| 远端日志目录 | `/root/autodl-tmp/sts-rl/logs/wholerun-cardvec-step10k-random-bs262144-e10-n128-hierhp/` |
| 远端 checkpoint 目录 | `/root/autodl-tmp/sts-rl/checkpoints/wholerun-cardvec-step10k-random-bs262144-e10-n128-hierhp/` |
| 目标总步数 | `10000000000`（全局环境步） |
| 当前恢复起点 | `1048576000`（全局环境步） |
| 最近验证的运行步数 | `1049624576`（本次恢复后的 update 0） |
| 最近可恢复 checkpoint | `hier_hp_ppo_1048576000_steps.zip` |

该运行的最后一次保存参数已另行归档为
`artifacts/paused-wrcv10h-20260905/hier_hp_ppo_1205862400_steps.zip`，SHA-256 为
`a91f85e7e125cffbe2361c748b0c8ae254d2fba3fce77de3999e47c0f29bb58e`；它不再继续
占用四台机器。新的四机训练不从该 WholeRun PPO checkpoint 续训，而是冻结已有
外层策略、单独训练战斗内 specialist。

暂停前最后一次状态检查时四个 rank 均在运行，显存约 16 GiB，未再出现 CUDA OOM。四个 rank 的 SSH 端点为：

```text
rank 0  connect.bjb2.seetacloud.com:21496
rank 1  connect.bjb2.seetacloud.com:20862
rank 2  connect.bjb2.seetacloud.com:15557
rank 3  connect.bjb2.seetacloud.com:46144
```

## 2. 数据与环境

当前 WholeRun 不读取旧的规则训练 JSONL 数据集。每个 rank 直接启动 128 个环境，从杀戮尖塔模拟器在线产生新的状态和转移；一轮 rollout 结束后立即用于 PPO 更新。

环境包装顺序为：

```text
SlayTheSpireEnv
  -> ShapedRewardWrapper
  -> TimeLimit(max_episode_steps=_MAX_EPISODE_STEPS)
```

本次使用默认的非语言观察：`card_language=false`、`combat_language=false`。卡牌规则向量以冻结的独立 encoder 提供，不在 WholeRun 中重新生成规则 token。

### 分层战斗回报

当前启用了 `--hierarchical-combat-hp`：

- 战斗阶段是低层 option。战斗结束时，若不是中断战斗，终止奖励取当前战士生命值的固定全局尺度；战斗中的普通点击奖励为 0。
- 战斗外阶段把整场战斗视作一个宏观转移，累计原始奖励后从战斗结束后的状态 bootstrap。
- PPO loss 对 combat 和 overworld 两类有效样本分别求均值，再等权合并，避免战斗点击数量淹没地图、商店和事件决策。

## 3. 模型结构

当前 checkpoint 的 policy geometry 为：

```text
policy                 SplitStSPointerPolicy
d_model                256
n_layers               8（组合层数）
n_heads                8
map_dag                true
combat specialist      4 Transformer layers
overworld specialist   4 Transformer layers
```

`SplitStSPolicyCore` 按 `phase` 路由样本。战斗样本只进入 combat encoder；地图、商店、事件、奖励等非战斗阶段只进入 overworld encoder。卡牌、遗物和药水的物理实体 tokenization 由 shared semantic tokenizer 完成；两个 specialist 共享这部分输入处理，但 actor、critic 和 specialist encoder 参数分开。

## 4. PPO 与分布式训练

实际启动参数如下：

```text
world size                         4
n_envs per rank                    128
n_steps per env                    2048
rollout per rank                   128 * 2048 = 262144 transitions
global rollout per update          262144 * 4 = 1048576 transitions
micro batch size                   8192
gradient accumulation               8
n_epochs                           10
effective batch per optimizer step 8192 * 8 * 4 = 262144 transitions
optimizer steps per rollout        (262144 / 8192 / 8) * 10 = 40
learning rate                      1e-4
gamma                              0.999
gae lambda                         0.95
entropy coefficient                0.01
value coefficient                  0.5
clip range                         0.2
AMP                                BF16 autocast，参数和同步梯度保持 FP32
float32 matmul precision           high
reasoning                          关闭
```

这里的有效 batch 是全局值：每个 rank 先对 8 个 8192 样本 micro-batch 累计梯度，然后通过 SSH ring 对四个 rank 做梯度平均。因此降低 micro-batch 后，统计意义上的有效 batch 没有改变，但单次前向/反向显存峰值显著降低。

梯度传输使用 `ssh-ring`，每个 rank 保持到下一个 rank 的持久 SSH 通道，环上的梯度平均完成后再执行 optimizer step。W&B 只由 rank 0 写入，四个 rank 的游戏统计和训练统计在写入前按固定指标集合做聚合。

当前没有独立的离线 validation 数据流；W&B 中的 `loss`、`combat_value_loss_mean`、`overworld_value_loss_mean`、rollout 速度和游戏进度指标来自在线 rollout。每个 rollout 在 PPO 内部按 `n_epochs=10` 重复遍历，这是同一轮 PPO 更新的内部 minibatch 重用，不是重复读取固定数据集。

## 5. 训练恢复日志

之前使用较大的 `batch_size=16384` 和累计步数 4 时，rank 2 在约 3.4 亿全局步附近发生 CUDA OOM；其他 rank 因 SSH ring 对端关闭而连带退出。该错误不是模型主动完成训练。

恢复时保留了原始权重和优化器状态，从：

```text
hier_hp_ppo_340787200_steps.zip
```

重新启动，并只覆盖 rollout buffer/训练超参数为 `8192 + accumulation 8`。恢复后的 update 0、1、2、3 均正常完成，随后持续运行到 update 56。启动参数中的 `--wandb-id 67vmdrx8` 使 W&B 继续写入同一个 run。

checkpoint 每 25 个 update 保存一次，并最多保留两个文件。因此内存中的最新步数可能高于磁盘上的最近 checkpoint；例如本文记录时运行到 `400556032`，磁盘上最近完整 checkpoint 为 `393216000`。恢复前应使用 rank 0 的最新文件，并同步到其他 rank。

2026-09-04 在 `hier_hp_ppo_1048576000_steps.zip` 保存完成后暂停四个 rank，
将该 checkpoint 从 rank 0 同步到另外三台机器，并验证四份 SHA-256 均为：

```text
3f7519e171fd870d832842cf9037fa0f430cc51b302e37efc2395bd970cc0ea7
```

随后从该 checkpoint 恢复，将学习率从 `3e-4` 降为 `1e-4`，其余训练参数不变，
并继续使用 W&B run `67vmdrx8`。旧日志分别保存在各机器的：

```text
logs/wholerun-cardvec-step10k-random-bs262144-e10-n128-hierhp/
  rank<N>.log.pre-lr1e-4-step1048576000
```

恢复后的首轮 update 已在四个 rank 完成：起点 `1048576000`，完成后全局步数
`1049624576`，每个 rank 均执行 40 个 optimizer step，SSH ring 正常。W&B 配置已
确认 `learning_rate=0.0001`、`start_timesteps=1048576000`。

## 6. 卡牌向量初始化存储位置

### 独立 card-rule encoder checkpoint

当前 WholeRun 使用的卡牌向量 checkpoint 是：

```text
远端：/root/autodl-tmp/sts-rl/.tmp-vector-rule-v7-contract-v10-step10000/last.pt
本地镜像：D:\sts-rl\.tmp-vector-rule-v7-contract-v10-step10000\last.pt
```

本地和远端 `last.pt` 的 SHA-256 为：

```text
C76A88C461A4C486BB7353E44E889D50F7D5BD9162AEC0B694C7F039B4B85A94
```

该文件是独立规则向量训练在 `step=10000` 保存的 checkpoint，内部元数据为
`schema_version=7`、`executor_contract_version=10`、独立训练有效 batch size
`1024`、梯度累计 `8`。它不是 WholeRun 的 PPO checkpoint。

本地目录还保留了便于检查的旁车文件：

```text
D:\sts-rl\.tmp-vector-rule-v7-contract-v10-step10000\config.json
D:\sts-rl\.tmp-vector-rule-v7-contract-v10-step10000\tokenizer.json
```

WholeRun 初始化只需要 `last.pt`。`last.pt` 内部包含 `config` 和 `model`，其中 `model` 的 `card_encoder.*` 参数会被提取；旁车的 `config.json` 和 `tokenizer.json` 不会被 WholeRun 运行时单独读取。

### card encoder 配置

当前 `config.json` 中的 card encoder 为：

```text
token_vocab_size  134
token_dim        64
output_dim       128
n_heads           4
n_layers          2
feedforward     256
max_tokens      256
dropout           0
```

它接收匿名卡牌规则 DSL token，不接收卡牌名称或卡牌 id 的语义 embedding。规则 Transformer 先把一串规则 token 编码，再池化并投影为一个 128 维卡牌向量。

### WholeRun 的加载与冻结流程

分布式入口收到 `--card-vector-checkpoint` 后，流程为：

```text
load_frozen_card_rule_encoder(last.pt)
  -> 检查 DSL 词表和卡牌表大小
  -> 载入 card_encoder.*
  -> eval()
  -> 所有 encoder 参数 requires_grad=False
  -> 注入 SplitStSPolicyCore
```

`SharedSemanticTokenizer` 随后在初始化阶段、无梯度地预计算所有卡牌的两个版本：

```text
card_vector_cache.shape = [环境卡牌数, 2, 128]
                         = [card id, base/upgraded, vector dim]
```

该 cache 是 `persistent=False` 的运行时 buffer，不会再次保存进 WholeRun checkpoint；重新启动时会从上面的 `last.pt` 重新构造。训练期间每次只做表查找，不重复执行 card-rule Transformer。

卡牌向量进入 WholeRun 前仍有一个可训练投影：

```text
[128 维冻结卡牌向量 + 1 维升级标记] -> Linear(129, 256)
```

所以 128 维是独立 encoder 的输出维数，256 维是 WholeRun Transformer 的统一 `d_model`；投影层和其后的 policy/value 参数参与 PPO 训练，卡牌 encoder 本身不更新。

## 7. 当前恢复命令

下面的命令对应本文记录的运行配置。远端路径必须使用 Linux 路径；本地脚本会把命令发往四个 rank。

```powershell
.\rl\launch_distributed_wholerun.ps1 `
  -Action start `
  -RunId wrcv10h `
  -Checkpoint /root/autodl-tmp/sts-rl/checkpoints/wholerun-cardvec-step10k-random-bs262144-e10-n128-hierhp/hier_hp_ppo_1048576000_steps.zip `
  -CardVectorCheckpoint /root/autodl-tmp/sts-rl/.tmp-vector-rule-v7-contract-v10-step10000/last.pt `
  -HierarchicalCombatHp `
  -NEnvs 128 `
  -NSteps 2048 `
  -BatchSize 8192 `
  -GradientAccumulationSteps 8 `
  -NEpochs 10 `
  -LearningRate 1e-4 `
  -TotalSteps 10000000000 `
  -WandbName wholerun-cardvec-step10k-random-bs262144-e10-n128-hierhp `
  -WandbId 67vmdrx8
```

状态查看：

```powershell
.\rl\launch_distributed_wholerun.ps1 `
  -Action status `
  -RunId wrcv10h `
  -WandbName wholerun-cardvec-step10k-random-bs262144-e10-n128-hierhp
```
