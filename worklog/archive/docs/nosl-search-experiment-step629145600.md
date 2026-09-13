# NOSL combat search 初步实验

更新时间：2026-09-03

本实验使用远端 WholeRun 最新完整 checkpoint，在相同 PPO deterministic 轨迹上采集 combat 状态，并对每个采样状态运行无模型 Flat Monte Carlo。搜索目标只有战斗结束后的预期剩余血量；未来 RNG 由 NOSL sampler 独立生成，不读取或延续当前世界线的 RNG 状态。

## 参数与复现信息

| 项目 | 值 |
| --- | --- |
| WholeRun checkpoint | `hier_hp_ppo_629145600_steps.zip` |
| 全局训练步数 | `629145600` |
| checkpoint SHA-256 | `bf97d064a7b4e3363f16adcad328c2b978a638b0ba5241c6d4a62acdc9a989e9` |
| card-vector checkpoint SHA-256 | `c76a88c461a4c486bb7353e44e889d50f7d5bd9162aec0b694c7f039b4b85a94` |
| sim build | `sim/sts_lightspeed/build-nosl311/slaythespire.cp311-win_amd64.pyd` |
| observation | `card_language=false`, `combat_language=false` |
| sampler | `blake2b-nosl-v1`, master nonce `0x4E4F534C7631` |
| 轨迹 | PPO deterministic，seed `910000..910003` |
| Flat-MC | 每个合法根动作 8 次独立样本，最多 128 个 thin steps |

运行命令（PowerShell）：

```powershell
$env:STS_LIGHTSPEED_BUILD_DIR=(Resolve-Path 'sim/sts_lightspeed/build-nosl311')
& .\.venv\Scripts\python.exe -m rl.collect_nosl_corpus `
  .tmp-hier_hp_ppo_629145600_steps.zip `
  --card-vector-checkpoint .tmp-vector-rule-v7-contract-v10-step10000\last.pt `
  --episodes 4 --seed 910000 --device cpu --max-states 32 `
  --mc-samples 8 --mc-max-steps 128 `
  --output .tmp-nosl-corpus-step629145600.jsonl `
  --summary .tmp-nosl-corpus-step629145600-summary.json

& .\.venv\Scripts\python.exe -m rl.compare_nosl_corpus `
  .tmp-nosl-corpus-step629145600.jsonl `
  --output .tmp-nosl-corpus-step629145600-compare.json
```

## 语料契约

`rl/collect_nosl_corpus.py` 每行保存一条 combat 状态：

- checkpoint、episode seed、thin step 和从 reset 开始的 `action_prefix`；
- policy observation、当前合法动作 mask、PPO deterministic action；
- `public_canonical` 调试视图；draw/discard/exhaust 均按卡牌 multiset 排序，不保留隐藏顺序；
- NOSL master/state nonce、每个根动作的 terminal HP 均值、完成数和超时数。

采集器会拒绝 `card_language=true` 的模型，因为该模式包含逐张 draw-pile 序列，不符合本实验的公开 observation 契约。原始 `BattleContext` 不写入语料；动作前缀是后续独立 replay 的唯一状态重建接口。

每个反事实分支先 clone 可见 combat 状态，再调用 `NOSLSampler.clone_and_redeterminize()`。C++ binding 会为 ai/card/misc/monster-hp/potion/shuffle 等流生成新种子，并先按卡牌 multiset 规范化 draw pile，再生成新的 Java-compatible permutation。因此同一状态的兄弟分支不会共享当前世界线的未来抽牌顺序。

## 初步结果

语料文件：`.tmp-nosl-corpus-step629145600.jsonl`（32 条状态）。摘要文件：`.tmp-nosl-corpus-step629145600-summary.json`。比较文件：`.tmp-nosl-corpus-step629145600-compare.json`。

| 指标 | 结果 |
| --- | ---: |
| combat 状态总数（4 局轨迹） | 1743 |
| 保存状态数 | 32 |
| NOSL 与 PPO 根动作差异率 | 40.625% |
| PPO 根动作的独立采样 terminal HP 均值 | 74.012 |
| NOSL 选择动作的独立采样 terminal HP 均值 | 75.383 |
| NOSL 相对 PPO 根动作的均值增益 | +1.371 HP |
| 状态中 NOSL 增益为正的比例 | 34.375% |
| Flat-MC 超时率 | 0% |

这只是固定状态集、随机后续策略和 8 次采样的条件估计，不等价于整局胜率或实际部署收益。4 局 PPO 轨迹本身均在 act 2 死亡；该事实只描述采样轨迹，不能作为 NOSL 策略的胜率结论。

另外用同一 checkpoint 做了 1 局真实 `CombatValueSearch` smoke（depth=1、node budget=16）。搜索默认开启 NOSL，549 个 combat 决策中有 55 次改变 PPO argmax（10.02%），整局在 floor 33 死亡；该运行用于验证分支契约和运行时开销，不作为策略效果结论。

## 单场完整战斗评估

从语料 `record_id=0` 的 episode seed 和 action prefix 重放到同一个 combat 状态，然后对 PPO 和 Flat-MC 分别运行完整战斗。每个 trial 使用相同的 NOSL future nonce，因而是 paired comparison；Flat-MC 在每个 thin combat decision 都重新搜索，搜索后的实际战斗继续执行 Flat-MC 选择的动作。

| 配置 | PPO 平均 projected exit HP | Flat-MC 平均 projected exit HP | 平均差值（Flat-MC - PPO） | PPO 胜率 | Flat-MC 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 trials，4 MC samples/decision | 80.00 | 78.75 | -1.25 | 100% | 100% |
| 8 trials，8 MC samples/decision | 80.00 | 79.25 | -0.75 | 100% | 100% |

结果文件：`.tmp-single-combat-step629145600.json` 和 `.tmp-single-combat-step629145600-mc8.json`。这说明单步 Flat-MC 估计中观察到的正增益没有直接转化为完整战斗收益；在这个初始战斗上，PPO 已经能稳定无伤通关，Flat-MC 只是在部分 trial 中额外受到 2--4 点伤害。提高 MC 样本数后差距缩小，但当前证据仍不支持 Flat-MC 优于 PPO。

## Act 1/2 分层多方法完整战斗评估

为避免把当前尚未稳定的 Act 3 能力混入比较，本轮评估显式使用 `--max-act 2`，只保留 Act 1 和 Act 2。语料来自 20 个 PPO seed，共保存 200 个 combat 起点；在指定楼层中按楼层 round-robin 抽取 20 个起点，并优先保留 Act 1 floor 16 和 Act 2 floor 33 的 boss。每个起点运行 2 个 paired NOSL future，因此每种方法有 40 个完整战斗样本，boss 子集有 14 个样本。

评估方法是 PPO continuation、uniform Random、Flat-MC、NOSL Beam Search 和 NOSL MCTS。搜索配置为 Flat-MC 每步 2 个样本，Beam depth/width 为 2/3，MCTS simulations/depth 为 8/2，随机 rollout 最多 96 个 thin steps；所有方法共享相同的 future nonce。`projected_exit_hp` 是战斗结束时的剩余血量，战斗失败记为 0；本轮没有发生 timeout。

复现命令（PowerShell）：

```powershell
$env:STS_LIGHTSPEED_BUILD_DIR=(Resolve-Path 'sim/sts_lightspeed/build-nosl311')
& .\.venv\Scripts\python.exe -m rl.eval_nosl_methods `
  .tmp-hier_hp_ppo_629145600_steps.zip `
  .tmp-nosl-stratified-step629145600.jsonl `
  --card-vector-checkpoint .tmp-vector-rule-v7-contract-v10-step10000\last.pt `
  --max-records 20 --max-act 2 --floors 1,8,16,18,24,27,33 `
  --trials 2 --methods ppo,random,flat_mc,beam,mcts `
  --mc-samples 2 --rollout-max-steps 96 --max-combat-steps 600 `
  --beam-depth 2 --beam-width 3 --mcts-simulations 8 --mcts-depth 2 `
  --output .tmp-methods-act2-step629145600-20c.json
```

### 总体结果

| 方法 | 样本数 | 平均终局 HP | 胜率 | 死亡率 | 超时率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PPO | 40 | 52.58 | 87.5% | 12.5% | 0% |
| Random | 40 | 20.58 | 57.5% | 42.5% | 0% |
| Flat-MC | 40 | 42.35 | 67.5% | 32.5% | 0% |
| Beam | 40 | 28.58 | 52.5% | 47.5% | 0% |
| MCTS | 40 | 38.20 | 60.0% | 40.0% | 0% |

在相同 record/future 配对后，搜索方法相对 PPO 的终局 HP 差值为：Flat-MC `-10.23`、MCTS `-14.38`、Beam `-24.00`、Random `-32.00`。按 20 个起点聚类、每个起点先平均 2 个 future 的 bootstrap 95% 区间分别为 Flat-MC `[-16.05, -5.18]`、MCTS `[-24.18, -6.25]`、Beam `[-33.78, -15.10]`、Random `[-41.15, -22.60]`；这些区间是当前样本规模下的描述性不确定性估计。

### 楼层结果

单元格格式为“平均终局 HP / 胜率”；`n` 是起点数（每个起点有 2 个 future）。

| Act/floor | 遭遇类型 | n | PPO | Random | Flat-MC | Beam | MCTS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1/1 | Small Slimes、Jaw Worm | 3 | 79.67 / 100% | 44.00 / 100% | 72.83 / 100% | 54.83 / 83.3% | 70.67 / 100% |
| 1/8 | Two Fungi Beasts、Exordium Thugs | 3 | 73.67 / 100% | 43.67 / 100% | 68.33 / 100% | 65.50 / 100% | 66.67 / 100% |
| 1/16（boss） | Slime Boss、The Guardian、Hexaghost | 4 | 34.13 / 87.5% | 0.00 / 0% | 18.13 / 50% | 2.00 / 12.5% | 6.25 / 25% |
| 2/18 | Shell Parasite、Two Thieves | 3 | 69.67 / 100% | 32.83 / 100% | 64.00 / 100% | 37.00 / 83.3% | 59.67 / 100% |
| 2/24 | Spheric Guardian、Three Cultist | 2 | 70.75 / 100% | 11.00 / 75% | 44.75 / 75% | 23.75 / 50% | 36.50 / 50% |
| 2/27 | Centurion And Healer、Cultist And Chosen | 2 | 38.75 / 100% | 14.00 / 50% | 34.75 / 50% | 22.00 / 50% | 37.50 / 50% |
| 2/33（boss） | Collector、Automaton | 3 | 9.00 / 33.3% | 0.00 / 0% | 0.00 / 0% | 0.00 / 0% | 0.00 / 0% |

### Boss 汇总

Act 1 floor 16 和 Act 2 floor 33 合计 7 个起点、14 个 paired samples：

| 方法 | 平均终局 HP | 胜率 | 死亡率 |
| --- | ---: | ---: | ---: |
| PPO | 23.36 | 64.3% | 35.7% |
| Flat-MC | 10.36 | 28.6% | 71.4% |
| MCTS | 3.57 | 14.3% | 85.7% |
| Beam | 1.14 | 7.1% | 92.9% |
| Random | 0.00 | 0% | 100% |

### 解释与限制

PPO 在总体、Act 1/2 的每个指定楼层以及 boss 子集都领先当前搜索配置；因此不能根据这批完整战斗结果认为 Flat-MC 比现有模型更好。这个结论与目标定义一致：PPO 的训练回报和搜索的 rollout 评分都是最大化战斗结束血量，差异来自策略质量和估计质量，而不是优化目标不同。

普通战斗中 Flat-MC 是搜索方法里最接近 PPO 的方法，但仍有约 10.2 HP 的 paired 劣势。boss 战的差距更大，说明浅层、低样本的随机 rollout 难以处理 boss 的长时序资源管理和高惩罚分支；Beam/MCTS 在本配置下没有弥补这一点。Act 2 floor 33 只有 3 个起点，楼层数字应视为方向性证据，不能替代更大 boss 样本。

本轮结果文件为 `.tmp-methods-act2-step629145600.json`（10 起点试跑）和 `.tmp-methods-act2-step629145600-20c.json`（20 起点主评估；后者使用相同配置扩充了起点数）。Act 3 不在本轮评估范围内。

## 扩大搜索范围（远端）

由于扩大搜索预算后本地 CPU 耗时过长，本组实验转移到远端 rank 0（RTX 5090，多核 CPU）执行。起点、paired future、方法集合和 Act 1/2 floor 筛选均与上一组完全相同，只增加搜索预算：Flat-MC 样本数 `2 -> 8`，Beam depth/width `2/3 -> 3/6`，MCTS simulations/depth `8/2 -> 32/3`，随机 rollout 上限 `96 -> 128`。远端 Linux NOSL binding 在运行前重新编译，7 个 NOSL/value-search 单测通过。

远端评估耗时 `919.19 s`，20 个起点、2 个 future、5 种方法共 200 行结果，Act 集合为 `[1, 2]`，指定 floor 集合为 `[1, 8, 16, 18, 24, 27, 33]`，timeout 为 0。结果文件为 `.tmp-methods-act2-step629145600-expanded-remote.json`。

### 扩展配置结果

| 方法 | 平均终局 HP | 胜率 | 死亡率 |
| --- | ---: | ---: | ---: |
| PPO | 52.58 | 87.5% | 12.5% |
| Flat-MC（扩展） | 50.48 | 77.5% | 22.5% |
| MCTS（扩展） | 45.63 | 75.0% | 25.0% |
| Beam（扩展） | 36.23 | 60.0% | 40.0% |
| Random | 20.58 | 57.5% | 42.5% |

在相同 paired 起点/future 上，扩展相对基线配置的平均终局 HP 变化为 Flat-MC `+8.13`、MCTS `+7.43`、Beam `+7.65`；PPO 和 Random 为 `0`。扩展后搜索方法相对 PPO 的差值为 Flat-MC `-2.10`、MCTS `-6.95`、Beam `-16.35`、Random `-32.00` HP。按起点聚类 bootstrap 95% 区间为 Flat-MC `[-6.83, 2.13]`、MCTS `[-11.95, -2.58]`、Beam `[-22.88, -10.40]`、Random `[-39.38, -24.75]`；Flat-MC 的区间包含 0，不能据此宣布超过 PPO。

### Boss 结果

Act 1 floor 16 和 Act 2 floor 33 的 7 个起点、14 个 paired samples：

| 方法 | 平均终局 HP | 胜率 | 死亡率 |
| --- | ---: | ---: | ---: |
| PPO | 23.36 | 64.3% | 35.7% |
| Flat-MC（扩展） | 26.43 | 57.1% | 42.9% |
| MCTS（扩展） | 17.07 | 50.0% | 50.0% |
| Beam（扩展） | 3.57 | 14.3% | 85.7% |
| Random | 0.00 | 0% | 100% |

扩展 Flat-MC 在 boss 子集的平均 HP 比 PPO 高 `+3.07`，但 paired bootstrap 95% 区间为 `[-3.72, 11.14]`，而且胜率仍低于 PPO；这应视为有价值的信号，不是确定性优越结论。逐 floor 看，Flat-MC 在 Act 1 floor 16（`34.50 vs 34.13 HP`）和 Act 2 floor 18（`70.00 vs 69.67 HP`）略高于 PPO，在 floor 33（`15.67 vs 9.00 HP`）也改善了基线搜索，但在 floor 1、8、24、27 仍低于 PPO。

### 解释

扩大搜索范围确实显著改善了搜索策略，尤其是 Flat-MC；它把总体差距从 `-10.23 HP` 缩小到 `-2.10 HP`，并把 boss 平均 HP 提升到 PPO 之上。但这不是“搜索已经优于 PPO”的充分证据：整体 paired 区间跨 0，胜率仍低 10 个百分点，且结果来自每起点仅 2 个 future。MCTS 也明显改善但仍落后，Beam 的提升不足以抵消其较弱的 rollout 估计。

这一结果支持下一步优先增加 Flat-MC 的 future 样本和 boss 起点，同时测量计算预算--收益曲线；在确认 Flat-MC 的优势能跨更多 seeds/futures 复现前，不应替换 PPO。所有扩展分支仍使用 NOSL 独立 RNG 和隐藏 draw pile multiset 规范化，Act 3 继续不纳入本轮实验。

## 后续实验

1. 继续只测 Act 1/2；将 boss 起点扩展到每个 boss 至少 8--16 个，并把 paired future 增加到 8--16 个。
2. 固定同一批起点和 future，分别提高 Flat-MC、Beam、MCTS 的 rollout 深度与预算，画出“计算预算--终局 HP”的曲线。
3. 将当前 PPO continuation 作为强基线，补充“搜索选第一步、之后 PPO continuation”的消融，区分搜索动作选择与随机 rollout policy 的影响。
4. 保持 NOSL sampler 的独立 RNG 和隐藏 draw pile multiset 规范化；任何不支持 `nosl_redeterminize()` 的分支都应继续拒绝执行。
