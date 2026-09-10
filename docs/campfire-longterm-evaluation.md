# 火堆长期评测入口

远端入口：

```bash
cd /root/autodl-tmp/sts-rl
./rl/run_campfire_longterm.sh
```

脚本会在 `checkpoints/wholerun-cardvec-step10k-random-bs262144-e10-n128-hierhp/`
中按文件名步数自动选择最新的 `hier_hp_ppo_*_steps.zip`，并在
`evaluations/campfire/<UTC时间>-step<步数>/` 新建结果目录。每次运行都不会覆盖
之前的结果。默认使用 CPU、80 局、seed 30000；可通过环境变量覆盖：

```bash
STS_CAMPFIRE_EPISODES=200 \
STS_CAMPFIRE_SEED=50000 \
STS_CAMPFIRE_DEVICE=cpu \
./rl/run_campfire_longterm.sh
```

可调整的变量包括 `STS_CAMPFIRE_CHECKPOINT_DIR`、`STS_CAMPFIRE_OUTPUT_ROOT`、
`STS_CAMPFIRE_EPISODES`、`STS_CAMPFIRE_SEED`、`STS_CAMPFIRE_ASCENSION`、
`STS_CAMPFIRE_DEVICE`、`STS_CAMPFIRE_BATCH_SIZE`、`STS_CAMPFIRE_MAX_STEPS` 和
`STS_CAMPFIRE_PROGRESS_EVERY`。

## 2026-09-03 复测结果

测试配置：80 局，seed 30000，A0，CPU。评测期间训练继续运行。

评测 checkpoint：
`hier_hp_ppo_550502400_steps.zip`。

结果目录：
`/root/autodl-tmp/sts-rl/evaluations/campfire/20260903T045014Z-step550502400`

80 局中有 79 局死亡、1 局获胜，共收集 396 个真实火堆状态。将预测限制在
每个状态的合法火堆选项后，选择分布为：

| 行动 | 次数 | 比例 |
| --- | ---: | ---: |
| 休息 `rest` | 365 | 92.2% |
| 锻造 `smith` | 20 | 5.1% |
| 召回 `recall` | 5 | 1.3% |
| 举重 `lift` | 3 | 0.8% |
| 吞噬 `toke` | 0 | 0.0% |
| 挖掘 `dig` | 0 | 0.0% |
| 离开 `proceed` | 3 | 0.8% |

按生命值比例分桶：

| 生命值区间 | 状态数 | 休息 | 锻造 | 召回 | 举重 | 离开 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0%–35% | 49 | 89.8% | 8.2% | 2.0% | 0.0% | 0.0% |
| 35%–55% | 47 | 93.6% | 4.3% | 2.1% | 0.0% | 0.0% |
| 55%–75% | 79 | 97.5% | 2.5% | 0.0% | 0.0% | 0.0% |
| 75%–100% | 221 | 90.5% | 5.4% | 1.4% | 1.4% | 1.4% |

总体比例会受到某些选项不合法的影响。更关键的条件统计是：休息在 370 个状态
中合法，模型选择了 365 次，`P(rest | rest legal)=98.6%`；在生命值 75%–100%
且休息合法的 200 个状态中，模型 200 次全部选择休息。锻造在 350 个状态中
合法，但只选择 20 次，其中 17 次发生在休息不合法时。`toke` 和 `dig` 分别有
6 和 4 个合法状态，但模型仍全部选择休息。

未限制到火堆动作子集时，396 个状态中也没有一次选择其他 phase 的旁路动作
（`raw_side_actions=0`），因此休息偏置不是评测 mask 造成的。当前模型对生命值
变化有输入，但火堆决策没有形成合理的生命值单调关系，高血量时的行为尤其异常。

20 次锻造的主要目标为 `Defend_R+0` 9 次、`Uppercut+0` 3 次、`Reaper+0`
2 次，其余目标各 1 次。完整逐状态概率和选择保存在 `predictions.csv`，汇总、
血量分桶和锻造目标分别位于 `summary.csv`、`hp_bins.csv` 和
`smith_targets.csv`。

与 2026-09-02 的 `314572800` 步结果相比，总体休息率从 91.8% 变为 92.2%，
而“休息合法时选择休息”的比例从 95.3% 增至 98.6%。两次虽然使用相同 seed，
但各自使用被测策略完整跑局收集状态，实际访问的状态集合不同，因此该差异只能作
描述性比较，不能当作同一状态集上的严格 checkpoint 对照。

## 2026-09-02 结果

测试时间：2026-09-02，80 局，seed 30000，CPU。

评测的最新已保存参数：
`hier_hp_ppo_314572800_steps.zip`。

结果目录：
`/root/autodl-tmp/sts-rl/evaluations/campfire/20260902T141635Z-step314572800`

共收集 354 个真实火堆状态。模型在合法火堆选项上的选择比例为：

| 行动 | 比例 |
| --- | ---: |
| 休息 `rest` | 91.8% |
| 锻造 `smith` | 3.4% |
| 召回 `recall` | 4.0% |
| 举重 `lift` | 0.8% |
| 吞噬 `toke` | 0.0% |
| 挖掘 `dig` | 0.0% |
| 离开 `proceed` | 0.0% |

按生命值比例分桶后，休息选择率分别为：`00-35%` 76.5%、`35-55%`
84.4%、`55-75%` 94.4%、`75-100%` 96.8%。本次样本没有出现带
`Peace Pipe` 或 `Shovel` 的合法火堆，因此 `toke` 和 `dig` 为零表示没有可选状态，
不是模型在这些状态下拒绝选择它们。

详细结果文件包括 `summary.csv`、`predictions.csv`、`hp_bins.csv`、
`smith_targets.csv` 和 `collection.csv`。
