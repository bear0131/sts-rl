# 战斗规则模型第一版

这一版单独训练战斗规则理解，不接入强化学习策略。模型接收动作前的完整战斗状态、完整动作、整副牌组规则和整套遗物规则，自回归输出：

```text
<OUTPUT_BEGIN>
<EFFECTS_BEGIN> 按模拟器执行顺序排列的原子状态变化 <EFFECTS_END>
<NEXT_STATE_BEGIN> 动作后的完整战斗状态 <NEXT_STATE_END>
<OUTPUT_END>
```

每行 JSONL 是一条完整动作样本，主要字段如下：

```text
state       动作前状态
action      完整动作，包括卡牌、目标或选牌集合
rules.deck  整副牌组及每种卡牌效果的一句话
rules.relics 整套遗物效果的一句话
effects     相邻原子快照的有序变化
next_state 动作后状态
```

## 采集

先构建带原子跟踪能力的专用 `slaythespire` 扩展。该开关默认关闭，因此普通 combat/wholerun 构建没有跟踪开销：

```powershell
$pythonExe = (Get-Command python).Source
cmake -S sim/sts_lightspeed -B sim/sts_lightspeed/build-rule-v1 -G Ninja `
  -DSTS_ENABLE_RULE_TRACE=ON "-DPYTHON_EXECUTABLE=$pythonExe"
cmake --build sim/sts_lightspeed/build-rule-v1 --target slaythespire -j 4
```

再从仓库根目录运行：

```powershell
$env:STS_LIGHTSPEED_BUILD_DIR = "D:\sts-rl\sim\sts_lightspeed\build-rule-v1"
$env:STS_LIGHTSPEED_DLL_DIRS = "D:\Dev-Cpp\MinGW64\bin"
python -m rl.collect_rule_data --states 10000 --out rl/data/rules-v1.jsonl
```

## Current v7 contract

The v7 tree is the active vector-rule training format. Cards are encoded from
anonymous rule programs; relics are not encoded as rule programs. The state
vector contains fixed binary relic-presence slots, and a stateful relic's exact
counter is returned only after an explicit environment read in a
`RELIC_COUNTER` node.

Each changed state impact has its own sparse `DELTA_SLOT`. The node's aggregate
`delta` is retained only to advance the environment snapshot between sibling
nodes. HP fraction follows HP, and block-presence follows block, so those
derived features do not create unrelated delta targets.

The local executor always emits the complete damage/block chain, including a
zero-block target. Runtime read responses and the response-to-register copy
table are environment-owned and masked from loss. The shared transformer is
trained on the read request and the subsequent symbolic calculations, including
their `CALC_VALUE` results.

每场战斗都会重新随机楼层、牌组、遗物和敌人，不进行车轮战。主轨迹随机选择合法动作推进；每个决策点会复制战斗状态并执行所有合法完整动作。多选牌会枚举合法子集；数据量过大时可用 `--max-actions-per-state` 随机限量。

## 训练

```powershell
python -m rl.train_rules rl/data/rules-v1.jsonl --out rl/runs/rules-v1
```

训练只在输出序列上计算下一个符号损失，并分别报告作用序列和下一状态的符号准确率。训练集和验证集按整场战斗切分，避免同一状态的不同枚举动作泄漏到两边。输出目录包含：

```text
tokenizer.json       固定词表
model_config.json    模型结构
run_config.json      训练参数
best.pt              验证损失最低的检查点
last.pt              最后一轮检查点
```

编码器最后一个有效输入位置的向量由 `RuleWorldModel.rule_latent()` 返回，供后续接入 split 策略时融合到手牌表示。当前版本不会用模型想象搜索未来，也不会修改 PPO 的决策方式。
