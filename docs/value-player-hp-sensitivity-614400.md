# 价值网络对自身 HP 的敏感性

测试 checkpoint：`artifacts/latest-nosl-az-s512-20260905/nosl_az_614400_steps.zip`。

基于同一个真实 Cultist 战斗 observation（seed 36，Bash+ + Strike，玩家最大 HP=80），
只改变玩家当前 HP 的编码字段：

- `player_scalars[0] = HP / 200`
- `player_scalars[5] = HP / max HP`
- `run_hp[0] = HP / 200`
- `run_hp[2] = HP / max HP`

敌人、手牌、能量、回合、牌堆和其他 observation 字段保持不变。原始结果见
`.tmp-value-player-hp-sensitivity-614400.json`。

| 玩家当前 HP | value |
|---:|---:|
| 1 | 0.23560479 |
| 10 | 0.23562458 |
| 20 | 0.23564613 |
| 40 | 0.23568803 |
| 60 | 0.23572820 |
| 80 | 0.23576665 |

统计量：

- `value(HP=1) - value(HP=80) = -0.00016186`
- 线性斜率约为 `+2.05e-6 / player HP`
- value 随玩家 HP 增加严格单调增加
- HP=80 是原始状态，因此该行没有人为改变字段

结论：价值网络确实对自身 HP 敏感，且方向符合“最大化玩家血量”的目标，但影响幅度
仍然很小：80 点 HP 变化只造成约 `1.6e-4` 的 value 变化（相对 value 约 0.07%）。
和敌人 HP 测试相比，自身 HP 的响应稍强，但仍不足以单独主导搜索中的 Q 差异。
