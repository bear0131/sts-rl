# PPO Combat Value HP 敏感性

checkpoint：`C:\Users\Tony2\Downloads\hier_hp_ppo_999817216_steps.zip`。

测试固定同一场真实 Cultist 战斗（seed 36，Bash+ + Strike），使用 observation-level
单变量干预。自身 HP 测试只修改 `player_scalars[0,5]` 和 `run_hp[0,2]`；敌人 HP 测试只
修改 `monster_scalars[0,1:3]`。其余 observation 字段逐项保持不变。

## 自身 HP

玩家最大 HP=80。

| 玩家 HP | value |
|---:|---:|
| 1 | 0.03316006 |
| 10 | 0.07166444 |
| 20 | 0.11578538 |
| 40 | 0.20726474 |
| 60 | 0.30114374 |
| 80 | 0.39503893 |

value 严格随自身 HP 增加而增加，`value(80)-value(1)=0.36187887`，线性斜率约
`0.00459 / HP`。

## 敌人 HP

Cultist 最大 HP=50。

| 敌人 HP | value |
|---:|---:|
| 1 | 0.43113735 |
| 5 | 0.42862377 |
| 10 | 0.42569941 |
| 20 | 0.41976926 |
| 30 | 0.41278705 |
| 40 | 0.40445003 |
| 50 | 0.39503893 |

value 严格随敌人 HP 增加而降低，`value(1)-value(50)=0.03609842`，线性斜率约
`-0.000721 / HP`。

## 与 NOSL-AZ checkpoint 对比

之前的 `nosl_az_614400_steps.zip` 在同样干预下：

- 自身 HP 1 到 80 只变化约 `0.00016186`
- 敌人 HP 1 到 50 只变化约 `0.00011292`，且方向曾经相反

PPO checkpoint 对自身 HP 的响应约大 2200 倍，对敌人 HP 的响应约大 320 倍，并且两个
方向都符合最大化最终玩家 HP 的目标。这说明原 PPO combat value head 已经明显学到血量
价值关系，而当前 NOSL-AZ value head 在这方面仍接近未形成有效表示。

原始 JSON：

- `.tmp-value-player-hp-sensitivity-999817216.json`
- `.tmp-value-enemy-hp-sensitivity-999817216.json`
