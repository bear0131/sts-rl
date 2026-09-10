# 价值网络对敌人 HP 的敏感性

测试 checkpoint：`artifacts/latest-nosl-az-s512-20260905/nosl_az_614400_steps.zip`。

测试状态是一场固定的 Cultist 战斗（seed 36，floor 1，Bash+ + Strike，玩家满血）。
以同一个真实 simulator observation 为基准，构造状态序列时只修改：

- `monster_scalars[0,1] = enemy_hp / 200`
- `monster_scalars[0,2] = enemy_hp / enemy_max_hp`

其余 observation 字段逐字段做 `array_equal` 检查，均保持不变。敌人最大 HP 固定为
50，测试当前 HP 为 1、5、10、20、30、40、50。原始结果见
`.tmp-value-hp-sensitivity-614400.json`。

## 结果

| 敌人当前 HP | value |
|---:|---:|
| 1 | 0.23565373 |
| 5 | 0.23566380 |
| 10 | 0.23567621 |
| 20 | 0.23570031 |
| 30 | 0.23572341 |
| 40 | 0.23574552 |
| 50 | 0.23576665 |

统计量：

- `value(HP=1) - value(HP=50) = -0.00011292`
- 线性斜率约为 `+2.30e-6 / enemy HP`
- value 随敌人 HP 增加严格单调增加
- 所有其他输入字段保持完全相同

## 解释

网络确实读取了敌人的当前 HP，变化不是浮点噪声；但敏感性很弱，敌人 HP 变化 51 点只
改变约 `1.17e-4` 的 value。更值得注意的是方向：敌人血量更低时 value 略低，而不是
更高。这与内层“最大化最终玩家 HP”的目标不一致，至少说明当前 value head 没有学到
一个可靠的单调敌人剩余血量价值关系。

这不是模拟器或 NOSL 的问题，因为这里没有执行任何动作或搜索，只对同一编码 observation
做 value forward。也不能简单解释为“敌人 HP 越高越危险所以价值更低”，实际方向相反，
因此更可能是 value head 受到其他强特征、训练分布偏差或尚未收敛的影响。

## 结论

当前 checkpoint 的价值网络对敌人 HP **有输入敏感性，但几乎不敏感且方向错误**。如果
需要确认这是否是普遍现象，下一步应在多个敌人、多个玩家 HP、多个 floor/战斗状态上做
同样的字段干预，并同时测 `value` 和合法动作 logits；还可以和客观的“固定策略继续战斗
后的最终玩家 HP”做相关性比较。
