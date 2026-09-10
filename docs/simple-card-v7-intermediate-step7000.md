# v7 中间计算结果诊断

source checkpoint: `.tmp-vector-rule-v7-contract-v9-inline-values-20260901/last.pt`; step=7000; contract=9

这里比较的是 local 序列中每个 `CALC_VALUE` 的数值，不经过最后的 `DELTA_SLOT`。模型仍是远端评测时的 greedy 输出；本报告只重新解析已保存的 token trace。

- CALC_VALUE 数值准确率：65/71 = 0.9155
- CALC_VALUE 数值 MAE：0.7324
- 排除最终 HP 计算后的准确率：55/55 = 1.0000
- 中间伤害量（`reg_unblocked_damage`/`reg_damage_after_vulnerable`）准确率：11/11 = 1.0000

## 按算子

| operator | correct / total | accuracy | MAE |
|---|---:|---:|---:|
| `ADD` | 14 / 14 | 1.0000 | 0.0000 |
| `FLOOR_DIVIDE` | 3 / 3 | 1.0000 | 0.0000 |
| `MIN` | 8 / 8 | 1.0000 | 0.0000 |
| `MULTIPLY` | 3 / 3 | 1.0000 | 0.0000 |
| `RELU` | 5 / 8 | 0.6250 | 3.2500 |
| `SUBTRACT` | 32 / 35 | 0.9143 | 0.7429 |

## 错误的中间结果

| case | path | operator | result register | expected | predicted |
|---|---|---|---|---:|---:|
| strike_vulnerable | root/1/0/0 | `SUBTRACT` | `reg_target_hp_after_subtract` | 91 | 99 |
| strike_vulnerable | root/1/0/0 | `RELU` | `reg_target_hp_after` | 91 | 99 |
| strike_tampered_6_to_12 | root/1/0/0 | `SUBTRACT` | `reg_target_hp_after_subtract` | 88 | 98 |
| strike_tampered_6_to_12 | root/1/0/0 | `RELU` | `reg_target_hp_after` | 88 | 98 |
| perfected_strike_vulnerable | root/1/0/0 | `SUBTRACT` | `reg_target_hp_after_subtract` | 91 | 99 |
| perfected_strike_vulnerable | root/1/0/0 | `RELU` | `reg_target_hp_after` | 91 | 99 |
