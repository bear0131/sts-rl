# v7 中间计算结果诊断

source checkpoint: `.tmp-vector-rule-v7-contract-v9-inline-values-20260901/last.pt`; step=3500; contract=9

这里比较的是 local 序列中每个 `CALC_VALUE` 的数值，不经过最后的 `DELTA_SLOT`。模型仍是远端评测时的 greedy 输出；本报告只重新解析已保存的 token trace。

- CALC_VALUE 数值准确率：53/71 = 0.7465
- CALC_VALUE 数值 MAE：2.3380
- 排除最终 HP 计算后的准确率：53/55 = 0.9636
- 中间伤害量（`reg_unblocked_damage`/`reg_damage_after_vulnerable`）准确率：9/11 = 0.8182

## 按算子

| operator | correct / total | accuracy | MAE |
|---|---:|---:|---:|
| `ADD` | 14 / 14 | 1.0000 | 0.0000 |
| `FLOOR_DIVIDE` | 3 / 3 | 1.0000 | 0.0000 |
| `MIN` | 8 / 8 | 1.0000 | 0.0000 |
| `MULTIPLY` | 3 / 3 | 1.0000 | 0.0000 |
| `RELU` | 0 / 8 | 0.0000 | 10.1250 |
| `SUBTRACT` | 25 / 35 | 0.7143 | 2.4286 |

## 错误的中间结果

| case | path | operator | result register | expected | predicted |
|---|---|---|---|---:|---:|
| strike_base | root/1/0/0 | `SUBTRACT` | `reg_target_hp_after_subtract` | 94 | 100 |
| strike_base | root/1/0/0 | `RELU` | `reg_target_hp_after` | 94 | 100 |
| strike_vulnerable | root/1/0/0 | `SUBTRACT` | `reg_target_hp_after_subtract` | 91 | 109 |
| strike_vulnerable | root/1/0/0 | `RELU` | `reg_target_hp_after` | 91 | 109 |
| strike_block | root/1/0/0 | `SUBTRACT` | `reg_unblocked_damage` | 2 | 4 |
| strike_block | root/1/0/0 | `SUBTRACT` | `reg_target_hp_after_subtract` | 98 | 100 |
| strike_block | root/1/0/0 | `RELU` | `reg_target_hp_after` | 98 | 100 |
| strike_vulnerable_block | root/1/0/0 | `SUBTRACT` | `reg_unblocked_damage` | 5 | 3 |
| strike_vulnerable_block | root/1/0/0 | `SUBTRACT` | `reg_target_hp_after_subtract` | 95 | 109 |
| strike_vulnerable_block | root/1/0/0 | `RELU` | `reg_target_hp_after` | 95 | 109 |
| strike_tampered_6_to_12 | root/1/0/0 | `SUBTRACT` | `reg_target_hp_after_subtract` | 88 | 99 |
| strike_tampered_6_to_12 | root/1/0/0 | `RELU` | `reg_target_hp_after` | 88 | 99 |
| perfected_strike_zero_strikes | root/1/0/0 | `SUBTRACT` | `reg_target_hp_after_subtract` | 94 | 100 |
| perfected_strike_zero_strikes | root/1/0/0 | `RELU` | `reg_target_hp_after` | 94 | 100 |
| perfected_strike_three_strikes | root/1/0/0 | `SUBTRACT` | `reg_target_hp_after_subtract` | 94 | 100 |
| perfected_strike_three_strikes | root/1/0/0 | `RELU` | `reg_target_hp_after` | 94 | 100 |
| perfected_strike_vulnerable | root/1/0/0 | `SUBTRACT` | `reg_target_hp_after_subtract` | 91 | 109 |
| perfected_strike_vulnerable | root/1/0/0 | `RELU` | `reg_target_hp_after` | 91 | 109 |
