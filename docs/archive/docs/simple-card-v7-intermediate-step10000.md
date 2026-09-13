# v7 中间计算结果诊断

source checkpoint: `.tmp-vector-rule-v7-contract-v10-step10000\last.pt`; step=10000; contract=10

这里比较的是 local 序列中每个 `CALC_VALUE` 的数值，不经过最后的 `DELTA_SLOT`。模型仍是远端评测时的 greedy 输出；本报告只重新解析已保存的 token trace。

- CALC_VALUE 数值准确率：71/71 = 1.0000
- CALC_VALUE 数值 MAE：0.0000
- 排除最终 HP 计算后的准确率：55/55 = 1.0000
- 中间伤害量（`reg_unblocked_damage`/`reg_damage_after_vulnerable`）准确率：11/11 = 1.0000

## 按算子

| operator | correct / total | accuracy | MAE |
|---|---:|---:|---:|
| `ADD` | 14 / 14 | 1.0000 | 0.0000 |
| `FLOOR_DIVIDE` | 3 / 3 | 1.0000 | 0.0000 |
| `MIN` | 8 / 8 | 1.0000 | 0.0000 |
| `MULTIPLY` | 3 / 3 | 1.0000 | 0.0000 |
| `RELU` | 8 / 8 | 1.0000 | 0.0000 |
| `SUBTRACT` | 35 / 35 | 1.0000 | 0.0000 |

## 错误的中间结果

| case | path | operator | result register | expected | predicted |
|---|---|---|---|---:|---:|
