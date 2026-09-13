# 远端 v6 参数评测汇总

本次使用远端训练目录中的稳定快照。远端文件虽然目录名包含
`contract-v5`，checkpoint 元数据为 `schema_version=7`、
`executor_contract_version=5`，对应当前历史上称作 v6 的 effect-tree 训练协议。
本地最新协议文档（contract v7）不直接用于加载这些参数，避免协议版本混用。

## 参数快照

| 模型 | checkpoint step | decoder | 本地参数 |
| --- | ---: | --- | --- |
| baseline | 8000 | d_model=256，4 层 | `.tmp-remote-v6-baseline/last.pt` |
| wide | 7500 | d_model=320，5 层 | `.tmp-remote-v6-wide/last.pt` |

两个快照均包含对应的 `tokenizer.json`、`config.json` 和 `metadata.json`。

## 自回归评测

评测包含两个固定例和随机种子 `20260831` 生成的 10 个随机例；模型在 CUDA
上运行，外层 header 完全自由贪心生成，local 节点使用真实 NodeSpec 前缀。

| 模型 | outer 完全正确 | outer token accuracy | local 节点 | local token accuracy | 活跃 delta MAE | `<UNK>` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline step 8000 | 5/12 | 33.07% | 82/82 | 100.00% | 0.019202 | 0 |
| wide step 7500 | 4/12 | 41.46% | 82/82 | 100.00% | 0.008979 | 0 |

固定例结果：

- 飞身踢（敌人 2 层易伤）：baseline、wide 均 100% 完全一致。
- 剑柄打击（洗牌并触发日晷）：baseline 3.17%，wide 2.38%；两者都在
  `RUNTIME_CONTEXT_END` 后错误地继续生成 `READ_REQUEST`，没有进入预期的
  effect-tree header。

10 个随机例的牌、遗物和 outer token accuracy（wide）：

| 编号 | 打出卡牌 | 遗物数量 | outer token accuracy |
| --- | --- | ---: | ---: |
| 1 | 重刃+ | 3 | 98.55% |
| 2 | 观察弱点 | 4 | 8.33% |
| 3 | 耸肩无视+ | 6 | 7.64% |
| 4 | 吞噬+ | 1 | 9.09% |
| 5 | 腐化+ | 8 | 48.48% |
| 6 | 无惧疼痛 | 4 | 100.00% |
| 7 | 燃烧 | 1 | 100.00% |
| 8 | 战斗专注 | 3 | 55.26% |
| 9 | 双发+ | 6 | 10.00% |
| 10 | 腐化 | 2 | 100.00% |

随机例的完整中文场景（楼层、敌人、状态、遗物计数、预测序列）见：

- [`vector-v6-wide-step7500-fixed-random-autoregressive-eval.md`](vector-v6-wide-step7500-fixed-random-autoregressive-eval.md)
- [`vector-v6-baseline-step8000-fixed-random-autoregressive-eval.md`](vector-v6-baseline-step8000-fixed-random-autoregressive-eval.md)

## 2000 条样本的错误位置

teacher-forced 统计使用同一随机种子和数据分布。主要问题不是 local 节点拓扑，
而是外层 header 和少数运行时数值响应：

| wide 参数位置 | 次数 | 错误率 |
| --- | ---: | ---: |
| `DRAW_COUNT` | 564 | 10.28% |
| `CALCULATION_SUBTRACT_HP/CALC_VALUE` | 4291 | 1.56% |
| `CALCULATION_SUBTRACT_BLOCK_DAMAGE/CALC_VALUE` | 3149 | 1.02% |
| `CALCULATION_SUBTRACT_BLOCK/CALC_VALUE` | 3706 | 0.84% |
| `CARD_COUNT` | 6483 | 0.42% |
| `CALCULATION_FLOOR_DIVIDE_VULNERABLE/CALC_VALUE` | 772 | 0.39% |

最高频混淆是敌人槽位选择（`ENTITY_ENEMY_SLOT_0` 与 slot 1/2），其次是
`REGION_PILE_SHUFFLE -> REGION_DRAW`，以及数字 token 的相邻位预测错误。
delta 误差最大的维度是低频的敌人 `present/gone` 和 `block_present`；常见的
抽牌堆、弃牌堆、消耗堆数量 delta MAE 分别为 0.00761、0.00074、0.00217。

### 为什么 `DRAW_COUNT` 偏差大

wide 的 2000 条样本中只有 188 个模型可评分的 `DRAW_COUNT` 数字，取值分布为
`1:108`、`2:43`、`3:24`、`4:9`、`5:4`。数字位错误 58 个（30.85%）；
`NUM_LSD_BEGIN` 和 `NUM_END` 均为 100% 正确。因此报告中 `DRAW_COUNT` 的
10.28% 是把字段名、数字开始、数字结束三个 token 合并后的比例，真正的数字
位准确率是 69.15%。

该值不是单纯的卡牌常数：当抽牌堆不够时，首次 `DRAW` 的数量为
`min(卡牌要求的抽牌数, 抽牌堆剩余数)`，洗牌后可能还有第二个 `DRAW`。模型需要
同时读取匿名卡牌向量和连续牌堆向量，并在较少出现的 4、5 等取值上做离散判断，
所以比固定的 `CARD_COUNT` 更难。

### 为什么 Outer accuracy 较低

teacher-forced 统计的 wide token accuracy 为 99.8725%，而自由自回归评测只有
41.46%。这是 exposure bias 和边界错误的组合：Outer 一旦在很早的位置生成错误
token，后续上下文已经偏离，位置对齐的 token accuracy 会连续下降。多个含遗物
计数的案例都在 `<RUNTIME_CONTEXT_END>` 后错误地产生额外 `READ_REQUEST`，因此
整条 header 几乎全部失配；固定的剑柄打击例就是这个情况。local executor 使用
真实 NodeSpec 前缀时仍为 100%，说明主要瓶颈在 Outer 的自由树结构和运行时读请求
边界，而不是局部计算 token。

完整机器可读结果：

- [`vector-v6-wide-step7500-error-positions-2000.json`](vector-v6-wide-step7500-error-positions-2000.json)
- [`vector-v6-baseline-step8000-error-positions-2000.json`](vector-v6-baseline-step8000-error-positions-2000.json)

## 协议文档

当前本地最新设计是 contract v7：

- [`rule-model-vector-v7-symbolic-executor.md`](rule-model-vector-v7-symbolic-executor.md)
- [`rule-model-vector-v7-effect-tree.md`](rule-model-vector-v7-effect-tree.md)
- [`rule-model-vector-v7-effect-tree-examples.md`](rule-model-vector-v7-effect-tree-examples.md)
- [`rule-model-vector-v7-contract-v5-relic-counter.md`](rule-model-vector-v7-contract-v5-relic-counter.md)
