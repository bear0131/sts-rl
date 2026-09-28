# 环境与 worker 搜索

`core/game_env.py` 是厚层整局状态机，`core/run_env.py` 是薄层 Gym 接口；`core/combat_env.py` 是单场战斗环境。它们调用显式选择的 slaythespire 构建。

| 模块 | 契约 |
|---|---|
| core/encode.py、vocab.py、cards_vocab.py | canonical 字段到固定观测；字段/形状/词表版本以源码为准 |
| core/action_space.py、actions.py、game_actions.py | 整局/战斗动作与候选映射，mask 和 native action 必须一致 |
| core/factories.py、encounters.py、spiremap.py | 初始场景、遭遇与地图构造 |
| core/run_env.py | reset/step、缓存、fork、native begin/backprop/packed leaf |
| [search](search/README.md) | 无模型的 Python NOSL session/nonce，供 worker 与学习侧共同使用 |
| scripts/ | 人工 play、binding 探针与接口文档生成；不是 pytest 目录 |

纯 encode/vocab/action 模块不加载 simulator；原生环境与工厂加载 simulator。env 不依赖 rl/apps/tools/tests，依赖方向由 tests/unit/test_repository_layout.py 检查。
共享状态规范化在 bridge/schemas；模型额外观测裁剪在 rl/policies/model.py。不要因当前模型不使用某字段就改变环境实际状态或动作语义。

测试在 ../tests。已知实体槽位、pending 来源、候选区域、gauntlet 持久状态问题仍见根 TODO F01/F02/F03/F07；目录整理没有修复这些语义缺陷。

## 完整战斗动作契约 native-v1

新策略可显式创建 `SlayTheSpireEnv(combat_action_contract="native-v1")`。默认 `legacy`
不变，旧 checkpoint 的动作、mask 和观测均不重映射。`native_combat_actions()` 返回当前
完整原生动作；`step_native_combat(bits)` 验证后复用 wrapper 的奖励和终局路径；
`combat_selection_candidates()` 保留 draw/discard/exhaust/generated 的真实候选身份。
共享 evaluation runner 通过 `Decision(..., native_action=bits)` 执行这一契约。

该模式不会给旧策略补齐 logits，也不是旧动作空间的透明升级。长选牌候选和 MULTI 子集
可以在 thin mask 之外继续执行；draw-select 的 NOSL 副本必须用原生实体身份重映射，不能
把 sampled 索引直接提交给真实环境。回归入口为
`tests/integration/test_native_action_contract.py` 与 `tests/unit/test_native_action_codec.py`。
