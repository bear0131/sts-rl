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
