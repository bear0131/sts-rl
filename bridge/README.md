# 真实游戏边界

`schemas/combat_state.py` 将真实通信帧/模拟器状态规范化；env 的编码器消费该契约。
`real_game_env.py` 适配真实游戏；`replay.py` 做动作回放与差分；`client/` 是 CommunicationMod 客户端和 agent orchestrator，操作说明见 client/README.md。

策略导入 canonical rl/policies，禁止在客户端复制动作/观测规则。真实帧兼容、错误分类和有界重试仍见 TODO F15。CI 的合成 checkpoint/模拟器验证不代表真实游戏端到端通过。
原始帧/log 放忽略的本地输出，不进入核心模块目录。
