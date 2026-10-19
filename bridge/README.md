# 真实游戏边界

`schemas/combat_state.py` 将真实通信帧/模拟器状态规范化；env 的编码器消费该契约。
`real_game_env.py` 适配真实游戏；`replay.py` 做动作回放与差分；`client/` 是 CommunicationMod 客户端和 agent orchestrator，操作说明见 client/README.md。

策略导入 canonical rl/policies，禁止在客户端复制动作/观测规则。真实帧兼容、错误分类和有界重试仍见 TODO F15。CI 的合成 checkpoint/模拟器验证不代表真实游戏端到端通过。
原始帧/log 放忽略的本地输出，不进入核心模块目录。

奖励领取以当前 `COMBAT_REWARD` 的合法项及按种类计数的索引生成命令；满物理药水槽或持 Sozu 时不可领取药水。直接调用 `command_for_semantic` 也拒绝不可领取项和错误画面，不只依赖策略 mask。模拟器侧 `SimGameEnv` 使用原生 `GameContext.legal_reward_options()` 的 `(kind, reward_group_index, item_index)` 过滤领取动作；更新 Python 后必须同步重建本 checkout 的扩展，旧扩展缺少此接口时不可混用。已有 action/observation 张量布局和 checkpoint 参数形状不变。

商店原生 Python 入口 `shop_buy_*`、`shop_remove`、`shop_proceed` 要求当前为 `SHOP_ROOM`；购买索引在编码前按实际货架范围检查，售罄/余额不足/已使用删牌服务抛 `ValueError` 且不修改状态。`legal_shop_options()` 在其他画面返回空。保留既有满槽或 Sozu 阻止取得药水时，对有效且可支付的药水槽直接调用为无副作用返回；枚举和validator均不将其视为可购买。需要重建本checkout扩展；不改变动作/观测张量布局。

`replay.Driver` 处理 REST/CHEST 的 `proceed` 时按模拟器当前画面推进：休息点使用合法 skip、未开宝箱使用不开箱选项、未领取奖励使用奖励退出。模拟器已经回图时，真实 UI 的房间收尾帧不重复执行；仍在选牌等未决画面时报告不一致，不强制改成地图。局部原生回归见 `tests/integration/test_bridge_room_replay.py`。
