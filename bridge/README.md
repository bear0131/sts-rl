# 真实游戏边界

`schemas/combat_state.py` 将真实通信帧/模拟器状态规范化；env 的编码器消费该契约。
`real_game_env.py` 适配真实游戏；`replay.py` 做动作回放与差分；`client/` 是 CommunicationMod 客户端和 agent orchestrator，操作说明见 client/README.md。

策略导入 canonical rl/policies，禁止在客户端复制动作/观测规则。真实帧兼容、错误分类和有界重试仍见 TODO F15。CI 的合成 checkpoint/模拟器验证不代表真实游戏端到端通过。
原始帧/log 放忽略的本地输出，不进入核心模块目录。

canonical怪物列表保留来源后端的位置：模拟器的INVALID槽成为PAD占位，真实列表保留死亡实体及客户端索引。只有差分报告的存活实体比较过滤gone槽，不将该视图传给策略。真实策略的观测/mask/PLAY命令使用同一个客户端槽位；这不等于录制回放已经建立客户端到native的目标映射，后者单列TODO BRIDGE31。

本地简化编辑器的 `BattleContext.apply_custom_state` 在空spec、空player/monster记录（含空powers/counters）时保持全部状态；显式空牌堆仍表示清空牌堆。有内容的修改先在副本完成，解析失败不留下部分写入；不存在的怪物槽不能凭hp等字段创建。有效非空编辑仍沿用编辑器的重置为玩家回合语义，不能用于恢复排队动作或完整真实快照。回归 `tests/integration/test_custom_combat_edits.py`；需要同步重建原生扩展，接口shape/观测/动作版本未改变。

战斗GRID/CARD_REWARD单选使用实际候选索引，经SELECT映射到CHOOSE命令，覆盖181动作的全部96个选牌槽，不再截到10张或压缩索引；超过96个候选明确拒绝。HAND_SELECT仍遵守十张手牌限制。动作容量与原生完整动作边界见 `env/README.md`，合成帧回归见 `tests/unit/test_bridge_selection_indices.py`；这不代表真实客户端验收。

奖励领取以当前 `COMBAT_REWARD` 的合法项及按种类计数的索引生成命令；满物理药水槽或持 Sozu 时不可领取药水。直接调用 `command_for_semantic` 也拒绝不可领取项和错误画面，不只依赖策略 mask。模拟器侧 `SimGameEnv` 使用原生 `GameContext.legal_reward_options()` 的 `(kind, reward_group_index, item_index)` 过滤领取动作；更新 Python 后必须同步重建本 checkout 的扩展，旧扩展缺少此接口时不可混用。已有 action/observation 张量布局和 checkpoint 参数形状不变。

商店原生 Python 入口 `shop_buy_*`、`shop_remove`、`shop_proceed` 要求当前为 `SHOP_ROOM`；购买索引在编码前按实际货架范围检查，售罄/余额不足/已使用删牌服务抛 `ValueError` 且不修改状态。`legal_shop_options()` 在其他画面返回空。保留既有满槽或 Sozu 阻止取得药水时，对有效且可支付的药水槽直接调用为无副作用返回；枚举和validator均不将其视为可购买。需要重建本checkout扩展；不改变动作/观测张量布局。

`replay.Driver` 处理 REST/CHEST 的 `proceed` 时按模拟器当前画面推进：休息点使用合法 skip、未开宝箱使用不开箱选项、未领取奖励使用奖励退出。模拟器已经回图时，真实 UI 的房间收尾帧不重复执行；仍在选牌等未决画面时报告不一致，不强制改成地图。局部原生回归见 `tests/integration/test_bridge_room_replay.py`。

原生战斗牌快照：`BattleContext.card_state()` / `set_card_state(state)` 使用版本2，保存 `version/next_uid/hand/draw/discard/exhaust/stasis`。四堆保持顺序，手牌最多10张，Stasis固定两个物理槽（空槽为null）。每个实体必须提供 `id/upgrades/uid/special_data/base_cost/cost/free_to_play_once/retain/exhausts`：id是原生枚举名，uid在全部牌堆和Stasis间唯一且小于next_uid；cost为未应用free-once前的当前费用，base_cost保留跨回合改费，SearingBlow完整升级次数须与special_data一致；exhausts是当前实例的真实属性，不是本次动作的exhaustOnUse。未知字段、缺项、错误类型及越界都会拒绝，整批验证后才替换并重算派生计数；恢复不推进RNG或回合。

这只是当前main已有CardInstance字段的精确接口，不是完整战斗/Java快照。恢复要求PLAYER_NORMAL且动作队列为空，待选牌/自动出牌中的引用必须与队列一起处理，不能仅替换牌堆。版本1缺少动态exhausts而被明确拒绝，不根据印刷属性补值。Java UUID映射及真实帧接入仍见TODO BRIDGE28。旧to_dict、apply_custom_state与模型张量布局不变；旧简化接口不会因此自动获得完整实例语义。回归 `tests/integration/test_combat_card_snapshot.py`；使用新接口需重建本checkout原生扩展。

主牌组快照：`GameContext.deck_state()` / `set_deck_state(records)` 保持物理牌序，每项必须且仅包含 `id/upgrades/misc/bottle`。`id` 为原生枚举名；`misc` 保留仪式匕首等永久数值，灼热的 `misc` 必须等于完整升级次数；`bottle` 是 `NONE/ATTACK/SKILL/POWER`，三种瓶装各至多一张且必须匹配牌类型。最多96张，非法类型、缺项、多余字段、越界或重复瓶装会在替换前整批拒绝；不触发取得卡牌效果或消耗随机数。此接口接受原生记录；CommunicationMod卡牌须经下述真实帧转换，不能直接传入原生接口。

`set_deck([(id, upgrades), ...])` 继续用于创建新卡，兼容list/tuple序列及二元项；不会恢复永久成长数值，并明确清除旧瓶装索引。它现在也先验证完整输入，再原子替换；负数/布尔升级数、普通牌多次升级、状态牌、非法ID及超出96张会拒绝。需要同步重建原生扩展，观测/动作张量与checkpoint形状不变。主牌组恢复后的真实伤害、首抽物理身份、成长提交与选择候选见 `tests/integration/test_master_deck_snapshot.py`、`test_deck_invariants.py`。

真实主牌组帧接入：`schemas/card_state.py` 接受同事分支 `fbfdf21` 起导出的逐卡 `simulator_card_state.version=1`（与原生战斗快照version=2是两个合同）。`replay._sync_gc` 使用其中的当前misc和三种瓶装标记恢复主牌组，Java灼热的misc=0转换为原生完整升级次数。主牌组必须提供唯一、规范拼写的UUID；卡牌ID、升级数、misc、当前费用、exhausts、ethereal及完整sidecar均需显式提供。缺项、未知字段、类型错误或不支持的版本会报错，旧日志不能通过补默认值用于精确回放。普通策略观测仍支持其原有帧格式。

逐卡sidecar字段为 `version/base_cost/base_damage/base_block/free_to_play_once/retain/self_retain/innate/bottled_flame/bottled_lightning/bottled_tornado`；字段读取对应 `AbstractCard` 当前实例，不用卡名重建。现有模型词表顺序不变；战士/无色卡的Java特殊ID通过 `_card` 统一映射，主牌组、战斗牌堆、商店/奖励/选择共享该入口。回归 `tests/integration/test_real_master_deck.py` 与 `tests/unit/test_bridge_card_state.py` 验证输入帧到实际原生伤害/首抽及现有观测token。此阶段没有接入完整战斗UUID映射、待选牌队列、遗物计数恢复；`_sync_gc` 也不是完整真实战斗根状态恢复。兼容的Java导出器仍在远端同事分支，本机未编译、部署或连接实机，不把合成帧回归当成客户端验收。
