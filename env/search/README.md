# Worker 搜索基础

`nosl_sampler.py` 派生 opaque nonce；`nosl_is_az.py` 提供只依赖 NumPy/观测编码的 Python tree/session。模型前推由外部 value callback/leaf request 提供，不导入 Torch、训练器或 policy。

env/core/run_env.py 在 worker 中驱动这些 session；学习侧通过同一接口消费。这里解决了过去 env 反向导入 rl 的耦合。原生严格 Full Gumbel 树仍归 sim/；checkpoint value 对照搜索归 rl/search/。

`overworld_rules.py` 是训练 worker 与 evaluation 共用的 Ironclad 公共状态上层规则，当前 canonical 版本由 `OVERWORLD_RULE_VERSION` 声明为 `ironclad-overworld-v3`。只对非战斗合法动作打分；未知事件/选择画面返回 `None`，由消费者显式回退。分层 NOSL 的冻结上层使用 `--nosl-is-az-upper-strategy rules`，或 canonical response file `@configs/train/nosl_ruleupper_latest.args`；该入口在代码更新后自动使用当前最新规则版本，不再绑定 `v1` 文件名。仅支持 inner-only NOSL；未覆盖画面使用冻结模型 masked argmax。默认 `model` 保留随机采样。该设置保存在 checkpoint，CLI 省略时沿用 checkpoint；不会改变战斗搜索或训练目标。

启用 `--nosl-is-az-train-shared-embeddings` 会按 run seed 重置并训练 split policy 的共享 card/relic/potion embedding，projection 与整个外层 specialist 仍冻结。共享实体既作为候选 token 直接影响 `USE_POTION`、`PICK_CARD`、card/relic/potion reward、shop 与 boss relic 动作，也作为常驻 deck/relic/potion token 进入每个外层 phase 的 Transformer，因此 map、event、proceed、chest、campfire、Neow 等固定按钮 logits 也可能间接受影响。规则上层只有返回 `None` 的状态会暴露给这条 masked-argmax fallback；v3 在 v2 的确定性事件与 Bonfire Spirits 基础上，让 native canonical state 公开 Falling、N'loth、We Meet Again、Note For Yourself、Designer 按钮上已经显示的具体卡牌、遗物、药水、金额及服务模式，并据此评分。该附加元数据不进入 observation tensor；旧扩展或缺项候选仍整屏返回 `None`，不会按固定 action index 猜测。stock CommunicationMod 目前只给本地化按钮文本，真实 bridge 在收到 action-aligned typed 扩展前仍对这五类 fail closed，不从文字猜实体。

`turn_planner.py` 提供无模型 native 回合规划器。它在独立 NOSL 样本中搜索完整战斗动作，
每次只提交首个动作并重新观察；`node_budget` 是每个 sampled world 的 transition 预算，
`worlds`、实际 transition、根覆盖下限和 fallback leaf 均进入诊断。人工状态价值不是胜率或
游戏 oracle。`ironclad-turn-v8-awakened-phase` 在既有 slime phase、终局优先级、最短等价
胜利和资源保护逻辑上，补充 Awakened One 可见阶段的剩余生命估值。Runic Dome 不进入
native 规划，adapter 回退 `rules-ironclad-v3`。主要回归入口为
`tests/integration/test_turn_planner*.py`。

## 当前选牌与隐藏未来采样

draw-select 的索引属于当前公开候选实体。`nosl_redeterminize` 会重排 draw，
而执行一次 selection 会继续运行 native 队列；例如 Secret Weapon 的耗尽会
触发 Dark Embrace 抽牌。因此不能先执行真实副本的选择、再重新采样：排队的
抽牌已经读了真实未来。

新 planner 应先 clone/redeterminize，然后调用
`sample.remap_draw_selection_from(source, action_bits)` 得到 sampled clone 中的
完整合法 uint32 动作。该只读 helper 通过 native `CardInstance.uniqueId` 定位，
不按卡名/升级猜测；同名重复牌不合并，持久 specialData 等字段必须保持一致。
源/目标必须是同类型的 live draw-select 界面；非法、丢失或歧义实体明确报错。
随后只在 sampled clone 执行重映射后的动作。不能把 sampled 索引直接提交给
真实环境；真实执行仍用原 root action。

`BattleContext.preview_end_turn_result(action_bits)` 只在内部 clone 上执行一次
合法 END_TURN，返回 post-enemy-turn boundary 或 terminal 的完整 raw state。
非终局但未到边界会抛错；source 不变。新搜索用它避免旧
`preview_end_turn_boundary` 在 terminal 时返回 None、调用方再次 execute 的重复
模拟。旧 API 和旧 NOSL 的重采样/训练语义均保持兼容；采用新 helper 的 planner
需要显式加载含这些绑定的新 native 构建。

## 无模型回合规划

`turn_planner.py` 的 `NativeTurnPlanner` 在独立 NOSL 样本中搜索当前玩家回合，
用 native transition 处理出牌、目标、选牌、药水和敌方完整行动，再用人工状态价值评价。
它属于规则与搜索结合的实验策略，不是纯规则 v3，也不是原有神经 Full Gumbel。
每次只执行首个完整动作，再观察并重新规划；每个首步动作有独立等额预算和有界 frontier，
跨样本平均各自找到的最好结果。样本内的乐观续局不等于精确信息集搜索。

`TurnPlannerConfig.node_budget` 是每个样本的原生 transition 上限；为覆盖全部合法根动作，
有效下限为 `2 * root_actions`。`effective_node_budget` 报告每样本上限，
`effective_total_budget` 乘以 worlds，`native_transitions` 报告实际总执行数。
`beam_width` 限制每个首步分支保留的 frontier，`max_depth` 限制玩家动作深度；
不会将原始字典当作完整状态做不可靠的 transposition 合并。搜索不足以完成 pending 时
使用并记录 `fallback_leaves` 近似值。标量值是人工偏好，不是胜率或经校准的 Q。

`ironclad-turn-v2` 增加保守的剩余输出检查，避免耗尽最后一张攻击后只剩无伤害牌，
却因本回合多打一点伤害而被评价为更优。`choose(..., public_relics=...)` 接收公开遗物 ID；
原生 `BattleContext.to_dict()` 不包含遗物，因此缺少该参数时关闭此检查，保持旧调用兼容。
评测 adapter 从公开 run state 传入遗物；manifest 记录该约定。判断仅在所有剩余手牌、
抽牌堆和弃牌堆均属于已知无输出集合时触发；身体冲撞、烈焰屏障、未识别牌或正数能力、
药水、伤害/造牌遗物、已启动炸弹、敌方毒/窒息/自动消失，以及未决选牌均保守放行。
这是狭窄的无输出惩罚（−10000），不能证明保留的输出足以在死亡前结束战斗，
也不估算一张消耗攻击是否足以打完剩余生命。真实胜利/失败仍按 native outcome 处理。

调用方必须保证当前敌人意图公开。评测 adapter 遇 Runic Dome 回退遮罩后的 v3 规则；
其遗留 thin 候选为空时按完整合法动作顺序选择，不窥探 native 当前意图。
根部抽牌堆选择先重采样，再按原生实体身份映射候选，不能先执行选择：后续 exhaust/draw
触发器可能读取真实牌堆顶。独立 master nonce 和决策计数不读取实际游戏 seed/RNG。

方法参考 bottled_ai 的当回合结果比较，没有复制其实现或独立制作卡牌模拟器。
参考版本 `7b47a877d626cdb4fbeeb423c46472b15dd2ac00` 的
[路径搜索](https://github.com/xaved88/bottled_ai/blob/7b47a877d626cdb4fbeeb423c46472b15dd2ac00/rs/calculator/play_path.py)
及 [结果比较](https://github.com/xaved88/bottled_ai/blob/7b47a877d626cdb4fbeeb423c46472b15dd2ac00/rs/common/comparators/core/comparisons.py)。


## Native one-turn execution and pending choices

`bc.execute_to_enemy_boundary(action_bits)` mutates a search clone using one legal
complete native action. It arms `stopAfterEnemyTurn` before execution and keeps
that control across ordinary player input and pending choices. A later action
that ends the turn, including Time Eater's twelfth-card trigger, stops after the
complete enemy phase and deterministic cleanup, before player start effects,
draw, and energy recharge. Cloning a pending state preserves the control flag.
The returned raw state is the clone's current state; ordinary playable input and
pending input are intermediate nodes, not value boundaries. Terminal states and
states with `post_enemy_turn_boundary=True` must not be expanded; the mutator
rejects actions there, and rejects illegal actions without changing state.

`bc.preview_end_turn_result(END_bits)` executes once on a temporary clone and
returns one of three distinct results: (1) a pending choice such as Nilry's Codex,
(2) a post-enemy boundary, or (3) a terminal outcome. It never mutates the source.
Its raw dictionary cannot be resumed as a simulator: searches needing to resolve
pending choices should clone explicitly and use the mutator above. The legacy
`preview_end_turn_boundary` API still returns `None` for a pending or terminal
result. `execute` and the existing NOSL search control remain unchanged.

Regression entrypoint: `tests/integration/test_native_boundary_execution.py`.
