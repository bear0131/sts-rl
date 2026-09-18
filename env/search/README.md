# Worker 搜索基础

`nosl_sampler.py` 派生 opaque nonce；`nosl_is_az.py` 提供只依赖 NumPy/观测编码的 Python tree/session。模型前推由外部 value callback/leaf request 提供，不导入 Torch、训练器或 policy。

env/core/run_env.py 在 worker 中驱动这些 session；学习侧通过同一接口消费。这里解决了过去 env 反向导入 rl 的耦合。原生严格 Full Gumbel 树仍归 sim/；checkpoint value 对照搜索归 rl/search/。

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
