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

`ironclad-turn-v2` 增加保守的剩余输出检查，避免耗尽最后一张攻击后只剩无伤害牌，
却因本回合多打一点伤害而被评价为更优。`choose(..., public_relics=...)` 接收公开遗物 ID；
原生 `BattleContext.to_dict()` 不包含遗物，因此缺少该参数时关闭此检查，保持旧调用兼容。
评测 adapter 从公开 run state 传入遗物；manifest 记录该约定。判断仅在所有剩余手牌、
抽牌堆和弃牌堆均属于已知无输出集合时触发；身体冲撞、烈焰屏障、未识别牌或正数能力、
药水、伤害/造牌遗物、已启动炸弹、敌方毒/窒息/自动消失，以及未决选牌均保守放行。
这是狭窄的无输出惩罚（−10000），不能证明保留的输出足以在死亡前结束战斗，
也不估算一张消耗攻击是否足以打完剩余生命。真实胜利/失败仍按 native outcome 处理。

`ironclad-turn-v3` 将人工能力收益时长固定到一次 `choose` 的公开根状态：
仍用根敌方总 HP 和根玩家 Strength 估算并限制到1–5，但所有样本、首步分支、
完整回合与未完成前缀共用该值。敌方 Strength/Ritual 和双方弱/易伤持续时间也用
同一时长，避免只冻结正收益而改变负收益的比较基准。它不改变搜索深度或预算。
这修正了伤害降低能力估值、从而奖励保留敌人 HP 的倒挂；分裂后子怪总 HP/数量
跳变、无触发器的能力高估和极端成长值超过胜利奖励仍是独立限制。
独立调用 `state_value` 时可显式传 `horizon` 比较同一次决策的分支；省略时保留
把该状态视作新根的兼容行为。评测 manifest 记录 `value_horizon` 约定。

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

`ironclad-turn-v4` adds a public phase-progress prior for splitting slimes. The
remaining damage potential includes expected descendant HP at the native half-HP
threshold (integer division), and the encounter-count term anticipates two medium
slimes per large slime, or four per boss. Native splitting at unchanged HP is
therefore continuous instead of falsely increasing remaining burden. This is an
approximation assuming threshold splits, not minimum damage or a combat oracle;
burst kills can skip phases. Native HP, loss of debuffs on split, actual incoming
damage and terminal outcomes still determine the other value terms. The root
power horizon remains the v3 public-state anchor. Regression entrypoint:
`tests/integration/test_turn_planner_phases.py`.


## Terminal priority candidate v5

`ironclad-turn-v5-avoid-sampled-death` keeps the v4 ordinary-state heuristic and
partial-frontier ordering. Within each sampled world, complete leaves compare
`(terminal_rank, heuristic)`, where death < live boundary < victory. Terminal
classification takes precedence over arbitrarily large Strength or other powers.
The original scalar `state_value` remains a heuristic, not a terminal utility.

Root actions compare `(-sampled_death_count, sampled_win_count, mean_heuristic)`
lexicographically across the same number of worlds. This explicitly prefers any
alternative with fewer sampled deaths, even if accepting a sampled death would
buy more wins in other worlds. It is a conservative risk preference, not a win
probability, calibrated risk estimate, or semantics-preserving optimization.
When all sampled leaves are live, the old arithmetic mean and ordinary ordering
are retained. The heuristic mean only breaks ties in both terminal counts.
Partial fallback leaves remain live/uncertain, never invented wins.

Finite `root_outcomes` trace records report deaths, wins, nonterminal worlds,
`unresolved` partial fallbacks, actual `boundary_live` leaves and heuristic mean
separately. `live` includes unresolved prefixes and is not a survival certificate; `root_values` remains the scalar mean for diagnostics and no
longer alone determines the chosen action. Structured fields never enter numeric
Decision metrics. Non-finite leaf values are rejected rather than written as
Infinity/NaN; an overflowing finite sum is averaged in scaled form.
Regression: `tests/integration/test_turn_planner_terminal_priority.py`, including
Strength500 versus immediate native victory and four actual NOSL Havoc futures
(one win/three deaths) versus Defend surviving every world.
## Optional two-turn planning

`ironclad-turn-v6` combines terminal-priority planning with
`TurnPlannerConfig(lookahead_turns=2)` / eval `--planner-turns 2`. The default
remains one turn; both horizons use the explicit sampled-risk contract above.
In a fresh independently sampled clone, first-turn actions use ordinary native
`execute`, including Codex choices and TimeWarp's early END, so native start
effects, energy and draw lead into the next player turn. Only actions in the
last requested player turn arm `execute_to_enemy_boundary`. Do not resample at
the intervening turn: that would erase publicly chosen Headbutt top cards and
replace one sampled world with an inconsistent continuation.

No new native binding is needed. Existing projected enemy-boundary states are
not resumable: projection settles end-of-round effects and erases next enemy
moves. The planner rejects those roots; two-turn mode also rejects an unexpected
early projection from a private pending clone already armed by the boundary API.
Use real game inputs or pending choices produced by ordinary execution. The
live BattleContext remains unchanged. Dome still bypasses all native planning.

`node_budget` covers both turns together per sampled world. The reported minimum
is `(lookahead_turns + 1) * root_actions`; root coverage does not guarantee a
complete horizon under every pending/depth limit. `native_transitions` counts
both execute APIs, including END/start-of-next-turn work and choices. Two-turn
`max_depth` counts every action over both turns and does not append an uncounted
END past its cap. `fallback_leaves` explicitly counts roots/worlds evaluated from
unfinished prefixes; `second_turn_entries`, `completed_horizon_leaves` and
`max_action_depth` distinguish actual horizon coverage from requested settings.
The per-decision maximum depth stays in the full trace, not the runner's additive
episode metrics; it is not meaningful to sum it and label that sum a maximum.
The root power horizon stays anchored throughout the complete decision.

More turns increase branching and determinization optimism; they do not certify
better decisions. A new real decision still samples the draw multiset afresh,
so preserving publicly known deck-top constraints across separate decisions
remains outside this change. Tests: `tests/integration/test_turn_planner_two_turns.py`.

`ironclad-turn-v7` retains the v5 terminal counts and
mean heuristic as the primary comparison. Only when all three are exactly equal,
it prefers fewer native actions needed to reach the selected winning leaves.
Within a world, equal winning leaf values retain their shortest observed depth;
across worlds the winning depths are summed (equivalent to comparing their means
when win counts tie). Live/death leaves contribute no winning length, preserving
the previous tie for roots with no proved wins. Higher-HP wins can still justify
more actions. This prevents repeatedly postponing an equally good available win.

`root_outcomes` additionally records finite `winning_steps_total` and
`winning_steps_mean`; the latter is conditional on winning worlds and is zero
when no win was found. These remain structured per-root trace data, never summed
as episode numeric metrics. Regression:
`tests/integration/test_turn_planner_winning_depth.py`. The Medical Kit/Hex/Dark
Embrace fixture verifies the current native simulator, not an independent audit
of the commercial game's event ordering or a replay of the original full run.

### Opt-in public binary powers candidate

`ironclad-turn-v7-public-flags-v1` adds a separate, read-only native
`BattleContext.public_player_power_flags()` contract. Schema 1 returns
`{"schema_version": 1, "flags": {...}}`, with boolean values for the explicit
public whitelist `CORRUPTION`, `BARRICADE`, `CONFUSED`, `HEX`, `PEN_NIB`, and
`SURROUNDED`. All six keys are present, including inactive effects. This is not
an export of arbitrary simulator bitsets or support for every character's powers.

Legacy `to_dict().player.powers`, canonical/encoded observations, packed leaf
arrays, action semantics, and native state hashing remain unchanged. The new
planner reads each sampled clone's flags into its own snapshot under
`player.public_power_flags`; its trace includes the same public field. Existing
checkpoints continue to receive the original observations. An old native build
can still run policy/rules paths; explicitly constructing the new planner fails
with a capability error instead of silently treating missing flags as false.

Only the new handwritten planner's normalized power lookup uses these flags.
Corruption and Barricade now activate their existing value terms after actual
native play, which changes strategy preferences and needs an independent eval.
Those terms remain human priors, not complete estimates of exhaust or block
retention value. Barricade, Pen Nib and Surrounded are explicitly classified as
not independently producing damage, so their newly visible flags do not invent
an output channel in the resource guard. Turn/risk/winning-depth priorities and budgets are unchanged.
The strategy manifest names the private flag contract explicitly. Native
mechanism and compatibility contracts are in
`tests/integration/test_public_power_flags.py`.
