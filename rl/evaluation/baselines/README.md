# 战斗规则基线

`CombatRuleStrategy` 通过 `DecisionStrategy` 接口决定战斗动作；`PhaseStrategy` 将所有
`combat` 请求（含目标、药水、选牌子步骤）交给它，其余请求仍交给指定的上层 checkpoint。
它是可解释的工程启发式，不代表人类平均水平，也不把分数解释成 Q 或真实终局 HP。

## 支持与信息边界

当前支持 simulator whole-run Ironclad、现有 thin action codec。75 张铁甲战士卡牌复用
`env/core/ironclad_cards.py` 的基础注释，组合价值由本模块手写。常见无色/事件牌有近似规则，
未知牌会记录 `unknown_card`/`rule_unknown_choices`；不宣称全卡牌、遗物或其他角色规则完整。
规则没有网络调用、训练参数、模拟器前瞻、随机搜索或未来敌人动作访问。

只使用当前手牌/费用、HP/格挡/能量、buff、公开敌人意图、可见牌堆内容、药水、遗物与合法动作。
普通抽牌堆进入策略前按牌名/升级排序，不能利用未来抽牌顺序；检索牌堆的选牌屏幕才保留
候选索引以正确提交选择。`Havoc` 按一般收益估计，不查看牌堆顶。原生怪物槽位保留空槽，
避免 canonical 旧路径的槽压缩问题 F01；这不表示全项目 F01 已修复。
从 v2 开始，Runic Dome 会隐藏真实敌人 move/damage/hits，改用明确标注的公开属性威胁估计。
历史 v1 没有这个遮罩；比较版本时应注明信息契约也发生修正，神经路径是否隐藏圆顶意图
仍需另行审计。

规则读的是玩家可见语义，而神经策略仍受现有编码表示限制（含 F01/F02/F03 已知缺陷）。
因此这个实验隔离“整个战斗决策组件”，不能把所有差异都归因于搜索算法或策略直觉；
需要进一步归因时，应另做观测表示与搜索预算的控制实验。

`HEADBUTT`/Liquid Memories 用弃牌堆，`EXHUME` 用消耗堆，Secret Weapon/Technique
用公开检索候选，Discovery/Codex 用屏幕生成的选项；不统一当作手牌。新 native build 的
`select_info.generated_cards` 是只读附加字段，仅在生成选牌屏幕读取，不改变神经观测或
动作版本。旧扩展缺少该字段时明确失败。MULTI 由现有 codec 单调选牌并确认。

## 决策方法

每个合法根动作在其合法目标中取最高分；目标子步骤使用显式 pending 来源重新评分，
相同输入以 action ID 打破平局。分项随 `--trace` 写入 JSONL：

- 估计伤害考虑力量、连击、虚弱、易伤、格挡及部分遗物；奖励击杀、取消敌人攻击和整场致死。
  削减敌人格挡也有价值，不能因为一张牌不扣 HP 就停止攻击。
- 只重点奖励覆盖当前预计伤害的格挡；低血致死风险提高防御权重，Barricade 允许持久格挡收益。
- 对抽牌、能量、易伤/虚弱、力量、消耗、回血、能力牌和可立即利用的组合加入分项。
  Flex 等免费铺垫尽量先于受益攻击，花费 HP 的效果计入代价。
- 药水有保留成本，紧急情况降低成本。选牌按所属区域和任务区分保留、消耗、升级、回收；
  End Turn/Confirm 为显式候选。
- Clash 会考虑先清除非攻击牌；力量铺垫考虑当前手牌的支付能力。死亡目标不再得到 debuff
  奖励。长战斗成长、睡眠期准备、Spore Cloud、Plated Armor、Apparition 与主要药水均有
  单独规则。Mercury Hourglass 在敌人行动之后才触发，不能算作当前回合免伤。

这些是近似值，不是第二套完整 simulator：例如动态 Rampage 数值、随机连击分配、飞行中途
解除、完整遗物/药水相互作用、无限循环和多回合资源规划尚未精确建模。评测中的差异应从
trace 对照真实动作/结果验证，不能直接把分项当作游戏规则证明。

## 可复现对照

从仓库根目录，使用 tests/README.md 的环境和显式 `STS_LIGHTSPEED_BUILD_DIR`。
位置参数始终是上层；`--combat-checkpoint` 指定下层，仅在 combat 阶段使用。

```bash
python -m rl.evaluation.eval UPPER.zip --solver rules --episodes 128 --seed 10000 --device cpu --num-envs 32 --max-steps 6000 --timeout 600 --output runs/local/rules-development.json
python -m rl.evaluation.eval UPPER.zip --combat-checkpoint COMBAT.zip --solver policy --episodes 128 --seed 10000 --device cpu --num-envs 32 --max-steps 6000 --timeout 600 --output runs/local/policy-development.json
python -m rl.evaluation.eval UPPER.zip --combat-checkpoint COMBAT.zip --solver native --search-mode strict --simulations 128 --depth 0 --leaf-batch-size 1 --episodes 128 --seed 10000 --device cuda --num-envs 32 --max-steps 6000 --timeout 3600 --output runs/local/search-development.json
```

整局候选筛选和冻结后验证每组至少 128 局；固定状态的机制回归测试可以更小。
上面是开发集示例，验证集每轮在 worklog 预先声明，不能将已看过的种子重新称为留出。
可以拆为四个不重叠的 32 局分片，但须保留相同 `num_envs`、各片 seed/config/源码与模型哈希，
合并时检查唯一 seed、自然结束、进程退出码及截断情况。分片评测不能声称是单次 128 宽 batch。
搜索可显式添加 `--leaf-inference cuda-graph`；其 FP32/CUDA 支持范围与验证边界见上级 README。

固定上层权重并不固定其之后的选择：战斗造成不同 HP/药水/奖励状态，上层可能改变路线和抓牌。
同 seed 整局比较因此衡量整个组合效果。为隔离战斗，沿明确的参考策略轨迹抽取前 N 个战斗开局，
从**同一完整初态和 RNG** 分别 fork 给规则、下层直接 policy 和 native 搜索。fork/推进只在
评测层，规则本身从不持有分支。现有 `runner` 负责三者步数、超时和结果，未复制 episode loop。

```bash
python -m rl.evaluation.eval UPPER.zip --combat-checkpoint COMBAT.zip --solver rules --episodes 2 --seed 10000 --compare-combats 12 --search-mode strict --simulations 128 --depth 0 --device cpu --max-steps 6000 --timeout 1800 --trace runs/local/paired-development.jsonl --output runs/local/paired-development.json
```

这里 `--solver rules` 是参考轨迹；用 policy/native 可采集另一方访问的状态。参考选择必须写入
报告，它会影响样本分布。每场各策略拿到同一隐藏随机状态，但行动不同可消耗不同随机数，
因此不是相同未来抽牌序列保证。当前复现方式是固定 checkpoint/native hash、seed、参考策略、
配置和版本重新运行；JSONL 保存可见状态和动作，不是可恢复完整 RNG 的跨版本快照语料。
分支 runner 从 step0 开始，因此 native 的决策 nonce 使用分支的局部步号；它不承诺与
参考整局同一位置的 nonce 相同。native 可以一次提交完整带目标动作，普通策略可能需要
两个 thin step，不能把两者的 steps 或 steps/s 当作等量原子动作的效率比较。

JSON 记录上下层/native 文件哈希、逐局结果和 `combat_rows`，包括初始/结束 HP、药水、
搜索量和用时；`hp_loss` 是初始减最终的净 HP 变化，包含战斗内治疗，可能为负，不能解释为
累计受击伤害。逐场同时保存 `initial_potions` / `final_potions`，便于检查消耗差异。
`combat_comparison` 在双方都赢的配对战斗上统计规则减对手的损血差，同时单列
双方胜场。最终 HP 是**胜利结算回血前**的战斗 HP，不与整局回血后的 HP 混用。
`step_limit`/超时必须单列，未完成组不能伪装成完整样本。CLI 超时是合作式；正式批次还需
外部进程监督硬超时。trace 拒绝覆盖现有文件。

只针对每轮声明的开发 bad case 调整规则，冻结版本后跑新留出；若查看留出并修改，就把它
标为新的开发集并另选留出。历史种子范围及其后续用途以 worklog 为准。报告样本数和不确定性，不能从
几局/单个怪物声称总体胜率优越，也不能拿 W&B 滚动训练指标代替固定 checkpoint 评测。

## 可选择的上层规则

```bash
python -m rl.evaluation.eval UPPER.zip --upper-strategy rules --solver rules --episodes 128 --seed 10000 --device cpu --num-envs 32 --max-steps 6000 --timeout 600 --output runs/local/joint-rules-development.json
python -m rl.evaluation.eval UPPER.zip --upper-strategy rules --combat-checkpoint COMBAT.zip --solver native --search-mode strict --simulations 128 --depth 0 --leaf-batch-size 1 --episodes 128 --seed 10000 --device cuda --num-envs 32 --max-steps 6000 --timeout 3600 --output runs/local/rule-upper-search-development.json
```

`OverworldRuleStrategy` 依据公开牌组缺口、重复牌和力量/消耗/格挡配合评估奖励及跳过，按价格
评估购买/移除，结合 HP、升级空间与公开地图选择路线，营火比较治疗和升级，评估 boss
遗物副作用，并处理源码已核对按钮含义的事件。永久诅咒、圆顶、咖啡滤杯、融合之锤等有
显式约束；没有访问未来事件、未展示奖励或 RNG。

未知事件/选择屏幕仍由指定上层模型处理，计入 `overworld_fallback_decisions`；已知规则决策
计入 `overworld_rule_decisions`。因此该组合是“上层规则加显式模型回退”，不能称为完全无模型。
无需 CLI 的使用方可传任意 `fallback` 策略；不传回退而遇到未知屏幕会明确报错。
`--trace` 同时记录 `overworld_decision` 的公开状态、排序、实际动作和来源；
`effective_strategy` 记录上下层组合、规则版本和回退策略。

`--upper-strategy demand` 选择独立候选 `ironclad-overworld-demand-v1`，与 `rules`
共享下层与模型回退。它只在第二幕部分奖励选牌中，加上公开牌组在假设均匀五张手牌中、
保留一点进攻能量后可支付的重复直接格挡覆盖增量；以已公开 boss 的一种普通攻击作上限。
此值不表示实际起手、完整战斗计划或预计 HP 损耗。未建模的消耗/格挡引擎及随机费用等
组合退回上层 v1，其余选牌、升级、路线与商店规则均保持 v1。
该候选尚无胜率改进结论；配对评估只切换 `--upper-strategy rules/demand`，固定所有战斗
参数、checkpoint、native 和种子。统计 `overworld_demand_decisions`、
`overworld_demand_changed_decisions`、模型回退数和自然终局；少量选择改变不代表胜率提升。
库调用可用 `strategies.UPPER_STRATEGIES` 或直接组合 `EnemyDemandUpperStrategy`，
不需要导入 CLI。默认仍为 `--upper-strategy model`。

`--upper-strategy energy` 选择独立 `ironclad-overworld-energy-v1`，只改 Boss 遗物
选择，仍以原上层 v1 为基础，不叠加 demand。它枚举公开无序牌组的假设均匀五张手牌，
计算现有能量和多一点能量下可支付的 v1 牌效先验总值，加入有界评分修正。
这是费用约束下的静态近似，不是实际出牌计划或预计损血；抽牌牌只保留其原先验，
不会免费增加手牌。当前临时能量、动态费用、Snecko/Pyramid/Choker 等未建模组合
退回 v1；原遗物副作用先验保留。痛苦印记的两张伤口作为抽牌稀释计入，医疗箱
不免此成本；该均匀分布也不表示原生的首轮起手，因为伤口在初始抽牌之后加入。
奴隶项圈按普通战斗/精英或 Boss 各半的显式先验处理。trace 写入原评分、边际容量
与修正；结果计数 `overworld_energy_decisions`、`overworld_energy_changed_decisions`
和 `overworld_energy_abstentions`。评测固定同一下层，仅切换 rules/energy。

`--upper-strategy potion-shop` 选择独立候选 `ironclad-overworld-potion-shop-v1`。
只在第二幕商店、v1本来会离店、整条实际药水带为空且无Sozu时，考虑补一瓶白名单战斗药水；
不叠加demand/energy。先保留v1已有购物优先级，要求售价不超过当前金币一半且买后剩至少75金。
伤害/格挡/力量等确定效果用公开牌组角色先验减去实际显示价格排序；分值不是预计HP收益。
不重算折扣，不缓存Courier库存；取得一瓶后停止本候选的补货。随机生牌等未覆盖效果沿用v1。
`overworld_potion_shop_decisions` / `overworld_potion_shop_changed_decisions` 记录实际补货；
trace给出原动作、药水角色、价格和剩余金币。共享下层决定何时使用，不能保证保留至精英战。
满槽/Sozu购买扣空钱是既有SHOP01；本候选只保护新增补货，v1原本会买的行为仍完整委托。
该候选未经过整局性能验收；至少128配对整局只切换rules/potion-shop并固定其他配置后再判断。

跨上层稳健性至少区分不同权重家族、路线和规则上层。不同文件名或不同下层训练步数不代表
上层不同，应核对 shared/overworld 张量身份。已检查的 20000–20007 在后续迭代中转为开发集，
下一轮留出声明在 worklog；每轮先冻结源码/哈希再查看新结果，旧版本通过固定 Git revision
重跑。全局胜率、不同参考轨迹的配对战斗、人工牌组压力测试分别报告，不混合样本数。

## 规则与原生回合搜索

`--solver turn-planner` 使用 [无模型回合规划](../../../env/search/README.md)，
沿用同一 upper checkpoint 和 runner，自动选择显式 `native-v1` 完整动作接口。
默认每次决策采样 2 个独立未来，每样本 512 次 transition；可调整
`--planner-nodes`、`--planner-worlds`、`--planner-beam`、`--planner-depth` 和
`--planner-nonce`。有效预算的根动作覆盖下限与实际用量都记录在诊断中。

```bash
python -m rl.evaluation.eval UPPER.zip --solver turn-planner --planner-nodes 2048 --planner-worlds 2 --episodes 128 --seed 150000 --device cpu --num-envs 8 --max-steps 6000 --timeout 1800 --output runs/local/turn-planner-development.json
```

仍可用 `--upper-strategy rules` 替换上层。战斗叶子完全由人工状态价值评价；
上层模型的使用不代表战斗搜索调用神经网络。`effective_strategy` 记录完整配置和信息边界。
需要重建包含 `execute_to_enemy_boundary` 与 `remap_draw_selection_from` 的 native 扩展；
旧模型维度不变。普通 `--trace` 记录完整原生首步、根值及排序后的 draw 多重集；
`--compare-combats N` 支持同初态配对：保留 rules、policy、native 对照，另建一个同配置的
planner 候选，与参考策略的搜索对象隔离。每个候选 fork 使用自身声明的动作契约；
legacy 候选遇到 F03 空动作表时保留 `dead_screen` 和截断标记，初态已不可操作则记零步，
不会把它改成死亡或把非法动作交给策略。嵌套 `PhaseStrategy` 递归合并可到达的战斗契约；
有 predicate fallback 时包含默认策略，`native-v1` 环境仍接受合法 thin 动作。
配对 fork 的决策计数沿用从零开始的协议，结果同时记录 `reference_step` 和
`fork_decision_step_origin`，manifest 标明 `fork-local-zero`；它不是原轨迹随机样本的逐步重放。
`combat_rows` 是参考路径上的战斗组件对照，不能与整局 `rows` 合并统计胜率或样本数。
步数是本接口下的完整动作及上层决策数量，不能直接与 legacy 自回归子步骤比较吞吐。

`SelectiveTurnPlannerStrategy` is an optional adapter over the same planner and
leaf values. Its base config must use one turn; a public Pocketwatch count of
exactly3, a live Time Eater's public Time Warp11, or a live Intangible Nemesis
while Fiend Fire is in hand, selects two turns with four
times the configured node budget. Worlds/beam/depth/nonce and other config fields
stay fixed. Each decision reselects the mode, so ordinary states use the original
planner. The manifest declares both configs; trace `selected_search` records the
actual config and triggering counters, alongside native effective-budget and
transition counts. `deep_search_decisions` is an additive episode count. Root
coverage floors still apply, so four times configured nodes need not mean exactly
four times actual transitions. Runic Dome retains the common visible-rule fallback.
This candidate reuses `decide`, pending-action handling, tracing and the runner;
it does not establish a win-rate improvement or change any state-value rule.
