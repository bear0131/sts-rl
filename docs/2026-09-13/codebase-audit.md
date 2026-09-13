# sts-rl 全仓诊断与重构评审

审计日期：2026-09-13。范围：GitHub `bear0131/sts-rl` 远端默认分支及其可达历史。最终提交和增量核验见下文“版本与验证”。

## 结论

**建议启动有验收门槛的全仓重构，并先修复环境、动作和训练标签的语义，再继续扩大训练。** 目前最重要的问题不是代码风格或吞吐量，而是同一游戏事实在模拟器、观测、策略、搜索、规则 teacher、评测之间存在不一致。若这些契约不统一，继续增加环境步数、模型层数或搜索次数，会让实验更贵，却不能可靠回答算法是否有效。

仓库也并非完全不可用：原生模拟器可以从源码构建；常规及分层 PPO 可以完成实际更新和保存恢复；归档 NOSL checkpoint 可以驱动 PUCT 和 Full Gumbel 搜索；已有文档中多处主动记录了负结果和评测局限。这些资产值得保留。**本报告没有证据认定所有旧实验都无效，也没有证据支持把当前系统视为已通过游戏语义验证的研究基线。**

建议暂时停止把以下指标单独作为成功判据：训练 loss 下降、teacher-forced token accuracy、纯环境 FPS、累计环境步数、单个局面的搜索偏好。它们可以诊断局部行为，不能替代游戏正确性、完整动作模仿能力和独立评测。

## 版本与验证

用户指定以 remote/master 为准；实际 `git ls-remote --heads` 仅返回 `main`，不存在远端 master，因此采用远端 main。

初始审计提交为 `f3696a96501b8249da17c0fb3723967f53dbf961`（2026-09-10）。审计期间 main 更新到 **`1325041dce1fb3e9ca6cb2641bb91645924947b6`（2026-09-13）**，本报告已覆盖这一个新增提交的 59 文件增量，重新构建扩展、运行全部测试及缺陷反例；结论和源码链接均以此最终快照为准。初始测试结果另存于证据包，避免把跨版本结果混成一次运行。

最终历史包含 92 个可达提交。作者字段为 bear0131 42 次、unknown 50 次；这些字段不足以可靠区分两位实际作者，所以本报告按子系统和演进阶段评判，不按人归责。最后的大提交打包大量实验和代码，也限制了精确重建每个实验当时源码的能力。

原工作目录 `D:\zxb\my\sts-rl-workbench\sts-rl` 保持原状；审查、原生构建和诊断脚本在独立审计副本中进行，没有修改生产源码、提交代码或启动远端训练。测试依赖安装在独立审计环境中。

本次扫描了目录、入口、依赖、历史和实验材料，并深入检查了原生状态导出、canonical schema、观测和动作编码、两个策略家族、普通/分层 PPO、梯度累积及分布式同步接口、Python/C++ 搜索、规则数据和评测链路。以下“全面”指覆盖各子系统及其边界，**不表示穷尽了所有卡牌、遗物、事件组合或证明了全部代码正确**。

| 子系统 | 跟踪文件数 | 代码行数（粗略） |
|---|---:|---:|
| rl | 162 | 60,275 |
| env | 38 | 10,646 |
| bridge | 7 | 2,975 |
| sim | 99 | 39,436 |
| docs | 151 | 0 |

统计排除了 vendored json/pybind11，代码行数按 `.py/.cpp/.h/.sh` 计算，不等于人工逐行证明覆盖率。`artifacts/` 下有 20 个 checkpoint ZIP，合计 1.13 GiB；最新提交还纳入了根目录临时状态、日志及额外模型文件。

实测基线：

| 验证 | 结果与解释 |
|---|---|
| C++/pybind11 从当前审计源码构建 | 成功；测试使用此次构建的扩展 |
| `pytest rl env/scripts` | 最终提交 190 passed；初始提交 173 passed / 1 failed，旧 dashboard 配置测试已在增量提交中更新 |
| `env.scripts.test_pure` | 通过；包括空间、词表、编码和动作编码器检查 |
| `env.scripts.test_combat` | 通过；包含 15 次随机战斗及特定规则检查 |
| `env.scripts.test_game` | 通过；6 个固定种子、地图与奖励流程 |
| `env.scripts.test_run_v2 --episodes 20 --seed 0` | 通过；2,023 次战斗阶段决策、819 次目标选择；这不是通关性能评测 |
| 实际 PPO 更新、保存和加载 | 普通/分层各 64 步、梯度累积 2；两者均有 112 个 state-dict 张量变化，全部有限，保存加载逐张量一致 |
| 归档 checkpoint + 原生搜索 | `nosl_az_307200_steps.zip`，PUCT/Full Gumbel/strict-paper 各 8 次模拟成功；最新版使用现有 dashboard 的旧观测过滤器适配，普通 eval 的失败见 F18 |
| 新 packed leaf 与 canonical 对照 | Jaw Worm / Two Louse / Gremlin Leader 各一个同 nonce、同动作叶；包括消费 Akabeko 后的状态，各字段一致。这不能证明全场景等价，也会共同继承 F01 |
| 针对缺陷的对照脚本 | 结果见附带 evidence 包中的 JSON；用于证伪局部契约，不用于估计胜率 |

运行环境为 Windows、Python 3.11、Torch 2.12.0+cu132、NumPy 2.4.6、Gymnasium 1.0.0、SB3/sb3-contrib 2.9.0、Ray 2.48.0、pytest 9.1.1；优化器和搜索集成验证使用 CPU。没有复跑远端四机、GPU/BF16 长跑、真实游戏桥接或十亿步训练。规则事实只定向查阅本地 `reference/cardcrawl` 反编译源码；未验证该副本与所有实验机器的游戏版本完全一致。

## 缺陷清单与修复验收

本次列出 **19 项：9 项 P1、9 项 P2、1 项 P3**。优先阅读 F01–F07、F18–F19。证据级别和适用路径在各项内分别说明。

| 编号 | 优先级 | 问题 |
|---|---|---|
| [F01](#f01) | P1 | 怪物观测槽位与实际攻击目标错位 |
| [F02](#f02) | P1 | 两步动作没有把已选中的牌交给目标策略 |
| [F03](#f03) | P1 | 战斗内选牌候选缺失，pointer 指向错误牌堆 |
| [F04](#f04) | P1 | 规则训练 teacher 本身不是可信游戏执行器 |
| [F05](#f05) | P1 | 原生搜索终值与 learner 的终局 HP 标签不一致 |
| [F06](#f06) | P1 | 原生搜索找到了目标，但蒸馏标签丢弃了目标信息 |
| [F07](#f07) | P1 | Gauntlet 跨战斗持久状态丢失 |
| [F08](#f08) | P2 | 自动 reset 后种子流不可复现 |
| [F09](#f09) | P2 | Python NOSL 的 depth=0 不执行搜索 |
| [F10](#f10) | P2 | 模拟器仍有确定的规则与索引错误 |
| [F11](#f11) | P2 | 分层 rollout 截断会永久漏掉一类上层决策 |
| [F12](#f12) | P2 | “有效 batch 不变”不等于 PPO 更新统计不变 |
| [F13](#f13) | P2 | checkpoint 不是完整且统一的可恢复实验包 |
| [F14](#f14) | P2 | 原生模块选择依赖目录存在，PUCT 兼容降级会丢配置 |
| [F15](#f15) | P2 | 真实游戏 bridge 的 schema 故障可能变成无限刷新 |
| [F16](#f16) | P2 | 评测入口和实验结论缺少统一的证据边界 |
| [F17](#f17) | P3 | 研究原型已超过当前组织方式的维护能力 |
| [F18](#f18) | P1 | 9 月 13 日观测变更破坏普通 eval 对旧 checkpoint 的兼容 |
| [F19](#f19) | P1 | 新 End-Turn value 边界丢失 Ice Cream 的持久能量 |


P1：优先修复，会改变状态、动作、学习目标或核心实验解释。P2：受条件限制的正确性问题、复现和评测缺口。P3：维护成本。这里不把所有设计取舍都称为 bug。

<a id="f01"></a>

### F01 · P1 · 怪物观测槽位与实际攻击目标错位

**已复现。** `normalize_sim` 会删除 `INVALID` 怪物槽位，后续 encoder 却按保留下来的列表位置编码；原生 action target 仍使用原槽位编号。

Gremlin Leader 例子：原生槽位为 `[INVALID, GremlinWizard, ShieldGremlin, GremlinLeader]`，canonical 被压成三个元素。合法目标仍为 `[1,2,3]`，但观测第 3 位已经是 PAD，其他目标也指向错误的观测实体。固定槽位的现有单元测试使用构造好的 canonical 输入，没有覆盖这条原生到 canonical 的路径。

证据：[bridge/schemas/combat_state.py:259](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/bridge/schemas/combat_state.py#L259-L287)；`probe-results.json/monster_slot_mismatch`。

影响：普通策略、split 策略及复用相同归一化的神经搜索叶观测。修复应保留槽位，或使用显式双向实体映射；用于差分比较的“删除空项”视图不能直接复用为策略输入。验收必须包含空首槽、死亡空槽、召唤和目标执行后受伤实体一致性。

<a id="f02"></a>

### F02 · P1 · 两步动作没有把已选中的牌交给目标策略

**已复现。** 常规配置先选手牌，再选择目标；进入第二步时 `_pending` 保留了手牌索引，但观测仅在 `interactive_reasoning` 开启时加入 `pending_card_slot`。

在同一个 Two Louse 状态，先选 Bash 和先选 Strike 的两个分支，所有观测数组及目标 mask 完全一致。无记忆策略无法根据“刚才选了什么牌”调整目标分布。药水来源也需要纳入同一个契约。

证据：[env/core/run_env.py:1623](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/env/core/run_env.py#L1623-L1662)；[env/core/encode.py:312](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/env/core/encode.py#L312)；`probe-results.json/pending_source_invisible`。

验收：把 pending action 的类型、来源实体和必要参数作为所有策略共有观测；测试不同来源动作进入 target 阶段时输入可区分。也可以统一改为原子动作，但需要新动作版本和旧 checkpoint 迁移方案。

<a id="f03"></a>

### F03 · P1 · 战斗内选牌候选缺失，pointer 指向错误牌堆

**已复现。** 原生 `select_info` 导出上下文和数量，没有完整候选实体；encoder 仅在战斗外 `card_select` 阶段填 `select_card_ids`；combat `PICK_CARD` 被策略统一当作手牌索引。

Headbutt 例子：弃牌候选为 `[Defend, Strike]`，手牌为 `[Defend, Bash]`。两个选项均合法，但 `select_card_ids` 全零，候选 1 的 pointer 实际读取 Bash 的 token，执行却取回 Strike。问题还涉及需要不同候选来源的其他选牌任务，不能仅给 Headbutt 加特例。

证据：[sim/sts_lightspeed/bindings/slaythespire.cpp:1215](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/sim/sts_lightspeed/bindings/slaythespire.cpp#L1215-L1240)；[env/core/encode.py:799](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/env/core/encode.py#L799-L804)；[rl/pointer_policy.py:175](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/rl/pointer_policy.py#L175-L194)；[rl/split_policy.py:944](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/rl/split_policy.py#L944-L959)。

验收：统一 `SelectionContext`，包含候选稳定 ID、来源区域、卡牌状态、数量限制、已选集合；所有选择头只读候选列表。覆盖 hand/discard/exhaust/generated/deck 以及多选顺序、容量溢出。

<a id="f04"></a>

### F04 · P1 · 规则训练 teacher 本身不是可信游戏执行器

**静态链路确认并已做原生对照。** v7 生成链路会调用 `build_record`，其 `_effect_record` 是 Python 手写近似规则，随后再编译成协议；不是对原生执行 trace 的无损转写。

本次输入相同卡牌、无遗物的受控状态，观察到：

| 场景 | teacher | 原生模拟器 |
|---|---|---|
| 打出 Carnage | 手牌 Carnage 仍在，exhaust 新增随机 Headbutt+ | Carnage 进入 discard |
| Frail 下打 Impervious | 获得 30 Block，exhaust 增加随机 Spot Weakness | 获得 22 Block，Impervious 进入 exhaust |
| 打出 Inflame | 手牌 Inflame 仍在，discard 新增随机 Bloodletting | 能力牌离开手牌，不进入 discard/exhaust |

代码还把 True Grit、Burning Pact、Sentinel、Sever Soul、Immolate 列入 self-exhaust；`_add_pile` 生成随机牌代替实际移动的牌；`_draw` 只删除 draw 中的牌，没有加入 hand。这不仅是少量伤害公式误差，也违反卡牌身份及数量守恒。部分协议只编码牌堆数量，并不能抵消错误目的地/数量和数值效果。

证据：[rl/make_vector_rule_training_data.py:240](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/rl/make_vector_rule_training_data.py#L240-L249)、[rl/make_vector_rule_training_data.py:504](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/rl/make_vector_rule_training_data.py#L504-L748)、[rl/make_vector_rule_v7_data.py:1208](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/rl/make_vector_rule_v7_data.py#L1208)；`extra-probe-results.json/teacher_card_conservation`。本地 Java `Carnage.java` 区分 Ethereal，`Impervious.java` 使用实际 block，作为定向事实核对。

验收：明确标记现有 teacher 为近似数据源；禁止以其自洽性证明游戏正确性。优先从受控原生 transition/trace 生成 labels，另用小规模 Java/真实游戏对照校准原生规则。以卡牌守恒、状态增减、触发次序和最终效果为硬断言；未知规则应拒绝生成“gold”，不能静默近似。

<a id="f05"></a>

### F05 · P1 · 原生搜索终值与 learner 的终局 HP 标签不一致

**已验证两种 HP 读数不同，并确认调用链。** C++ 搜索直接使用战斗内 `curHp/200`；Python 搜索使用 `projected_exit_hp`；learner 在战斗结算后取下一阶段 `run_hp` 给整场样本打标。

50 HP、Burning Blood、击杀获胜：战斗原始 HP 为 50，退出投影为 56。因此原生已终止叶值为 0.25，而对应 post-exit 标签为 0.28。Burning Blood 只是简单例子；条件回血使不同终局的偏差不再只是常量。固定除以 200 不能解决“结算前还是结算后”的定义差异。

证据：[sim/sts_lightspeed/src/sim/search/NoslCombatTrainer.cpp:1123](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/sim/sts_lightspeed/src/sim/search/NoslCombatTrainer.cpp#L1123-L1129)、[sim/sts_lightspeed/src/sim/search/NoslCombatTrainer.cpp:1919](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/sim/sts_lightspeed/src/sim/search/NoslCombatTrainer.cpp#L1919-L1924)；[rl/nosl_is_az.py:102](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/rl/nosl_is_az.py#L102-L105)；[rl/accumulating_ppo.py:1381](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/rl/accumulating_ppo.py#L1381-L1394)。

验收：由同一个 combat outcome 接口返回 raw HP、post-exit HP、胜负和奖励，配置中明确选用哪一个；Python/C++ 搜索、GAE/AZ labels、评测必须使用同一定义。覆盖回血上限、Burning Blood、Meat on the Bone、死亡及超时。

<a id="f06"></a>

### F06 · P1 · 原生搜索找到了目标，但蒸馏标签丢弃了目标信息

**代码和真实 checkpoint 均确认。** 原生搜索 action 包含 card+target；`thinActionId` 只留下手牌来源，多个目标的 prior 平分同一 card prior；适配层把完整动作分布加总为同一个 PLAY 槽位。执行通过 `_nosl_cpp_pending` 一次完成，没有随后 target 观测供 learner 收集条件标签。

用归档 checkpoint 跑 Two Louse：PUCT 和 Full Gumbel 分别选 target 0、1；两次蒸馏标签的 target 段概率质量都是 0，执行后 pending 均为空。当前机制能训练“哪张牌”，不能完整教会无搜索策略“这张牌对谁用”。这不是说 C++ 搜索没有区分目标价值；丢失发生在搜索结果到策略训练的接口。

证据：[sim/sts_lightspeed/src/sim/search/NoslCombatTrainer.cpp:1136](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/sim/sts_lightspeed/src/sim/search/NoslCombatTrainer.cpp#L1136-L1158)；[env/core/run_env.py:1029](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/env/core/run_env.py#L1029-L1124)；[rl/accumulating_ppo.py:1306](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/rl/accumulating_ppo.py#L1306-L1338)；`integration-results.json/checkpoint_native_search`。

如果产品目标始终要求带搜索运行，这可以是明确声明的限制；若目标包括 policy-only 的 AlphaZero 式完整动作蒸馏，则必须修复。选择原子候选动作头，或同时学习 `P(card)` 与 `P(target|card,state)` 及多选条件分布。验收应比较完整动作分布，不能仅比较 card-level KL。

<a id="f07"></a>

### F07 · P1 · Gauntlet 跨战斗持久状态丢失

**已复现。** `SimCombatBackend._settle` 只复制当前 HP，直接从旧 `GameContext` 创建下一场战斗，不完成持久状态同步。

Feed 击杀前为 80/80，击杀后为 83/83，下一场为 **83/80**。遗物计数、消耗药水和永久牌组变化也处于同一个遗漏同步的边界，需要分别测试。设计中“无战后治疗”可以保留，但不能因此丢掉最大 HP 等持久状态。相同 floor/seed 重建同一 encounter 的随机流也应单独界定。

证据：[env/core/combat_backend.py:80](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/env/core/combat_backend.py#L80-L92)；`extra-probe-results.json/gauntlet_feed`。

验收：抽出 battle-to-run 状态提交；将是否回血作为独立规则，而不是跳过整个 exit-battle；加入 `0 <= hp <= max_hp` 及持久效果跨场测试。此项主要影响 gauntlet 预训练，不能据此声称完整跑局环境也丢失相同状态。

<a id="f08"></a>

### F08 · P2 · 自动 reset 后种子流不可复现

**已复现。** 薄环境每次 reset 都新建 backend。第一次传入显式 seed 正确；后续向新 backend 传 `None`，使用新熵源，而没有从已固定的 Gym RNG 派生下一局 seed。

两个独立环境执行 `[reset(42), reset(None), reset(None)]`，第一局 game seed 一致，后两局不同。SB3 等自动 reset 的训练因而不能仅凭初始 `--seed` 重建 episode 流。每一局显式设 seed 的评测不受这一个路径影响。

证据：[env/core/run_env.py:275](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/env/core/run_env.py#L275-L278)；`probe-results.json/seed_stream`。

验收：使用持久 RNG 派生 episode seed，记录 base seed、rank/env ID、episode ID 和真实 game seed；覆盖自动 reset 和恢复后的序列。精确续训还需环境/采样 RNG 状态，单独恢复模型 optimizer 不等于完整续训。

<a id="f09"></a>

### F09 · P2 · Python NOSL 的 depth=0 不执行搜索

**已复现。** 配置允许 0 表示无深度限制，但 Python session 和同步搜索循环使用 `range(depth)`。8 次模拟均在未推进动作的 root 请求叶值，最终退回均匀合法动作分布。

证据：[rl/nosl_is_az.py:193](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/rl/nosl_is_az.py#L193)、[rl/nosl_is_az.py:359](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/rl/nosl_is_az.py#L359)；`probe-results.json/python_depth_zero`。

C++ 有明确的 `maxDepth==0` 分支，本次实际 depth=0 搜索能推进；已归档 NOSL 元数据均启用 C++，**不能用这个 Python 缺陷否定那些 C++ 搜索实验**。验收对 Python/C++ 同一深度配置做最小一致性测试。二者随机状态节点组织也不同，不应未经验证就当成可互换实现。

<a id="f10"></a>

### F10 · P2 · 模拟器仍有确定的规则与索引错误

**Impatience 已复现**：遍历手牌发现 ATTACK 时仍写 `hasAttack=false`，导致有攻击牌也抽牌。本次 4 张 Strike + Impatience、draw 5 张，打牌后 hand 6 张、draw 3 张。Java `ConditionalDrawAction.checkCondition` 明确要求手牌不存在受限类型。

**另有静态确定错误**：`exhaustSpecificCardInHand` 的 fallback 循环遍历 `i`，却继续比较 `cards.hand[idx].uniqueId`。当原索引失效、卡牌已移位时无法按唯一 ID 找到牌；该触发场景尚未动态构造，不能把它报成每次 Exhaust 都失败。

证据：[sim/sts_lightspeed/src/combat/BattleContext.cpp:1422](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/sim/sts_lightspeed/src/combat/BattleContext.cpp#L1422-L1432)、[sim/sts_lightspeed/src/combat/BattleContext.cpp:2678](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/sim/sts_lightspeed/src/combat/BattleContext.cpp#L2678-L2695)；`extra-probe-results.json/impatience_with_attacks`。

验收：规则最小 fixture + 队列引起手牌位移后的 unique-ID 查找测试。模拟器来自成熟上游或已经做过旧审计，均不能替代这些断言。

<a id="f11"></a>

### F11 · P2 · 分层 rollout 截断会永久漏掉一类上层决策

**行为已由代码和现有测试确认；实际影响大小未测。** 当 rollout 恰好结束在一场 combat 中间，开启该场战斗的 overworld macro transition 被标 NaN，从 PPO loss 中排除。函数只处理当前 buffer，没有跨 buffer 结算这一条上层行为的机制。

“每个 env 每轮至多一个”只限制绝对数量；其入选概率与战斗持续时间相关，可能对长战斗造成选择性漏样。不能直接据此解释火堆偏置。长 `n_steps` 会减轻比例，但不是语义修复。

证据：[rl/hierarchical_returns.py:52](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/rl/hierarchical_returns.py#L52-L58)、[rl/hierarchical_returns.py:133](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/rl/hierarchical_returns.py#L133-L185)。

验收：记录 unresolved 比例和按 option 时长的分布；设计跨 buffer macro transition 或明确的 option continuation bootstrap。若跨多次策略更新再结算，需处理行为策略版本/重要性比率，不能只把旧样本塞回新 PPO buffer。

<a id="f12"></a>

### F12 · P2 · “有效 batch 不变”不等于 PPO 更新统计不变

**已做数值反例。** `ppo_loss_components` 在每个 microbatch 内标准化 advantage；分层时还在 microbatch 内分别对 phase 求均值。之后平均这些梯度，不等于在合并的全局样本上计算一次同一损失。不同 rank 的 phase 有效样本分布会进一步改变权重。

相同 4 个样本、相同参数：完整 batch 梯度 `-2.165343`，拆成两个 microbatch 后平均梯度 `-1.414213`。这是微批统计定义差异，不是梯度累积代码没有执行，也不是数值精度误差。普通 PPO 按 minibatch 标准化本身可以合理，但调整 microbatch 不应被当作纯显存优化的严格等价实验。

证据：[rl/accumulating_ppo.py:153](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/rl/accumulating_ppo.py#L153-L156)、[rl/accumulating_ppo.py:195](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/rl/accumulating_ppo.py#L195-L219)；[docs/current-wholerun-training.md:90](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/docs/current-wholerun-training.md#L90-L103)；`extra-probe-results.json/accumulation_not_full_batch`。

验收：决定统计边界，固定 rollout/global-phase normalization，按有效样本数同步 loss 分子分母；或保留现有目标但明确记录其定义。以单 batch / 累积 / 多 rank 同步的参数更新一致性作为测试。非分层 `std()` 的 singleton 尾批也需要 finite guard。

<a id="f13"></a>

### F13 · P2 · checkpoint 不是完整且统一的可恢复实验包

归档 WholeRun checkpoint 保存了 Linux 绝对 `card_vector_checkpoint` 路径；独立 encoder 不在该 ZIP 中。`distributed_sb3` 在应用 `--card-vector-checkpoint` 覆盖之前，已经先 `load()` 了一次旧模型，迁移机器后可以在这一步因缺文件失败。`rl.eval` 的候选路径还没有包含原 saved path 本身，即使原路径可用也需额外配置。

证据：[rl/distributed_sb3.py:1754](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/rl/distributed_sb3.py#L1754-L1807)；[rl/eval.py:100](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/rl/eval.py#L100-L140)；`checkpoint-inventory.json`。

归档 NOSL checkpoint 的对应字段为 null，并不意味着所有 checkpoint 都依赖丢失文件；本次实际加载一个 NOSL checkpoint 成功。需要区分不同家族。

验收：先只读元数据、解析并校验依赖，然后只构建一次模型；保存依赖 hash 或内嵌冻结 embedding。manifest 应包含代码/原生 build SHA、obs/action/reward/search contract、实际覆盖后的配置、训练预算口径及 RNG/恢复级别。已有部分 SHA 文档值得保留，但不能替代自动强校验。

<a id="f14"></a>

### F14 · P2 · 原生模块选择依赖目录存在，PUCT 兼容降级会丢配置

默认优先 `build-gumbel-current`，其次 `build-nosl311`，最后 `build`，只检查目录；“刚编译成功”不能保证随后 import 的就是它。适配层在旧扩展 TypeError 时回退到 5 参数接口，可丢掉非默认 `root_value` 和 `prior_temperature`。当前代码对旧扩展的 Full Gumbel 请求会报错，这个保护是正确的，但普通 PUCT 的回退仍不是严格配置等价。

证据：[env/core/__init__.py:34](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/env/core/__init__.py#L34-L65)；[env/core/run_env.py:1177](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/env/core/run_env.py#L1177-L1250)。

验收：启动打印/记录实际扩展路径、能力版本和构建 hash；请求了扩展无法实现的非默认参数时拒绝启动。CI 使用明确 build 路径；版本兼容集中到一个适配模块。

<a id="f15"></a>

### F15 · P2 · 真实游戏 bridge 的 schema 故障可能变成无限刷新

`RealGameAdapter` 捕获 `normalize_real` 所有异常后继续以 `{}` 构造 combat/global 状态，并添加仅含 counters 的 player。下游 encoder 要求 `player.hp` 等字段，这个“恢复”分支会再次报错。最外层 orchestrator 捕获任意异常并发送 `STATE`；持续的 schema/模型兼容问题可以因此反复刷新，缺少重试上限和明确的故障状态。日志保留了异常，不能把它描述成完全静默或已经证实会执行错误游戏动作。此项为静态故障链确认，没有启动真实游戏验证出现频率。

证据：[bridge/real_game_env.py:488](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/bridge/real_game_env.py#L488-L512)；[bridge/client/agent_orchestrator.py:346](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/bridge/client/agent_orchestrator.py#L346-L358)。

验收：区分瞬态未就绪、schema 不兼容和模型错误；无效帧进入显式不可行动状态，有限重试并保留原帧供重放。建立真实 frame fixture，测试 simulator/real adapter 在同一可观察游戏状态上的等价性，避免复制两套 phase/mask 规则后自行漂移。

<a id="f16"></a>

### F16 · P2 · 评测入口和实验结论缺少统一的证据边界

`rl.eval` 默认执行 policy-only，显式打开的是 beam value search；dashboard 还有原生 NOSL PUCT/Gumbel 路径。加载同一个 NOSL checkpoint 不等于运行相同算法。`rl.eval` 还刻意取消步数限制，遇到循环策略时整次评测可能没有结果。

证据：[rl/eval.py:1](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/rl/eval.py#L1-L6)、[rl/eval.py:178](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/rl/eval.py#L178-L190)、[rl/eval.py:243](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/rl/eval.py#L243-L254)；[rl/dashboard.py:659](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/rl/dashboard.py#L659-L696)。

验收：统一 evaluator，明确 `policy_only / beam / nosl_puct / legacy_gumbel / full_gumbel`；结果逐局记录实际 solver 参数、checkpoint hash、种子和预算。设置 wall-clock/decision budget，把 timeout 单列，不隐式算作死亡。采用固定 held-out 种子和保存状态的 paired 对照，避免用不同策略访问到的不同状态解释局部行为变化。

<a id="f17"></a>

### F17 · P3 · 研究原型已超过当前组织方式的维护能力

规则表示 v5/v6/v7、多个训练器、多个搜索器、真实桥接和 dashboard 都在一个仓库快速演进。多个核心文件超过千行，实验启动及恢复混入核心逻辑；README 仍引用不存在的 `rl.train` / `rl/train.py`。测试分散、缺少根 CI 门禁；大 checkpoint 直接进入 Git，使普通 fetch/review 也承担大文件成本。

证据：[README.md:17](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/README.md#L17)、[README.md:49](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/README.md#L49)；仓库文件与 checkpoint 清单。这里的风险不是“文件长就错误”，而是契约变更没有一个必经、统一的验证入口。

验收：建立受支持入口表、最小开发依赖及分层 CI；历史实验保留到有 manifest 的 archive。大模型迁到版本化 artifact store/LFS，保留 hash 和获取脚本；不需要为此立即重写 Git 历史。

<a id="f18"></a>

### F18 · P1 · 9 月 13 日观测变更破坏普通 eval 对旧 checkpoint 的兼容

**在最终提交中运行正式入口复现。** `obs_space()` 无条件增加 `post_enemy_turn_boundary`；旧 checkpoint 的 observation space 没有该 key。运行 `python -m rl.eval artifacts/latest-nosl-az-s512-20260905/nosl_az_307200_steps.zip --episodes 1 --device cpu`，模型加载成功，但第一次 `model.predict` 抛 `KeyError: 'post_enemy_turn_boundary'`。

新增的 `CombatEncoder` 动态选择字段使模型本身能加载；dashboard 也增加了 `_observation_for_policy` 过滤器，但普通 eval 没有使用。说明“旧 checkpoint 可以 load”不能替代“旧 checkpoint 能运行/恢复”的兼容验收。其他恢复入口也必须逐个检查环境空间匹配，不能假定 dashboard 的补丁全局生效。

证据：[env/core/encode.py:183](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/env/core/encode.py#L183)；[rl/eval.py:256](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/rl/eval.py#L256)；[rl/dashboard.py:98](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/rl/dashboard.py#L98-L116)；`latest-probe-results.json/legacy_checkpoint_actual_eval`、`legacy-eval.log`。

验收：统一 observation contract adapter，在所有预测/评测/恢复入口复用；矩阵覆盖每个保留 checkpoint 家族的加载、至少一步预测、环境接入及保存恢复。对搜索 boundary 的新语义，应明确旧模型是否支持，而不只是删 key 后继续推断。

<a id="f19"></a>

### F19 · P1 · 新 End-Turn value 边界丢失 Ice Cream 的持久能量

**在最终提交中已复现。** `enterPostEnemyTurnBoundary()` 无条件把 `player.energy=0`，声称在清理过期状态。但 Ice Cream 下剩余能量是未来资源。

同一 Jaw Worm 状态、相同 RNG、持有 Ice Cream：结束回合前分别剩 1/3 能量，生成的全部 boundary 观测数组相同、energy 都是 0；实际执行 END_TURN 后下一回合分别有 4/6 能量。该 value 输入无法表达两种未来机会的区别，给它最终 HP 标签也无法恢复被删掉的信息。

证据：[sim/sts_lightspeed/src/combat/BattleContext.cpp:2233](https://github.com/bear0131/sts-rl/blob/1325041dce1fb3e9ca6cb2641bb91645924947b6/sim/sts_lightspeed/src/combat/BattleContext.cpp#L2233-L2283)；`latest-probe-results.json/icecream_boundary_aliasing`。此项影响启用 `end_turn_as_leaf` 的路径；不应笼统否定默认关闭该选项或显式 `no-end-turn-as-leaf` 的实验。

验收：边界状态必须保留所有影响后续转移的公开资源；清理逻辑应从真实 turn transition 拆分并共享，不能另手写一个近似状态。用 Ice Cream、Barricade、Blur、Calipers、跨回合 counters 等验证“相同观测确实代表相同的可观察未来条件”。

## 对既有工作和实验的重新判断

### 早期环境、模拟器和 PPO 基线

薄/厚环境分层、canonical schema、动作 mask、词表和 bridge 差分思路是可以继续使用的基础。但 F01/F02/F03/F07 说明，早期的“能跑通”和“观测在 space 内”未覆盖实体与动作的语义一致性。这些缺陷应优先追溯为基线契约问题，不能全归因于后续复杂模型。

也不应照抄旧 audit 的所有问题：当前版本中部分历史问题已经修复，例如 Akabeko/Vigor 导出、若干卡牌数值，以及原先的部分怪物/伤害逻辑问题。本次 Vigor 等实际测试通过。`rl.eval.is_win` 目前判断三幕结束，不应再把旧的一幕结束判断当作现存错误。旧报告是线索，当前源码与可复现结果才是结论依据。

### 后期分层 PPO、性能优化和分布式训练

阶段专用网络、释放不需要的激活、梯度累积和同步指标具有工程价值；本次小规模更新能跑通，现有相关测试多数通过。但 F11/F12 使“只改 batch，统计方法完全不变”的说法需要收窄。多机训练提供吞吐，不会自动改善 credit assignment。

`current-wholerun-training.md` 明确说明 online rollout 重用、没有独立 validation；这不是偷用测试集的证据，却意味着在线 reward/loss 不能当作泛化效果。火堆文档记录高血量也高度偏向休息，并明确指出两个 checkpoint 到达的状态集合不同，不能严格配对比较。这些负结果应保留为待解释事实。尚无受控证据把休息偏置单独归因于某个 bug、奖励设计或网络结构。

### 规则学习、匿名向量和 embedding 迁移

已有 `ironclad-attack-damage-accuracy-epoch14756.md` 把 100% teacher-forced HP 输出与只有 58.33% 的基础伤害读取区分开；精确覆盖基础/升级两版本的固定伤害卡仅 4/29。`simple-card-v7-eval-step10000.md` 列出近乎完美 token accuracy，但数值效果 MAE 9.0686，并指出 Perfected Strike/Frail teacher 缺口。**这些文档本身已经不支持“模型完整学会游戏规则”。**

F04 进一步给出 teacher 的卡牌守恒反例。评估必须分开：协议语法、条件局部补全、无 gold prefix 的自回归推理、真实状态最终效果、迁移到游戏策略的因果收益。不要把 token accuracy 或 embedding PCA 分离当作最终能力指标。

这也不直接证明冻结 embedding 对 PPO 一定有害：decoder 数值差不等于表示毫无价值。需要在修正后的相同环境、相同预算下比较随机可训练 embedding、随机冻结 embedding、手工语义特征、当前规则 embedding、原生 trace 训练 embedding。当前无法从仓库材料恢复出足以支持因果迁移收益的完整对照链。

### NOSL、AlphaZero 和 Gumbel

隔离真实隐藏 RNG、原子动作搜索、叶评估 batch 化是有意义的方向。本次实际原生搜索并未退化为完全不运行。仓库的 order-sensitivity 文档也正确指出强 policy prior 压制探索，两个换位 fixture 不足以证明长期规划能力；Python depth=0 的缺陷不适用于其中的 C++ 路径。

但 F05/F06 是现存目标与蒸馏契约缺陷。当前应描述为“使用神经先验/值的战斗局部随机搜索及 card-level 蒸馏”，并说明目标分布限制。Full Gumbel 本次验证覆盖可运行性与接口，不构成对论文算法完整等价性的证明；模式命名、公式单测和长期性能评测应分开验收。

9 月 13 日增量进一步区分实验性 Full Gumbel 与 strict-paper，增加显式 boundary 特征、零 policy weight 的 value-only 样本、replay 新鲜样本配额和多次同步 AZ 更新。新增测试覆盖其中若干边界并全部通过；不应再报告“所有 AZ 分布式路径只支持一次更新”等旧限制。另一方面，F18/F19 表明契约变化的影响仍未通过全入口测试约束。strict 模式已做 8 次模拟的运行验证，未与官方数值实现逐节点对照；文档中的“严格复现”应保留这一证据边界。

## 推荐重构结构

先在现有代码上建立正确性边界，再迁移文件；不建议直接重写整个模拟器或同时替换所有训练框架。

| 边界 | 唯一职责及必须输出的契约 |
|---|---|
| Simulator adapter | 原生状态、实际扩展身份、原子动作执行、可重放 transition、battle outcome |
| Observable state | 稳定实体 ID、公共信息、pending action、selection candidates；禁止隐式删槽 |
| Action codec | 语义动作、原子/自回归映射及双向转换；与 policy/search/bridge 共用 |
| Policy core | 纯 `observation + legal candidates -> logits/value`；不承载 checkpoint 寻址、SSH、界面逻辑 |
| Objectives / trajectory | reward contract、GAE、macro transition、AZ 完整动作 labels、timeout 语义 |
| Search | 显式 sampler/value/prior/action 接口，返回完整动作统计及预算，不直接决定训练标签格式 |
| Learner/runtime | 一个明确支持的主训练后端；SB3/RLlib/分布式包装层只负责采样、同步和持久化 |
| Rule-data pipeline | 数据来源/规则覆盖可追踪，原生 trace 与近似 teacher 分开版本化 |
| Evaluation/artifacts | 固定场景、held-out seeds、solver 身份、paired comparison、manifest、checkpoint 完整性 |
| Dashboard / real bridge | 使用上述接口；业务 UI 和外部协议不得复制核心规则 |

短期以已经能运行的 SB3 + 原生模拟器 + 一条明确的搜索路径为受支持基线较稳妥；RLlib、独立 gauntlet、旧规则协议可保留为实验性后端。这是降低同时维护的契约数量，不是宣布某种框架天然优劣。

## 推进顺序与放行条件

| 阶段 | 工作 | 放行条件 |
|---|---|---|
| 0：保存可追溯基线 | 固定 commit、构建、checkpoint 和依赖；保留本次反例与旧结果 | 干净 checkout 能一条命令构建并复现诊断；旧结果明确标注 contract |
| 1：修环境/动作契约 | F01–F03、F07、F08、F10、F18/F19；bridge 必需字段验证 | 最小规则 fixtures、实体/目标一致性、选择候选、跨战斗持久状态、旧 checkpoint 和 reset 序列通过 |
| 2：统一学习目标 | F04–F06、F11–F12；确定支持的动作表达 | 同一终局各路径 label 相同；完整动作能被蒸馏；buffer/分片变化影响可解释 |
| 3：整理运行和证据 | F13–F17；模块抽离、manifest、CI、统一 eval | 迁移机器可加载依赖；请求算法与实际执行一致；超时与失败可定位 |
| 4：重做最小实验矩阵 | 小模型、少量固定场景，逐项开启分层/embedding/search | 多个训练随机种子、固定 held-out 集及 paired 场景；报告置信区间和实际计算预算后再扩规模 |

第一轮测试矩阵应包含：槽位错位的 Gremlin Leader；Bash/Strike 条件目标；Headbutt/Exhume/生成牌/多选；Frail/Dexterity/Weak/Strength 与消耗牌；Feed 与持久遗物；终局回血；不同 rollout 切点；强先验下的搜索探索；有/无搜索的完整动作一致性。每个测试都应能在对应缺陷恢复时失败，避免仅镜像实现。

重新评测历史 checkpoint 时应同时保留“原环境复现”和“修正环境迁移”两条结果：修复 observation/action 契约后直接横比旧曲线会混入分布变化，不能把差异全归为模型改进。无需一开始重跑十亿步；先用有明确期望的小场景判断哪个环节失败。

## 交付与未覆盖边界

附带 evidence 包保存诊断脚本、原始 JSON、pytest/JUnit、冒烟日志、checkpoint 元数据清单、源码/参考文件身份信息及构建命令。脚本需要本地构建的 `slaythespire` 扩展和对应 Python 依赖；它们是局部反例与集成检查，不是全游戏 oracle。没有分发反编译游戏源码或模型大文件。

未确认的内容包括：四机故障恢复和跨 rank 位级一致性、GPU/BF16 长跑稳定性、全部事件/卡牌/遗物组合的 Java 一致性、真实 CommunicationMod 操作回放、历史 W&B/远端日志的完整性、所有 checkpoint 的完整胜率，以及 Full Gumbel 与外部参考实现的数值等价性。对应模块已纳入静态和接口审查；未实测部分应保持为后续验收项，不能写成通过。

## 外围工程追加：性能、维护性与重构准备

本节承接用户追加的四项要求。上文诊断的固定基线仍为远端 main `1325041`；下文是随后在本地 `chore/refactor-foundation` 分支进行的工程干预及实测。因此，上文的“未修改源码”描述原始审计阶段，不再描述本节实施阶段。没有推送远端或启动集群作业。

### 训练性能：应按瓶颈和计算预算评价

测量机器为 Intel Core Ultra 7 155H（16 物理核、22 逻辑核）、约 31.47 GiB RAM、RTX 4060 Laptop GPU（8 GiB）；Windows、Python 3.11.15、Torch 2.12.0+cu132、SB3/sb3-contrib 2.9.0、Gymnasium 1.0.0，原生扩展为本机 GCC/Ninja 构建。不是训练集群，不据此预测四机加速比。

固定工作量：完整 SlayTheSpireEnv + ShapedReward + TimeLimit，带全图观测和 map DAG 的 SplitStSPointerPolicy，d_model=128、4 层、8 heads、8 环境、每环境 rollout=64、microbatch=128、累积=2、2 epochs、seed=42、Torch/OMP/BLAS 线程=1。预热 1 个 rollout，测量 6 个 rollout，共 3,072 个环境 transition；PPO 无搜索、无 AMP。这里一次 transition 可以是选牌、选目标或地图决策，不等于一张完整牌或一场战斗。

| 运行顺序 / 配置 | transitions/s | 采样 s | 优化 s | 测量区间 s | 进程墙钟 s |
|---|---:|---:|---:|---:|---:|
| CPU / dummy，首轮 | 172.9 | 5.40 | 12.37 | 17.77 | 25.29 |
| CUDA / dummy，首轮 | 204.1 | 8.58 | 6.47 | 15.05 | 23.14 |
| CUDA / 8 worker | 100.5 | 18.18 | 12.38 | 30.57 | 53.48 |
| CUDA / 4 worker，每个 2 env | 78.0 | 22.59 | 16.81 | 39.40 | 60.01 |
| CUDA / dummy，复测 | 77.1 | 22.78 | 17.08 | 39.86 | 55.23 |
| CPU / dummy，复测 | 61.6 | 15.28 | 34.62 | 49.90 | 64.63 |

速度结论仅适用于该配置和短程单次对照。阶段时间含 CUDA 同步，进程墙钟还包含 Python 导入、模型建立、预热和退出；表中的吞吐仅为预热后的训练区间。没有把 saved checkpoint I/O、跨 rank all-reduce 或 NOSL 搜索开销悄悄算作零。

**主要发现是性能波动和串行工作占比，而不是稳定的 CPU/GPU 排名。** 同一 CPU 配置复测从 172.9 降至 61.6 transitions/s，同一 CUDA 配置从 204.1 降至 77.1；两者均接近 2.7–2.8 倍变化。后续多进程运行的优化阶段也变慢，不能把全部差异归因于 IPC。没有锁定电源/频率和机器其他负载，具体原因未确认。因此不得选择首轮最好数字宣布加速比，也不依据这几次运行断言 grouped 一定慢于 subproc。

仍然能观察到：训练进程树通常只使用约一个逻辑核；设备 GPU 平均利用率约两三成；多进程方案明显增加 RSS 和启动时间。在这一小模型/小 batch 配置中，8 GiB 显存远未用满，主要约束不表现为模型装不下。首轮 CUDA 的优化比 CPU 短，但采样更长；复测仍是相同方向。应优先拆解小批推理/传输、编码和环境推进成本，再决定增加 GPU 或 worker。设备级 GPU 指标可能含其他进程，因此这仍是有边界的观测而非完整归因。

| 配置 | CPU % | RSS 峰值 GiB | GPU util 均值 / 峰值 % | 设备 VRAM 峰值 MiB | Torch allocated / reserved 峰值 MiB |
|---|---:|---:|---:|---:|---:|
| CPU / dummy，首轮 | 98.5 | 0.81 | 未采集 | 不适用 | 不适用 |
| CUDA / dummy，首轮 | 100.6 | 1.72 | 26.8 / 53 | 312 | 161.2 / 186.0 |
| CUDA / 8 worker | 114.4 | 5.75 | 23.3 / 58 | 314 | 161.2 / 188.0 |
| CUDA / 4 worker，每个 2 env | 110.2 | 3.74 | 25.0 / 64 | 312 | 161.2 / 186.0 |
| CUDA / dummy，复测 | 101.7 | 1.73 | 24.9 / 56 | 312 | 161.2 / 186.0 |
| CPU / dummy，复测 | 98.1 | 0.81 | 未采集 | 不适用 | 不适用 |

另做了实际原生搜索训练：2 env、rollout=64、d_model=32/2 层/4 heads、batch=32、8 simulations、depth=0、replay capacity=2,048，预热 1 次、测量 4 次更新。配置和算法与上表不同，以下仅证明完整训练路径及其短程成本，不能用于 CPU/GPU 加速比或算法优劣比较。

| 搜索训练冒烟 | 测量 transitions | transitions/s | 采样 / 优化 s | RSS GiB | 累计 replay labels（含预热） |
|---|---:|---:|---:|---:|---:|
| CPU PUCT | 512 | 86.9 | 4.59 / 1.30 | 0.63 | 497 |
| CUDA strict Full Gumbel | 512 | 28.6 | 13.86 / 4.01 | 1.76 | 499 |

搜索冒烟中 CPU PUCT 进程树 CPU 均值约 97.7%；CUDA strict 的 CPU 均值为 98.7%，GPU util 均值/峰值为 19.3%/34%，设备 VRAM 峰值 160 MiB，Torch allocated/reserved 峰值 26.8/34.0 MiB。两次搜索运行分配的 replay 数组均约 8.60 MiB；完整样本与配置保存在证据包。

资源口径：CPU 100% 代表占满一个逻辑核，不能把它误读成 22 核全满。RSS 是训练进程及子进程的采样和，共享 DLL/库页面可能重复计数，峰值也可能漏采。GPU util/设备 VRAM 来自 nvidia-smi、每约 0.5 秒一次，含桌面和其他进程；Torch allocated/reserved 是本训练进程的分配器统计。GPU util 不等于浮点单元效率，也不能由低 util 单独断定 GPU 太弱。原始采样与不可用指标说明随证据包提供。

内存规划不能只看模型参数。本配置模型参数约 6.40 MiB，512 行 rollout 数组已约 9.04 MiB，即约 18.1 KiB/transition（实际观测配置相关）。相同字段、262,144 行就约 4.52 GiB；这是容量外推，不是本次实际分配，也不是当前 CLI 默认值。`distributed_sb3.py` 当前默认 n_envs=32、n_steps=256，为每 rank 8,192 行。同样观测约 145 MiB/rank，随后还要另算 replay、临时 batch、激活、优化器、进程和通信副本。NOSL replay 已采用较紧凑的整数/float16 存储；不要为了“规范”退回全 float32 存储，也不要把 float16 replay 与原始观测宣称逐位等价。

下一轮性能工作按以下顺序开展：

1. 保持语义和模型配置不变，先拆开采样中的环境推进、编码、mask/IPC、policy forward 和设备传输。新基准已分采样/优化；现有 profiler 有更细阶段，之后应统一字段和统计口径。
2. 对目标训练硬件扫 env 数、worker 分组和 microbatch，并记录有效 transition、search simulations/leaf、optimizer updates。较大 batch 可能提高 GPU 使用率，也会改变 rollout/staleness 和学习统计，须分别报告。
3. 检查 rollout 内反复 `.cpu().numpy()`、小 batch 推理和 Python/C++ 往返。NOSL 的根/叶 batch 化与 packed exporter 有价值，应在等价测试之后继续优化；本轮 packed/canonical 对照只是指定场景的数组一致性，不证明两者共同的语义正确。
4. 分布式必须另测 rank 采样不均衡、同步等待、all-reduce 和每 rank 常驻内存。需要实际集群长跑才能判断速度/利用率，当前不代替这个验收。

### 实用维护性：目前增加一个功能仍然危险

问题主要在职责和证据边界，而不是空格、命名风格或是否使用某个框架。

| 维度 | 当前风险与实际例子 | 本轮建立的约束 / 后续方向 |
|---|---|---|
| 审查粒度 | 最新基线一次提交涉及 59 文件、19,480 行新增、288 行删除，混入实验和运行材料，难以隔离原因 | 每项可验收工作独立 commit，日期 worklog 保留原始意图和验证；移动文件与改行为分开 |
| 定位入口 | `distributed_sb3.main` 约 1,284 行，`trainer.main` 874 行，NOSL collector 585 行；加载、配置、算法和运行管理相互缠绕 | 按前文架构边界渐进抽离；先提取配置验证和纯算法函数，不先全仓改名 |
| 添加观测 | 一个 boundary 字段破坏旧 checkpoint 的正式 eval，dashboard 局部补丁没保护其他入口 | 本轮共享额外字段适配与保存/加载/eval 回归；后续建立 schema/version 和兼容矩阵 |
| 添加动作/卡牌 | canonical、packed、policy pointer、搜索原子动作、bridge 各有映射，空怪物槽和 Headbutt 反例均可触发 | 正式 tests/ 中保留严格预期失败；先确立实体 ID/候选契约，再扩卡牌 |
| 修改学习目标 | 搜索终值与 learner、完整动作与 card-level labels 不一致，改性能参数可能同时改统计方法 | 各 objective 建独立 oracle 和 trajectory fixtures；PPO/GAE、AZ、teacher 的验收不可互相替代 |
| 迁移和复现 | 原生构建目录自动选择、旧绝对依赖路径、checkpoint 和源代码脱节 | CI 从源码显式构建；基准记录加载路径/版本；完整 manifest、外部依赖寻址仍在 TODO |
| 文档和进度 | 大量一次性实验文档/根日志难以区分现行规范；报告和实现可能互相过期 | 单一 TODO + 日期 worklog；独立文档只留持续复用的入口/契约，保留历史而不批量删除 |
| 运行成本 | 纯 FPS 与真实训练不同；无预算 eval/长跑容易把死循环当慢实验 | 正式训练基准具备步数和父进程超时，资源指标有单位；统一生产 eval 超时仍待重构 |

本轮没有引入全仓 formatter/linter 清洗或为了“好看”重排旧代码。冻结的是新的回归入口和实用协作规则，不是声称现有所有模块已符合新架构。根 AGENTS.md 要求主动本地提交，同时明确用户指令优先、不擅自 push、不把“嘴硬”变成争吵。

### 已实现的防线与小修

- 规范：根 AGENTS.md、唯一 TODO.md、worklog/2026-09-13.md，已独立提交；不批量清理原历史材料。
- 测试与 CI：正式 tests/、统一 pytest 配置、主要 CI 依赖、GitHub Actions、可复用开发指南、受父进程监督的训练/资源基准。
- 小修：F08 reset 随机流、F09 Python depth=0、F10 Impatience 与 exhaust identity fallback、F18 旧 checkpoint 额外字段兼容、F19 Ice Cream boundary。每一项附测试、日志与独立 commit。
- 训练：普通/分层 PPO 的更新、checkpoint/Adam 保存恢复、两次 CPU bitwise 对齐；原生 PUCT/strict Full Gumbel 的已完成战斗标签、非空 replay 和 combat 参数更新；基准超时退出码测试。
- 最终本机统一回归：**219 passed, 8 xfailed in 53.75s**。前一阶段为 216 passed / 8 xfailed，新增搜索 learner 与预算测试后的最终汇总以上述结果为准。

本地提交序列（尚未 push，故不生成指向不存在远端 commit 的链接）：

```text
43ae9cf docs: establish agent worklog and reviewable commit conventions
392aa75 test: add native contracts bounded training checks and resource benchmarks
1f35bfa fix: preserve seeded episode stream across automatic resets
1b1b3e9 fix: advance Python NOSL searches with unbounded depth
62a925c fix: correct Impatience draw condition and exhaust identity lookup
495d4e1 fix: retain Ice Cream energy in end-turn value boundaries
e870fe6 fix: share legacy observation projection across inference entrypoints
debad3f test: exercise native NOSL learner updates and enforce benchmark budgets
9d2c6e1 docs: record validation results performance limits and refactor handoff
```

已有测试仍纳入统一 pytest。严格预期失败必须对应具体断言与 TODO ID；依赖缺失、fixture 错误或异常类型变化会失败，修好后仍留 xfail 也会失败。剩余 8 个失败标记覆盖 F01/F02/F03/F04/F05/F07；它们是明确债务，并未从 CI 汇总中隐藏。F06、F11/F12 等尚缺精确的端到端 oracle，不应误认为没有红测试就已正确。

本轮保存恢复验收比较了权重、Adam 状态和预测，重复 CPU 训练还比较 rollout action/reward。这不等于恢复一个中断训练现场后继续训练与不中断运行逐位相同：后者还涉及环境、RNG、回放和 rank 状态，仍需专门设计。也不承诺跨 CPU/GPU、跨编译器、跨依赖版本逐位相同。

GitHub Actions 工作流已建立 CPU 依赖安装、显式原生构建、已有/新增契约、真实训练、预算基准和 JUnit/JSON artifact 上传。本机完成验证；由于没有 push，本报告不把尚未运行的托管 runner 写成绿色 CI。requirements/ci.txt 固定主要依赖版本，尚不是含全传递依赖哈希的锁文件。

交付分支保留原审计基线和逐项提交。全仓重构应从这条带测试的分支继续，以 TODO 中尚未解决的核心契约为下一批工作；不建议先跑更大实验来绕过红色反例。
