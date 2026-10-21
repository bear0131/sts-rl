# COMBAT01：冻结上层与整局回报战斗层实现设计

本设计承接[原始计划](frozen-upper-combat-objective-plan.md)，交付其中五项设计结果。任务状态统一见 [TODO COMBAT01](../../TODO.md)，核查过程见[工作日志](../../worklog/2026-10-21.md#combat01-design-intention)。本轮只完成设计与只读加载核查，尚未提取 teacher、实现新目标或训练。

本文使用三种标记：**已定合同**来自用户确定的方向及其必要语义约束；**建议默认**是可审查的首版技术选择，不冒充用户已指定；**待核验接口**必须在实现阶段用实际代码和测试关闭。用户补充“模拟器改动还需要继续等候，会一同并入”：设计继续，正式实现基线等待模拟器交付并整合后固定。

## 1. Teacher 身份、依赖与正式源码入口

**已定合同**：提取当前最强且可核验候选的上层 actor、critic 及依赖并全程冻结。A0、Ironclad、第三幕自然通关，保留原上层奖励、`gamma=1`、`upper gae_lambda=1` 身份。上层实际 optimizer update 数必须为零。

本轮选择 TRAIN47/876、step 688914432 为 teacher 来源。核对了 TODO、最近日志及本地 EVAL30/31 归档，没有找到口径可比且更强的已验收候选。EVAL31 的 186/200（93%，Wilson 95% 88.59%—95.78%）仅证明其**完整模型＋Full Gumbel 128＋确定性上层＋噪声 0**在该模拟器、该种子集上的表现；没有固定同一下层后的上层独立排名，也没有实机 93% 证据。881 和噪声分叉不自动替换 teacher。

| 已重新核验的对象 | 固定身份 |
|---|---|
| 模型 | `runs/local/eval31-train47-latest-deterministic-gumbel0-seeds200200-200399-20261018/train47-step-000688914432-rank0-model.zip`，115656088 bytes |
| 模型 SHA256 | `693670e4405f73a2d41d1b92002d520331fb8720b2298e29b5e839b355369e3a` |
| 卡牌依赖 | `runs/local/real-game-train47-20261011/old-wholerun-v7-step10000-card-vectors.pt` |
| 卡牌依赖 SHA256 | `5b1689cd201562a99a312171edf97a6e0ee5538a4d89924b5bed0a44daf5d4da` |
| 参考 native 扩展 | EVAL31 `native/slaythespire.cp311-win_amd64.pyd`，SHA256 `b9eb3613b4e51c0e30faf9cf6f31f6f7fb7f887f423e4b2045f96897a9b64a3d` |
| 参考源码 | baseline `e0b5491ef959bc8a3bf645f8c08d978a10faaa58`；overlay `a3c4204efce107f15d845d012258230174d4cfb5` |
| overlay 文件 | `bridge/schemas/combat_state.py`、`env/core/spiremap.py`、`tests/unit/test_combat_state_names.py` |
| 本轮加载 | `.venv-eval-gpu/Scripts/python.exe`，Python 3.11.15、Torch 2.11.0+cu128，CPU、2线程、`inference_only=True`、显式卡牌路径与 native 构建 |

25 个关键源码文件的字节 SHA 与 runtime manifest 一致；与上述 Git 对象在换行归一化后相等（归档多数为 CRLF，不声称全部 Git 字节相等）。实际 checkpoint 加载成功。原训练保存身份为 Linux/Python 3.11.15/Torch 2.12.0+cu132；本轮加载不是跨平台前推等价验收。详细清单在忽略目录 `runs/local/combat01-design-20261021/audit.json`，不提交模型、依赖或源码包。

**正式源码落点与整合合同**：后续修改进入受版本控制的一代 canonical 包 `env/core`、`env/search`、`rl/policies`、`rl/algorithms`、`rl/training`、`rl/evaluation` 和 `sim/sts_lightspeed`，复用 `python -m rl.training.trainer` / `python -m rl.training.distributed_sb3` 及 `python -m rl.evaluation.eval`。不使用 `rl.training.sts2`，不把 EVAL31 忽略目录作为唯一实现落点。

核查时根分支为 `codex/sts2-training@931086eaf41c1dac35bc76e3a4c1b342fafad3b5`；一代工作树为 `codex/train32-blue-search-simfix-20261003@8698cc81217d58922644da30a4586e518220a052`。后者相对参考 baseline 的 rl/env/sim 已有 70 文件变化，不能当作 EVAL31 等价版本。**按用户补充，等待模拟器改动完成后一同并入，再固定正式整合 commit、构建 SHA、依赖锁和运行入口；本轮不回退模拟器，也不从旧 baseline 另行启动实现。** 上述固定参考只用于 teacher 身份、提取对照及差异审查。

**待核验接口 S1**：整合后逐项核对旧观察/动作编码、`exitBattle`、结算回调、恢复 journal、搜索随机流。模拟器有意修复的行为差异需记录具体例子与验收，不能要求修复后复现旧错误；teacher 同输入前推仍应一致。所有主比较必须在同一整合后模拟器重跑旧下层基线，EVAL31 的 93% 仅作历史参考。正式 commit 尚待该依赖交付，不虚填完成。

## 2. 提取边界、存储隔离与冻结不变量

本轮实测加载对象，非仅按类名推断：

| 组件 | 实测内容 | 提取处理 |
|---|---|---|
| `core.upper_tokenizer` | 8 个参数 tensor、61000 参数；1 个 `card.vectors` buffer | 完整复制，保留 tokenizer/projection 语义 |
| `core.overworld` | 123 个参数 tensor、4844665 参数；含 encoder、fixed/pointer actor、value 分支 | 完整复制，actor/critic 不拆掉共享编码器 |
| `core.combat.card` ↔ `upper_tokenizer.card` | 同一个 `FrozenCardVectorTable` 对象；唯一跨层共享存储是 `[371,2,128]` 的 `card.vectors` buffer | teacher 拥有独立物理副本；下层另持副本，禁止跨对象 storage alias |
| `upper_tokenizer.card_vector_cache` | 本 checkpoint 加载时为 `None`，非持久 buffer | 不能宣称已有缓存；将来物化时也纳入内容哈希与恢复检查 |
| codec、词表、观察适配 | 不是 optimizer 参数，但决定输入含义 | 清单中记录文件/版本 SHA、字段 shape/dtype、卡牌行序、动作映射、phase 路由 |

teacher 共 4905665 参数；卡表 94976 个 FP32 元素，379904 bytes。当前共享对象没有可训练参数，故仅检查 `upper_parameters()` / optimizer 参数集合会遗漏真正共享点。完整参数名、shape、dtype、内容 SHA 与存储关系已留存在审计结果中。

**建议默认**：独立 `FrozenUpperTeacher`（拟议接口，放 `rl/policies/entity_joint.py`），提供 `actor_logits(upper_obs, legal_mask)` 和 `outer_value(upper_obs)`；value 的接口不能以 mask 缺失为由走 combat 路由。拒绝 combat、terminal、缺字段及版本不符的输入。父策略只负责按 phase 调用，不把 teacher 注册进可训练 combat optimizer。加载来源模型只供转换，不继续其 learner/Adam。

**已定合同**：

1. teacher 所有参数 `requires_grad=False`，无 grad，和所有 optimizer 参数、下层参数/buffer 的 storage 集合都不相交；不创建上层 optimizer、不收集上层训练样本、不调用上层更新。
2. `.train()`、父 `.to()`、AMP、编译包装、同步广播和恢复均不得静默改变 teacher 的模式、精度和 tensor。teacher 保持 eval；value 真正 FP32 前推（退出 autocast），累加和标签 FP32。建议 actor 也固定 FP32，与 EVAL31 上层精度一致。
3. 冻结检查遍历 parameters、persistent/nonpersistent buffers、派生表、配置与依赖；哈希按稳定名称＋dtype＋shape＋连续 tensor 字节，不以 zip 文件哈希代替 tensor 哈希。合法的设备搬运保持同 dtype 数值；重建缓存也必须内容一致。
4. manifest 含源模型/提取件/tensor/依赖 SHA、结构、输入/动作/奖励/边界版本、上层动作模式与 tie-break、精度、整合源码/native 构建及来源训练身份。`lambda=1` 留作来源身份，不意味着继续计算上层 PPO。

**静态缺口已确认**：`upper_ppo.enabled()` 目前仅看 `joint_upper`；`AccumulatingMaskablePPO.train()` 和 distributed 更新循环即使 `inner_only=True` 仍调用 `update_upper`。旧 `--nosl-is-az-inner-only` 不是冻结开关；现有 `EntityJointPolicy._build()` 还会创建两个 optimizer。新模式须有明确目标配置与冻结路由，不能只设上层 LR=0。

**待核验接口 S2 / 前推验收**：覆盖 map/reward/shop/rest/event/card-select/boss-relic/单合法动作和不同候选数，用相同可达观察同时前推原模型和提取对象；比较完整 logits、mask、合法概率、argmax 与 V。固定同设备、dtype、线程/确定性设置。建议同运算图 CPU FP32 要求 tensor 精确相等；若拆批改变浮点归约，须先定位原因，另批准容差合同（建议上限 atol=1e-6、rtol=1e-5），不静默放宽或忽略近 tie 的动作翻转。再做下层真实更新前后、两轮上报、保存/恢复、父 `.train()` 后的 hash/前推不变测试。旧 BF16 生产差异单独报告。**本轮只完成原对象加载与共享审计，未做提取前推实验。**

## 3. 真实出口、reward 时点与统一目标

### 3.1 一条奖励账本

**已定合同**（`gamma=1`，状态 t 在当前动作之前）：

```text
r_k = strategic_reward(before_act, before_floor, after_run_stats,
                       done=natural_run_terminal, censored=False)
Y_t = sum(r_k for k in [t, tau)) + B_tau
B_tau = 0                         if natural_run_terminal
      = frozen_V_outer(obs_tau)    if settled_upper_decision
```

自然终局包括死亡与第三幕通关；collection 切断不属于终局。`r_k` 的唯一来源是原 `upper_ppo.strategic_reward` 语义：新楼层差×0.002、跨第1/2幕分别0.1/0.2、自然整局胜利1。死亡没有另加 `-1`。原 `SimGameEnv.step*` 返回的终局 ±1 以及训练 reward wrapper/HP shaping 不是这个账本，不能相加。建议将纯奖励函数下移到 `env/core` 的共用库，旧 `upper_ppo` 委托同一函数；先做旧新函数等价测试。

实际加载还确认旧模型顶层 `gamma=0.999`，而 `upper_config.gamma=1`、`gae_lambda=1`。新模式必须显式设定整条战斗回报/搜索边折扣为1，不能继承旧顶层gamma；旧上层lambda只保留身份，不参与更新。

固定 before/after 的真实 run_stats 与原缩放 observation 解码进行一致性检查，之后用整数元数据计算。进入战斗的上层动作已经得到的楼层奖励归该动作；战斗从进入后的状态开始，不重复计入。autoregressive 选择前缀、未确认多选等未推进 simulator 的 token 边奖励为零；同一真实动作/自动结算只记一次。搜索回传每条边 reward＋叶子 continuation，不能把整条累计奖励反复加到每层。

| 真实边界 | 接口应返回的内容与目标 |
|---|---|
| 普通/精英胜利 | `bc.exit_battle(gc)` 完成持久变化和自动触发，停在首个上层决策（通常 combat_reward）；不替上层领奖。常见战内 r=0，因此 Y=V(reward screen) |
| Act1/2 Boss 胜利 | 当前 `afterBattle` 先开 Boss reward，再由后续选择进入 Boss treasure/transitionToAct；尚未发生的过幕奖励和回血留给 V，不能在击杀时预支0.1/0.2或满血 |
| Act3 最终胜利 | 当前 `enterAct3VictoryRoom` 增一层并设 victory 或进act4；以真实 after stats 为准。典型该原子动作 r=1.002、B=0，不能只写1或再加Boss胜利1 |
| 自然死亡 | 同账本实际增量（通常0）、B=0；无需查询上层；不得接 vec-env 自动 reset 后的新局 observation |
| 复活/敌人阶段切换 | 仍是 ongoing combat，无出口标签；Awakened One 中间阶段不是死亡或通关 |
| 战斗中的 pending/选择子页面 | 依据拥有该选择的 BattleContext/selection owner，不仅依据屏幕名称；保持战斗样本来源、候选区域、prefix 与实体ID |
| 直接进入下一场战斗 | 属连续战斗段，下一叶仍用同单位新 combat V。首版仅A0；A20/Act4入口拒绝。A0事件若出现直接连战也须续段；无法正确续接时明确失败，不用旧HP或伪上层值 |
| collection 切断 | pending 持久化：episode/fight/transition ID、每样本奖励前缀、动作/策略标签版本、RNG/环境续接；结束后才回填 |
| 环境步数上限/dead screen/故障 | truncated/error/censored 单列；保留可恢复 pending，否则不进入完整出口监督。不能用最后HP、死亡0或reset后的V制造完整标签 |

learner 对实际执行轨迹记录每个 t 的累计奖励前缀 `R_t`，出口后用 `R_tau-R_t+B_tau` 回填。若战内奖励不同，不能整场所有根直接贴同一个数。搜索深度/预算截断用新 combat V 估叶，不等于真实轨迹完整标签。新模式不追加旧 pre-settlement combat-terminal HP 样本，也不经过旧 `outcome_targets`/`_fill_nosl_combat_value_targets` 的 HP 路径。

**另一处已确认的迁移缺口**：旧回填函数把target clip到`[0,1]`；native `NoslNeuralSearchSession` 的root value、backprop/批量backprop、propagated value、sample/network fallback等多处也clamp到`[0,1]`，非有限值还可能退回HP或0。新目标须按objective显式选择不裁剪的有限标量路径，非有限值报错；所有聚合与completed-Q消费者一起覆盖，不能只修改terminal callback。softmax数值稳定clamp和概率范围约束保留，它们不是回报裁剪。

### 3.2 完整分支结算协议

**已确认的现状**：Python `_CombatBranchGameEnv` 浅拷贝其他属性、仅 `bc.clone()`，共享 `gc` 并故意不 `exit_battle`；native `NoslNeuralSearchSession` 的叶请求仅携 BattleContext/动作。参考扩展无 `GameContext.clone`。`projected_exit_hp` 仅用于投影HP，不满足目标。实际 `BattleContext::exitBattle` 回写药水、多个 RNG、HP/maxHP/gold、遗物计数、永久牌组及偷金，再触发 `regainControl`；遗漏任一项都可能改变出口观察/后续收益。

**建议默认的实现协议**（名称待实现；不是现有 API）：

1. 每个搜索根建立独立 `RunBranchRoot`：root/episode ID、完整 GameContext 及其结算 continuation、当前 BattleContext、thin/prefix/多选状态、地图可见性、奖励账本起点、搜索世界线身份。存于 worker/native，不向神经网络暴露隐藏字段。
2. 普通 combat 叶沿用合批路径，携带下层上下文版本。分支结束时产生类型化 `ExitRequest(root_id, request_id, terminal_bc, worldline_id)`，交给该根的完整游戏快照执行与真实环境共用的结算函数一次；返回 natural-terminal / upper-decision / next-combat，reward_delta、公开 observation、mask/版本和审计 digest。
3. upper-decision 叶进入独立 FP32 teacher 批次；natural-terminal 直接 continuation=0；next-combat 回新 combat 叶。响应按 root/request ID 回传，允许两类叶交错、空批、不同尺寸；cancel/error 必须释放树和请求，不填0应付。
4. Python 搜索与 native 绑定复用同一结算 primitive/编码器，不能另写药水、回血、永久升级规则。cache key 必须含完整根上下文、teacher/目标/观察版本及世界线，不能仅以战斗HP或旧 state key 命中。
5. 搜索使用其自己的 sampled worldline；BattleContext 回写的 RNG 和未被回写的 reward/事件 RNG 都要有明确来源。既有 NOSL 对战内隐藏未来重采样不自动覆盖结算 RNG。建议每根从公开信息和 nonce 生成一致的结算随机流，simulation 间按 fixed_worldline 合同复用；真实执行仍用真实环境 RNG。禁止用后来实际抽出的奖励候选给未执行分支估值；分支不强求与另一条实际随机路线相等。

**待核验接口 S3（最关键的实现门槛）**：整合后 native 是否提供完整可复制/可恢复状态、如何正确重绑 continuation。参考 `GameContext` 有 `shared_ptr<Map>` 和 `std::function regainControlAction`，并存在捕获实例成员的闭包；C++ 默认复制或 Python deepcopy 不是隔离证明。需要显式分支 clone：可变 map 深拷贝或验证只读；闭包改成可重建的 continuation 描述或逐类安全重绑；保存所有 deck instance、永久增量、RNG 和事件来源。若采用 journal 重建，只能先用相同版本/seed重建根，再施加该搜索分支的世界线，不能重放 live RNG 冒充采样分支。整合后的 API 若已具备这些能力则复用，不再造平行接口。

测试须让同一根、同一显式世界线、同一动作序列经真实环境副本/Python/native 到达相同完整状态、候选、reward、mask、teacher V，同时检查根及兄弟分支不变。普通/精英/Boss/事件结算、药水消耗与获得、Feed/Ritual Dagger/永久升级、战后遗物触发、逃跑偷金、复活/连战全部列入。分支能力未通过时禁止开启新目标 native 搜索；不能退回HP。

### 3.3 版本与恢复

**建议默认版本组合**：`objective_id=combat-outer-return-v1`、`reward_id=train47-strategic-return-v1`、`boundary_id=settled-run-exit-v1`、`lower_obs_id=combat-public-run-context-v1`、`teacher_manifest_version=1`；动作仍为原 `autoregressive-v2`，但同时记录 codec 文件 SHA。这些是待落地的新身份，不假称当前 loader 已识别。

replay 每条记录包括上述身份、teacher tensor SHA、policy/search 版本、初始化、上层采样模式、precision、episode/segment/transition、reward前缀/出口类型、target来源（真实出口或搜索叶）与有效掩码。旧HP标签、旧HP搜索 policy target、旧 Adam 和未知身份一律拒绝混用；不是仅给旧 quantile 头换标签。历史轨迹暂不导入，除非另有完整原始状态/边界和重标注审计。

建议 recovery bundle 新 schema 3、replay 新 schema 2（实现前检查整合分支是否已占用，若占用分配新号），保存独立 teacher、下层 optimizer/replay、各rank pending 和 reward 前缀、环境 journal/RNG、采样器与搜索 nonce、模式/精度、完整配置。manifest 最后原子发布；恢复先验身份再分配模型/optimizer。旧 bundle 仅允许显式“导入初始化”通道，禁止自动 resume 到新模式。参考源码已有 schema2 环境 journal 恢复；根目录 README 的“环境暂不序列化”不能代替当前实现核查。

## 4. 下层初始化、value 与可见上下文

**建议默认主路线**：`init=warm-start-876-actor-encoder`。复制旧下层 encoder、actor、卡牌表及必要状态，删除其效用分布头，新建无 sigmoid/clip 的 FP32 标量回报头；MSE 学条件均值，初版不保留HP/胜负辅助 loss。清空 lower Adam、replay、步数/探索日程；旧模型只作为来源和基线。这是新目标重训的 warm start，不称从零训练。随机初始化是独立后续对照，不列为首轮必跑。

新 value 不限制 `[0,1]`，原整局 shaping 可令剩余回报超过1。不要继续乘 `boss_win_weight=0.9` 或使用HP量化/胜率混合。旧 `EntityCombatNetwork.scalar_keys` 会因 boss weight 是否非零改变输入维度：warm start 须把 `combat_is_boss` 保留为独立的公开输入特征，和新效用的无Boss人工权重分离，按参数 shape 明确迁移，不能将配置直接改0后盲目 `strict=False` 丢权重。

校准阶段以冻结旧下层 actor 的 masked argmax 采集真实完整战斗出口，仅训练新 critic/value-context 分支；不采用旧HP搜索标签。actor/旧encoder 前推应不变。新 critic 在独立校准分割上通过有限值、尺度、残差和资源反例检查后，再启用新目标搜索并联合更新下层。校准数据的状态分布是旧actor分布，搜索开启后必须另报分布变化，不能以低MSE当胜率证明。

**上层动作建议默认**：训练采样、验证、最终评测均使用同一个 teacher 的 masked argmax、固定 tie-break。来源 critic 在原 categorical 上层、旧下层后续行为下训练；固定 argmax 与新下层都改变 continuation 分布。此差异写入实验身份，不能借校准上层或更新 teacher 来消除。随机上层为另立预算的敏感性实验。

**下层输入建议默认**：保留既有 combat 实体/槽位/候选区域/动作codec，新增公开 run context：act、floor、已知 next_boss、当前/最大HP、永久牌组ID/升级/永久成长（按已公开事实）、角色/ascension；已有gold、药水、遗物/计数直接复用。目前 combat encoder 不消费 act/floor/永久deck/next_boss，即便统一 Gym Dict 有这些字段也不等于网络看见。新上下文编码放下层独立分支：critic 接旧 combat 表示＋context；actor 新增零输出初始化的 context residual，初始 logits 对齐旧actor。校准时冻结旧encoder/actor和actor residual，只更新新critic/context；联合训练再解冻下层。teacher 输入完全沿旧 schema 投影，不注入新字段。

v1 不增加路线记忆或泄漏地图未来、抽牌顺序、隐藏随机种子、未生成奖励；不改 UPPER09 的地图输入。其剩余回报是对当前可见输入的条件期望，不宣称完整 Markov。若同输入状态因缺少已可见信息而出现系统性不可区分，记录反例，后续以新 lower_obs 版本扩展，不能静默改变本轮输入。

完整结算状态也不等于冻结teacher能感知其中每个变量：当前upper字段白名单有`deck_ids/deck_upgrades`，并不证明Ritual Dagger等卡牌的永久数值被编码。需要将原始状态变化、teacher输入变化、teacher V变化分开核验。若某成长被旧schema丢弃，记录teacher盲区及真实续打影响，不能补字段进teacher冒称仍是同一冻结模型；严重盲区可阻止扩大试训，不通过人工成长bonus掩盖。

**待核验接口 S4**：compact/native 叶编码目前大量复用根 global template；必须逐字段标明不可变上下文与分支可变部分。药水、maxHP、gold、遗物计数和已发生永久成长不能只从根复制；临时生成牌不能当永久牌。需要实际战斗状态到永久deck实例的映射验证；若整合接口暂时不能暴露永久成长，先关闭该训练模式，不把字段填0当支持。Watch/bridge 也需完整可见上下文与新目标版本；bridge 缺少完整分支结算能力时可支持已验证的actor推理，新目标native搜索应明确拒绝。

## 5. 模块改动、合同测试与有限预算

### 5.1 可审查的实现顺序

下表是实施顺序/验收边界，任务进度仍只维护 TODO COMBAT01。每阶段独立提交代码、测试和日志，并普通push；不直接启动训练。

| 阶段与模块 | 最小改动及验收 |
|---|---|
| 整合与基线：`sim/sts_lightspeed`、`env/core`、依赖锁 | 等模拟器改动一起并入，锁正式commit/构建；差异清单、旧模型加载/一步预测/环境接入、明确支持A0与可拒绝配置；不覆盖用户正在进行的修复 |
| 合同测试先行：`tests/unit`、`tests/integration`、共享fixture | 先表达奖励时点、冻结与branch隔离预期；未实现测试不冒称通过，按阶段一起交付实现；不在运行时代码旁放test |
| 提取：`rl/policies/entity_joint.py`、`split_policy.py`、`checkpoint.py`、`tools/checkpoints` | 独立teacher/manifest与转换；复用已有checkpoint工具入口，只有无法承载时才增CLI；保留旧checkpoint读取；拒绝模式/依赖不匹配 |
| 边界：`env/core/game_env.py`、`run_env.py`、共用reward/outcome，native bindings/GameContext/BattleContext | 真实动作和搜索共用结算、完整分支、类型化边界、选择归属；`boss_value.py`只服务旧目标分派 |
| 下层：`rl/policies/entity_combat.py`、`rl/algorithms/accumulating_ppo.py`、`combat_distribution.py` | 新标量头/context、目标配置、实际轨迹回填；旧分布实现保留给旧模型，新模式无HP效用路径；复用`algorithms/updates.py` |
| 搜索：`env/search/nosl_is_az.py`、`rl/search/native.py`、native `NoslCombatTrainer`/bindings、`cached_mask_vec_env.py` | typed叶、完整根状态、异步teacher合批、同单位回传、上下文编码与缓存隔离；不复制trainer |
| 生命周期：`rl/training/trainer.py`、`distributed_sb3.py`、`checkpoints.py`、replay | 单机/分布式都禁上层更新；版本校验、pending恢复、统计payload、原子保存；新目标配置对象集中校验，不散布布尔开关 |
| 消费者：`rl/evaluation/strategies.py`、runner、Watch、bridge、模块README/configs | `DecisionStrategy`/`PhaseStrategy`组合冻结上层和新下层；runner无模型名分支；报告目标/teacher/构建身份；参数放configs，机器/run身份留launchers |

### 5.2 合同测试矩阵

| 合同 | 必须验证的预期行为 |
|---|---|
| 提取与冻结 | 同输入完整actor/V等价；存储无alias；无上层optimizer/grad/update；下层更新、父train、恢复后参数/buffer/表哈希不变；多rank一致 |
| 单一reward | 入场楼层奖励不重复；普通胜利不新增1；Boss奖励/回血按实际时点；Act3典型1.002；死亡通常0；正负/大于1的人工测试值不会被value/search截断 |
| 边界与选择 | 自然死亡/通关/步数截断/错误/collection切分分别断言；选择来源、空怪槽、multi prefix、Awakened复活、A0事件连战；不读取autoreset新局 |
| 分支结算 | 同显式世界线同动作的真实副本/Python/native完整出口与reward一致；Feed、药水、永久牌、遗物回调、事件/奖励候选、RNG；根和兄弟不变；拒绝未知continuation |
| 搜索与learner | 解析小树中`edge r + leaf V`与手算一致；不同出口请求正确匹配；混合/空批/取消；同真实执行路径的标签相等，未执行分支不获得实际轨迹标签；旧HP/Boss参数不能进入新模式 |
| pending/replay/恢复 | 跨至少两次collection后回填，save/reload后reward前缀不漏不重；不同teacher/目标/动作/观察/构建拒绝；旧replay与Adam拒绝；同版本恢复续接 |
| 兼容消费者 | 保留的876 entity-joint及其他现役checkpoint家族清单逐项加载/一步预测/环境接入；新模型evaluation/Watch/bridge能力矩阵，缺上下文显式失败；不误入二代 |
| 真实生命周期 | 真实PPO与NOSL各至少两轮采样→更新→聚合→真实payload→checkpoint，外部W&B可用collector；NOSL新目标/冻结teacher、恢复后再一轮，覆盖pending；PPO回归不启用冻结上层训练 |
| GPU生产长度 | 实际新目标配置完整512步collection，shape变化/缓存淘汰/临时存储释放后重放、真实同步与保存；检查teacher hash及内存，无CPU结果替代GPU验收 |

明确加载扩展绝对路径与SHA；缺扩展/依赖是未执行，不算测试通过。语义测试用可手算reward/可控假teacher验证代数，真实teacher测试只验证冻结和真实状态；资源增加不必然令这个teacher V增加，排序质量用真实续打另测。合同错误、teacher变化、身份不符、NaN或错误终局任何一项均停止后续阶段。

### 5.3 建议预算与停止条件（尚未执行）

机器建议为本机CPU和已存在的RTX 4060 Laptop，独占使用条件须开跑前核对。所有阶段新增租机费用上限为0；若本机不能满足则停在该阶段另立资源预算，不恢复旧四机/SSH/heartbeat。记录实际硬件、线程、依赖、batch和峰值；下面是上限，命中任一上限即保存并退出，不自动加量。

| 阶段 | 建议固定工作量、种子与上限 |
|---|---|
| CPU合同/提取 | `seed=42101`，2线程，最多30分钟，0环境训练步；真实fixture只执行有界回放；输出≤1GiB |
| 真实生命周期 | CPU或单GPU、8env×128步×2轮，每算法独立；恢复后再1轮，NOSL128可先以8 simulations作链路检查但不作质量评测；seed从9000000开始；每算法20分钟，learner每轮最多2个实际optimizer update，microbatch32 |
| critic校准 | 上层和旧下层actor均确定性，32env，最多128整局/250000真实transition/90分钟；seed9010000起，按episode固定3:1拟合/校准分割；新critic最多200次Adam、batch256、microbatch128、建议LR1e-4；未取得足够各类自然出口则不晋级 |
| GPU完整collection | 首版生产配置建议单GPU、32env、n_steps512、16384个vector槽位step；128 simulations、strict Full Gumbel、depth0、fixed_worldline；seed9001000起；最多60分钟、1个完整collection及2次更新/保存。它验证此新配置，不能声称覆盖旧四rank×384env规模 |
| 有限联合试训 | 同上32×512，最多4轮/65536槽位step/30分钟；每轮最多10次Adam，batch1024、microbatch128、建议LR1e-4；seed9002000起；lower探索root Gumbel=1，执行搜索选择动作，噪声/探索日程固定，非上层采样变化 |
| 开发诊断 | seed9020000—9020031，旧/新下层各32局，上层固定；每臂30分钟、单局20000动作上限；用于校准与选择一个候选，不作留出结论 |
| 最终主评测 | 封存seed9100000—9100199，固定一个候选；两臂各200整局、32env、128 simulations、Gumbel0/确定性上层、同整合源码/精度/搜索配置；每臂120分钟、单局20000动作；任何未完成自然终局使质量结论待定 |
| 无搜索对照 | 同封存集前100 seed，两臂actor masked argmax、每臂30分钟；单列，不与128搜索结果合并 |

校准先建议 eager FP32；正式比较也先固定同精度。若性能预算要求lower BF16/cuda-graph，须先完成同精度对照及GPU生命周期验收、作为统一配置固定后才跑两臂，teacher/V/reward/target仍为FP32。新回报Q与旧HP数值意义不同，Gumbel value scale建议先保留1、rescale_q=false并记录Q范围/根策略熵；需要调参只能用开发集，冻结配置后才触碰留出集。

首个联合试训建议只均匀使用本轮新完成的segment标签（包括早先pending在本轮完成的记录），不重采以前已完成批次，不混入校准集/旧搜索policy target；实际每轮样本数和policy版本另报。新replay/恢复容器仍保存版本与pending；后续引入旧样本比例是单独的统计方案变更。校准数据按episode分组保存且不超过250000条，输出达到容量上限立即停止，不靠丢pending继续跑。

校准建议门槛：目标/预测有限率100%，保留集整体MSE优于仅训练集均值预测；按act、普通/精英/Boss、胜败/用药/永久成长报告残差和覆盖，稀有类别无样本写明而非填0；严重可复现资源估值反例必须复核真实续打。门槛只允许进入小预算试训，不承诺最终胜率。不得为通过门槛训练teacher。

最终比较三种身份：原876完整模型为参考A；提取teacher＋旧下层为B，先用旧EVAL31种子的一小组（建议8局、20分钟上限）验证组合等价；提取teacher＋新下层为C。主配对为B/C，确保同冻结teacher、上层模式、模拟器和预算。来源完整模型的历史200局不是新B/C分母。若A/B等价失败先修提取/路由，不用其差异解释新目标效果。

统计建议预注册：每seed一对，胜率差 `mean(win_C-win_B)`，报告两臂Wilson区间、配对重采样10000次的95%区间、双方不一致胜负计数和精确McNemar检验；同样报告真实整局奖励差、自然死亡楼层、药水/永久资源取舍、步数和耗时。无不一致对或极少不一致对时bootstrap可能退化，不据零宽区间宣称非劣。初始可接受退化阈值建议3个百分点；晋级须自然终局完整、胜率点差≥0且可靠配对区间下界>−3pp，真实回报无明显退化；“改善”另需差值区间下界>0。200对只是预算上限，不是检测3pp的功效保证；结果不确定即保留B，不反复加种子试到显著。若分析方法需调整，在查看留出结果前版本化。

这些种子是拟议分区；正式启动前与历史run manifest完整去重并封存清单，不能因数值较大就宣称从未使用。拟议输出根 `runs/local/combat01-{phase}-{run_id}/`；每次启动保存配置/seed/依赖/teacher manifest、退出码和超时原因，原子checkpoint保留2代并校验备份。禁止覆盖876/881及既有产物。

**当前实际资源缺口**：本轮D盘剩余7626932224 bytes（约7.10GiB），没有满足建议的至少20GiB且足够“新输出＋两代完整bundle＋恢复临时副本”的空间门槛。这里只记录，不删除用户备份。模拟器整合commit、磁盘容量、GPU可用性、native完整分支接口均未验收，因此本设计不提供可直接执行的试训命令。

监控同时记录环境transition、codec token/完整原子动作、搜索simulation/leaf/出口teacher请求、有效combat样本、optimizer update；上层更新数单列恒0。计时覆盖warmup、采样、编码/IPC、teacher与lower前推、反向/同步、聚合/保存及总时长；GPU计时前后同步。每2秒采集进程树CPU时间/利用率、RSS（共享页可能重复）、系统可用内存、Torch allocated/reserved/peak、设备VRAM/util；取不到写null及原因。teacher分数上升不替代真实整局结果，UPPER09/10/11不会因本实验自动关闭。
