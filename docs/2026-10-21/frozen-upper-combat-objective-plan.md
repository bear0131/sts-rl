# 冻结最强上层并以外层价值重训战斗层的设计计划

用户已确定方向：**提取当前最强参数的上层并冻结，用它的外层价值定义战斗目标，重新训练下层。** 新会话首先完成实现设计，随后按独立预算推进实现和实验。本阶段上层策略、上层价值及其输入含义保持固定，不交替训练上下层。

本计划对应 [TODO 的 COMBAT01](../../TODO.md)。它是设计交接，不表示已经提取 teacher、实现新目标或启动训练。进度只追加 [工作日志](../../worklog/2026-10-21.md#frozen-upper-combat-retraining-plan-intention)，任务状态只维护 TODO。

本计划末尾五项交付的具体设计已写入[COMBAT01 实现设计](frozen-upper-combat-design.md)，区分已定合同、建议默认值和待核验接口。用户补充要求等待模拟器改动完成后一同并入；正式实现 commit 在整合后固定，EVAL31 仅作为 teacher 身份及对照依据。本轮尚未进入实现或训练。

## 已确定的范围

- 面向一代杀戮尖塔 TRAIN47 模型；首先覆盖既有 Ironclad、A0、第三幕通关目标与真实自然终局。
- 上层包括 **actor 和 critic**：actor 继续处理地图、奖励、火堆、商店等；critic 提供战斗退出后的价值。两者及其依赖全程冻结。
- 战斗层使用新的整局回报目标训练，不再以最终 HP 或 Boss 的人工胜负与 HP 加权作为决策效用。
- 保持原外层奖励定义与 `gamma=1`。保留 teacher 的 `upper lambda=1` 配置身份，不通过改 lambda 修复目标问题；冻结后不再执行上层 PPO。
- 本实验不同时实施 UPPER09 的上层观察/会话重构，也不通过强制拿金币、手写药水成本或资源奖励改变问题。
- 用户将自行新开 session。本文件不要求创建新聊天，不授权沿用已经结束的通宵租机计划。

## Teacher 候选与身份核验

当前有完整验收证据、应优先核验的候选是 **TRAIN47 第 876 轮，step 688914432**。EVAL31 在模拟器 A0、确定性上层、Full Gumbel 128、Gumbel 噪声 0 的条件下完成 200 局，186 胜、14 死，胜率 93%，Wilson 95% 区间约 88.59%—95.78%。这是完整模型与搜索组合的表现，不能单独证明该上层优于所有其他上层，也不是实机胜率。

| 身份 | 本地证据 |
|---|---|
| 归档目录 | `runs/local/eval31-train47-latest-deterministic-gumbel0-seeds200200-200399-20261018/` |
| 模型文件 | 上述目录内 `train47-step-000688914432-rank0-model.zip` |
| 模型 SHA256 | `693670e4405f73a2d41d1b92002d520331fb8720b2298e29b5e839b355369e3a` |
| 模型大小 | 115656088 bytes；本次写文档时重新核对 SHA 通过 |
| 参数与依赖身份 | `source-manifest.json`、`runtime-manifest.json`、`protocol.json` |
| 评测验收 | `acceptance-summary.json`、`result.json`、`comparison.json` |
| 对应实现 | 归档内 `source/` 与 `native/` |
| 实现来源 | baseline `e0b5491ef959bc8a3bf645f8c08d978a10faaa58`，overlay `a3c4204efce107f15d845d012258230174d4cfb5`；细节以 runtime manifest 为准 |

新会话先检查是否存在更强且口径可比的已验收候选；没有更强证据时，以以上 876 模型作为本计划的 teacher。第 881 轮是通宵噪声实验共同起点，更新、更大的 step 或固定火堆面板改善均不能替代完整评测，不自动升级为“最强”。EVAL30 的 91.5% 还涉及旧缓存问题，不能作为纯权重对照。

这些大文件位于忽略目录，不随 Git 文档传输。新会话须核对本地可用性及 SHA；不重新下载或复制整套 GB 级源码包来代替文件核验。

## 提取与冻结上层

冻结源码的 `rl/policies/entity_joint.py` 中，上层由 `core.upper_tokenizer` 和 `core.overworld` 组成，后者包含 actor、critic 及其编码器。`upper_parameters()` 是核对起点，不足以代替完整依赖审计。

一个实际耦合点是：启用冻结卡牌向量时，`upper_tokenizer` 构造时传入了 `self.combat.card`。提取包必须包含这份依赖及词表/观察/动作协议，不能留下指向未来可训练下层对象的引用。推荐独立 teacher 对象，必要的共享数据复制或作为严格只读依赖装载。

提取后应具备两个职责清晰的接口：合法动作下的上层 actor 前推，以及上层非终局状态的 value 前推。接口名称由实现阶段决定，不为这次实验复制整套 trainer。

必须满足以下冻结合同：

1. 所有 teacher 参数不求梯度、不进入任何下层 optimizer；禁用上层更新入口，而不是仅把学习率设为零。
2. teacher 始终处于推理模式，不能被父模型的 `train()` 切换影响；参数、buffer、归一化统计、依赖表的语义内容保持不变。
3. 对下层训练前后、保存恢复后，逐项验证 teacher tensor 内容哈希不变。测试相同输入时 actor logits、合法动作概率和 value 不变。
4. 提取前后的比较使用相同设备、dtype 和推理设置。FP32 对照与原 BF16 运行的差异另报，不伪称跨精度 bitwise。
5. 保存 teacher manifest：来源模型 SHA、提取 tensor 清单与 SHA、结构参数、词表与依赖 SHA、观察/动作版本、奖励定义、gamma、动作采样规则、推理精度。

teacher 的 value、target、reward 累加和回报计算采用 FP32。切换精度应固定为明确配置，并在原模型与提取模型的同精度测试中验证。不要把 `.float()` 转换后的 BF16 输出冒称为 FP32 前推。

## 战斗目标及结算边界

设 `t` 为战斗内某个真实状态，`tau` 为本次战斗退出到下一上层决策的边界。对于 `gamma=1`：

```text
Y_t = sum(global_reward[k], k=t..tau-1)
      + (0 if run_terminal else frozen_V_outer(observation_at_tau))

V_combat(x_t) learns E[Y_t | x_t]
```

这里的 `global_reward` 与 teacher 原先学习的外层奖励使用同一语义和时点。冻结实现的 `upper_ppo.py::strategic_reward` 包含每新楼层 0.002、第一/二幕通过奖励 0.1/0.2、真正整局胜利 1；设计时以固定源码和真实 transition 重新核对。不能新增一份独立的“战斗胜利 1”，也不能把已经在进入战斗前收到的奖励再加给战斗。

| 边界 | 目标处理 |
|---|---|
| 普通/精英战斗结束、游戏继续 | 累加该段实际整局奖励，再取真实自动结算后的上层 value |
| 中间 Boss | 使用相同定义；已经发生的回血、奖励、过幕由真实流程体现，尚未发生的部分留在后续 value 中 |
| 真正整局通关或死亡 | 按原整局奖励定义结算，后续 bootstrap 为 0；不重复终局奖励 |
| 多阶段敌人、Awakened One 复活等仍在同一战斗 | 不当作战斗或整局结束 |
| 连续 Boss 直接进入下一场战斗 | 不能调用上层网络的非战斗路由，也不能接旧 HP value；优先设计为连续战斗段直到合法上层出口，或明确同单位新 combat target 的接续方案 |
| collection 切断、步数上限、故障 | 显式区分 pending/truncated/error；不得标为自然死亡或伪造完整战斗标签 |

初版支持范围先固定为既有 A0；若连续 Boss 等未实现，入口应明确拒绝相应配置，不能静默用错误终点。跨 collection 的未结束战斗须持久化并续接；真正丢失续接能力的轨迹单列为 censored，不制造假标签。

退出状态必须包含真实 HP、药水、牌组、永久成长、遗物计数等已发生变化。奖励选卡等自由决策仍由冻结上层执行，不预先替它领取奖励。外层输入继续使用旧 teacher 的观察合同，不能悄悄补入地图等新字段改变其含义。下层是否需要补齐持久可见上下文，应在设计阶段审计；允许版本化改下层输入，不能泄漏隐藏未来信息。

## 搜索与学习使用同一个效用

冻结源码当前的 `EntityCombatCore.training_objective` 为 `combat-outcome-quantiles-v1`，value 路径涉及胜负与条件 HP 分布，并可通过 `boss_win_weight` 混合。这一身份与新整局回报不同，不能只给旧 HP 头塞入新标签。

初版建议采用预测整局回报的独立标量 value 头，输出不限制在 `[0,1]`；如保留分布式 value，必须改成同单位整局回报分布。旧 HP/胜负预测只能作为显式辅助任务，不能继续进入动作效用。

搜索内部未结束的战斗叶子使用新 combat value；搜索分支真正退出战斗时，使用该分支真实结算状态的冻结外层 value，并沿边累计同一整局奖励。实际执行轨迹由自己的真实退出结果产生训练标签，不能把实际选择路线的结果贴给未执行搜索分支。

原生搜索目前主要操作战斗状态。设计阶段必须回答：**怎样从一个搜索分支可靠取得对应的完整战后 GameContext 与上层观察？** 如果当前接口缺少药水消耗、永久牌组变化、遗物计数、结算 RNG 等，就补充明确状态契约和批量 teacher 推理通路；不能只修改 HP 后调用外层，也不能用另一条轨迹的奖励候选代替。

应沿以下路径完成目标语义审计，具体正式实现位置以新会话选定的受维护源码为准：

| 路径或组件 | 设计要解决的内容 |
|---|---|
| `env/core/game_env.py`、`run_env.py`、`boss_value.py` | 真实退出、奖励时点、终局与截断、完整结算状态 |
| `rl/algorithms/accumulating_ppo.py`、`combat_distribution.py` | 实际轨迹回填、新 value loss、旧目标退出 |
| `env/search/nosl_is_az.py`、`rl/search/native.py`、native bindings/search | 搜索终点与 learner 同单位，分支状态可重建，批量出口估值 |
| `rl/policies/entity_joint.py`、`entity_combat.py`、`split_policy.py` | teacher 提取、依赖隔离、新下层 value 与输入合同 |
| `rl/training/trainer.py`、`distributed_sb3.py`、checkpoint/replay | 上层完全冻结、新初始化、恢复/版本拒绝、实际生命周期 |
| evaluation、Watch、bridge 的对应消费者 | 同一动作/观察/目标身份；新旧模型显式区分 |

以上文件定位来自 EVAL31 冻结 `source/`，不代表当前仓库根目录都有相同实现。根目录还包含二代训练，不能把一代权重装进二代入口。新会话先选择可维护源码分支或完成必要兼容导入；忽略目录只作证据和运行材料，不作为正式修复唯一落点。

旧 `--nosl-is-az-inner-only` 与 `--nosl-is-az-reset-combat` 可作为复用入口的调查起点，但存在这些参数不等于已支持 entity-joint 的独立 teacher、新目标和所有恢复流程。须检查实际消费者后再给出可运行命令。

## 下层初始化和数据处理

用户已确定重新训练下层，但未指定“全部随机初始化”还是“从旧下层迁移”。新会话要将此选择写入实验身份，不能称 warm start 为从零训练。

建议首先设计低成本迁移路线：保留旧战斗 actor/编码器作初始化，重建新回报 value 头、清空下层 Adam 与 replay；用旧行为策略采到的真实战斗出口拟合新 critic，校准后再让新 critic 驱动搜索及下层联合更新。全过程 teacher 冻结。完全随机的下层可作为独立对照，是否运行取决于后续预算。

旧 HP 标签和旧目标搜索产生的 policy target 不直接混入新训练。历史真实轨迹只有在具有完整边界、reward 时点、动作/观察版本且能够正确重标注时才可使用；旧搜索策略标签不能仅通过换 value 数字变成新目标标签。

新 replay/checkpoint 至少记录 `objective_id`、teacher SHA、观察/动作/边界版本、下层初始化方式、目标精度、动作采样模式、pending 战斗、环境/RNG 与运行配置。恢复时不匹配应明确拒绝，不能退回 HP target 或自动加载旧 Adam。

上层训练时的 categorical 采样与 EVAL31 的 masked argmax 不同，关联 EVAL01。建议首个主对照在采样与评测都使用固定确定性上层，另把随机上层作为分布敏感性实验；设计时须明确 teacher value 原本对应的后续策略及这一差别，不能只写“权重一样”。下层搜索噪声、预算及探索日程也须显式固定。

## 分阶段交付与验收

| 阶段 | 可验收交付 |
|---|---|
| 设计 | 固定 teacher 身份与正式源码，列出参数/存储共享关系；确定边界、value 头、初始化、上层采样模式、接口和实验预算。先补具体合同测试设计，再实现 |
| 提取与冻结 | 独立 teacher 及 manifest；原模型/提取模型同输入同精度前推一致；下层连续更新与保存恢复后 teacher tensor 哈希不变 |
| 目标迁移 | reward 不漏不重、终局/截断/嵌套状态正确；真实出口、Python 与 native 搜索、learner 对同一分支给出同单位目标；旧 HP/Boss 效用不再参与新搜索 |
| 完整生命周期 | 新训练真实执行至少两轮采样、更新、聚合、指标和保存恢复；覆盖未结束战斗跨批续接。正式 GPU 配置至少完成一个生产长度 collection，检查图缓存、内存与 teacher 冻结 |
| 有预算试训 | 新 critic 校准与资源取舍检查通过后训练下层；记录完整目标/配置/资源数据，完成配对基线比较 |
| 晋级判断 | 同一冻结上层、搜索预算、动作模式和新留出种子下比较完整游戏质量；不能只按 teacher 分数或少量战斗 HP 判断优劣 |

关键测试放 `tests/unit` 与 `tests/integration`，覆盖保药与耗药、永久成长、战后遗物触发、中间 Boss 回血、最终通关、死亡、跨批 pending、多阶段敌人、选择子页面、指定 native 扩展和旧 checkpoint 的兼容加载。teacher 自身对某些资源的估值可能错误，因此语义测试验证状态与回报合同，学习测试用真实续打检查排序，不能把“teacher 必须认为所有资源增加都更好”写成无条件断言。

评测保留原完整模型、同冻结上层加旧下层，以及新下层；主比较固定搜索预算，同时另报关闭搜索的 actor 表现。EVAL31 的 seeds 200200—200399 可用于复现，不能反复调参后仍当全新留出。新验证种子、胜率差的统计方法及可接受退化阈值在训练前固定，样本量由预算和要检测的效应决定。

必须监控 teacher 来源与参数哈希、上层实际更新数为 0、上下层真实更新计数、退出类别、目标范围/NaN/截断、critic 校准误差、搜索 Q 尺度、药水/永久资源取舍、自然终局与步数截断，以及端到端时间、CPU/RSS、GPU/Torch 内存。不同分支要同时报告环境 transition、有效战斗样本和 optimizer update，不能只用一个 step 数表示全部工作量。

## 冻结 teacher 的局限与实验边界

冻结的外层 value 是旧后续策略下的近似价值。下层更新后会改变战后状态分布与未来战斗能力，固定 critic 不会自动变成新策略的真实整局 value；它可能在分布外状态失准，甚至被下层利用。**teacher 分数提高只是优化结果，完整游戏真实表现才是最终检验。** 本轮用户选择的是固定 teacher 实验，不能为缓解该问题擅自更新上层。

UPPER09/10/11 中已知的地图输入、回忆偏好和奖励选择问题会随 teacher 保留。本实验不能以重训下层宣称修复上层，也不能强制领取资源来掩盖偏差。严重估值错误应形成明确反例，影响是否继续放大训练，而不是静默换 teacher 或修改全局奖励。

每个训练/评测阶段必须先填写机器、独立种子、环境数、n_steps、更新数、microbatch、时长/费用上限、退出条件、结果目录和备份方案；暂不指定未核验的租机配置或沿用旧 SSH 信息。本机 GPU 可能被其他任务使用，开跑前核对实际占用。10 月 21 日 08:00 截止的通宵授权已经结束，四机已执行关机，本计划不恢复旧队列或 heartbeat。

本地大备份较多，训练前检查磁盘余量。只引用原始 teacher，提取件与新训练输出使用独立路径，保留原 876/881 快照与用户改动。

## 新会话的第一轮任务

先读 `AGENTS.md`、TODO 的 COMBAT01 与本计划，并核对工作区。完成以下设计输出后再进入实现：

1. 确认最强候选、模型 SHA、加载依赖与正式源码入口，明确“完整模型最强”和“上层独立最强”的证据范围。
2. 给出 teacher 提取边界、共享依赖隔离、冻结不变量及前推等价验证方法。
3. 确定战斗出口与 reward 时点、native 分支如何构造完整战后状态、learner/search/replay 的统一目标版本。
4. 明确下层 warm start 或随机初始化、value 头形式、上层动作模式与上下文支持范围。
5. 给出按模块的修改清单、合同测试、有限试训及独立评测预算。不要只给一个未经验证的启动命令。

用户可将以下文字作为新会话开场：

> 请读取 D:/sts-rl/AGENTS.md、TODO.md 的 COMBAT01，以及 D:/sts-rl/docs/2026-10-21/frozen-upper-combat-objective-plan.md。按照已确定的方向启动设计：提取当前最强参数的上层 actor 和 critic 并全程冻结，用同一外层回报定义重训下层。先完成文档末尾的五项设计交付，区分已定合同、建议默认值和待核验接口；不要把旧 HP 目标续训、解除上层冻结或恢复旧通宵任务当作实现。

## 原始设计依据

- [战斗目标与外层价值](../../worklog/2026-10-19.md#combat-objective-from-overworld-value)
- [终局与外层回报单位一致性](../../worklog/2026-10-19.md#combat-terminal-and-outer-return-consistency)
- [完整交互会话与出口价值](../../worklog/2026-10-20.md#upper-design-revision-results)
- [既有共享参数与梯度诊断](../../worklog/2026-10-21.md#upper-shared-parameter-full-batch-directions)

实施后的长期接口说明应归入所属模块 README，本计划保留为本次阶段设计与交接记录。
