# 仓库总 TODO

本文件只维护持久问题与验收；工作进度见 worklog/。F 编号的完整问题、证据与验收见 [codebase-audit.md](docs/2026-09-13/codebase-audit.md)；每行编号直接链接相应问题。报告是固定快照，修复状态以本表为准。

| ID | 状态 | 范围 / 验收 |
|---|---|---|
| BASE01 | 初版已验收 | 固定上层，仅替换 combat 的可解释规则策略；DecisionStrategy/PhaseStrategy/runner、独立下层 checkpoint、目标/选牌区域与 trace 已接入；留出8整局规则/policy/native分别0/0/1胜，24同初态战斗分别23/19/24胜。方法与局限见 [基线协议](rl/evaluation/baselines/README.md)，结果见 [实验记录](worklog/2026-09-18.md#base01-held-out-evaluation) |
| BASE02 | 本轮确认完成；v3保留，Artifact候选独立保存 | `rl/evaluation/baselines/combat_rules.py`：384局规则/native128/native32为24/22/8胜；Artifact独立1024确认固定上层48→52胜、规则上层79→88胜，但固定上层双方同seed各1次F03 dead_screen，未过全自然晋级门槛且胜率优势尚不稳定。hand-finish跨上层结果不一致，其他开发候选未晋级。下一步先解决F03选牌覆盖，再验证安全斩杀前回血；新策略须冻结后至少128新种子、按上层分别验证，不覆盖已有未决样本。见 [本轮结果](worklog/2026-09-18.md#resumed-campaign-results-and-final-artifact-validation) 与 [确认及诊断](worklog/2026-09-18.md#artifact-confirmation-result-and-f03-draw-selection-reproduction) |
| BASE03 | 上层v1保留，独立候选未晋级 | `rl/evaluation/baselines/overworld_rules.py`，独立 `--upper-strategy rules`；v2成长权重在旧验证退步；重复伤害牌获取候选b5fcbf4在128开发集胜场/均楼层均与v1相同（9胜/33.3047），不晋级。未知事件显式回退计数；与固定上层分开评估，新改动须冻结后新种子验证 |
| EVAL01 | 待实验 | 当前 inner-only NOSL rollout 的冻结上层按 masked categorical 随机采样，正式 eval 默认按 masked argmax；固定上层/战斗 checkpoint、combat solver、环境种子后，分别用 deterministic 上层和多个独立 action RNG nonce 跑足量配对评测，按 phase 记录 `p_max`、entropy、非 argmax 率及牌组/路线/血量/药水和战斗初态分布；部署动作规则作为主口径，另一模式作为分布敏感性消融，不把单次随机8局归因为战斗算法差异 |
| TRAIN01 | 已优雅停止并固定最新快照；性能恢复已验收 | `u40x4-ruleupper-v1-fresh-20260918` 在共同 step1622016 停止，四机完整 model/replay/Adam/RNG 已固定于各自 `runs/snapshots/u40x4-ruleupper-v1-fresh-20260918/step-000001622016/rankN`，不受滚动保留清理影响；本地另保存 model 与配置 manifest。此前 PERF02 修复在40 updates/128 simulations不变下使 collection 671.02s→112.59s、wall862.7s→267.9–273.6s，graph零eviction。旧曲线的早期整局效果落后仍是独立风险，不能以 TRAIN02 的新架构 run 改写旧曲线。见 [快照](worklog/2026-09-19.md#original-training-snapshot-preserved)、[在线诊断](worklog/2026-09-18.md#ruleupper-online-speed-config-and-curve-diagnosis) 与 [性能恢复](worklog/2026-09-18.md#train01-rollout-efficiency-recovery) |
| TRAIN02 | 四机首轮已验收，MLP 训练运行中 | `u40x4-ruleupper-v1-mlp-20260919` / W&B `u40x4mlpf19`：716566参数combat MLP（256/256主干、独立128 policy/value头），保留132个冻结上层/shared张量，step0/空replay/新Adam开始。学习/search/replay/调度预算不变，40 updates、128 strict simulations、10M全局transition；step49152 bundle/策略与价值更新/Adam40/有限性/跨rank policy和Adam一致均验收，rollout93.9–95.0s、backward约9s、wall107–126s、graph零eviction。W&B已到98304且running；模型效果和长程稳定性待观察，不能把架构变化称为等价优化。见 [实验及验收](worklog/2026-09-19.md#train02-small-combat-mlp-experiment) |
| [F01](docs/2026-09-13/codebase-audit.md#f01) | 待重构 | 空怪物槽不改变 action-target 映射；canonical/packed/real 路径一致 |
| [F02](docs/2026-09-13/codebase-audit.md#f02) | 待重构 | 非 reasoning 策略也能观察 pending card/potion 来源 |
| [F03](docs/2026-09-13/codebase-audit.md#f03) | 待重构，选牌截断已复现 | 统一候选实体列表；Headbutt/Exhume/generated/multi 等来源正确。Secret Technique/Weapon原生draw候选index>=10被71-action绑定/薄层丢弃，可使存活战斗dead_screen；验收须覆盖draw/discard/exhaust长候选、明确候选→native索引映射、71/181 checkpoint兼容及同seed110104独立回归。具体strict xfail：`tests/integration/test_environment_contracts.py::test_draw_selection_remains_actionable_beyond_hand_slots`；[新增证据](worklog/2026-09-18.md#artifact-confirmation-result-and-f03-draw-selection-reproduction) |
| [F04](docs/2026-09-13/codebase-audit.md#f04) | 待重构 | teacher 来源显式；卡牌守恒、状态最终效果与原生/参考一致 |
| [F05](docs/2026-09-13/codebase-audit.md#f05) | 待重构 | Python/C++ 搜索与 learner 共用结算前/后 HP 定义 |
| [F06](docs/2026-09-13/codebase-audit.md#f06) | 待重构 | 搜索完整动作的 target/selection 分布可被无搜索策略学习 |
| [F07](docs/2026-09-13/codebase-audit.md#f07) | 待重构 | Gauntlet 提交 max HP、药水、遗物计数等持久状态 |
| [F08](docs/2026-09-13/codebase-audit.md#f08) | 已修 | 显式 seed 后的自动 reset 序列可复现；普通/分层 PPO CPU 重复训练逐位对齐 |
| [F09](docs/2026-09-13/codebase-audit.md#f09) | 已修 | Python NOSL depth=0 真正推进搜索；同步和分批入口在新叶处停止 |
| [F10](docs/2026-09-13/codebase-audit.md#f10) | 已修 | Impatience 条件抽牌及 exhaust unique-ID fallback；覆盖过时、负数和越界 hint |
| [F11](docs/2026-09-13/codebase-audit.md#f11) | 待设计 | rollout 切点不选择性漏掉上层 macro transition |
| [F12](docs/2026-09-13/codebase-audit.md#f12) | 待设计 | 明确 PPO normalization/phase weighting 的统计边界；累积不是自动等价 |
| [F13](docs/2026-09-13/codebase-audit.md#f13) | 部分完成 | policies/checkpoint 共享 metadata 预检与依赖覆盖；仍需跨入口有效配置/依赖 manifest、版本化移植协议 |
| [F14](docs/2026-09-13/codebase-audit.md#f14) | 待重构 | 显式 native build 身份/能力校验；不静默丢搜索参数 |
| [F15](docs/2026-09-13/codebase-audit.md#f15) | 待重构 | bridge schema 故障分类、有界重试和可重放帧 |
| [F16](docs/2026-09-13/codebase-audit.md#f16) | 部分完成 | episode eval/dashboard 共用 strategy/runner/outcome，step_limit 独立、合作式 timeout；仍需完整固定评测集、有效 solver manifest 和进程级硬超时 |
| [F17](docs/2026-09-13/codebase-audit.md#f17) | 部分完成 | 规范、小提交、正式测试/基准与 CI 已建立；一阶段目录/职责/配置/文档归位；大 artifact 外置与巨型 collector/main 内部拆分仍待做 |
| [F18](docs/2026-09-13/codebase-audit.md#f18) | 已修 | 共享 additive-field 适配接入 eval/watch/search/dashboard/bridge；旧 schema 保存加载及正式 eval 回归通过（未连接真实游戏） |
| [F19](docs/2026-09-13/codebase-audit.md#f19) | 已修 | End-Turn value boundary 保留 Ice Cream 能量；普通能量仍归零 |
| VAL01 | 部分完成 | 真实执行路径的 End-Turn boundary 与战斗胜利 terminal 已无条件采为 value-only 样本，policy mask/target/weight 均不参与 policy 监督；搜索截断与采样已分离并覆盖开关两态，新增 policy root/boundary/terminal 计数。仍需固定边界语料按怪物身份及 HP/分裂阶段验证校准及 near-terminal 覆盖，禁止用所选路线回报标注假想叶子 |
| VAL02 | 待实验 | 在固定 combat corpus 上审计 policy prior → Full-Gumbel 搜索 → improved-policy target 的反馈：按合法完整动作数/候选裁剪分层，统计未访问动作、prior/Q 排名冲突、搜索前后 TV/KL、低 prior 高 Q 翻转率；以只改根 prior 且保留内部 policy 的 matched-nonce 消融判断根部影响，不能用全树 uniform 冒充根部消融 |
| SEARCH01 | 已修 | `paper_full_gumbel` 在训练、Watch、Eval 与前端显式一致；strict 模式强制 mctx canonical 参数、单叶顺序评估及无额外 visit 采样，原生回归覆盖 considered-visits、completed-Q、improved-policy target 与最终动作；NOSL chance/redeterminization 和卡→目标接口仍明确属于项目适配 |
| SEARCH02 | 已修 | NOSL packed leaf 的合法动作矩阵按当前 batch 最大动作数动态定宽；Exhume 等大牌堆选择状态不再受错误的 128 动作上限约束，普通与 packed 路径逐 action bits 对齐 |
| CI01 | 已修并通过托管 CI | 历史路径 PermissionError 已修；df65017 的 correctness #2 全部成功，运行链接见 worklog/2026-09-14.md。生产 PPO/NOSL 上报、保存/恢复、旧部署模块遮蔽和 SSH 路径边界已补入 CI；CUDA 合同另需 GPU runner |
| PERF01 | rollout 单机对照与四机恢复验收完成 | 共享 CUDA graph 分桶、grouped IPC 和零主动等待；384env/strict128 固定 checkpoint 的同机完整 collection 从803.56s降至241.75s，四机恢复后的首轮rollout239–251s、整轮808–813s，step2162688模型/Adam跨rank一致。CPU/GPU/内存与数值边界见 [rollout 实测](worklog/2026-09-16.md#rollout-optimization-bounded-graphs-ipc-and-measured-scheduling)。不减少simulation，不宣称改变batch形状后bitwise；多seed长程吞吐与收敛仍需后续验证；本轮 learner/worker/transport 等价优化已过固定输入与完整单机 collection，见 [09-17验证](worklog/2026-09-17.md#perf01-decode-grouped-leaves-once-per-contiguous-source-span)，四机已恢复至step3342336，40updates与模型/Adam跨rank一致，见 [恢复验收](worklog/2026-09-18.md#perf01-four-rank-optimized-run-recovery-accepted) |
| PERF02 | 修复已部署到 TRAIN01 并四机验收 | `u40x4-u10` 派生 preset 曾遗漏 CUDA graph 分桶及调度参数，96env 生产轮发生约 1.7–1.98 万次 capture/eviction。修正版先在 `u10x4` 验证，再以协调 save/resume 部署到 TRAIN01 的40-update preset；step835584 在0.883M leaves/rank 下 mean collection 112.59s、capture约9.75次/0.256s、零 eviction，相对同 run 前一轮 collection 671.02s 提升5.96x。学习/search/replay预算未变，bundle、replay/Adam/RNG、跨rank模型同步及存活均已验收；batch shape 改变后的 BF16 不宣称 bitwise。见 [性能恢复](worklog/2026-09-18.md#train01-rollout-efficiency-recovery) |
| PERF03 | 冻结评测已验收，可选开启 | `rl/evaluation/eval.py`、`strategies.py`：`--leaf-inference cuda-graph`，默认 eager；独立策略 cache、同设备数值/完整搜索动作及释放后回放测试通过。32局、1410289叶子评估，682s→416s，逐局行除耗时外一致；显存采样峰值903→2841MiB。仅冻结评测范围，不代表训练 collection 或跨设备验收，见 [最终记录](worklog/2026-09-18.md#perf03-final-frozen-source-evaluation-and-candidate-reporting) |
| OPS01 | 已完成 | 四机 NOSL replay 由本地 `manage_nosl_replay_4way.ps1` 统一预检、启动、状态和指定 recovery step 验收；密码不入库，恢复前校验四 rank bundle 身份与 checksum |
| OPS02 | 待修 | 分布式训练单轮约 20 分钟且只在轮末调用 `wandb.log` 时，W&B `heartbeatAt` 停在上一轮指标、run 会在活进程仍正常运行时被误标为 `crashed`；需用受测的独立低频 heartbeat 或显式 lifecycle 更新保持 Running 状态，且不得改变训练 step/图表语义 |
| R01 | 已落实 | canonical 运行时/工具/apps、tests、configs、归档分离；退役旧链路/CLI，20 个 checkpoint 库别名保留；迁移查询 tools/validation/migrations.json |
| R02 | 部分完成 | 更新原语/梯度同步与生产环境构造已共享；per-rank replay/RNG/Adam 保存恢复及四机同步验收已完成，见 [恢复交接](docs/2026-09-14/nosl-performance-recovery-handoff.md)。仍需拆巨型 collector/learner 与 main；活 simulator/未完成战斗不支持精确恢复 |
| R03 | 共享后端已落实 | episode runner/策略组合、native eval/rollout/watch、固定场景库已共享并合批；仍需 corpus/历史基准协议迁接、完整评测集与统计定义，关联 F05/F16/PERF01 |
| PERF04 | 搜索优化已验收；条件交接已武装 | 原生 NOSL 搜索等价常数优化；明确 baseline/candidate 构建，验证叶子顺序、动作、Q/visits/targets、父状态及完整预算性能。独立协调器已在 rank0 等待 2026-09-19 06:23:29+08:00：届时判断 TRAIN02 完整历史 `game/floor_mean` 是否曾严格超过29；仅在未超过且历史、PID、构建及四机 step1622016 快照均可核查时保存停机、从 TRAIN01 replay/Adam/RNG 恢复40-update训练，否则 fail closed。见 [意图和验收](worklog/2026-09-19.md#perf04-equivalent-native-search-optimization-and-conditional-train02-handoff) |

后续大重构验收：先建立固定 game fixtures 和版本化契约，再改接口。旧 checkpoint 的原环境复现与新环境迁移评测分开；跨平台/多 GPU/多机准确性不得从单机 CPU 结果推断。
