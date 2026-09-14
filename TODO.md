# 仓库总 TODO

本文件只维护持久问题与验收；工作进度见 worklog/。F 编号的完整问题、证据与验收见 [codebase-audit.md](docs/2026-09-13/codebase-audit.md)；每行编号直接链接相应问题。报告是固定快照，修复状态以本表为准。

| ID | 状态 | 范围 / 验收 |
|---|---|---|
| [F01](docs/2026-09-13/codebase-audit.md#f01) | 待重构 | 空怪物槽不改变 action-target 映射；canonical/packed/real 路径一致 |
| [F02](docs/2026-09-13/codebase-audit.md#f02) | 待重构 | 非 reasoning 策略也能观察 pending card/potion 来源 |
| [F03](docs/2026-09-13/codebase-audit.md#f03) | 待重构 | 统一候选实体列表；Headbutt/Exhume/generated/multi 等来源正确 |
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
| CI01 | 已修并通过托管 CI | 历史路径 PermissionError 已修；df65017 的 correctness #2 全部成功，运行链接见 worklog/2026-09-14.md。生产 PPO/NOSL 上报、保存/恢复、旧部署模块遮蔽和 SSH 路径边界已补入 CI；CUDA 合同另需 GPU runner |
| PERF01 | GPU 采样实测已建立 | 固定 RTX 5090/512 simulations，最终 CUDA graph 采样提速 1.37×且摘要逐位相同；叶推理初测 6.51×；含 CPU/RSS/显存/设备采样及完整长度故障回归，见 docs/2026-09-14/nosl-performance-recovery-handoff.md。多 seed 长程吞吐/收敛和 IPC 优化仍待验证 |
| R01 | 已落实 | canonical 运行时/工具/apps、tests、configs、归档分离；退役旧链路/CLI，20 个 checkpoint 库别名保留；迁移查询 tools/validation/migrations.json |
| R02 | 部分完成 | 更新原语/梯度同步与生产环境构造已共享；per-rank replay/RNG/Adam 保存恢复及四机同步验收已完成，见 [恢复交接](docs/2026-09-14/nosl-performance-recovery-handoff.md)。仍需拆巨型 collector/learner 与 main；活 simulator/未完成战斗不支持精确恢复 |
| R03 | 共享后端已落实 | episode runner/策略组合、native eval/rollout/watch、固定场景库已共享并合批；仍需 corpus/历史基准协议迁接、完整评测集与统计定义，关联 F05/F16/PERF01 |

后续大重构验收：先建立固定 game fixtures 和版本化契约，再改接口。旧 checkpoint 的原环境复现与新环境迁移评测分开；跨平台/多 GPU/多机准确性不得从单机 CPU 结果推断。
