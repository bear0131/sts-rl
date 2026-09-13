# 仓库总 TODO

本文件只维护持久问题与验收；工作进度见 worklog/。F 编号来自 2026-09-13 全仓审计（基线 1325041），修复后保留记录。

| ID | 状态 | 范围 / 验收 |
|---|---|---|
| F01 | 待重构 | 空怪物槽不改变 action-target 映射；canonical/packed/real 路径一致 |
| F02 | 待重构 | 非 reasoning 策略也能观察 pending card/potion 来源 |
| F03 | 待重构 | 统一候选实体列表；Headbutt/Exhume/generated/multi 等来源正确 |
| F04 | 待重构 | teacher 来源显式；卡牌守恒、状态最终效果与原生/参考一致 |
| F05 | 待重构 | Python/C++ 搜索与 learner 共用结算前/后 HP 定义 |
| F06 | 待重构 | 搜索完整动作的 target/selection 分布可被无搜索策略学习 |
| F07 | 待重构 | Gauntlet 提交 max HP、药水、遗物计数等持久状态 |
| F08 | 已修 | 显式 seed 后的自动 reset 序列可复现；普通/分层 PPO CPU 重复训练逐位对齐 |
| F09 | 待修 | Python NOSL depth=0 真正推进搜索 |
| F10 | 待修 | Impatience 条件抽牌及 exhaust unique-ID fallback 正确 |
| F11 | 待设计 | rollout 切点不选择性漏掉上层 macro transition |
| F12 | 待设计 | 明确 PPO normalization/phase weighting 的统计边界；累积不是自动等价 |
| F13 | 待重构 | checkpoint 依赖与有效配置 manifest；覆盖前不能先 load 缺失依赖 |
| F14 | 待重构 | 显式 native build 身份/能力校验；不静默丢搜索参数 |
| F15 | 待重构 | bridge schema 故障分类、有界重试和可重放帧 |
| F16 | 待重构 | 统一 solver 身份、固定评测集和 timeout 口径 |
| F17 | 进行中 | 规范、小提交、正式测试/基准与 CI；大 artifact 外置尚未迁移 |
| F18 | 待修 | 新 obs 字段不破坏旧 checkpoint 的正式预测入口 |
| F19 | 待修 | End-Turn value boundary 保留 Ice Cream 能量 |
| CI01 | 进行中 | 从源码构建、核心回归、固定预算训练/保存恢复、严格已知失败列表 |
| PERF01 | 进行中 | 可复用 CPU/GPU 资源监测和端到端训练基准；受控硬件基线 |

后续大重构验收：先建立固定 game fixtures 和版本化契约，再改接口。旧 checkpoint 的原环境复现与新环境迁移评测分开；跨平台/多 GPU/多机准确性不得从单机 CPU 结果推断。
