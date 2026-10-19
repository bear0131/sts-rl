# 环境与 worker 搜索

`core/game_env.py` 是厚层整局状态机，`core/run_env.py` 是薄层 Gym 接口；`core/combat_env.py` 是单场战斗环境。它们调用显式选择的 slaythespire 构建。

| 模块 | 契约 |
|---|---|
| core/encode.py、vocab.py、cards_vocab.py | canonical 字段到固定观测；字段/形状/词表版本以源码为准 |
| core/action_space.py、actions.py、game_actions.py | 整局/战斗动作与候选映射，mask 和 native action 必须一致 |
| core/factories.py、encounters.py、spiremap.py | 初始场景、遭遇与地图构造 |
| core/run_env.py | reset/step、缓存、fork、native begin/backprop/packed leaf |
| [search](search/README.md) | 无模型的 Python NOSL session/nonce，供 worker 与学习侧共同使用 |
| scripts/ | 人工 play、binding 探针与接口文档生成；不是 pytest 目录 |

纯 encode/vocab/action 模块不加载 simulator；原生环境与工厂加载 simulator。env 不依赖 rl/apps/tools/tests，依赖方向由 tests/unit/test_repository_layout.py 检查。
共享状态规范化在 bridge/schemas；模型额外观测裁剪在 rl/policies/model.py。不要因当前模型不使用某字段就改变环境实际状态或动作语义。

测试在 ../tests。实体槽位、pending 来源、候选区域和 gauntlet 持久状态的验收范围见根 TODO F01/F02/F03/F07。

## 完整战斗动作契约 native-v1

新策略可显式创建 `SlayTheSpireEnv(combat_action_contract="native-v1")`。默认 `legacy`
仍提供分步接口；当前候选语义遵守下述版本，历史 checkpoint 由推理适配层投影观测。`native_combat_actions()` 返回当前
完整原生动作；`step_native_combat(bits)` 验证后复用 wrapper 的奖励和终局路径；
`combat_selection_candidates()` 保留 draw/discard/exhaust/generated 的真实候选身份。
共享 evaluation runner 通过 `Decision(..., native_action=bits)` 执行这一契约。

该模式不会给旧策略补齐 logits，也不是旧动作空间的透明升级。长选牌候选和 MULTI 子集
可以在 thin mask 之外继续执行；draw-select 的 NOSL 副本必须用原生实体身份重映射，不能
把 sampled 索引直接提交给真实环境。回归入口为
`tests/integration/test_native_action_contract.py` 与 `tests/unit/test_native_action_codec.py`。

## 公开候选与实体槽位

OBS_VERSION=20、ACTION_VERSION=8、native compact layout v5 必须配套使用。
v18将真实布尔能力/减益写入player_buffs（激活为1，再按BUFF_SCALE归一化），
包括腐化、壁垒、混乱、Hex及钢笔尖就绪；数值buff仍保留实际层数。
canonical与packed共享原生读取逻辑；新训练使用新合同，旧checkpoint推理按声明空间投影，
不允许跨版本恢复优化器或replay，也不给旧动作头补造新增动作分数。
怪物数组保留原生槽位，INVALID 用 PAD 填充；完整动作的目标编号不压缩。
战斗 PICK_CARD 指向 `options.select.candidates` 的公开顺序，各候选携带 zone、
牌 ID、升级、当前费用和 `native_index` 执行句柄。draw 候选按公开牌面排序；
神经网络只能使用句柄是否存在，不能把数值作为特征。CODEX skip 是无牌的显式候选。
单选最多96个候选，超出范围明确报错；多选合法子集由原生统一枚举，包含空集，
可超过128个完整动作。普通与批量叶观测使用同一候选信息。
pending_card_slot/pending_potion_slot 常驻观测，0为无来源，槽位+1为来源。
旧模型可以加载并预测，但不会因此获得新增候选/来源的表达能力；重新训练须迁移新架构。

## 联合战斗策略 joint-v2

`SlayTheSpireEnv(combat_action_contract="joint-v2")` 使用无损策略ID：181个原有整局控制，
随后是 End Turn、10×6牌/目标、5×6药水/目标、96单选和1024个子集地址，共1392。
第6目标为NONE，怪物槽0仍为真实目标。`combat_legal` 携带后1211个地址的合法性，
模型不把执行句柄或该mask嵌入棋盘特征。native扩展必须声明joint_policy_size=1392。

网络以来源分数和来源条件目标分布组成联合分布；多选共享自回归“递增手牌下标/STOP”
解码器，每个集合恰有一条路径。1024是搜索和传输的最大地址域，不是1024个独立学习类别。
初始残差为0时按合法后续集合数分配概率，因此对合法集合均匀，避免偏向短序列。
根和叶的原生先验都读取完整动作分数；Full Gumbel未访问动作的概率也保留。
replay按行稀疏存储合法ID和float32概率（含合法但零概率动作），采样时恢复batch，
保存时使用数值CSR数组并校验policy/value contract。value-only样本不需要策略mask。

当前运行时支持Split Transformer、标准PPO和C++ NOSL；不支持旧交互推理步骤、
历史Boss战后回血价值目标。旧thin模型保留旧几何；迁移必须重新收集replay。
训练CLI用 `--combat-action-contract joint-v2 --init-from CHECKPOINT` 热启动；
`--resume` 禁止跨合同。迁移按命名特征映射全局投影列，初始化新增因子，优化器和replay重建。
distributed/evaluation按checkpoint空间恢复环境合同。bridge直接展开牌/药水的完整目标，
多选集合转为降序UI选择，逐帧核对确认和候选变化，不为每张牌重新调用模型。
NOSL inner-only在唯一合法End Turn根旁路前推和搜索，保留同权重根价值样本、边界样本与
采样计数；缺失的预测不进入HP-order诊断，单列forced_end_turn_roots/root_value_predictions_skipped。
标准PPO及非inner-only NOSL仍计算GAE需要的V；搜索内部叶也保留V。
25CPU/RTX5090、96×128环境transition的完整collection和GPU图资源生命周期已经验证，
结果见 `worklog/2026-09-29.md#joint-contract-production-length-validation`；不代表新策略已重训收敛。
新模型可作为旧战斗规则/规划器的整局上层，只有非战斗阶段适配空combat mask；
旧CombatTask及Python value search不支持joint-v2，观看中的规划器切换须使用独立评测。
