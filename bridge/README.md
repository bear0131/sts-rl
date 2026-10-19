# 真实游戏边界

`schemas/combat_state.py` 将真实通信帧/模拟器状态规范化；env 的编码器消费该契约。
`real_game_env.py` 适配真实游戏；`replay.py` 做动作回放与差分；`client/` 是 CommunicationMod 客户端和 agent orchestrator，操作说明见 client/README.md。

策略导入 canonical rl/policies，禁止在客户端复制动作/观测规则。真实帧兼容、错误分类和有界重试仍见 TODO F15。CI 的合成 checkpoint/模拟器验证不代表真实游戏端到端通过。
原始帧/log 放忽略的本地输出，不进入核心模块目录。

## 完整对接词表

`schemas/codec_contract.py` 的 `complete_word_tables()` 返回展开后的真机→canonical、
原生→canonical、逐怪物行动字节、选择来源、事件选项、命令、奖励、三钥匙、
意图、角色、房间/动作阶段、卡牌类型/稀有度/目标关键词及明确拒绝的身份集合。
`validate_codec_contract()` 检查目标存在、全集覆盖、哨兵隔离和能力/选择来源的完整性。
它们复用 `entity_ids.py` 与 `event_contract.py` 的运行时数据，不另维护一份文档副本。
`relic_counter_contracts` 为180件遗物逐项记录计数来源/含义/数值映射和审查状态；
29个Java类实际读写counter，其他类继承显示哨兵。`pending_phase_and_numeric_review`
明确表示数值/阶段尚未核实，不是运行时用PAD或0代替缺失状态的许可。

PAD/EMPTY 只表示实际不存在的固定槽位、空药水槽或显式 skip 候选；未知身份、
缺失字段、越界枚举、无法完整表达的原生状态必须报错。紧凑观测导出完整枚举名称表，
解码按名称映射；卡堆列顺序必须与冻结词表逐列一致，禁止截断或夹到合法索引。

OBS20 / ACTION8 / compact layout5 / `simulator-parity-v5` 配套使用。
layout5 追加百年积木/蜥蜴尾巴的实时可用标志；九件实时遗物在搜索叶观测中覆盖根计数。
永久与战斗计数使用不同对接表（例如钢笔尖永久9、战斗-1），不能按名称直接复制。
旧 checkpoint 只投影到其原有字段与动作几何，不补造不存在的输入或新动作 logits；
训练恢复和 replay 不支持跨合同。当前真机原生搜索仅支持 Ironclad 及明确实现的
卡牌能力集合；不支持的内容明确失败。

事件词表覆盖 52 个入口（含 Neow），表示每个身份都有显式处置。
`event_options[*].effect_status=unverified` 的选项已识别但效果确认尚未完成，发送前报错；
Match and Keep 的板面传输仍待接入，动态事件必须携带明确的候选身份。
`implemented` 表示已有处理代码，仍须逐分支核对副作用和实机结果，不能据覆盖数宣称全部等价。
