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
29个Java类实际读写counter，其他类继承显示哨兵。数值映射与`phase_audit_status`
分开：有映射不等于完整效果/阶段已验收；待处理项不是运行时用PAD或0代替缺失状态的许可。
卡牌词表逐项列原serializer的12字段、7项附加字段合同及原生4种specialData来源。
parity-v7通过第7个只读serializer补充11项typed字段，misc包括真实0也明确导出；
主牌组`deck_state`保存有序身份/升级/当前native misc/瓶装物理槽位，恢复必须走
`set_deck_state`并核对旧模型的deck投影，不以`set_deck`创建fresh牌代替当前快照。
后者也改为原子替换，明确清除旧瓶装索引；未知字段/缺值/错型/超容量/重复瓶装类型拒绝，失败不改变原牌组。
原生`BattleContext.card_state/set_card_state`完整读写8项实例成员和7项牌堆快照字段，
保留四个牌堆、两个青铜球暂扣槽、稳定UID和后续生成编号；整批验证后替换并重算派生缓存。
该接口的同状态回灌保持模型输入、合法动作、压缩观测与搜索future key逐项一致。
真实UUID到native UID、动态卡牌标志及暂扣牌导出仍须完成后，实时搜索才能使用此完整路径；
现有`apply_custom_state`简化tuple不得被视为实例快照，新增接口本身不代表实时接入完成。
347个真机卡牌构造字节码中仅GeneticAlgorithm/RitualDagger写非零misc，分别为1/15；
fresh初值和370个native CardId一一核对，但这不代替当前misc、UUID实例映射、永久/回合费用和瓶装状态的完整对接。
`pending_transport/pending_mapping` 明确是未完成项，不得解释成可以丢字段的支持路径；
解决这些已知卡牌缺口前，不部署此分支或宣称实机环境已全部对齐。

PAD/EMPTY 只表示实际不存在的固定槽位、空药水槽或显式 skip 候选；未知身份、
缺失字段、越界枚举、无法完整表达的原生状态必须报错。紧凑观测导出完整枚举名称表，
解码按名称映射；卡堆列顺序必须与冻结词表逐列一致，禁止截断或夹到合法索引。

OBS20 / ACTION8 / compact layout8 / `simulator-parity-v7` 配套使用。
layout8 携带百年积木/蜥蜴尾巴标志、怀表边界计数和涅奥实时余量；19件实时/回合派生遗物在搜索叶观测中覆盖根计数。
六项player scalar逐列有名称/来源/消费者词表，实时gold必须覆盖根journal；layout8保持payload宽度、模型输入形状不变，拒绝旧解码合同。
敌方回合后、下次抽牌前的搜索专用边界保留怀表的上回合出牌数，不能随普通回合计数清零。
永久与战斗计数使用不同对接表（例如钢笔尖永久9、战斗-1），不能按名称直接复制。
`set_relics` 明确创建fresh初始计数，不重放获得副作用；恢复必须用完整的
`set_relic_state([{id,counter}, ...])`，缺字段/未知额外字段/重复ID/负数哨兵拒绝，
失败不修改原状态。巫毒娃娃的派生计数须与先恢复的主牌组一致。
`native_state.make_native_combat_root` 是搜索/replay共用的全局恢复入口；
`native_combat_root_fields` 逐字段列真实来源、原生去向与限制，
`real_room_classes` 完整列11种具体房型，事件战保留EVENT，未知房型禁止子串猜测。
进阶等级、HP、金币、三钥匙和药水槽必须显式恢复；腰带容量按进阶门槛核对，缺槽/越界报错，不能夹取或重放onEquip。
原生战斗金币当前是int16，超32767明确拒绝，不能截断。replay应使用
`python -m bridge.replay` 并由启动环境明确指定扩展构建，不再抢占旧build路径；
共享全局恢复不代表replay已恢复隐藏RNG或已解决所有入战时序差异。
涅奥奖励/代价由Java对象导出类型关键词、UI索引和缓存HP单位，接受真实2/4选项；
不再解析本地化文字或猜测NONE代价。单位与当前上限不一致时明确拒绝当前不支持的状态。
涅奥内部画面也有完整处置表：0/1对话、3奖励、99完成；2的祝福/跳过分叉、
每日10、无尽999尚未实现对应原生路径，明确拒绝，不因“只有一个按钮”就自动点击。
旧 checkpoint 只投影到其原有字段与动作几何，不补造不存在的输入或新动作 logits；
训练恢复和 replay 不支持跨合同。当前真机原生搜索仅支持 Ironclad 及明确实现的
卡牌能力集合；不支持的内容明确失败。

## 现役权重的评测保真

桥接验收的目标是保留权重已学对的能力，而非仅确认ZIP可加载。冻结权重、卡向量、
词表顺序、归一化、动作含义及评测搜索配置；当前最终评测为Full-Gumbel128/Gumbel0，
不能以checkpoint默认Gumbel1代替实际评测参数。基线源码/native/参数/种子与原成绩一起保留，不覆盖。
对物理状态和规则等价的局面，逐字段比较checkpoint实际收到的tensor、合法动作与动作还原；
仅维度一致或一次预测成功不足以证明保真。搜索结果比较还须固定随机身份、worldline、预算和同设备/依赖/精度，
不伪称能重建真机未公开的未来RNG。真机对齐必须修正的原生规则差异单独列账、单独对照，
不将旧模拟器成绩写作修正后或实机成绩；先评估差异，再判断是否有训练需求，不擅自重训或改策略配置。

事件词表覆盖 52 个入口（含 Neow），表示每个身份都有显式处置。
`event_options[*].effect_status=unverified` 的选项已识别但效果确认尚未完成，发送前报错；
Match and Keep 的板面传输仍待接入，动态事件必须携带明确的候选身份。
`implemented` 表示已有处理代码，仍须逐分支核对副作用和实机结果，不能据覆盖数宣称全部等价。
