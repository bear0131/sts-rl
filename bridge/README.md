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
`hit_reaction_contracts`记录已审受击回调的Java来源、原生入口、触发条件和排队顺序，
区分Angry/Enrage、攻击/反伤和零伤/非致命条件；其中明确列出的pending缺口仍未解决。
`card_damage_timing_contracts`逐卡记录本批战士/无色攻击的伤害捕获时点、真实Java动作与原生入口，
区分群攻固定矩阵、飞剑逐段计算及恶魔之火随机消耗；源码核对不等于行为回归或完整pending恢复。
`combat_victory_action_contracts`按Java类/ActionType列战后保留规则，`monster_death_contracts`
列首领、小怪、复活边界和死亡回调的排队合同；不能用MINION_LEADER死亡直接替代全部动作结算。
`event_combat_rule_contracts`列六类事件战的阶段、战前奖励、战后药水/卡牌时序与精英/Boss遗物触发。
事件精英触发不等于把房间改成ELITE；竞技场第一战无展示奖励，烟雾后也返回第二轮选项。
默认模拟器仍排除Colosseum，不以补执行函数为由改变旧训练事件池；分支实机覆盖见TODO BRIDGE74/75。
卡牌词表逐项列原serializer的12字段、11项附加字段合同及原生4种specialData来源。
parity-v7通过第7个只读serializer补充11项typed字段，misc包括真实0也明确导出；
主牌组`deck_state`保存有序身份/升级/当前native misc/瓶装物理槽位，恢复必须走
`set_deck_state`并核对旧模型的deck投影，不以`set_deck`创建fresh牌代替当前快照。
后者也改为原子替换，明确清除旧瓶装索引；未知字段/缺值/错型/超容量/重复瓶装类型拒绝，失败不改变原牌组。
原生`BattleContext.card_state/set_card_state`快照v2完整读写9项实例成员和7项牌堆快照字段，
保留四个牌堆、两个青铜球暂扣槽、稳定UID和后续生成编号；整批验证后替换并重算派生缓存。
该接口的同状态回灌保持模型输入、合法动作、压缩观测与搜索future key逐项一致。
实时普通决策已接入UUID映射和暂扣牌导出；选择中实例和其他动态字段的剩余审查单列，
不能将`apply_custom_state`简化tuple视为实例快照，严格真实根必须提供完整`card_state`。
选择候选通过`instance` sidecar和原生`selection_card_state()`核对完整实例，
单选重排显式映射策略索引；未知临时UUID、多选候选顺序不一致明确拒绝，不猜测匹配。
`combat_selection_transport_fields`列选择来源、数量、任意张标志和已选实例的传输合同。
HAND_SELECT的`any_number`来自Java真实字段，不能仅凭GAMBLE等来源推断；
joint multi以UUID及完整实例同时校验手牌移除和已选区追加回执。
这不等于恢复所有pending/limbo/暂存牌或GRID多张进度；这些仍为未完成路径。
347个真机卡牌构造字节码中仅GeneticAlgorithm/RitualDagger写非零misc，分别为1/15；
fresh初值和370个native CardId一一核对，但这不代替当前misc、UUID实例映射、永久/回合费用和瓶装状态的完整对接。
`pending_transport/pending_mapping` 明确是未完成项，不得解释成可以丢字段的支持路径；
解决这些已知卡牌缺口前，不部署此分支或宣称实机环境已全部对齐。

PAD/EMPTY 只表示实际不存在的固定槽位、空药水槽或显式 skip 候选；未知身份、
缺失字段、越界枚举、无法完整表达的原生状态必须报错。紧凑观测导出完整枚举名称表，
解码按名称映射；卡堆列顺序必须与冻结词表逐列一致，禁止截断或夹到合法索引。

当前源码要求OBS20 / ACTION8 / compact layout9 / `simulator-parity-v28`配套使用。
v28补六火亡魂实际激活/灼伤升级/火球数/Divider基础伤害，周期不再被通用能力清零；Inferno升级抽牌与弃牌堆灼伤并加入3张Burn+，Sear不从回合猜升级。旧v27帧拒绝，需Java/native配套构建，模型布局不变。
v27补巨人头实际count，A18提前进入大攻击且后续伤害按真实倒计数成长；旧v26帧拒绝，需Java/native配套构建。
v26补红绿虱虫实际biteDamage、黑暗小鬼实际nipDmg和守护者实际dmgThreshold，非攻击/防御根也保留。
黑暗小鬼不重复加高进阶伤害，守护者阈值在进入防御时递增；旧v25帧拒绝，需Java/native配套构建。
v25补刺杀之书实际stabCount，单刺意图也完整保留；先恢复计数再核对当前意图hits，冲突拒绝。
旧v24帧拒绝，必须配套重建Java和原生，不改权重张量或动作编号。
v24新增六种怪物的实际一次性行动/次数/阶段字段，完整映射见`monster_move_state_fields`；
不从行动历史或能力数量猜状态，旧v23帧拒绝，必须配套重建Java和原生。
v23导出蠕动团块实际usedMegaDebuff，防止搜索重复预测植入；非该物种显式null，旧v22帧拒绝。
真实根已同步主牌组，已植入标志不会在原生战后重复添加寄生或扣御守；需配套新原生扩展。
v22独立导出GRID确认预览的实际原卡UUID；升级/变牌/单张移除清空selected_cards时，
确认仍能正确进行，不伪造已选列表、不重复发送确认；旧v21帧拒绝。
v21将最近一次怪物攻击扣血限定为NORMAL攻击，按受击前HP封顶，缠绕等THORNS不覆盖；
配套原生包含HP_LOSS的Buffer防护、自伤来源与复活最低生命修正，模型张量/动作编号不变。
v9新增HAND_SELECT真实`any_number`，v10新增明确的怪物生命周期字段与觉醒者死亡保护状态；
必须重新构建Java补丁，旧v8/v9不能配当前Python。
`monster_lifecycle_fields`逐项列出死亡、动画结束、半死、逃跑、下回合逃跑及消费者。
canonical保留零HP实体的能力和行动历史；OBS20只在固定模型投影中屏蔽gone能力，不删除原始规则信息。
根注入必须提供完整生命周期布尔值，失败原子回滚；不再接受`gone=true`代替一整只怪物。
真实空槽通过完整实体映射证明后传`present=false`。尸体槽复用会丢实体或更换种类但缺私有状态时明确拒绝，
不能伪造PAD或随机初始化。Awakened私有阶段与完整复活pending仍待BRIDGE37，不宣称全状态对齐。
原生扩展也必须包含`selection_card_state()`与`selected_card_state()`；接口与源码未统一验证前不部署。
神圣树皮液态记忆的两张GRID选择逐次使用现有PICK动作，记录已选物理UID，选齐才按顺序取牌。
候选排除已选实例但保留真实CHOOSE句柄；完整已选实例独立于模型投影进行核对。
回执核对源牌堆未提前变化、最终四牌堆/费用/身份，重复帧只等待，不重复选牌。
满手不移动/改费，并按Java刷新明确的动态base字段；这不代表其他pending动作均可恢复。
`native_card_queue_fields`列出待执行牌的13项内部状态；same-UUID复制不等于同一物理对象。
能力复制存在于limbo，死灵之书复制不在limbo，成长传播按真实成员范围处理。
选择中搜索key保留当前牌、复制队列的顺序/身份/参数；未知Java action队列的完整恢复仍未完成。
普通战斗决策的实时搜索使用完整 `card_state` 恢复四牌堆和两个青铜球暂扣槽，
禁止退回只含身份/升级/手牌费用的 fresh tuple。主牌组 UUID 映射当前物理槽；
临时 UUID 从原生主牌组容量 96 之后分配，战斗内不随牌堆移动重新编号、不复用。
保留 base/current cost、freeOnce、retain、实际exhaust、成长与生成游标；每个 settled 根核对原生快照，
同时刷新主牌组当前成长值。Java 每个怪物明确导出 `simulator_stasis_card`（确实未暂扣才是 null）。正常状态/错误回包两个实际JSON出口保留null，避免Gson默认删除字段；current_action仅在确实没有动作时导出null，v14对现存动作使用真实Java binary class name，保留包名及匿名/嵌套类的`$`后缀，禁止用匿名类的空simple name伪装无动作。未开放CHOOSE时choice_list明确为空列表。接收端在自动动作前区分正常帧与错误帧、实际空集合与缺字段；画面/库存缺失、列表别名或奖励记录冲突直接拒绝，不按非空优先或按钮文字补实体。完整存在性词表见`codec_contract.complete_word_tables()['json_presence_contracts']`。v17 Python必须配套v17 jar；原生要求包含此前实例/计数/睡眠修复，本次v17不改变其接口或冻结权重张量布局。

v15的`simulator_thief_state`对两种盗贼携带实际`stolenGold/slashCount/goldAmt`，保留死亡/逃跑后的账本，非盗贼才明确为null。三项经canonical、严格根恢复、完整dict/差分与搜索状态键传递，不能由玩家金币、房间回合数或显示Thievery能力反推。字段源表见`thief_private_state_fields`；packed叶仍是冻结权重的输入投影，不是完整可恢复状态。源码接通不等于已构建/行为验收，完整pending与死亡奖励捕获仍在TODO。

v16只补拉加维林影响行动预测的睡眠状态和已睡回合数：`simulator_lagavulin_state`经严格恢复重新设置ASLEEP，不能被普通powers覆盖清除。自然醒来消除8层Metallicize，零HP伤害不触发醒来；继续使用原权重的sleep/stun合并token，不要求动画内部状态逐项一致。
v17在CARD_REWARD明确导出`reward_item_present`（实际`CardRewardScreen.rItem != null`）。战外无父奖励项时是独立的一组三选卡，不应读取不存在的外层奖励列表；有父项则必须关联已经观察的奖励页，不能用空库存掩盖缓存缺失。战内宝典/发现仍属于战斗选择。多份奖励的跳过记录按剩余真实卡组数同步，已跳过的卡组不能因领取其他组而重新开放。独立奖励的PROCEED对应真实SKIP，SKIP_CARDS在颂钵可用时对应颂钵，两种模型动作不混同。
v18补充商店物理槽位和实际售出证据。只读ShopScreen.init/purchaseCard钩子记录7张货架牌原槽位及信使替换；遗物/药水读取Store对象的slot/isPurchased，保留已被UI列表移除的售出对象证据。Python按7/3/3固定槽投影并将SHOP_BUY映回当前CHOOSE句柄，缺槽必须有明确售出记录才为空，不能压缩重排或猜PAD。purge_available=false时删牌价格为原生的-1。词表见shop_transport_fields。本次必须配v18 jar；不改变原生接口/权重形状，编译不是实机验收。
v19为明确登记的12种事件导出实际阶段/当前执行标志，及四个效果队列中真实未完成的拿牌效果。20项分阶段回执统一登记于`event_contract.STAGED_EFFECTS`，同步到完整词表；不泄露尚未揭示的随机结果、不把阶段新增为策略输入。Bonfire的专用献祭上下文优先于通用删牌标志，Java额外对话只作为传输确认。具体分支仍需实机覆盖，源码核对和编译不是验收。
v20读取三种召唤首领的实际引用槽，按固定4/3/5槽恢复并将目标映回真实列表；未入列表的排队对象明确pending。已被引用槽替换的历史召唤物只有确认死亡、不复活、不持卡时才退出有限搜索投影，完整原始状态及历史快照仍保留，不填PAD、不将复制的真实状态冒充原生恢复。七种已审召唤物允许无RNG的完整种类替换，巫师实际currentCharge经完整状态/diff恢复到原生miscInfo/search key。需重新构建原生扩展；模型OBS/ACTION/compact布局不变。

史莱姆首领双分裂采用独立的精确七对象→五槽合同（BRIDGE107）：真实首领、两大型父体、四中型子体必须完整；只有已经死亡且非复活/无Stasis的两大型父体退出搜索投影并保留历史账本。首领死体与四个中型实例（含已死者）均保留，以免错误开放烟雾弹或丢目标身份。按物种内实际左右次序映射回Java索引，不依靠HP匹配或旧下标。原生扩展必须配套保留首领和中型实例；其他无法表示的阵容仍明确拒绝，不用PAD替代未知实体。
实际exhaust独立保存，包含蓝蜡烛/医疗箱使用后的修改；ethereal/selfRetain与原生固有语义不一致时拒绝。其他数值变更和选择中实例
仍须完成审查，不能将普通根恢复视为所有卡牌/所有选择路径已经对齐。
layout9 携带百年积木/蜥蜴尾巴标志、怀表边界计数和涅奥实时余量；19件实时/回合派生遗物在搜索叶观测中覆盖根计数。
六项player scalar逐列有名称/来源/消费者词表，实时gold必须覆盖根journal。
相较layout8，layout9增加2个已选卡类别列与2个升级数值列，复用OBS20既有selected槽；
模型输入形状、词表与ACTION8不变，传输宽度改变，必须同步重建原生及两解码器，拒绝旧格式。
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

烟雾弹使用独立原生终局`PLAYER_ESCAPE=3`，既不算击杀胜利也不算死亡；原有0/1/2值不变。
合法性按所有实际Boss实体判断（包括史莱姆Boss分裂后留下的尸体），Act4背袭战仍在排除范围。
烟雾消耗后搜索以战斗结束前HP结算，不再模拟为丢弃一瓶药水继续挨打；评测保留escape标签，
胜负分布监督不将逃跑标作loss。真实客户端等待同局同层药水消耗及战斗结束，不在逃跑动画中再次搜索。
胜利回调与逃跑同样会触发的战后遗物效果仍需共用规则核对；不提供普通战利品，不伪造击杀/金币回收。
这些改动需要重建native，但不更改OBS20/ACTION8/layout9、Java v20字段或最终权重；具体覆盖见TODO BRIDGE71。

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

## 奖励、商店与房间回放

parity-v12新增 `simulator_current_node`（每个画面的真实坐标/房间类型）与 `simulator_map_leaf_nodes`（原版hasEdges过滤遗漏的实际叶节点）。第四幕心脏(3,3)回到全图原位，胜利展示节点(3,4)保留为 `terminal_map_nodes`，不伪造为可行动房间/PAD；Boss按钮真实入房位置(-1,15)与地图图标坐标显式区分。地图移动收据在ready稳定帧同时确认落点、楼层和飞靴次数；缺新协议/字段拒绝。`Map.fromSeed(..., act=4)`现在返回实际固定终幕图，而非随机15行图。现役SimGameEnv训练/评测仍在第三幕结束；这些辅助地图修复不延长episode、不改变reward/终局定义，也不把第四幕作为前三幕实机启动门槛。

地图候选由原生 `GameContext.legal_map_options()`、动作 validator 和实际 transition 共用连接规则。飞靴有剩余次数时可选同下一行的其他有效房间，只在非正常连线移动时扣一次；首行/正常连接/Boss不耗次，Java耗尽值-2明确映射到native0。更新此路径须重建扩展，不兼容缺少该接口的旧构建，但OBS20/ACTION8/layout9和冻结参数形状不变。全图的children仍是真实连线，choice/reachable从实际可选节点展开；不可把飞靴跳跃伪造成永久边。真实current_node只有坐标/房型，边从game_state.map查取；缺children明确拒绝。移动命令收据等待同seed/act的下一层及精确飞靴次数，迟到帧不能再次点击，错误扣次/超时保持拒绝；不声称仅靠这些字段验证全部入房副作用。

奖励领取以当前 `COMBAT_REWARD` 的合法项及按种类计数的索引生成命令；满物理药水槽或持 Sozu 时不可领取药水。直接调用 `command_for_semantic` 也拒绝不可领取项和错误画面，不只依赖策略 mask。模拟器侧 `SimGameEnv` 使用原生 `GameContext.legal_reward_options()` 的 `(kind, reward_group_index, item_index)` 过滤领取动作；更新 Python 后必须同步重建本 checkout 的扩展，旧扩展缺少此接口时不可混用。已有 action/observation 张量布局和 checkpoint 参数形状不变。

商店原生 Python 入口 `shop_buy_*`、`shop_remove`、`shop_proceed` 要求当前为 `SHOP_ROOM`；购买索引在编码前按实际货架范围检查，售罄/余额不足/已使用删牌服务抛 `ValueError` 且不修改状态。`legal_shop_options()` 在其他画面返回空。保留既有满槽或 Sozu 阻止取得药水时，对有效且可支付的药水槽直接调用为无副作用返回；枚举和validator均不将其视为可购买。需要重建本checkout扩展；不改变动作/观测张量布局。

`replay.Driver` 处理 REST/CHEST 的 `proceed` 时按模拟器当前画面推进：休息点使用合法 skip、未开宝箱使用不开箱选项、未领取奖励使用奖励退出。模拟器已经回图时，真实 UI 的房间收尾帧不重复执行；仍在选牌等未决画面时报告不一致，不强制改成地图。局部原生回归见 `tests/integration/test_bridge_room_replay.py`。

远端版本1卡牌快照的语义已由本分支版本2覆盖；同步时保留动态exhausts、UUID映射与选择合同，不降级快照版本。
公开`set_card_state`只能在稳定玩家回合且两队列为空时单独替换牌堆，防止破坏pending引用。
内部完整根恢复仍由`apply_custom_state`统一校验；公开单独牌堆接口不能冒充完整pending恢复。
