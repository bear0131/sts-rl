# sts_lightspeed 接入 / 差分测试 开发笔记

行号基于 vendored 的 `sts_lightspeed/`（我们改过的版本）。上游文件行号稳定；我们自己加的代码按符号名找（编辑后行号会漂）。

## 构建

- venv 在 `sts-rl/cpp/.venv`（py3.11）；build 目录 `sts_lightspeed/build`，产物 `slaythespire.cp311-win_amd64.pyd`。
- 配置命令：`cmake -S sts_lightspeed -B sts_lightspeed/build -G Ninja -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DPYBIND11_FINDPYTHON=ON -DPython_EXECUTABLE=<venv python>`；编译：`cmake --build sts_lightspeed/build --target slaythespire`。
- vendored pybind11 已从 2.7.1 升到 2.13.6（2.7.1 不支持 py3.11 的不透明 `PyFrameObject`）。
- `PYBIND11_FINDPYTHON=ON` 必须加，否则老式 FindPythonInterp 扫 PATH 抓到坏的 chocolatey python3.13。
- mingw 编出的 `.pyd` 运行时依赖 `libstdc++-6.dll / libgcc_s_seh-1.dll / libwinpthread-1.dll`，已从 `C:\Strawberry\c\bin` 复制到 build 目录旁，import 不用动 PATH。
- python 里加载：`os.add_dll_directory(r"C:\Strawberry\c\bin")` + `sys.path.insert(0, <build dir>)` 再 `import slaythespire`。

## 模拟器战斗内核（关键位置）

- 战斗开局 `aiRng/monsterHpRng/shuffleRng/cardRandomRng` **无状态重置**成 `Random(gc.seed+gc.floorNum)`（`src/combat/BattleContext.cpp:29-34`）→ 抽牌、怪物 HP 是 (seed,floor) 纯函数；但 `miscRng/potionRng` 是 `gc.miscRng/gc.potionRng`（**有状态拷贝**，同处 :33-34）。
- 虱子颜色 `getLouse(bc.miscRng)`（`src/combat/MonsterGroup.cpp:555` 用 `randomBoolean`）→ **吃 miscRng**，故 vanilla gc 强开复现不出真机虱子；**必须让 gc 真走过 Neow→进图把 miscRng 推到位**（固定组成的遭遇战如 Small Slimes 不吃 miscRng，vanilla 也能对）。
- `BattleContext::init` 末尾调 `executeActions()`（`src/combat/BattleContext.cpp:76`），所以建好直接停在 `PLAYER_NORMAL`，可立即接受输入。
- `BattleContext` 战斗状态字段：6 个 RNG `include/combat/BattleContext.h:61-66`，`outcome/inputState` `:69-70`，`turn/player/monsters/cards` `:87-90`。
- `search::Action` 是 32-bit 打包（type 高位、idx 低位），构造与访问见 `src/sim/search/Action.cpp:11-44`。
- `Action::execute` 末尾 `bc.inputState=EXECUTING_ACTIONS; bc.executeActions();`（`src/sim/search/Action.cpp` 行 ~459），即施加动作后自动 pump 到下一输入态，不用手动 pump。
- `Action::isValidAction` 含 `outcome!=UNDECIDED 即非法`（终局判断），`src/sim/search/Action.cpp`。
- 合法动作枚举逻辑：`enumerateActionsForNode`（`PLAYER_NORMAL`→出牌/药水/结束；`CARD_SELECT`→选牌）见 `src/sim/search/BattleScumSearcher2.cpp:175`，出牌枚举 `:205`。
- `sts::Random` 是真机 RNG 的忠实复刻（xorshift128+ + murmurHash3，带 `counter`），`include/game/Random.h:66`；洗牌用 `java::Random`（同文件 `:10`）。
- 玩家：`Player` 结构 `include/combat/Player.h:33`，力/敏/集中/神器是独立字段 `:50-54`，其余 buff 在公开的 `statusMap` `:59`。
- 怪物：`Monster` 结构 `include/combat/Monster.h:29`，buff 读取用 `getStatusInternal(MonsterStatus)`（`:104` 声明，对不存在的返回 0，安全），意图伤害 `getMoveBaseDamage` `:160`，最近两步 `moveHistory[2]` `:48`，怪物专属计数在 `uniquePower0/1`+`miscInfo`。
- 牌堆：`CardManager` 手牌定长数组 `hand`+`cardsInHand` `include/combat/CardManager.h:29-30`，抽/弃/消耗是 `std::vector<CardInstance>` `:39-41`。
- 卡访问器：`getId/getName/getUpgradeCount/requiresTarget` `include/combat/CardInstance.h:42-54`，`canUseOnAnyTarget(bc)` `:79`。

## id / buff 字符串表

- 卡枚举名 `cardEnumStrings`（"STRIKE_RED" 等）`include/constants/Cards.h:420`。
- 怪 id 串 `monsterIdStrings` `include/constants/MonsterIds.h:81`——注意 sim 用 `GREEN_LOUSE/RED_LOUSE`，对应真机 `FuzzyLouseNormal/FuzzyLouseDefensive`。
- 怪 buff 名 `enemyStatusStrings` `include/constants/MonsterStatusEffects.h:66`，枚举 `:10`（`MonsterStatus::INVALID` 是末尾哨兵）。
- 玩家 buff 名 `playerStatusStrings`（在**全局命名空间**，不在 `sts::`）`include/constants/PlayerStatusEffects.h:117`。
- 遭遇战名 `monsterEncounterStrings` `include/constants/MonsterEncounters.h:88`，枚举 `:12`（`INVALID=0`，act1 弱怪如 "Cultist"/"Jaw Worm"/"Two Louse"）。

## 我们加的绑定（`bindings/slaythespire.cpp`）

- 匿名命名空间里的 helper（`using namespace sts;` 之后）：`rawCombatState`（导出裸状态 dict）、`legalCombatActions`、`createBattle`、`encounterNames`。
- 模块内 `.def`（`#ifdef VERSION_INFO` 之前）：`create_battle(gc, 名字)` / `encounter_names()` / `BattleContext.{to_dict,legal_actions,execute,turn,outcome,input_state,is_battle_over}` / `Action(ActionType,src,tgt)` / `ActionType` 枚举。
- 战斗外进度（新增到 gameContext 链）：`choose_neow_option(idx)`（应用 `gc.info.neowRewards[idx]`）、`transition_to_map_node(x)`（进入下一行 column x 的房间，自动推进 floor/room/miscRng）。
- `pick_reward_card` 吃的是 **`Card` 对象**不是下标（文档串了）；用 `gc.get_card_reward()[idx]` 取。
- `Action` 有歧义（`sts::Action` in `ActionQueue.h:20` vs `sts::search::Action` in `Action.h:33`），写绑定时必须全限定 `sts::search::Action`。
- 已知小瑕疵：`NNInterface.observation_space_size` 只读属性 getter 是零参 lambda，pybind 2.13 读取会报错，用 `len(getObservation(...))` 绕过。
- 当前**未导出**：药水动作枚举、怪物 intent 字符串/move_id（只导出了 base_damage/hits）。

## schema / diff（`sts-rl/bridge/schemas/combat_state.py`）

- `normalize_real(game_state)`：真机 CommunicationMod 帧 → canonical（已用 phase0 的 154 帧验证）。
- `normalize_sim(raw)`：sim `BattleContext.to_dict()` → canonical；映射表 `_MONSTER_SIM2REAL`（未知 id 直接 raise）、`_CARD_OVERRIDE`+通用变换（`ENUM_NAME`→`"Enum Name"`，仅红色 Strike/Defend 特例）、`_POWER_SIM2REAL`（默认透传 + `Weak→Weakened`）。
- `diff(real, sim, pile_mode)`：逐字段比对，返回 `(path, real, sim)` 列表；手牌按序、抽/弃/消耗按多重集、非攻击意图 base/hits 统一归零去噪。
- 比对字段集 `_MONSTER_CMP_FIELDS` 不含 intent/move_id（sim 暂未导出）。
- `normalize_sim` 把 `turn` **+1**：sim turn 0-based、真机 1-based。

## 单战斗回放（`sts-rl/bridge/replay.py`）

- `python replay.py <orchestrator.log> [encounter]`：`progress_to_first_combat` 从日志走 **Neow→选牌→进图**把 gc 推到 floor 1 真实状态，`create_battle` 重建首战，回放 `play/end` 命令、每步 `diff`；encounter 名默认从 `gc.encounter` 反推（`SMALL_SLIMES`→`"Small Slimes"`）。
- 命令翻译：`play N [T]`→`Action(CARD, N-1, T)`（卡 cmd 里 1-based、sim 0-based；怪 T 两边都 0-based），`end`→`Action(END_TURN)`；`wait` 跳过；`choose/potion`（战斗内）暂不支持。
- `progress_to_first_combat` 识别：EVENT 且选项>1=Neow bonus、CARD_REWARD=选牌、MAP=`transition_to_map_node(sorted(next_x)[idx])`；talk/ack(1选项)/confirm/proceed 跳过。仅覆盖首战所需。
- `extract_first_combat`：瞬态帧（无 combat_state 且 screen 为空/NONE）跳过，只在遇到真实非战斗屏（COMBAT_REWARD 等）才收尾；日志读取 utf-8 失败回退 gbk。
- 日志编码：CommMod stdin 是 GBK，但本次 golden 日志用户手动转成了 **utf-8**；replay 两者都吃。
- seed 串↔数字：`get_seed_long(str)->int` / `get_seed_str(int)->str`（**与各自 docstring 描述相反**）；replay 直接从日志帧读数字 seed。

## 对齐事实与风险

- 真机 CommunicationMod **只暴露 `seed`、无 RNG 计数器**，故差分只能固定 seed 喂相同动作、比对可观测状态（间接反证 RNG 对齐）。
- 一律用英文 `id` 比对，不用 `name`（真机 `name` 是 GBK 本地化的，乱码且不稳定）。
- 真机 `draw_pile` 顺序未必是真实抽牌序——未验证；故非手牌牌堆暂按多重集比，不逐位比。
- 重建战斗要复刻卡组**内容+顺序**+遗物+hp+floor，洗牌 = f(初始顺序, shuffleRng)，初始顺序不对则即便 RNG 对齐手牌也会差（首个差分测试要重点看这个）。

## 首个差分用例（取自 `sts-rl/bridge/client/orchestrator.log` 第一场战斗）

- seed `1298`，ascension 0，floor 1，encounter = Two Louse（`FuzzyLouseDefensive` + `FuzzyLouseNormal`），卡组 = 初始 10 张（5 Strike/4 Defend/1 Bash，未升级），遗物 `Sozu`（即 Neow 换了起始遗物、卡组没动，利于重建）。
- **旧日志(`orchestrator.log`)不能当 golden**：288 RECV / 56 SEND、154 战斗帧只有 17 play——大多手点没记命令。已重开干净一局 → `orchestrator_new.log`（seed 串 "ABC" = 数字 12647，全程走 orchestrator）。
- **首战差分已 PASS**（2026-06-23）：`orchestrator_new.log` 首战 = Small Slimes（`SpikeSlime_S 12` + `AcidSlime_M 31`），Neow 选了 THREE_CARDS→拿 Clash（牌组 11 张）。8 步全 `[OK]`：开局洗牌/抽牌、出牌、block、伤害、怪物 AI、回合结束逐帧一致。
- 教训：vanilla `create_battle` 复现不出真机（牌组缺 Neow 加的牌、miscRng 没推进）；走 `progress_to_first_combat` 后逐字段对齐。
- 下一步：扩到整局到 boss——需补战斗外决策（事件/休息/商店/grid/boss遗物）的绑定+派发，及战斗结果回填 gc（`exitBattle`）让 gc 主循环连起来。

## 整局回放（`replay.py` 重写）+ 事件 RNG desync（2026-06-23）

- **绑定新增**（`slaythespire.cpp`，已重编）：`BattleContext.exit_battle(gc)`（回写 hp/gold/deck/rng + 胜利时开奖励屏）；GameContext 上 `deck_ids()`/`relic_ids()`（relic 用 `sts::relicIds[]` 即真机风格 id，**不是** `relicNames[]`）、`screen_state_name`、`cur_event_name`、`choose_event_option`/`choose_campfire_option`/`choose_select_card_option`/`choose_treasure_room`/`choose_boss_relic`、`get_select_cards`、`get_reward_info`/`claim_gold`/`claim_relic`/`claim_potion`/`reward_proceed`、`shop_buy_card/relic/potion`/`shop_remove`/`shop_proceed`。
- **驱动架构**：`parse_log` 丢弃出错命令（紧跟带 `error` 的 RECV 的 SEND）；`build_steps` → `(cmd, 命令后稳定帧)` 序列；`Driver` 按 `doc/0623_game_flow.md` 的屏幕→API 映射逐步驱动 sim。战斗内逐帧 `diff`，战斗外 `diff_overworld`（deck 多重集/relics/hp/gold/floor）。命令的语义靠真机帧的 `choice_list`（gold/card/purge/rest/smith/x=N…）+ 卡类屏幕的英文 `id`。篝火按字符串映射固定下标（rest0/smith1/recall2/lift3/toke4/dig5），grid 选牌按 (id,upgrade) 匹配 `get_select_cards`。
- **结果**：floor 0→2 进房**全 0 diff**——Neow、整场 floor-1 战斗(8步)、combat/card reward、地图导航、floor-2 房间类型全部逐字段对齐。
- **首个 desync = 事件选择 RNG**：floor-2 的"?"房间，房间类型对齐（都判定为 EVENT），但 sim 抽到 `The Woman in Blue`、真机是 `Transmorgrifier`(shrine)。`eventRng` 在 floor 2 才首次被消耗（`getEventRoomOutcomeHelper` 1 掷 + `generateEvent` 的 shrine-chance 掷），房间类型 EVENT 是 85% 大概率桶、不能证明那一掷相同 → `eventRng` 的种子/消耗序列与真机不一致，**疑似版本相关的事件 RNG 差异**（战斗 RNG 已验证一致，事件/地图生成 RNG 是另一回事）。combat 引擎是 RL 训练的核心、已验证；事件/地图 RNG 对齐是独立子课题。
- `replay.py --natural` = 上面这条自然驱动模式（停在 floor-2 事件 desync）。

## 解耦逐战斗验证（`replay.py` 默认模式，2026-06-23）

因事件/地图 RNG 未对齐，改为**每场战斗独立验证**：`extract_combats` 切出每场战斗，`_sync_gc` 用日志里该战斗首帧的真机快照强制对齐 sim 的 deck(`set_deck`)/relics(`set_relics`)/hp/gold/floor，`create_battle`(遭遇战名由怪物组成反查 `build_encounter_map`) 重建后逐帧 `diff`。战斗内无状态 RNG(抽牌/HP/洗牌) = `Random(seed+floor)`，与事件 RNG 独立，故此法能验证整局所有战斗。新增绑定 `set_deck`/`set_relics`（real id→sim 枚举的反查在 C++ `cardIdFromEnum`/`relicIdFromString`；real→sim 枚举名的逆映射 `real_to_sim_card_enum` 在 Python）。
- **结果（seed 12647 整局到 boss，8 场战斗）：7/8 逐帧字节级一致** —— Small Slimes / Cultist / Jaw Worm / Three Sentries / Blue Slaver / Lagavulin / Gremlin Nob。
- normalizer 修正：`Enrage→Anger`(Gremlin Nob)、丢弃 sim 内部 power `Asleep`(真机不暴露)、**hp≤0 的死怪 powers 归零**(真机死亡清空、sim 保留 Artifact → 哨兵战因此全过)。
- **HAND_SELECT 帧不参差分**：战斗内选牌(如 Burning Pact 消耗/抽牌)真机延到 CONFIRM 才结算、sim 在 `SINGLE_CARD_SELECT` 即结算，mid-selection 帧是时序假象。
- **floor 10 Lagavulin（已修复，抓到一个 sim bug）**：真机 Lagavulin 全程带 `Strength 2`——它是这 seed 的**带火精英(emerald-key elite)**，强化类型 0 = +Strength。sim 支持(`MonsterGroup::applyEmeraldEliteBuff`，触发于 `BattleContext.cpp:58` `burningEliteX/Y==curMapNodeX/Y`)。解耦验证须在 `_sync_gc` 里把带火精英那层(`floor==burning_elite_y+1`)的 `cur_map_node_x/y` 设成带火坐标(新增绑定 `burning_elite_x/y`)才会触发。但触发后 sim 给 `Strength 1`、真机是 2 → 查出 `MonsterGroup.cpp:607` 是 `buff<STRENGTH>(act)`(act1→1)的 **off-by-one**(Metallicize/Regen 都带 `act*2+…` 缩放，独 Strength 只 `act`）。**改成 `act+1` 后 27 个 diff 全清、floor 10 逐帧一致**——差分测试抓到并修掉的第一个 sim 源码 bug。
- **floor 16 The Guardian（START-DESYNC: deck order）**：deck 多重集一致但开局抽牌顺序不同 → 重建的 deck 顺序 ≠ 真机内部 master-deck 洗牌顺序（CommunicationMod 的 `deck` 数组顺序不可靠，早记的 pile-order 风险）。按位置回放会打错牌、损坏 sim，故开局牌堆顺序不一致即跳过该场。**是解耦强制对齐法的固有局限，非战斗引擎 bug**。
