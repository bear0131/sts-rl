# Observation 设计文档（v5，完整版）

> 状态：**已实现并落地**（`env/core/encode.py`，`OBS_VERSION=5`，`spaces.Dict`）。本文档是 RL 模型输入的**契约**，与 `encode.py` 的真实 layout 一一对应；改 encode.py 必须同步改本文档。
>
> v3 相对 v2：新增 `arg_step` 键（薄层 `SlayTheSpireEnv` 的 autoregressive 子步，见 `action_design.md` §1）。
> v4 相对 v3：新增商店货架组 `shop_*`（卡/遗物/药水 id + 价格 + 移牌价），与 SHOP_* 动作段 positional 对位。
> v5 相对 v4：新增选牌屏契约组（`select_context`/`pick_remaining`/`can_pick_zero`/`can_pick_any`/`selected_card_ids`，§4.1），`select_card_ids` 的 `G_MAX` 12→64（全卡组选牌屏）。其余不变。
>
> 关联：`env/core/encode.py`（实现）、`env/core/vocab.py`（词表）、`doc/day_0623/action_design.md`（动作面，姊妹文档）、`bridge/schemas/combat_state.py`（canonical 生产）。
>
> 决策点 A–E（原 §7）**已拍板**，结论见各节与文末。

---

## 0. 设计原则（先读这个）

### 0.1 observation 的职责边界

env 输出**类型化的原始状态**，不做表示学习：

- **连续量**（hp、block、伤害、楼层…）→ 归一化 **scalar**（float，大致落在 [0,1]）。本身有序、有距离，网络该直接看到大小关系。
- **无序类别**（卡 id、怪种、move id、遗物 id、buff 名…）→ **整数 index**（categorical id）。类别之间无序无距离。
- **绝不**把 categorical id 当 scalar 塞进 float 向量（v1 手牌的错误：`Bash=25, Defend=130` 会让网络以为 130 比 25“大”，臆造了虚假序结构）。
- **绝不** one-hot 或 embedding。`onehot(i) @ E == E[i]`：one-hot + 线性层 ≡ index + embedding 查表，二者信息等价；但 embedding 矩阵 `E` 是**要学的参数**，必须活在模型里被优化器更新。env 是无参数纯函数，一旦 embedding 就把可学参数冻死在环境里。**embedding 是模型第一层的事。**

→ **结论：env 给 categorical id（int）+ scalar（float），分组、带类型标记。模型侧自己 embed + attention。**

### 0.2 为什么是 dict obs，不是扁平向量

模型必须知道“哪几维是 categorical id、哪几维是 scalar、哪些 id 共享同一个词表（卡牌）”。扁平 float 向量丢掉了这些类型信息。所以 obs 是 **结构化 dict**（gymnasium `spaces.Dict`），categorical 与 scalar 分开成组：

- 训练用 SB3 `MultiInputPolicy`（识别 Dict obs）。
- 这也是上 transformer 的天然形态：手牌 / 卡组 / 怪物 / 遗物各是一组 token（id 序列），模型把它们 embed 成 token 再互相 attention。env 只管吐 id 序列 + mask。

### 0.3 版本与稳定性纪律

- `OBS_VERSION`：dict 的**键集合 / 每个字段的形状 / 词表顺序 / scalar 归一化方式**任一改变 → bump，作废旧 checkpoint。
- **词表 append-only**（卡/怪/遗物/药水/buff/move）：只能往后加，不能重排或删除，否则旧 index 失效。
- **保留容量、零/PAD 填充**：词表预留富余长度（见各节）、容量槽（手牌 10、怪物 N）定长。**往保留槽填真数据不算 bump**，只有改形状/重排才算。

### 0.4 backend 无关

encode 是 canonical dict → obs 的纯函数，不 import slaythespire。sim 后端（训练）和真机后端（推理）产出同形 canonical，共用同一 encode。词表用 canonical id（bridge 已把 sim/真机命名归一）。

### 0.5 mask 不是 obs（autoregressive 多选的关键纪律）

`MaskablePPO` 的 `MultiInputPolicy` **前馈无记忆**（无 LSTM）。autoregressive 每一步是一次独立 forward，**步间唯一的记忆载体是 obs 本身**。两条通道必须分清：

- **action_mask** 作用在网络**输出 logits** 上，把非法动作压成 −∞。它**不进网络输入**，模型的内部表征**看不到 mask 内容**——只在出口删掉不许选的选项。
- **obs** 喂进网络前几层，是模型**用来决策**的信息；模型“看得见”、会基于它推理。

→ **推论**：mask 能阻止重选已选的牌（把那个槽 logit 设 −∞），但**不会让模型“知道”自己选过什么**。多选屏（选 k 张 / 任意张）里“已经选了哪些、还差几张、能否收手”这些决策依据，**必须显式进 obs**（见 §4.1）。少了这一半，多选训练就是瞎选 + 无记忆。已选的牌要**两条通道各标一次**：mask 管“不准再选”，obs 管“让模型看见”。

---

## 1. 词表清单（categorical id 空间）

每个 categorical 字段引用一张词表，index 0 一律是 `PAD`/`EMPTY`/`NONE`（占位 / 空槽 / 无）。

| 词表 | 现状 | 预留长度 | 来源 |
|---|---|---|---|
| `CARD_VOCAB` | **已有**，371 项（index 0=EMPTY） | 现长，append-only | `cards_vocab.py` |
| `MONSTER_VOCAB` | **已建**，40 项 | 96（act1-3 + 富余） | canonical `monster.id`，`bridge` 的 `_MONSTER_SIM2REAL` 键 |
| `MOVE_VOCAB`（意图） | **已建**，197 项（move_id 已 C++ 导出，见 obs_v2_landing） | 224（MonsterMoveId 197 + 富余） | sim `MonsterMoveId` 枚举 |
| `RELIC_VOCAB` | **已建**，181 项 | 208（全遗物 + 富余） | `gc.relic_ids()` 稳定 id 串 |
| `POTION_VOCAB` | **已建**，43 项 | 56 | `gc.overworld_potions`/`bc.potions` 名 |
| `PLAYER_BUFF_VOCAB` | **已扩**，85 项 | 稠密向量 110（玩家 buff 82 + debuff 18 + 富余） | canonical power id |
| `MONSTER_BUFF_VOCAB` | **已扩**，41 项 | 稠密向量 50（怪 buff 33 + debuff 11 + 富余） | canonical power id |
| `INTENT_CATEGORY`（派生） | **已建**，9 项 | 固定 9 | 从 move_id 派生：attack/defend/buff/debuff/strong_debuff/sleep/stun/escape/unknown |
| `CHARACTER_VOCAB` | **已建**，5 项 | 5（IRONCLAD/SILENT/DEFECT/WATCHER + INVALID） | 构造时已知 |
| `BOSS_VOCAB` | **已建**，64 项 | 72（`MonsterEncounter` 枚举名） | `gc.boss`（MonsterEncounter 枚举） |
| `ROOM_VOCAB`（房型） | **已建**，10 项 | 固定 10 | sts.Room：MONSTER/ELITE/REST/SHOP/EVENT/TREASURE/BOSS/NONE… |

> buff/debuff 不分两张表，合一张 `*_BUFF_VOCAB`（debuff 也是一种 power）。词表外的 power 丢弃（不改形状）。

---

## 2. 战斗 observation（dict）

只在 `phase == "combat"` 时有效（非战斗时这些组以 PAD/零填充，靠 §4 的 `phase` 区分）。

### 2.1 `player`（玩家）

| 键 | 类型 | 形状 | 内容 / 归一化 |
|---|---|---|---|
| `player_scalars` | scalar | (6,) | hp/200, max_hp/200, block/100, energy/10, turn/30, **hp_frac = hp/max_hp**（新增，比绝对值更稳） |
| `player_buffs` | scalar | (len(PLAYER_BUFF_VOCAB),) | 按词表的 buff 量 / BUFF_SCALE（有符号，如 Strength 可负） |

> buff 用“词表长度的稠密量向量”（每个 buff 一个槽放层数），不是 id 序列——因为 buff 是“有哪些 + 各多少层”，量是关键。这与卡牌（要 id 做 attention）不同。

### 2.2 `monsters`（怪物，固定 N=5 槽，**不再滤死怪压位**）

**改动**：v1 把死怪滤掉导致槽位漂移（slot0 这步是 Cultist 下步变 Jaw Worm），破坏槽位语义。v2 **固定槽位**：怪 i 始终在槽 i，死了用 `present=0` 标记。

每槽：

| 子键 | 类型 | 形状 | 内容 |
|---|---|---|---|
| `monster_id` | **categorical** | (N,) | MONSTER_VOCAB index（怪种！v1 缺失） |
| `monster_move` | **categorical** | (N,) | MOVE_VOCAB index（意图 move_id，**已 C++ 导出**：`moveHistory[0]`，见 obs_v2_landing §2） |
| `monster_intent_cat` | **categorical** | (N,) | INTENT_CATEGORY index（attack/defend/...，从 move 派生） |
| `monster_scalars` | scalar | (N, 6) | present, hp/200, hp_frac, block/100, base_damage/100, hits/10 |
| `monster_buffs` | scalar | (N, len(MONSTER_BUFF_VOCAB)) | 每怪的 buff 量向量 |

> `base_damage`/`hits` 是 sim 已导出的“本回合来袭攻击”，配合 `monster_intent_cat` 一起看：意图类别说明它要干嘛，base_damage 说明打你多少。二者都需要才完整（你说的“别只学会它防我就打它打我就防”）。

### 2.3 `combat_piles`（战斗牌区）

| 键 | 类型 | 形状 | 内容 |
|---|---|---|---|
| `hand_ids` | **categorical** | (10,) | CARD_VOCAB index，**位置敏感**（动作 play hand[i] 对应槽 i），空槽=0 |
| `hand_upgrades` | scalar | (10,) | 各手牌升级数 / 某 scale（升级影响数值，单列） |
| `draw_counts` | scalar | (371,) | 抽牌堆每种卡计数（顺序无关） |
| `discard_counts` | scalar | (371,) | 弃牌堆计数 |
| `exhaust_counts` | scalar | (371,) | 消耗堆计数 |

> 手牌为何 id 序列而非计数：手牌是要**逐张选择打出**的对象，且位置=动作槽，必须保 id+位置。牌堆为何计数：抽牌堆真实顺序跨后端不保证一致（见 replay 的 deck-order caveat），只能给 multiset。
>
> **待决策点 A**：`hand_upgrades` 单列 vs 把“升级”并进 CARD_VOCAB（如 `Strike_R` 和 `Strike_R+` 占两个 index）。后者词表翻倍但 attention 更干净。倾向：单列升级（词表小），但听你的。

---

## 3. 战斗内的全局信息（你的第 6 点：战斗时也要看到全局）

战斗 obs 里也带，因为战斗决策依赖全局（该囤血还是莽、留不留爆发给 boss）：

| 键 | 类型 | 形状 | 内容 | 来源 / 碰 C++？ |
|---|---|---|---|---|
| `character` | categorical | (1,) | CHARACTER_VOCAB（IRONCLAD…） | 构造已知 ❌ |
| `ascension` | scalar | (1,) | ascension/20 | 构造已知 ❌ |
| `act` | scalar | (1,) | act/3（或 4） | `gc.act` ❌ |
| `floor` | scalar | (1,) | floor/56 | `gc.floor_num` ❌ |
| `next_boss` | categorical | (1,) | BOSS_VOCAB（**很重要**：Hexaghost? Guardian?） | `gc.boss` ❌ |
| `relics` | **categorical 多热** | (R_MAX,) | 持有遗物的 RELIC_VOCAB index 列表，PAD 到 R_MAX（如 30） | `gc.relic_ids()` ❌ |
| `potions` | categorical | (P_MAX,) | 药水槽的 POTION_VOCAB index，空槽=0，P_MAX=5 | `bc.potions`/`gc.overworld_potions` ❌ |
| `gold` | scalar | (1,) | gold/999 | `gc.gold` ❌ |
| `deck_counts` | scalar | (371,) | **整局卡组**每种卡计数（区别于战斗内三堆的临时牌） | `gc.deck_ids()` ❌ |

> 战斗内 `gc` 是否拿得到这些：战斗由 `create_battle(gc, …)` 起，env 仍持有 `gc`，故 act/floor/boss/relics/deck 都能从 gc 读。`bc.potions` 给战斗内药水槽。**全部纯 Python，除意图外不碰 C++。**
>
> **待决策点 B**：`relics` 用“多热 id 列表（PAD 到 R_MAX）”还是“RELIC_VOCAB 长度的 0/1 向量”。前者紧凑、适合 attention（你要 transformer）；后者定长简单。倾向前者。

---

## 4. 整局（overworld）observation —— 为 SlayTheSpireEnv 预留

整局训练时，非战斗屏也要决策（选牌/选路/商店）。这些组与战斗组**共存于同一个 dict**，用 `phase` 指示当前活跃组；非活跃组 PAD。

| 键 | 类型 | 形状 | 内容 |
|---|---|---|---|
| `phase` | categorical | (1,) | PHASE_VOCAB：combat/map/combat_reward/card_select/rest/event/chest/boss_relic/shop/neow |
| 全局信息（§3 那批：character/act/floor/next_boss/relics/potions/gold/deck_counts） | —— | —— | **整局始终有效**，战斗与非战斗共用 |

各屏的“当前选项”（动作相关 obs）：

| 屏 | 键 | 类型 | 形状 | 内容 |
|---|---|---|---|---|
| card_reward | `offer_card_ids` | categorical | (3,) | 三张候选牌 CARD_VOCAB index（已能拿到，见上次 commit） |
| card_select(grid) | `select_card_ids` | categorical | (G_MAX,) | grid 候选牌（转化/升级等），PAD。**G_MAX 须容纳满卡组**：全卡组选牌屏（Smith/Transform/Remove/Toke）候选 = 整副卡组，中后期 30~40+ 张，`G_MAX=12` 会截断 → 模型看不到第 13 张往后。一代正常游玩卡组上限保守取 **64**（二代有「一变二」复制可上千张，但本项目只做一代，64 足够）。截断 = 丢候选 = 必须避免。 |
| map | `map_node_rooms` | categorical | (3,) | 可达下一节点房型 ROOM_VOCAB index（≤3 条边），PAD |
| map | `map_node_x` | scalar | (3,) | 各可达节点列号 /6 |
| shop | `shop_card_ids`/`_prices` (7,) · `shop_relic_ids`/`_prices` (3,) · `shop_potion_ids`/`_prices` (3,) · `shop_remove_cost` (1,) | cat+scalar | —— | 货架 positional（slot i == SHOP_* 买入 action id）。售罄槽（price=-1）→ id PAD + price 0；价格 /999 |
| event | `event_id` | categorical | (1,) | 事件 id 词表（**选项数待 C++**，先只 phase 标记） |
| boss_relic | `boss_relic_ids` | categorical | (3,) | 三个 boss 遗物 RELIC_VOCAB index |

> 这些“选项 obs”与**动作 mask** 配套：模型在 card_reward 看到三张牌 id + 用 mask 限定只能 take#0/1/2/skip/proceed。动作侧编码（变长语义动作→定长 id+mask）是薄层 SlayTheSpireEnv 的另一份契约，单独文档。

### 4.1 选牌屏的语义 + 进度（select_context / pick_remaining / selected_card_ids）—— ✅ 已落地

`select_card_ids` 只给了“候选是哪些牌”，但模型还缺几类信息才能在选牌屏正确决策。各按其类型编码（§0.1），现已全部进 obs v5：

| 键 | 类型 | 形状 | 内容 / 来源 |
|---|---|---|---|
| `select_context` | **categorical** | (1,) | 这屏选牌**干嘛的**——升级/消耗/复制/转化/移除/从弃牌堆拿回/检索…。一张合并词表 `SELECT_CONTEXT_VOCAB`（战内 `CardSelectTask` 20 项 + 战外 `CardSelectScreenType` 8 项，index 0=PAD，共 29）。✅ **已绑 C++**（`bc.select_info()['context']` / `gc.select_screen_info()['context']`）。为何不能只靠 mask：升级一张 vs 移除一张，**可点的牌集合可能完全相同**，目的却相反（升级挑最值钱、移除挑最废）——语义不进 obs，模型在信息上无法区分这两屏。 |
| `pick_remaining` | scalar | (1,) | **还要选几张** = `pickCount − 已选数`，`/PICK_SCALE=10`。✅ **已绑 C++**。⚠️ 仅**战外有效**：sim 的 `cardSelectInfo.pickCount` 只对战内 MULTI 任务（Gamble/Exhaust-Many）有意义，战内单选任务（Armaments/Dual Wield…）pickCount 留 0（选一张即结束屏，不需进度量）——故战内此字段恒 0，靠 `select_context` 区分屏即可。 |
| `can_pick_zero` / `can_pick_any` | flag scalar | (1,)×2 | 能否合法停手 / 是否任意张。✅ **已绑 C++**（`bc.select_info()`）。主要服务战内多选（下一批的 Gamble/Exhaust-Many accumulate+confirm）；战外恒 False（必须选满 pick_count）。 |
| `selected_card_ids` | **categorical** | (3,) | **本屏已选牌的 id**（CARD_VOCAB index，PAD 到 3）。✅ **纯 Python + C++**：战外读 `gc.select_screen_info()['selected']`，战内恒空。 |

> **关键实现纪律（与文档草案的偏离，已在落地时拍板）**：草案设想 `selected` 是对齐 grid 的 `(G_MAX,) flag`，前提“多选途中候选网格不重排”。**绑定时发现该前提对战外不成立**：sim 的 `chooseSelectCardScreenOption(idx)`（`GameContext.cpp:3641`）每选一张就 `toSelectCards.remove(idx)`——**把已选牌从候选网格删除并重排**，自己在 `haveSelectedCards`（`fixed_list<SelectScreenCard,3>`）里跨 env step 累积。
>
> 推论：① grid-aligned flag 在自剪枝网格上**恒为 0**（已选牌已不在网格里），是死字段；② “网格在缩小”本身已是前馈模型的记忆载体（配合 `pick_remaining`，满足 §0.5）；③ 故 `selected` 改为**直接读 `haveSelectedCards` 的已选牌 id**（`selected_card_ids(3)`，上限 3 = 该 `fixed_list` 容量，正好覆盖最大战外多选 Astrolabe=3）。重复牌歧义不再是问题——已选牌根本不在候选里，无需 positional 对齐。
>
> **战内为何不累积**：战内单选（SINGLE_CARD_SELECT）一步选完；战内多选（Gamble/Exhaust-Many 的 MULTI_CARD_SELECT）是**一次性 bitset** 催收整屏，sim 不走 `haveSelectedCards`。故战内 `selected_card_ids` 恒空、`pick_remaining` 恒 0。战内多选“选哪几张”（薄层 accumulate+confirm 子状态机 + CONFIRM 动作 id + C++ MULTI bitset 执行）是**下一批**，不在本契约内。
>
> 这套与 §0.5 呼应：**mask 把已选/非法牌设非法（防重选/防越界），`selected_card_ids`/`pick_remaining`/`select_context` 进 obs（让前馈模型看见自己选了啥、还差几张、这屏干嘛）**——两条通道各管一半。

---

## 5. 维度小结（预估，待词表定稿后精确）

| 组 | 量级 |
|---|---|
| 牌堆/卡组计数 4×371 | ~1484 |
| 怪物 5×(3 cat + 6 scalar + ~50 buff) | ~295 |
| 手牌 10 id + 10 up | 20 |
| buff/全局/选项 | ~300 |
| **合计（扁平化后）** | **~2100** |

> 注意：这是“摊平后”的量级参考。实际 obs 是 dict，categorical 组以 int index 存（小），模型 embed 后才膨胀。真正进网络的维度由模型的 embedding dim 决定，不是这里。

---

## 6. 落地顺序（审定后执行）

1. **纯 Python 可补**（不碰 sim）：
   - 建 `monster_vocab.py` / `relic_vocab.py` / `potion_vocab.py` / 扩 buff 词表 / `room_vocab` / `phase_vocab` / `character_vocab`。
   - canonical 扩展：combat dict 加 `character/act/floor/next_boss/relics/potions/gold/deck_counts`；怪物固定槽位 + present。
   - 重写 `encode.py` → 输出 dict obs（v2），`OBS_VERSION=2`。
   - 更新 `SimCombatEnv` / `SimGameEnv` 的 observation_space 为 `spaces.Dict`。
2. **C++ 绑定补丁清单**（碰 sim，重建 .so）：
   - ✅ **意图**：已导出 `monster.moveHistory[0]`（当前 MonsterMoveId，草案误写 `[1]`）→ `MOVE_VOCAB` + 派生 `INTENT_CATEGORY`（obs_v2_landing §2，已落地）。
   - ✅ 商店货架内容（卡/遗物/药水 + 价格）list getter（`gc.shop_contents()`，obs v4）。
   - ✅ 事件合法选项数（`gc.legal_event_options()`）。
   - ✅ 选牌屏语义/进度（`bc.select_info()` / `gc.select_screen_info()` → select_context/pick_remaining/selected，obs v5，§4.1）。
   - （可选）战斗内遗物计数器 `Relic.data`。

---

## 7. 决策点（已拍板，记录在案）

| | 议题 | 结论 | 落地 |
|---|---|---|---|
| **A** | 手牌升级编码 | `hand_upgrades` **单列 scalar**（词表不翻倍） | `encode.py` `hand_upgrades (10,)`，`up / UPGRADE_SCALE` |
| **B** | 遗物编码 | **多热 id 列表，PAD 到 R_MAX=30**（适合 attention，非 0/1 长向量） | `relics (R_MAX,)` categorical |
| **C** | obs 容器 | **`spaces.Dict` + `MultiInputPolicy`** | `obs_space()` 返回 `spaces.Dict` |
| **D** | 意图绑定时机 | **改为现在就绑**（草案原是「先 PAD 训一版」，落地时反转——远程编译机可用，不留 PAD 债）：`monster_move` 由 C++ 导出真实 move_id（`moveHistory[0]`）填入；`monster_intent_cat` 仍由纯 Python 的 `intents.derive` 派生 | obs_v2_landing §1/§2，已落地 |
| **E** | 怪物槽位 N | **固定 5**（act1-3 普通战足够；boss 单体）。扩容 = bump OBS_VERSION | `MAX_MONSTERS=5` |

> A/B/C/D/E 均已实现。`monster_intent_cat` 由 `intents.derive(move, base_damage)` 派生（不需 C++），给出 attack/defend/buff/debuff 粗类别；`monster_move` 现已带真实 move_id，提供细粒度招式身份（同为 attack 的不同招式）。

---

## 8. 实现对照（`encode.py` 键 → 本文档）

obs 是单层 `spaces.Dict`（无嵌套），共 41 个键。下表是 `obs_space()` 的权威清单，便于审计 layout 漂移：

| 键 | kind | 形状 | 词表/scale | 文档 |
|---|---|---|---|---|
| `player_scalars` | scalar | (6,) | hp/200,maxhp/200,block/100,energy/10,turn/30,hp_frac | §2.1 |
| `player_buffs` | scalar | (PLAYER_BUFF_VEC,) | /BUFF_SCALE | §2.1 |
| `monster_id` | cat | (5,) | MONSTER_VOCAB | §2.2 |
| `monster_move` | cat | (5,) | MOVE_VOCAB（真实 move_id） | §2.2 / D |
| `monster_intent_cat` | cat | (5,) | INTENT_CATEGORY | §2.2 |
| `monster_scalars` | scalar | (5,6) | present,hp/200,hp_frac,block/100,dmg/100,hits/10 | §2.2 |
| `monster_buffs` | scalar | (5,MONSTER_BUFF_VEC) | /BUFF_SCALE | §2.2 |
| `hand_ids` | cat | (10,) | CARD_VOCAB | §2.3 |
| `hand_upgrades` | scalar | (10,) | /UPGRADE_SCALE | §2.3 / A |
| `draw_counts`/`discard_counts`/`exhaust_counts` | scalar | (N_CARDS,) | 计数 | §2.3 |
| `phase` | cat | (1,) | PHASE_VOCAB | §4 |
| `arg_step` | cat | (1,) | ARG_STEP_VOCAB（PAD=root） | action_design §1 |
| `character` | cat | (1,) | CHARACTER_VOCAB | §3 |
| `ascension`/`act`/`floor`/`gold` | scalar | (1,) | /20,/4,/56,/999 | §3 |
| `next_boss` | cat | (1,) | BOSS_VOCAB | §3 |
| `relics` | cat | (30,) | RELIC_VOCAB | §3 / B |
| `potions` | cat | (5,) | POTION_VOCAB | §3 |
| `deck_counts` | scalar | (N_CARDS,) | 计数 | §3 |
| `offer_card_ids` | cat | (3,) | CARD_VOCAB | §4 |
| `select_card_ids` | cat | (G_MAX=64,) | CARD_VOCAB | §4 |
| `select_context` | cat | (1,) | SELECT_CONTEXT_VOCAB | §4.1 |
| `pick_remaining` | scalar | (1,) | /PICK_SCALE=10（战外有效） | §4.1 |
| `can_pick_zero`/`can_pick_any` | flag | (1,)×2 | 0/1 | §4.1 |
| `selected_card_ids` | cat | (3,) | CARD_VOCAB（已选牌，战外） | §4.1 |
| `map_node_rooms` | cat | (3,) | ROOM_VOCAB | §4 |
| `map_node_x` | scalar | (3,) | /6 | §4 |
| `boss_relic_ids` | cat | (3,) | RELIC_VOCAB | §4 |
| `event_id` | cat | (1,) | EVENT_VOCAB | §4 |
| `shop_card_ids` | cat | (7,) | CARD_VOCAB | §4 |
| `shop_card_prices` | scalar | (7,) | /999 | §4 |
| `shop_relic_ids` | cat | (3,) | RELIC_VOCAB | §4 |
| `shop_relic_prices` | scalar | (3,) | /999 | §4 |
| `shop_potion_ids` | cat | (3,) | POTION_VOCAB | §4 |
| `shop_potion_prices` | scalar | (3,) | /999 | §4 |
| `shop_remove_cost` | scalar | (1,) | /999 | §4 |

> **缺口（与动作面共享，详见 action_design）**：(1) ~~`select_context` 缺~~ ✅ 已落地（v5，§4.1）：选牌屏语义/进度（select_context/pick_remaining/can_pick_*/selected_card_ids）全部进 obs，战内单选 + 战外多选覆盖。**仅剩战内多选（Gamble/Exhaust-Many）“选哪几张”** —— 一次性 bitset，需薄层 accumulate+confirm 子状态机 + CONFIRM 动作 id + C++ MULTI bitset 执行，是下一批（动 ACTION_VERSION）。(2) `shop_*` 货架已绑定并进 obs（v4），但商店牌的**升级态**未编码（与 `offer_card_ids` 一致地省略，留后续版本）。(3) `event` 只有 id，选项数/选项语义未绑定。
