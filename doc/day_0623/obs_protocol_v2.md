# Observation 协议 v2（OBS_VERSION = 2）

> ⚠️ **已被取代,勿当现行契约**。现行 obs 是 `OBS_VERSION = 4`、**36 组**(v3 加 `arg_step`,v4 加 `shop_*` 货架组;本文档是 v2 的 28 组快照,二者皆缺)。权威清单见 `doc/day_0623/obs_design.md` §8 与 `env/core/encode.py`。本文件保留作 v2 冻结记录。

RL 模型输入契约的事实参考。源：`env/core/encode.py`（`obs_space()` / `encode()`）、
`env/core/vocab.py`（自动生成）、`env/core/intents.py`、`bridge/schemas/combat_state.py`。

obs 是 gymnasium `spaces.Dict`，共 **28 组**。两类条目：
- **categorical**（`dtype=int64`）：词表 index，`Box(low=0, high=容量-1)`。
- **scalar**（`dtype=float32`）：`Box(low=-inf, high=inf)`，已按下方除数归一。

`encode(canon)` 是纯函数；canon 缺某组时该组填 PAD/0。

---

## 1. 容量常量（`encode.py`）

| 常量 | 值 | 含义 |
|---|---|---|
| `MAX_MONSTERS` | 5 | 怪物固定槽数 |
| `HAND_SLOTS` | 10 | 手牌槽 |
| `N_CARDS` | 371 | `CARD_VOCAB` 长度（牌堆计数向量维度） |
| `R_MAX` | 30 | 持有遗物槽 |
| `P_MAX` | 5 | 药水槽 |
| `G_MAX` | 12 | grid 选牌候选槽 |
| `OFFER_SLOTS` | 3 | 战斗卡奖励 / boss 遗物候选槽 |
| `MAP_EDGES` | 3 | 可达下一节点槽 |
| `PLAYER_SCALARS` | 6 | player_scalars 宽度 |
| `MON_SCALARS` | 6 | 每怪 scalar 宽度 |

## 2. 归一除数（`encode.py`）

| 常量 | 值 | | 常量 | 值 |
|---|---|---|---|---|
| `HP_SCALE` | 200.0 | | `BUFF_SCALE` | 30.0 |
| `BLOCK_SCALE` | 100.0 | | `UPGRADE_SCALE` | 5.0 |
| `ENERGY_SCALE` | 10.0 | | `ASCENSION_SCALE` | 20.0 |
| `TURN_SCALE` | 30.0 | | `ACT_SCALE` | 4.0 |
| `DMG_SCALE` | 100.0 | | `FLOOR_SCALE` | 56.0 |
| `HITS_SCALE` | 10.0 | | `GOLD_SCALE` | 999.0 |
| | | | `MAP_X_SCALE` | 6.0 |

## 3. 词表（`vocab.py`，自动生成）

| 词表 | 当前长度 | 容量(=Box high+1) | index 0 | 备注 |
|---|---|---|---|---|
| `CARD_VOCAB` | 371 | 371 | `EMPTY` | 在 `cards_vocab.py` |
| `MONSTER_VOCAB` | 40 | `MONSTER_VOCAB_CAPACITY`=96 | `PAD` | canonical 怪 id |
| `MOVE_VOCAB` | 197 | `MOVE_VOCAB_CAPACITY`=224 | `PAD` | sim `MonsterMoveId` 枚举名 |
| `RELIC_VOCAB` | 181 | `RELIC_VOCAB_CAPACITY`=208 | `PAD` | canonical 遗物 id |
| `POTION_VOCAB` | 43 | `POTION_VOCAB_CAPACITY`=56 | `PAD` | 药水显示名 |
| `EVENT_VOCAB` | 53 | `EVENT_VOCAB_CAPACITY`=72 | `PAD` | 事件 id |
| `BOSS_VOCAB` | 64 | `BOSS_VOCAB_CAPACITY`=72 | `PAD` | `MonsterEncounter` 枚举名 |
| `PLAYER_BUFF_VOCAB` | 85 | 稠密向量 `PLAYER_BUFF_VEC`=110 | 非 PAD（`Strength`） | 量向量，无 sentinel |
| `MONSTER_BUFF_VOCAB` | 41 | 稠密向量 `MONSTER_BUFF_VEC`=50 | 非 PAD（`Artifact`） | 量向量，无 sentinel |
| `PHASE_VOCAB` | 12 | 12 | `none` | |
| `CHARACTER_VOCAB` | 5 | 5 | `INVALID` | |
| `ROOM_VOCAB` | 10 | 10 | `NONE` | |
| `INTENT_CATEGORY` | 9 | 9 | `unknown` | |

- id 词表 index 0 = `PAD`（缺省/越界 id 编码为 0）。append-only。
- buff 词表无 sentinel：每槽一个 power id，值是层数；缺省=0 层。
- `*_INDEX` 是各词表的 name→index 反查 dict。

固定词表内容：
- `PHASE_VOCAB` = `[none, combat, map, combat_reward, card_select, rest, event, chest, boss_relic, shop, neow, invalid]`
- `CHARACTER_VOCAB` = `[INVALID, IRONCLAD, SILENT, DEFECT, WATCHER]`
- `ROOM_VOCAB` = `[NONE, MONSTER, ELITE, EVENT, REST, SHOP, TREASURE, BOSS, BOSS_TREASURE, INVALID]`
- `INTENT_CATEGORY` = `[unknown, attack, defend, buff, debuff, strong_debuff, sleep, stun, escape]`

---

## 4. obs Dict 字段（全 28 组）

### 4.1 player（战斗）

| 键 | 类型 | 形状 | 内容 |
|---|---|---|---|
| `player_scalars` | scalar | (6,) | `[hp/200, max_hp/200, block/100, energy/10, turn/30, hp/max_hp]`（hp、max_hp 先 clip 到 200） |
| `player_buffs` | scalar | (110,) | 按 `PLAYER_BUFF_VOCAB` 的层数 / 30（有符号） |

### 4.2 monsters（战斗，固定 5 槽，怪 i 恒在槽 i，死怪 present=0）

| 键 | 类型 | 形状 | 内容 |
|---|---|---|---|
| `monster_id` | cat | (5,) | `MONSTER_VOCAB` index |
| `monster_move` | cat | (5,) | `MOVE_VOCAB` index（= sim move_id，缺省 0） |
| `monster_intent_cat` | cat | (5,) | `INTENT_CATEGORY` index（派生，见 §5） |
| `monster_scalars` | scalar | (5, 6) | 每行 `[present, hp/200, hp/max_hp, block/100, base_damage/100, hits/10]`（hp clip 200） |
| `monster_buffs` | scalar | (5, 50) | 每怪按 `MONSTER_BUFF_VOCAB` 的层数 / 30 |

### 4.3 combat_piles（战斗）

| 键 | 类型 | 形状 | 内容 |
|---|---|---|---|
| `hand_ids` | cat | (10,) | `CARD_VOCAB` index，位置敏感（play hand[i]↔槽 i），空槽 0 |
| `hand_upgrades` | scalar | (10,) | 各手牌升级数 / 5 |
| `draw_counts` | scalar | (371,) | 抽牌堆每种卡计数 |
| `discard_counts` | scalar | (371,) | 弃牌堆计数 |
| `exhaust_counts` | scalar | (371,) | 消耗堆计数 |

### 4.4 run-global（每相位都有效）

| 键 | 类型 | 形状 | 内容 |
|---|---|---|---|
| `phase` | cat | (1,) | `PHASE_VOCAB` index |
| `character` | cat | (1,) | `CHARACTER_VOCAB` index |
| `ascension` | scalar | (1,) | ascension / 20 |
| `act` | scalar | (1,) | act / 4 |
| `floor` | scalar | (1,) | floor / 56 |
| `next_boss` | cat | (1,) | `BOSS_VOCAB` index |
| `relics` | cat | (30,) | 持有遗物 `RELIC_VOCAB` index 列表，PAD 到 30，持有顺序 |
| `potions` | cat | (5,) | 药水槽 `POTION_VOCAB` index，空槽/`INVALID`=0 |
| `gold` | scalar | (1,) | gold / 999 |
| `deck_counts` | scalar | (371,) | 整局卡组每种卡计数 |

### 4.5 overworld 决策上下文（按相位填，其余 PAD）

| 键 | 类型 | 形状 | 活跃相位 | 内容 |
|---|---|---|---|---|
| `offer_card_ids` | cat | (3,) | combat_reward | 候选牌 `CARD_VOCAB` index |
| `select_card_ids` | cat | (12,) | card_select | grid 候选牌 `CARD_VOCAB` index |
| `map_node_rooms` | cat | (3,) | map | 可达节点房型 `ROOM_VOCAB` index |
| `map_node_x` | scalar | (3,) | map | 各可达节点列号 / 6 |
| `boss_relic_ids` | cat | (3,) | boss_relic | 三个 boss 遗物 `RELIC_VOCAB` index |
| `event_id` | cat | (1,) | event | 事件 `EVENT_VOCAB` index |

---

## 5. INTENT_CATEGORY 派生规则（`intents.py`）

输入 `(move 枚举名, base_damage)`，按序判定，首中即返回：
1. `base_damage > 0` → `attack`
2. move 名为空 / `INVALID` → `unknown`
3. 否则关键词匹配（对 move 名大写做子串匹配，按下列顺序）：
   - `stun` ← `STUNNED`
   - `sleep` ← `SLEEP` / `ASLEEP` / `DORMANT`
   - `escape` ← `ESCAPE` / `SMOKE_BOMB`
   - `strong_debuff` ← `MEGA_DEBUFF` / `HEX` / `DEBILITATE`
   - `debuff` ← `DEBUFF` / `ENTANGLE` / `SPORES` / `ENFEEBL` / `CONSTRICT` / `SPIT_WEB` / `GLARE` / `MOCK` / `TAUNT` / `SCREAM` / `WITHER` / `PERPLEXING`
   - `defend` ← `DEFEND` / `HARDEN` / `PROTECT` / `FORTIFY` / `DEFENSIVE` / `STANCE`
   - `buff` ← `BUFF` / `BELLOW` / `INCANTATION` / `RITUAL` / `GROW` / `BOOST` / `CHARG` / `RALLY` / `ENCOURAGE` / `PREPARING` / `ACTIVATE` / `REGROW` / `REINCARNATE` / `REBIRTH` / `SPLIT` / `SPAWN` / `SUMMON` / `HEAL` / `HASTE` / `ANGER` / `GLOAT` / `FURY` / `CIRCLE_OF_POWER` / `SQUARE_OF_PROTECTION` / `VENT_STEAM` / `COUNT` / `IT_IS_TIME` / `ROAR` / `DROOL` / `FLY` / `CAW` / `IMPLANT` / `ECHO`
4. 无匹配 → `unknown`

---

## 6. canonical 输入（`encode()` 消费的 dict）

顶层可选键，缺省即 PAD/0：

```
phase     : str（PHASE_VOCAB）
turn      : int
player    : {hp, max_hp, block, energy, powers:{power_id:amt}}
monsters  : [{id, hp, max_hp, block, gone, base_damage, hits, powers:{}, move, move_id}, ...]
hand/draw/discard/exhaust : [{id, up}, ...]
global    : {character, ascension, act, floor, boss, relics:[id...],
             potions:[name...], gold, deck:[(id,up)...], boss_relics:[id...]}
options   : map      -> {nodes:[(x, room_name), ...]}
            combat_reward / card_select -> {cards:[(id, up), ...]}
            event    -> {event: event_id}
```

组装：
- `bridge/schemas/combat_state.py` `normalize_sim(bc.to_dict())` 产战斗块（含每怪 `move`/`move_id`）。
- `normalize_sim_global(gc)` 产 global 块（战斗内外都可调）。
- `SimCombatEnv._obs_canon()` / `SimGameEnv.obs_canonical()` 合并战斗/overworld 块 + global + phase + options。

## 7. 后端说明

- sim 后端：`monster.move`/`move_id` 来自 binding 的 `to_dict`（当前意图 = `moveHistory[0]`）。
- 真机后端：`normalize_real` 的 `move`=`INVALID`、`move_id`=0（真机 move_id 是另一套 id 空间，不喂 `MOVE_VOCAB`），意图退化为靠 base_damage 派生 attack；真机原值存 `real_move_id`。
- 训练 policy：`MultiInputPolicy`。
