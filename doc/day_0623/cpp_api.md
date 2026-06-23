# `slaythespire` Python 接口文档

C++ 模拟器(`sim/sts_lightspeed`)通过 pybind11 导出的 Python 模块 `slaythespire`。
**这是把 sim 当黑盒使用时的完整接口面** —— 凡是 Python 侧能调的,都在这里。

来源:`sim/sts_lightspeed/bindings/slaythespire.cpp`(逐 `.def` 抄录)。若绑定改了,以那个文件为准。

> **约定**:C++ 侧一律导出 **sim 自己的 enum 名串**(卡:`STRIKE_RED`;遭遇战:`SMALL_SLIMES`)或**游戏稳定 id**(遗物:`Art of War`)。映射到真机 CommunicationMod id 是在 Python 侧 `bridge/schemas/combat_state.py` 做的,绑定层故意保持最小。

---

## 加载模块

`.pyd`/`.so` 和依赖 dll 都在 `sim/sts_lightspeed/build/` 旁边。

```python
import os, sys
_BUILD = ".../sim/sts_lightspeed/build"
os.add_dll_directory(_BUILD)      # Windows/mingw 专用;mac/linux 上无此函数,需改
sys.path.insert(0, _BUILD)
import slaythespire as sts
```

> ⚠️ 当前加载逻辑是 Windows/mingw 写死的(产物 `slaythespire.cp311-win_amd64.pyd`)。在 mac 上需先重新构建出 `.so` 并改掉 `add_dll_directory`。

---

## 模块级函数

| 函数 | 说明 |
|---|---|
| `play()` | 跑控制台版游戏(交互),与差分测试无关。 |
| `get_seed_str(int) -> str` | 数字 seed → seed 字符串。⚠️ **与 docstring 描述相反**(见 DEVNOTES)。 |
| `get_seed_long(str) -> int` | seed 字符串 → 数字 seed。⚠️ 同上,实际行为与 docstring 反。 |
| `getNNInterface() -> NNInterface` | 取 NN 观测接口单例。 |
| `create_battle(gc, encounter_name: str) -> BattleContext` | 从 GameContext + **遭遇战名**强制构造一场战斗。名字必须是 `encounter_names()` 里的串。构造完已 pump 到 `PLAYER_NORMAL`,可直接接受输入。 |
| `encounter_names() -> list[str]` | `create_battle` 可用的全部遭遇战名(已跳过 `INVALID`)。 |

---

## `GameContext` —— 整局(战斗外)状态

### 构造

```python
gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed: int, ascension: int)
```

### 进度推进 / 决策派发

这些方法镜像 GameContext 自己的屏幕状态机,按当前 `screen_state` 调对应的:

| 方法 | 说明 |
|---|---|
| `choose_neow_option(idx)` | 应用第 idx 个 Neow 奖励(0–3)。 |
| `transition_to_map_node(x)` | 进入下一行 column x 的房间(自动推进 floor/room/miscRng)。boss 节点任意 x 均可。 |
| `choose_event_option(i)` | 事件屏选项 i。合法的 i 由 `legal_event_options()` 给出。 |
| `legal_event_options() -> list[int]` | 当前事件的**真实合法选项下标**(绑 `getValidEventSelectBits`)。已正确处理多阶段事件(Golden Idol 二阶段返回 `[2,3,4]` 而非只 0)。少数事件(MATCH_AND_KEEP/BONFIRE_SPIRITS)返回空。 |
| `choose_campfire_option(i)` | 篝火选项 i。固定下标:rest0/smith1/recall2/lift3/toke4/dig5。 |
| `legal_campfire_options() -> list[int]` | 篝火当前**真实可用选项下标**(绑 `getAllActionsInState`,relic/key 门控)。index 0–5 同上,6=skip。 |
| `choose_select_card_option(i)` | CARD_SELECT/grid 屏选第 i 张(配合 `get_select_cards()`)。 |
| `choose_treasure_room(open: bool)` | 宝箱房:开/不开。 |
| `choose_boss_relic(i)` | 选 act-boss 宝箱的第 i 个 boss 遗物。 |

### 卡牌奖励屏

| 方法 | 说明 |
|---|---|
| `get_card_reward() -> list[Card]` | 当前卡牌奖励列表。 |
| `pick_reward_card(card: Card)` | 选牌。⚠️ 吃的是 **`Card` 对象,不是下标**;用 `gc.get_card_reward()[idx]` 取。 |
| `skip_reward_cards()` | 跳过(有 Singing Bowl 时 +2 max_hp)。 |

### 战斗奖励屏(REWARDS)

| 方法 | 说明 |
|---|---|
| `get_reward_info() -> dict` | 当前可领取项,见下方结构。 |
| `claim_gold(idx)` | 领第 idx 份金币。 |
| `claim_relic(idx)` | 领第 idx 个遗物(已处理 Bottled* 这类会开二级屏的遗物,不会吞掉返图 action)。 |
| `claim_potion(idx)` | 领第 idx 瓶药水。 |
| `reward_proceed()` | 离开奖励屏(`regainControl()`)。 |

`get_reward_info()` 返回:
```python
{"gold": [int, ...], "relics": [str, ...], "potions": [str, ...], "card_count": int}
```

### 商店(SHOP_ROOM / SHOP_SCREEN)

| 方法 | 说明 |
|---|---|
| `shop_contents() -> dict` | 当前货架(**positional**:槽 i 即 `shop_buy_*` 的买入下标)。结构见下。已售/空槽 `price == -1`。 |
| `legal_shop_options() -> list[(kind, idx)]` | **真实可买/可用选项**(绑 `getAllShopActions`,gold + 售罄双重门控)。`kind` ∈ `"card"/"relic"/"potion"/"remove"/"skip"`;`idx` 为该 kind 下的槽位(跨 kind 会碰撞,故成对返回)。`"skip"` 恒在。 |
| `shop_buy_card(idx)` | 买第 idx 张牌(idx 是**完整未过滤列表**里的下标,= `shop_contents()["cards"]` 的槽号)。 |
| `shop_buy_relic(idx)` | 买第 idx 个遗物。 |
| `shop_buy_potion(idx)` | 买第 idx 瓶药水。 |
| `shop_remove()` | 用删牌服务(purge)。 |
| `shop_proceed()` | 离开商店(切到 MAP_SCREEN)。 |

`shop_contents()` 返回:
```python
{
  "cards":   [{"id": str, "upgrades": int, "price": int}, ...],   # 7 槽
  "relics":  [{"id": str, "price": int}, ...],                    # 3 槽
  "potions": [{"id": str, "price": int}, ...],                    # 3 槽
  "remove_cost": int,
}
```
> `price == -1` = 该槽已售/无货。id 命名同其余绑定(卡 `cardEnumStrings` / 遗物稳定 id / 药水显示名)。

### 卡组 / 遗物 读写

| 方法 / 属性 | 说明 |
|---|---|
| `deck_ids() -> list[(enum_str, upgrade)]` | 卡组,sim enum 名 + 升级数。 |
| `relic_ids() -> list[str]` | 遗物,**稳定 id 串**(`Art of War`,非显示名)。 |
| `set_deck([(enum_str, upgrade), ...])` | 整体替换卡组(force-sync 用)。 |
| `set_relics([id_str, ...])` | 整体替换遗物(force-sync 用)。 |
| `deck` (属性) | 卡组的 `Card` 对象列表副本。 |
| `obtain_card(card: Card)` | 加一张牌进卡组。 |
| `remove_card(idx)` | 删卡组里第 idx 张(越界打印 error 并忽略)。 |
| `relics` (属性) | `Relic` 对象列表副本。 |
| `get_select_cards() -> list[(enum_str, upgrade)]` | 当前 grid 屏提供的牌(sim 顺序)。 |

### 药水(战斗外)

| 方法 / 属性 | 说明 |
|---|---|
| `overworld_potions` (属性) | 战斗外药水槽名列表(空槽是 `"Potion Slot"` 之类)。 |
| `discard_potion(slot)` | 丢掉第 slot 槽(0-based)的药水。 |

### 可读写字段(`def_readwrite`)

直接 `gc.xxx` 读/写:

```
outcome              GameOutcome
act                  int
floor_num            int
screen_state         ScreenState
seed                 int
cur_map_node_x/y     int
cur_room             ...
boss                 ...
cur_hp / max_hp      int
gold                 int
ascension            int                # obs v2 新增导出
blue_key / green_key / red_key   bool
card_rarity_factor / potion_chance / monster_chance / shop_chance / treasure_chance
shop_remove_count    int
speedrun_pace        ...
note_for_yourself_card  ...
```

### 只读属性(`def_property_readonly`)

```
encounter            当前遭遇战枚举(_encounter_name 反推用)
screen_state_name    屏幕状态名串(如 "...BATTLE" / "...REST_ROOM" / "...EVENT_SCREEN")
cur_event_name       当前事件 id 串
character             角色枚举名串(IRONCLAD/SILENT/DEFECT/WATCHER)  # obs v2 新增
burning_elite_x/y    带火精英(emerald-key elite)的地图坐标;无图时 -1
```

> obs v2 还新增方法 `boss_relic_ids()` → boss 遗物三选的 stable id 列表(读 `info.bossRelics[3]`)。

### 调试用(`_dbg_*`,谨慎)

- `_dbg_set_card_rng(n)` — 把 `cardRng` 重置成 `Random(seed, n)`。
- `_dbg_card_reward()` — 返回 MONSTER 房卡牌奖励 `[(enum_str, upgrade), ...]`。

---

## `BattleContext` —— 一场战斗的状态

由 `create_battle(gc, name)` 构造。构造末尾已调 `executeActions()`,停在 `PLAYER_NORMAL`。

| 方法 / 属性 | 说明 |
|---|---|
| `to_dict() -> dict` | 导出裸战斗状态,见下方结构。 |
| `legal_actions() -> list[Action]` | 当前合法动作(终局返回空)。 |
| `execute(action: Action)` | 施加动作,并**自动 pump 到下一输入态**(不用手动 pump)。 |
| `turn` (属性) | 回合数(**0-based**;真机 1-based)。 |
| `outcome` (属性) | int:`UNDECIDED`/胜/负 的整数值。 |
| `input_state` (属性) | int:`1=PLAYER_NORMAL`,`2=CARD_SELECT`。 |
| `is_battle_over` (属性) | `outcome != UNDECIDED`。 |
| `exit_battle(gc)` | 把战斗结果(hp/gold/deck/rng)写回 gc,胜利时开奖励屏。 |
| `drink_potion(idx, target=0)` | 用第 idx 槽药水。⚠️ **不 pump action 队列**;要立即结算(如 Swift Potion 抽牌)请走 `execute(Action(POTION, slot, target))`。 |
| `discard_potion(idx)` | 丢第 idx 槽药水。 |
| `potions` (属性) | 战斗内药水槽名列表。 |

### `to_dict()` 返回结构

```python
{
  "turn": int,            # 0-based
  "outcome": int,
  "input_state": int,     # 1=PLAYER_NORMAL, 2=CARD_SELECT
  "player": {
    "hp": int, "max_hp": int, "block": int, "energy": int,
    "powers": [(power_name, amount), ...],   # Strength/Dexterity/Focus/Artifact 是独立字段,其余来自 statusMap
  },
  "monsters": [
    {
      "id": str,                 # sim monster enum 串(如 "SPIKE_SLIME_S")
      "hp": int, "max_hp": int, "block": int,
      "gone": bool,              # isDeadOrEscaped()
      "half_dead": bool,
      "base_damage": int,        # getMoveBaseDamage().damage
      "hits": int,               # .attackCount
      "move_id": int,            # 当前意图 = moveHistory[0] 的 MonsterMoveId 枚举值(0=INVALID)
      "move": str,               # 对应 monsterMoveStrings 枚举名(如 "JAW_WORM_CHOMP")
      "powers": [(name, amount), ...],
    }, ...
  ],
  "hand":    [(card_enum, upgrade), ...],   # 定长手牌数组里前 cardsInHand 张
  "draw":    [(card_enum, upgrade), ...],
  "discard": [(card_enum, upgrade), ...],
  "exhaust": [(card_enum, upgrade), ...],
}
```

> **未导出**:药水动作枚举(`legal_actions` 不含药水)。
> **意图已导出**(2026-06-23,obs v2):`move_id`/`move` = 当前回合 rolled move。
> ⚠️ 当前意图在 **`moveHistory[0]`**(`setMove` 把旧的挪到 `[1]`、新的放 `[0]`),不是草案里写的 `[1]`。
> 上游无富信息 intent 枚举(只有 `isMoveAttack`),attack/defend/buff/… 分类在 Python 侧从 move 名派生(`env/core/intents.py`)。

---

## `Action` / `ActionType` —— 战斗动作

```python
sts.Action(ActionType)                # 如 END_TURN
sts.Action(ActionType, src)           # 如 CARD 不需选怪 / SINGLE_CARD_SELECT
sts.Action(ActionType, src, tgt)      # 如 CARD 需选怪
```

`ActionType` 枚举值:`CARD` / `POTION` / `SINGLE_CARD_SELECT` / `MULTI_CARD_SELECT` / `END_TURN`。

只读属性:`action_type` / `source_idx` / `target_idx`。

**下标约定**(见 replay.py):
- `play N [T]` → `Action(CARD, N-1, T)` —— 卡在真机命令里 1-based、sim 0-based;怪 T 两边都 0-based。
- `end` → `Action(END_TURN)`。
- `choose K`(HAND_SELECT)→ `Action(SINGLE_CARD_SELECT, K)`。
- 药水:`Action(POTION, slot, target)`;`target > 5` 表示丢弃(discard)。

---

## `Card`

| 成员 | 说明 |
|---|---|
| `Card(CardId)` | 构造。 |
| `upgrade()` | 升级。 |
| `misc` (读写) | sim 内部值(如 Ritual Dagger 伤害)。 |
| `id` | `CardId` 枚举。 |
| `upgraded` (bool) / `upgrade_count` (int) | 是否升级 / 升级次数(Searing Blow 可 >1)。 |
| `innate` / `transformable` / `upgradable` (bool) | |
| `is_strikeCard` / `is_starter_strike_or_defend` (bool) | |
| `rarity` (CardRarity) / `type` (CardType) | |

---

## `Relic`

读写字段:`id`(`RelicId`)、`data`(int)。

---

## `SpireMap`

```python
m = sts.SpireMap(seed: int, ?, ?, bool)   # init(seed, int, int, bool)
m.get_room_type(x, y) -> Room
m.has_edge(x, y, x2) -> bool
m.get_nn_rep() -> list[int]
repr(m)                                    # 地图字符画
```

---

## `NNInterface` —— RL 观测

| 成员 | 说明 |
|---|---|
| `getObservation(gc) -> array[412]` | 观测向量。 |
| `getObservationMaximums() -> array[412]` | 各维上界。 |
| `observation_space_size` | ⚠️ pybind 2.13 下这个只读 getter 读取会报错;用 `len(getObservation(gc))` 绕过。 |

---

## 枚举一览

下列枚举完整导出(具体值很多,按需查绑定文件):

- **`GameOutcome`**:`UNDECIDED` / `PLAYER_VICTORY` / `PLAYER_LOSS`
- **`ScreenState`**:`INVALID` `EVENT_SCREEN` `REWARDS` `BOSS_RELIC_REWARDS` `CARD_SELECT` `MAP_SCREEN` `TREASURE_ROOM` `REST_ROOM` `SHOP_ROOM` `BATTLE`
- **`CharacterClass`**:`IRONCLAD` `SILENT` `DEFECT` `WATCHER` `INVALID`
- **`Room`**:`SHOP` `REST` `EVENT` `ELITE` `MONSTER` `TREASURE` `BOSS` `BOSS_TREASURE` `NONE` `INVALID`
- **`CardRarity`**:`COMMON` `UNCOMMON` `RARE` `BASIC` `SPECIAL` `CURSE` `INVALID`
- **`CardColor`**:`RED` `GREEN` `PURPLE` `COLORLESS` `CURSE` `INVALID`
- **`CardType`**:`ATTACK` `SKILL` `POWER` `CURSE` `STATUS` `INVALID`
- **`CardId`**:全卡枚举(数百个,见绑定 ~L521 起)。
- **`MonsterEncounter`**:全遭遇战枚举(见绑定 ~L894 起)。
- **`RelicId`**:全遗物枚举(见绑定 ~L960 起)。
- **`ActionType`**:`CARD` `POTION` `SINGLE_CARD_SELECT` `MULTI_CARD_SELECT` `END_TURN`

---

## 已知坑(汇总)

1. `get_seed_str` / `get_seed_long` 的实际行为**与各自 docstring 相反**。
2. `pick_reward_card` 收 **`Card` 对象**,不是下标。
3. `drink_potion` **不 pump** action 队列 → 走 `execute(Action.POTION, ...)`。
4. `NNInterface.observation_space_size` getter 在 pybind 2.13 下读取报错。
5. 未导出:战斗内药水动作枚举。怪物 intent/move_id **已于 obs v2 导出**(当前意图 = `moveHistory[0]`,非 `[1]`)。
6. 加载逻辑目前是 Windows/mingw 写死(`add_dll_directory` + `.pyd`),mac/linux 需调整。
