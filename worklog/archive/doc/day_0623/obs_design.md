# Observation 与双模型契约（OBS v11）

> 实现来源：`env/core/encode.py`、`rl/split_policy.py`、
> `bridge/schemas/combat_state.py`。当前 `OBS_VERSION=11`，不兼容旧 checkpoint。

## 1. 总体结构

环境输出一个类型化的 `spaces.Dict`：

- categorical 字段使用整数词表 index，0 为 PAD；embedding 只存在于模型中。
- scalar 字段使用 `float32`，在 encoder 中按固定 scale 归一化。
- sim 和真机 bridge 先生成相同 canonical dict，再调用同一个纯函数 `encode()`。
- 非当前 phase 的字段保持 PAD/0。模型侧还有严格白名单，不能误读另一分支字段。

整局策略由两个独立 actor-critic specialist 组成：

| 分支 | 独立参数 | 输入范围 |
|---|---|---|
| 战斗 | `CombatEncoder`、pointer/fixed policy head、value MLP/head | 手牌、战斗牌堆、怪物、玩家状态，以及会影响战斗的运行资源 |
| 战斗外 | `OverworldEncoder`、pointer/fixed policy head、value MLP/head | 完整主卡组、当前页面候选、地图/事件/商店和持久运行状态 |

`SharedSemanticTokenizer` 是两个分支唯一共享的实体表示层：

- 卡牌 id embedding、升级版 card-language program、卡牌 projection；
- 遗物 id embedding、计数器和 projection；
- 药水 id embedding 和 projection。

因此同一张牌/遗物/药水在两个模型中有相同 token 语义，但两个 Transformer、policy head 和 value head 完全独立。运行时按 `phase` 只执行对应 specialist；PPO mixed batch 会按 phase 分组执行并恢复原顺序。

默认 `d_model=256`、总层预算 `n_layers=4`，拆成战斗 2 层和战斗外 2 层。战斗预训练默认直接构造 2 层的 combat-only core。

## 2. 容量与归一化

| 常量 | 值 | 含义 |
|---|---:|---|
| `MAX_MONSTERS` | 5 | 怪物固定槽，和 target action 对齐 |
| `HAND_SLOTS` | 10 | 手牌槽 |
| `R_MAX` | 208 | `RELIC_VOCAB_CAPACITY`，足以容纳全部可能持有遗物 |
| `P_MAX` | 5 | 药水槽 |
| `DECK_SLOTS` | 96 | `sts::Deck::MAX_SIZE`，完整主卡组硬容量 |
| `G_MAX` | 96 | 全卡组选牌屏硬容量 |
| `OFFER_SLOTS` | 5 | 战后卡牌奖励槽 |
| `SELECT_PICK_SLOTS` | 3 | 战外多选过程中已选牌槽 |

卡组、遗物和已选牌超过硬容量会抛出 `ValueError`，不允许静默截断。split encoder 会在 forward 时裁掉实体序列尾部 PAD，所以 `R_MAX=208` 不会让普通样本按 208 个遗物做 attention；batch 中最后一个实际持有遗物之后的槽不会进入模型。

主要 scale：HP 200、block 100、energy 10、turn 30、damage 100、buff 30、升级数 5、战斗/遗物计数 10、ascension 20、act 4、floor 56、gold 999。

## 3. 战斗分支

战斗模型不接收 `phase`、`act`、`floor`、`next_boss`、完整主卡组、地图、商店、事件或战外候选字段。

### 3.1 玩家和运行资源

| 字段 | shape | 内容 |
|---|---:|---|
| `player_scalars` | `(6,)` | 当前 HP、max HP、block、energy、turn、HP 比例 |
| `player_buffs` | `(PLAYER_BUFF_VEC,)` | 玩家 powers 稠密向量 |
| `combat_counters` | `(24,)` | 会影响后续战斗结算的内部计数 |
| `character` | `(1,)` | 角色 |
| `ascension` | `(1,)` | 难度 |
| `gold` | `(1,)` | **战斗中的实时金币**，支持 Hand of Greed 等效果 |
| `relics` | `(208,)` | **全部当前持有遗物**，不按“战斗相关”类别筛选 |
| `relic_counters` | `(208,)` | 与遗物槽对齐的持久/实时计数 |
| `potions` | `(5,)` | 战斗中的实时药水槽，喝药后立即更新 |
| `arg_step` / select flags | 小定长 | 定向牌/药水和战内选牌子步骤 |

`combat_counters` 的固定顺序为：

```text
happy_flower, incense_burner, ink_bottle, inserter,
nunchaku, pen_nib, sundial, necronomicon_used,
cards_played, attacks_played, skills_played,
orange_attack, orange_skill, orange_power,
combust_hp_loss, deva_form_energy_per_turn,
echo_form_cards_doubled, panache_counter,
cards_discarded, times_damaged,
bomb_1, bomb_2, bomb_3, last_attack_unblocked_damage
```

sim 从 `BattleContext::player` 导出真实值。百年拼图和蜥蜴尾巴触发后会关闭运行时遗物位，其实时二值 0/1 有效状态覆盖对应 `relic_counters` 槽；不增加 token。真机 CommunicationMod 直接提供的弃牌数、受击数和遗物 counter 会进入 canonical；协议未暴露的内部计数保持 0。

### 3.2 怪物、手牌和战斗牌堆

| 字段 | shape | 内容 |
|---|---:|---|
| `monster_id` / `monster_move` / `monster_intent_cat` | `(5,)` | 怪物、精确招式和粗粒度意图 |
| `monster_scalars` | `(5,6)` | present、HP、HP 比例、block、调整后伤害、hits |
| `monster_buffs` | `(5,MONSTER_BUFF_VEC)` | 怪物 powers |
| `hand_ids` / `hand_upgrades` | `(10,)` | 每个手牌槽的卡 id 与升级数 |
| `hand_costs` / `hand_damage` / `hand_block` | `(10,)` | 当前费用和基础数值 |
| `draw_counts` / `discard_counts` / `exhaust_counts` | `(N_CARDS,)` | 三个战斗牌堆的完整卡种计数 |
| `select_hand_mask` | `(10,)` | 战内多选已经挂起的手牌 |

可选 card-language observation 为 draw/discard 提供紧凑、升级感知的语义序列；exhaust 仍由完整 count vector 经共享卡牌 embedding 聚合。该设计保持战斗牌堆 token 数较小。

## 4. 战斗外分支

战斗外模型不接收手牌、抽牌堆、弃牌堆、消耗堆、怪物、玩家战斗 buff 或 `combat_counters`。

### 4.1 持久运行状态

| 字段 | shape | 内容 |
|---|---:|---|
| `phase` / `arg_step` | `(1,)` | 当前页面与自回归子步骤 |
| `character` | `(1,)` | 角色 |
| `ascension` / `act` / `floor` / `gold` | `(1,)` | 持久进度与金币 |
| `run_hp` | `(3,)` | 当前 HP、max HP、HP 比例，所有战外 phase 有效 |
| `next_boss` | `(1,)` | 当前幕 boss |
| `relics` / `relic_counters` | `(208,)` | 全部持有遗物及计数 |
| `potions` | `(5,)` | 当前药水槽 |
| `deck_ids` | `(96,)` | **完整铺平主卡组，每张实体牌一个 token** |
| `deck_upgrades` | `(96,)` | 每个实体牌槽的升级数 |

`deck_ids/deck_upgrades` 取代旧 `deck_counts`。重复卡牌占不同槽，升级牌和未升级牌可区分；因此 value head 可以识别火堆升级前后的卡组价值。

### 4.2 当前页面候选

战斗外 encoder 只添加当前 phase 的候选 block：

| phase | 字段 |
|---|---|
| `combat_reward` | `offer_card_ids/upgrades`、gold/relic/potion rewards |
| `card_select` | `select_card_ids/upgrades`、`selected_card_ids/upgrades`、选择语义和进度 |
| `map` | 下一节点；可选完整 15x7 map DAG |
| `shop` | 卡/遗物/药水货架 token、升级数和价格、移牌价 |
| `boss_relic` | 三个 boss 遗物 token |
| `event` | event id；可选 Neow/普通事件 action-aligned option token |

选牌、删牌、升级、转化等页面都使用与战斗手牌相同的共享卡牌 tokenizer，再由 `PICK_CARD[k]` pointer 对候选 token 评分。卡牌奖励和商店卡牌也携带升级数。

## 5. 可选 observation

| 开关 | 新增字段 | 版本常量 |
|---|---|---:|
| `global_map_view` | `map_full_rooms/features` | 12 |
| `card_language` | draw/discard/deck 的语义卡序列 | 13 |
| `combat_language` | 玩家/怪物状态与怪物意图 program | 随基础布局校验 |
| `deck_size_observation` | `deck_size` | 14 |
| `neow_observation` | Neow bonus/drawback/features | 随当前 space |
| `event_option_observation` | 普通事件 option ids/features | 随当前 space |

## 6. 不变量

1. 怪物槽不因死亡而重排，slot 始终与 target action 对齐。
2. 所有卡牌候选均携带升级数；完整主卡组不做桶化。
3. 所有持有遗物都进入战斗模型；超过容量报错，不截断。
4. action mask 只约束输出，不替代 `select_context`、进度和已选牌 observation。
5. 战斗和战斗外字段由显式白名单隔离；新增字段必须明确归属。
6. 两分支只有 semantic tokenizer 共享，policy/value 参数不共享。
