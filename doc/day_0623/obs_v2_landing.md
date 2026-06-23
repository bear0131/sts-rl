# obs v2 落地记录（2026-06-23）

`doc/day_0623/obs_design.md` 的接口落地完成。本文件记录**实际做了什么**、**拍板的决策**、**C++ 绑定的改动**，以及**怎么验证的**。

---

## 1. 决策点拍板（obs_design.md §7）

| 点 | 决定 | 理由 |
|---|---|---|
| **A** 手牌升级 | `hand_upgrades` 单列 scalar（/UPGRADE_SCALE=5） | 词表不翻倍；升级是数值修饰，scalar 足够 |
| **B** 遗物 | 多热 id 列表 PAD 到 `R_MAX=30` | 紧凑、适合后续 transformer attention |
| **C** obs 容器 | `spaces.Dict` + SB3 `MultiInputPolicy` | 已跑通训练 smoke |
| **D** 意图绑定 | **现在就碰 C++**，导出真实 move_id | 远程编译机可用，战斗 obs 一步到位，不留 PAD 债 |
| **E** 怪物槽位 | 固定 `N=5` | act1 够；固定槽位（present 标记）不漂位 |

---

## 2. C++ 绑定改动（已编译进 .so，远程机）

文件：`sim/sts_lightspeed/bindings/slaythespire.cpp`

1. **怪物意图**（唯一硬依赖）：`rawCombatState` 的每个怪物 dict 增加
   - `move_id`：`m.moveHistory[0]` 的 int（`MonsterMoveId` 枚举值）
   - `move`：对应的 `monsterMoveStrings[...]` 枚举名字符串

   关键修正：obs_design 草案写的是 `moveHistory[1]`，**错了**。`Monster::setMove`
   是 `moveHistory[1]=moveHistory[0]; moveHistory[0]=moveId;`，所以**当前意图在 `[0]`**，
   `[1]` 是上一手。新增 `#include "constants/MonsterMoves.h"`。

2. **全局信息 getter**（让 `normalize_sim_global` 纯 Python 拿全）：
   - `gc.character`（property）→ 角色枚举名字符串（IRONCLAD/…）
   - `gc.ascension`（readwrite）→ 进位等级
   - `gc.boss_relic_ids()`（method）→ boss 遗物三选的 stable id 列表（`info.bossRelics[3]`）

   `act`/`floor_num`/`boss`/`relic_ids()`/`overworld_potions`/`deck_ids()`/`gold` 已有，未动。

> 重建（远程 autodl，见 `[[autodl-remote-env]]`）：
> `cmake --build sts_lightspeed/build --target slaythespire -j4`（Unix Makefiles，~42s）。
> 注意 binding lambda 里用 `pybind11::list` 不要用 `py::`（该作用域下 `py` 有歧义）。

意图**无富信息枚举**：上游只有 `isMoveAttack`，没有 attack/defend/buff 分类枚举。
故 `INTENT_CATEGORY` 在 Python 侧从 move 名 + base_damage 派生（见 `env/core/intents.py`）。

---

## 3. Python 侧新增/改动

**新增**
- `env/core/_gen_vocab.py`：词表生成器，解析 sim 头文件 + 复用 bridge 归一映射。
- `env/core/vocab.py`（**自动生成**，勿手改）：全部 categorical 词表 + `*_INDEX` + `*_CAPACITY` + `*_BUFF_VEC`。
- `env/core/intents.py`：move 名 → `INTENT_CATEGORY` 的纯函数派生。

**重写**
- `env/core/encode.py`：v1 扁平 Box → **v2 `spaces.Dict`**，`OBS_VERSION=2`。
  - categorical 组 = int index 数组（模型侧 embed），scalar 组 = float32。
  - `obs_space()` 给出 gym Dict；`encode(canon)` 纯函数，缺组自动 PAD。
  - 怪物**固定槽位**：怪 i 恒在槽 i，死了 `present=0`，不滤不漂。

**扩展**
- `bridge/schemas/combat_state.py`：
  - `_norm_sim_monster` 带上 `move`/`move_id`（diff 的 `_MONSTER_CMP_FIELDS` 不含它们，差分不受影响）。
  - 新增 `normalize_sim_global(gc)`：全局信息块（character/ascension/act/floor/boss/relics/potions/gold/deck/boss_relics），战斗内外都有效。
  - `normalize_real` 的怪物：`move`/`move_id` 填 PAD（**真机 move_id 是另一套 id 空间，不能直接喂 sim 的 MOVE_VOCAB**），真机原值存 `real_move_id` 备查。
- `env/core/combat_env.py` / `game_env.py`：`observation_space` 接 `encode.obs_space()`；
  组装 obs canonical（combat + global + phase + options）；新增 `SimGameEnv.obs()` / `obs_canonical()` / `observation_space()`。
- `rl/train.py`：`MlpPolicy` → `MultiInputPolicy`（Dict obs）。

---

## 4. 维度（实测）

词表长度（含 PAD）：monsters 40 / moves 197 / relics 181 / potions 43 / events 53 / boss 64；
buff 向量 player 85（预留 110）/ monster 41（预留 50）。obs Dict 共 **28 组**。

---

## 5. 验证（远程真 sim）

- `python -m env.scripts.test_pure`：词表不变量 / 意图派生 / obs-空间一致 / 固定槽位不漂 / 全局+选项，全过。
- SimCombatEnv：reset+step，`space.contains(obs)` 每步成立；意图 `JAW_WORM_CHOMP`→attack 正确。
- SimGameEnv：**12 个 seed × 共 1346 步，0 个非法 obs**，覆盖 combat/combat_reward/map/rest/event/chest/shop/card_select 全相位。
- `rl.train --overfit --steps 2048`：MaskablePPO + MultiInputPolicy 整环跑通、保存、评估（overfit Jaw Worm mean_reward≈0.88）。

---

## 6. 已知后续（不在本次范围）

- **模型侧 embedding**：SB3 默认 extractor 把 int Box 当连续值；categorical id 组需要自定义
  features-extractor 做 embedding + attention（obs_design §0.1 的“embedding 是模型第一层的事”）。这是下一步模型侧工作，env 契约已就位。
- **真机意图映射**：真机推理后端接入时，需建 CommMod intent/move → sim MonsterMoveId 的映射，
  现 `normalize_real` 的 move 填 PAD、靠 base_damage 派生 attack。
- **商店货架 / 事件合法选项数**：仍是 stub（obs 的 `shop_*` 未建、`event_id` 只标 phase+事件 id），
  需要时再补 C++ 货架 getter（obs_design §6 第二批）。
