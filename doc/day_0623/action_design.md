# Action 设计文档（v2）

> 状态：**已实现并落地**（薄层 `SlayTheSpireEnv`，`env/core/action_space.py` + `env/core/run_env.py`，`ACTION_VERSION=2`）。这是 RL 模型**输出面**的契约，与 `doc/day_0623/obs_design.md`（输入面）配套。
>
> 关联：`env/core/action_space.py`（v2 定长 id 编解码 + autoregressive 子状态机 + mask）、`env/core/run_env.py`（薄层 gym.Env）、`env/core/actions.py`（战斗内 71-id sts↔id 映射，被 action_space 复用）、`env/core/game_actions.py`（语义动作协议，backend 契约）、`env/core/game_env.py`（厚层 `legal_semantic_actions` / `step`）。
>
> 验证：`python -m env.scripts.test_run_v2`（masked-random 整局，断言 obs 在 space 内、mask 非空至终局、autoregressive 目标步不变量）；`python -m env.scripts.play_run_v2`（交互整局）；`python -m rl.train --run`（MaskablePPO 整局训练）。

---

## 0. 设计原则（先读这个）

### 0.1 动作面的两层，和 obs 对称

和 obs 一样分厚/薄两层，**语义动作面是 backend 契约，定长 id 面是模型契约**：

- **厚层（SimGameEnv / 未来 RealGameEnv）** 只说**语义动作** `(verb, *args)`：`(COMBAT, 17)`、`(MAP, 3)`、`(REWARD_CARD, 1)`…（见 `game_actions.py`）。它 `legal_semantic_actions() -> list[tuple]` 枚举当前合法语义动作，`apply(action)` 把语义动作翻成自己后端的调用。**两个后端共用这套词汇**，差异吸收在厚层。
- **薄层（SlayTheSpireEnv）** 把变长语义动作压成**定长 id 面 + mask**，喂给模型。压缩方式由薄层独占，后端不锁死它（`game_actions.py` 顶注释已写明这一点）。

> 本文档定义的就是**薄层的 id 面**。`actions.py`（71 维战斗平面）是它的前身，只覆盖战斗、且是平面笛卡尔积——本文档要把它换掉。

### 0.2 核心病：平面笛卡尔积反结构（你的吐槽）

v1 战斗动作面 `actions.py` 是**平面**的：

```
END_TURN                       1
PLAY card h on monster t       HAND_SLOTS × MAX_MONSTERS = 10×5 = 50
PLAY card h (no target)        HAND_SLOTS = 10
CARD_SELECT pick k             SELECT_SLOTS = 10
                               ────── 71 维
```

“打哪张牌”和“打哪个怪”是两个语义无关的维度，平面化成 `10×5` 乘积有两个毛病：

1. **稀疏 + 不可泛化**：模型学不到“第 3 槽的 Strike”和“第 7 槽的 Strike”是同一回事——它们是 50 个乘积里互不相干的两个 id。
2. **爆维**：一旦出现多参数动作（X 费牌选费量、选两张牌、Gamble 弃任意张），平面就指数膨胀。

### 0.3 解法：autoregressive（分步出动作），不是平面，也不是 MultiDiscrete

模型**分步**决策，每步一个独立的小决策 + 独立 mask：

```
第 1 步：选“打哪张牌 / 结束回合 / 用哪个药水 / …”   ← 在 card/verb 维度上选
第 2 步（仅当上一步选的动作需要参数）：选目标 / 选费量 / 选第二张牌 …
```

- 定向牌：步1 选牌 → 步2 选怪。无定向牌：步1 选牌，无步2。
- “选两张牌放入手牌”：步1 选第一张 → 步2 选第二张（每步独立 mask，已选的置非法）。
- 这正是从一开始就要的“结构化分步”，也是 transformer 的天然 pointer 形态：模型 attend 到一个 card token、再 attend 到一个 monster token。

**为什么不用 MultiDiscrete `(card_slot, target)` 一步出？** target 合法性依赖选了哪张牌（无定向牌没 target、不同牌的合法目标不同），一步出就得做条件式联合 mask，脏且无法干净扩展到 3+ 参数动作。autoregressive 把“条件”变成“下一步的 mask”，是唯一能干净扩展的形态。

---

## 1. 模型怎么知道“这一步在干嘛”——两个 obs 字段（关键）

autoregressive 的合法性 mask 只告诉模型**能选什么**，不告诉它**为什么选**。语义必须进 obs（见 obs_design §0.1：env 给类型化原始状态）。两个新增 categorical 字段：

| 字段 | 词表 | 作用 |
|---|---|---|
| `arg_step` | `ARG_STEP_VOCAB`（固定 8，slot0=`root`） | 当前在 autoregressive 的第几步、正在为哪个 verb 填第几个参数：`root`（步1，选主动作）/`pick_target`（选攻击目标）/`pick_card`（选牌屏单选）/`pick_card_2`/`pick_discard`/`pick_scry`/…（后几项**预留**给 C++ select_context 多选屏）。否则模型在“选目标”那步看到一堆 monster id，不知道自己是在选攻击目标还是别的。**已落地**：纯 Python，薄层 `arg_step_name()` 算，encode v3 加 `arg_step` 键（PAD=root，故单战 SimCombatEnv 也合法）。 |
| `select_context` | `SELECT_TASK_VOCAB`（战内）+ `SELECT_SCREEN_VOCAB`（战外） | 这个**选牌屏的目的**：升级一张(ARMAMENTS)/消耗(EXHAUST)/复制(DUAL_WIELD)/从弃牌堆拿回(HEADBUTT)/Gamble弃/检索(SEEK)…（战内 20 种 CardSelectTask）；转化/升级/移除/获取/装瓶(战外 CardSelectScreenType 6~8 种）。**需 C++ 绑定（§5，未做）**：未绑前 `pick_card` 屏只知“在选牌”不知目的。 |

> `arg_step` 已加进 obs（encode v3，`spaces.Dict` 第 29 个键，§obs_design §8 同步）；`select_context` 仍待绑定。`arg_step` 纯 Python（薄层子状态机自己知道）；`select_context` 需绑定。

---

## 2. 单策略一张大表 + phase-gated mask（不预设几个脑子）

**架构决定：env 永远只暴露 `phase` + 一张按 phase 分区的定长 action 表 + 该 phase 下的 mask。脑子有几个是模型侧的事，env 一行不改。**

这一条同时满足你的两个诉求（统一端到端 vs 专才拼装 baseline）——它们是同一个 env 的不同读法：

- **统一策略**：一个网络吃全 obs、用全表，mask 按 phase 切。
- **专才拼装 baseline**：N 个网络各只看自己 phase 的 obs 子集 + action 子分区（只会打架的 / 只会买东西的 / 只会选牌的），一个 trivial 的 phase-router 按 `phase` 派发。因子分区互不重叠，天然可缝。
- **分层 HRL**：manager 看战外大局选策略，worker 执行。同一套接口。

> **唯一硬约束**：别把“单策略”假设焊死进 env——(1) action id **不跨 phase 复用**（否则无法干净切片成子分区）；(2) reward 别把战内战外揉成不可分的单标量（保留可分项，便于专才各自训）。只要 phase 边界干净、表按 phase 分区，三种读法的门一直开着。

---

## 3. 定长 action 表（薄层 id 面，append-only）—— 已落地

平面 71 维废弃，换成**按 phase 分区的 autoregressive id 面**（`action_space.py` 的 `_SEGMENTS`）。每个 timestep 模型只在“当前 phase × 当前 arg_step”激活的那段 id 里选，其余全 mask 掉。

**实测 layout（`N_ACTIONS=96`，段顺序与宽度已冻结、永不重排）**：

| 段 | base | 宽度 | 用在 | 含义 |
|---|---|---|---|---|
| `END_TURN` | 0 | 1 | combat / ROOT | 结束回合 |
| `PLAY_CARD[h]` | 1 | 10 | combat / ROOT | 打手牌第 h 张（**不含目标**！定向牌目标进步2） |
| `USE_POTION[p]` | 11 | 5 | combat & overworld / ROOT | **预留**：用药水槽 p（厚层暂无喝药语义动作，mask 恒 0；待 C++ #potion 解锁，填预留段不算 bump） |
| `PICK_TARGET[t]` | 16 | 5 | combat / PICK_TARGET | 步2：攻击/指向目标怪 t |
| `PICK_CARD[k]` | 21 | 12 | combat & overworld / PICK_CARD | 步2/选牌屏：在牌堆/grid 选第 k 张（战内 SINGLE_CARD_SELECT、战外 grid 复用同段） |
| `MAP_NODE[x]` | 33 | 7 | map / ROOT | 进下一行第 x 列节点（mask 由 RunMap 可达性给） |
| `REWARD_GOLD[i]` | 40 | 2 | combat_reward / ROOT | 拿金币堆 i（`Rewards.gold` 是 array<int,2>） |
| `REWARD_RELIC[i]` | 42 | 3 | combat_reward / ROOT | 拿遗物 i（array<,3>） |
| `REWARD_POTION[i]` | 45 | 5 | combat_reward / ROOT | 拿药水 i（array<,5>） |
| `REWARD_CARD[j]` | 50 | 5 | combat_reward / ROOT | 拿候选卡 j（`cardRewards` array<,5>） |
| `REWARD_SKIP` | 55 | 1 | combat_reward | 跳过卡奖励 |
| `CAMPFIRE[opt]` | 56 | 6 | rest / ROOT | 营火 rest/smith/recall/lift/toke/dig（厚层暂只暴露 rest/smith，余者预留） |
| `EVENT_OPT[i]` | 62 | 10 | event / ROOT | 事件选项 i（**预留宽 10**；厚层暂只给 option 0，待 C++ #7） |
| `NEOW[i]` | 72 | 4 | neow(event) / ROOT | Neow 四选一 |
| `CHEST[open/skip]` | 76 | 2 | chest / ROOT | 开(+0)/跳过(+1)宝箱 |
| `BOSS_RELIC[i]` | 78 | 3 | boss_relic / ROOT | 选 boss 遗物 i |
| `SHOP_CARD/RELIC/POTION/REMOVE` | 81 | 7+3+3+1 | shop / ROOT | ✅ 货架买卡/遗物/药水/付钱移牌（厚层枚举真实货架，gold/售罄由 sim 门控；filled 预留宽，未 bump `ACTION_VERSION`） |
| `PROCEED` | 95 | 1 | 多屏 | 离开奖励/营火/宝箱屏（shop 屏走 shop_proceed，同一 id） |

> 总宽 96（定长）。每个 id 永久属于一个段（→ 切子分区 = 取若干连续段，干净）。预留段（USE_POTION/EVENT_OPT 尾部）现在就占好位，C++ 绑定批解锁后**填预留宽不 bump `ACTION_VERSION`**（SHOP_* 段即如此解锁：现已 live，`ACTION_VERSION` 仍 2）。
>
> **autoregressive 落点**：战斗里一次“打定向牌打 t 号怪”= 两个 timestep（`PLAY_CARD[h]` → `PICK_TARGET[t]`）。薄层 `run_env.py` 内部维护 `_pending=('play',h)`，第一步 `resolve` 返回 `(None, pending)`、**不调后端、reward 0**，第二步集齐目标才向厚层 `step` 一个完整语义动作 `(COMBAT, encode_play(h,t))`。对厚层、对 sim 完全透明。实测 30 局共 1067 次两步定向出牌，目标步只暴露 `PICK_TARGET` 段 id。

### 3.1 mask 怎么来

`legal_semantic_actions()`（厚层，已实现）枚举当前合法语义动作 → 薄层把每个语义动作映射回它占的 id 段并置 1：

- ROOT 步：把所有“主动作”id 置 1（能打的牌、能用的药、能走的节点、能拿的奖励…）。
- 参数步：根据 `pending` 的 verb，只把该参数的合法取值置 1（如定向牌 → 活着的怪槽；选两张牌的第二步 → 除已选外的候选）。

sb3-contrib `MaskablePPO` 每个 timestep 调 `action_masks()` 拿当前 mask——autoregressive 天然契合（它本就是“每步一个分布 + 每步一个 mask”）。

---

## 4. 落地顺序

1. **薄层重构（纯 Python，不碰 sim）—— ✅ 已完成**：
   - `env/core/action_space.py`（替换平面 `actions.py` 的对外角色，但仍复用其战斗内 sts↔id 映射）：定义上表的 id 分段（`_SEGMENTS`，append-only）、`resolve(id)->(语义动作或None, pending)` 的 autoregressive 子状态机（ROOT→PICK_TARGET）、`legal_ids()`/`arg_step_name()` 从 `legal_semantic_actions()` 构建 mask 与子步。
   - `env/core/run_env.py` 的 `SlayTheSpireEnv`（薄层 gym.Env）：持 SimGameEnv（厚层），`step(id)` 走子状态机，集齐参数才向厚层 `step` 语义动作；`action_masks()` 暴露当前 mask；`_obs()` 给厚层 canonical 注入 `arg_step` 再 `encode`。
   - `ACTION_VERSION=2`、`OBS_VERSION=3`。战斗内沿用已验证的 71-id `combat_env.legal_mask` 路径，只是改由两步 pointer 拼出 `(COMBAT, action_id)`。
   - 验证脚本 `env/scripts/test_run_v2.py`（断言通过）、交互 `play_run_v2.py`；`rl/train.py --run` 接通 MaskablePPO（GPU 实测可学）。
   - 顺手修了厚层 `normalize_sim_global` 的 latent bug：`boss_relic_ids()` 仅在 boss_relic 屏读（否则空指针 RuntimeError）。
2. **战外 obs 补全（纯 Python）**：obs_design §4 已大半就位（map/card_reward/boss_relic/event_id/**shop 货架**）；`arg_step` 已补，event 选项语义（select_context）待 C++。
3. **C++ 绑定批（碰 sim，重建 .so，单独提交）—— 未做**：见 §5。

---

## 5. C++ 待绑清单（与 obs 的绑定批合并，一次重编 .so）

调查记录（`sim/sts_lightspeed/include/...`，均为 POD 字段/枚举，绑定是纯加法零侵入）：

> **进度（本批已落地，重编 .so 已验证）**：#7 事件 + 火堆 recall/lift/toke/dig 已绑。
> 关键发现：sim 自带 `search::GameAction::getAllActionsInState(gc)` / `getValidEventSelectBits(gc)`——
> 一套 relic/gold/phase 感知的**权威战外合法动作枚举器**（战斗侧 `search::Action` 的战外对偶），
> 已正确处理多阶段事件（Golden Idol 二阶段返回 `0b11100` 而非只 option 0，正是旧死循环根因）。
> 故不手抄各事件/火堆合法性（脆弱、复刻 sim 的 `assert(false)` 雷），而是绑定这一个原语，
> 暴露 `gc.legal_event_options()` / `gc.legal_campfire_options()` 两个 getter，厚层直接消费。
> 顺带修了一个被深局暴露的**既有战斗引擎 bug**：大史莱姆在 `placeIdx==2` 分裂时把
> `monsterTurnIdx` 推到 5，`doMonsterTurn` 的 `extraRollMoveOnTurn.test(5)` 越界 `bitset<5>`
> （`MonsterGroup.cpp:583` 加边界守卫修复）。
> 仍待绑：#2/#3/#5 select_context。（#1 move_id、#6 shop 货架已落地，见下表。）

| # | 项 | 位置 | 用途 | 词表 |
|---|---|---|---|---|
| 1 | ✅ **意图 move_id** | `monster.moveHistory[0]`（MonsterMoveId，197 项；草案误写 `[1]`） | obs `monster_move` 细粒度招式身份（obs v2 已导出） | `MOVE_VOCAB` |
| 2 | **战内 select_context** | `bc.cardSelectInfo.cardSelectTask`（`CardSelectTask`，20 项：ARMAMENTS/EXHAUST_ONE/EXHAUST_MANY/GAMBLE/DUAL_WIELD/HEADBUTT/NIGHTMARE/SEEK/…） | obs `select_context`：这屏选牌干嘛 | `SELECT_TASK_VOCAB` |
| 3 | **选几张** | `bc.cardSelectInfo.{pickCount, canPickZero, canPickAnyNumber}` | autoregressive 选牌还要选几张、能否选零/任意 | scalar/flag |
| 4 | ✅ **火堆可用选项** | `gc.legal_campfire_options()`（绑 `getAllActionsInState`） | rest/smith/recall/lift/toke/dig/skip 的真实可用性（relic/key 门控） | — |
| 5 | **战外 select_context** | `gc.info.selectScreenType`（`CardSelectScreenType`，6~8 项：TRANSFORM/UPGRADE/REMOVE/OBTAIN/BOTTLE/BONFIRE_SPIRITS） | obs `select_context`：商店/营火/事件选牌目的 | `SELECT_SCREEN_VOCAB` |
| 6 | ✅ **商店货架** | `Shop`（`include/game/Shop.h`）→ `gc.shop_contents()` + `gc.legal_shop_options()`（绑 `getAllShopActions`） | obs 货架卡/遗物/药水 + 价格（OBS v4）；action SHOP 买卖段（已 live） | `CARD_/RELIC_/POTION_VOCAB` |
| 7 | ✅ **事件选项数 + 语义** | `gc.legal_event_options()`（绑 `getValidEventSelectBits`） | action `EVENT_OPT` 真实合法选项（多阶段事件已解锁） | — |

> #1（move_id）原是战斗 obs 完整性的唯一硬依赖（obs_design D），**已于 obs v2 批导出**；#2/#3/#5 是 autoregressive 选牌的关键（不绑就只能盲选）；#6 商店货架**已落地**（绑定 + 买卖动作 + 货架进 obs v4）。**#1/#4/#6/#7 已完成，仅剩 #2/#3/#5。**

---

## 6. 整局完整度清单（离“真能整局训练”差多远）

> 澄清一个易混点：**obs 整局化已完成**——`SimGameEnv.obs_canonical()` 给 `_canonical()`（按 phase 战斗或战外）拼上 `global` 块喂 `encode`，map/选牌/boss/event 屏全 phase 都吐 obs（obs_design §8）。差的不是 obs，是**动作面**：把厚层 `SimGameEnv`（吐语义 tuple）包成“定长 id + mask”的薄层 `SlayTheSpireEnv` 还没建（§4 第 1 条），而 `rl/train.py` 目前还指着 `SimCombatEnv`（单场战斗）。

逐 phase 的真实完整度（✅ 能训 / ⚠️ 瘸腿 / ❌ stub）：

| phase | obs offer | 动作 | 状态 | 缺口 |
|---|---|---|---|---|
| combat | 全 | 71-id mask（已验证） | ✅ | move_id 已绑（#1，obs v2），intent_cat 另有派生 |
| map | 可达节点房型+列号 | `MAP_NODE[x]`，RunMap 可达性 mask | ✅ | — |
| combat_reward | 三张候选牌 id（**变长**，截 3） | take/skip/proceed | ✅ | — |
| boss_relic | 3 个 boss 遗物 id | `BOSS_RELIC[i]` | ✅ | — |
| chest | — | open/skip | ✅ | — |
| card_select | grid 候选牌 id（**变长**，截 G_MAX=12） | `select(k)` | ⚠️ | **缺 `select_context`**：能选但不知为何选（升级/消耗/转化/复制…）→ C++ #2/#5；缺 `pickCount`（要选几张）→ #3 |
| rest | — | rest/smith/recall/lift/toke/dig/skip | ✅ | 选项可用性由 sim `legal_campfire_options()` 门控（relic/key 感知）；缺 toke/dig 后续选牌屏仍需 select_context（#2/#5） |
| event | event_id（事件身份齐） | `event(i)` 真实合法选项 | ✅ | 由 sim `legal_event_options()` 枚举，多阶段事件已解锁；少数事件(MATCH_AND_KEEP/BONFIRE_SPIRITS)枚举器返回空，薄层无合法动作（不崩，老实停在该屏） |
| shop | 货架卡/遗物/药水 id + 价格（OBS v4，positional 对位买入 id） | `shop_buy(kind,idx)` / `shop_remove()` / `shop_proceed()`，sim 枚举真实货架 | ✅ | 商店牌升级态未编码（与 offer_card_ids 一致省略，后续版本）；移牌后选牌屏仍缺 select_context（#2/#5） |

**可变长 offer 序列**：`combat_reward.cards`（`gc.get_card_reward()`）和 `card_select.cards`（`gc.get_select_cards()`）都是**变长**——canonical 给全长，encode 用定长槽（OFFER_SLOTS=3 / G_MAX=12）PAD/截断装入。其余 offer（map 节点 ≤3、boss 遗物 3、event 选项）也同此纪律。

### 卡点排序（按“离整局训练的阻塞程度”）

1. ~~**event 选项数（#7）**~~ ✅ 已绑：sim `legal_event_options()` 枚举真实合法选项，多阶段事件解锁，旧死循环消除。
2. ~~**商店货架（#6）**~~ ✅ 已落地：`gc.shop_contents()` + `gc.legal_shop_options()`（绑 `getAllShopActions`）→ 厚层枚举真实货架，`shop_buy`/`shop_remove` 语义动词 + codec 接 SHOP_* 段（填预留宽，未 bump `ACTION_VERSION`）+ 货架进 obs v4。顺带修了被商店分流暴露的两个既有缺口：combat `MULTI_CARD_SELECT`（EXHAUST_MANY/GAMBLE）漏 mask、事件战斗（Mushrooms 等）未在 `_settle` 起战。
3. **select_context + pickCount（#2/#3/#5）** — 选牌屏能选但模型瞎选（不知目的、不知选几张）。**当前最高优先。**
4. ~~**火堆 recall/lift/toke/dig**~~ ✅ 已绑：sim `legal_campfire_options()` 给出 relic/key 门控后的真实可用选项（index 0–5 映射 CAMPFIRE_OPTIONS，6=skip→proceed）。
5. ~~**move_id 细粒度（#1）**~~ ✅ 已绑（obs v2 批）：`monster_move` 带真实 move_id；不阻塞整局，intent_cat 另有派生顶上。

> ~~1–3 + 火堆 recall 全在 §5 的同一批 C++ 绑定里~~ → event + 火堆已落地，move_id 已于 obs v2 批落地，**shop(#6) 已落地**（绑定 + 买卖动作 + 货架进 obs v4）。**仅剩 select_context(#2/#3/#5)**。其余（map/combat_reward/boss_relic/chest/event/rest/shop）已 ✅，整局已能端到端跑“能打架+能选路+能拿奖励+能选牌(瞎选)+能玩事件+能用火堆+能逛商店买卖”。`--run` MaskablePPO 训练全程跑通（rollout→learn→eval→save 无崩）。

---

## 7. 已知 stub / 缺口（速查）

- ~~**shop**~~ ✅：`gc.shop_contents()` + `gc.legal_shop_options()` → 厚层枚举真实货架，`shop_buy(kind,idx)`/`shop_remove()`/`shop_proceed()`，codec 接 SHOP_* 段，货架进 obs v4。
- ~~**event**~~ ✅：`gc.legal_event_options()` 枚举真实合法选项（多阶段事件解锁）。
- ~~**campfire**~~ ✅：`gc.legal_campfire_options()` 给 relic/key 门控后的 rest/smith/recall/lift/toke/dig/skip。
- **potion 使用**：v1 动作面无“喝药”。§3 已规划 `USE_POTION` 段，落地随薄层重构。
- **select_context / pickCount**：选牌屏缺“目的”和“选几张”，模型瞎选。需 C++ #2/#3/#5。
