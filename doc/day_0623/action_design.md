# 整局动作契约（ACTION v6）

> 实现来源：`env/core/action_space.py`、`env/core/run_env.py`、
> `env/core/game_env.py`。当前 `ACTION_VERSION=6`、`N_ACTIONS=181`，不兼容旧 checkpoint。

## 1. 两层动作模型

厚层环境使用可读的语义动作，例如：

```text
(COMBAT, combat_action_id)
(USE_POTION, slot, target_or_none)
(MAP, x)
(REWARD_CARD, index)
(SELECT, index)
(SHOP_BUY, kind, index)
```

薄层 `SlayTheSpireEnv` 对外暴露固定 `Discrete(181)` 和 action mask。需要多个参数的动作通过 `pending` 子状态分步收集：

- 定向牌：`PLAY_CARD[h]` -> `PICK_TARGET[t]`；
- 定向药水：`USE_POTION[p]` -> `PICK_TARGET[t]`；
- 战内多选：若干 `PICK_CARD[h]` -> `CONFIRM_SELECT`。

中间步骤不调用厚层 backend，reward 为 0；参数收集完整后才提交一个语义动作。`arg_step`、`select_hand_mask`、选择语义和剩余数量进入 observation，模型能看见当前子步骤。

## 2. 固定动作表

| segment | base | width | 含义 |
|---|---:|---:|---|
| `END_TURN` | 0 | 1 | 结束回合 |
| `PLAY_CARD` | 1 | 10 | 选择手牌槽 |
| `USE_POTION` | 11 | 5 | 使用药水槽 |
| `PICK_TARGET` | 16 | 5 | 选择怪物槽 |
| `PICK_CARD` | 21 | 96 | 战内手牌选择或战外 grid 候选 |
| `MAP_NODE` | 117 | 7 | 选择地图列 x |
| `REWARD_GOLD` | 124 | 2 | 领取金币奖励 |
| `REWARD_RELIC` | 126 | 3 | 领取遗物奖励 |
| `REWARD_POTION` | 129 | 5 | 领取药水奖励 |
| `REWARD_CARD` | 134 | 5 | 领取卡牌奖励 |
| `REWARD_SKIP` | 139 | 1 | 跳过卡牌奖励 |
| `CAMPFIRE` | 140 | 6 | rest/smith/recall/lift/toke/dig |
| `EVENT_OPT` | 146 | 10 | 事件选项 |
| `NEOW` | 156 | 4 | Neow 选项 |
| `CHEST` | 160 | 2 | open/skip |
| `BOSS_RELIC` | 162 | 3 | boss 遗物 |
| `SHOP_CARD` | 165 | 7 | 商店卡牌 |
| `SHOP_RELIC` | 172 | 3 | 商店遗物 |
| `SHOP_POTION` | 175 | 3 | 商店药水 |
| `SHOP_REMOVE` | 178 | 1 | 商店移牌 |
| `PROCEED` | 179 | 1 | 离开/继续 |
| `CONFIRM_SELECT` | 180 | 1 | 提交战内多选 |

`PICK_CARD` 宽度与 `Deck::MAX_SIZE=96` 相同。全卡组删牌、升级、转化页面不会截断；观察或动作候选超过容量会显式报错。

## 3. Pointer head 对齐

双模型各自拥有独立 policy head。候选实体动作使用 action-aligned token pointer：

| 分支 | segment -> token |
|---|---|
| 战斗 | `PLAY_CARD -> hand`、`PICK_TARGET -> monster`、`USE_POTION -> potion`、`PICK_CARD -> hand` |
| 战斗外 | `PICK_CARD -> select`、reward segments -> reward entity、shop segments -> shelf entity、`BOSS_RELIC -> boss_relic`、`NEOW/EVENT_OPT -> option` |

地图不是按候选数组下标输出，而是把可达节点 token 按 x 列聚合到 `MAP_NODE[x]`。没有实体候选的动作由该 specialist 自己的 fixed head 从 `[CLS]` 输出。

这使战外删牌、升级、转化等卡牌选择复用已经验证的战斗手牌选取设计：动作 logit 直接由对应卡牌 token 评分，而不是让固定输出列猜测槽内实体。

## 4. Mask 不变量

1. 每一步至少有一个合法 action；非法动作 logit 在分布前被 mask。
2. target 子步骤只开放当前已选牌/药水可作用的怪物。
3. 战内多选为单调集合：已选手牌不可重选，到上限后只保留 confirm。
4. `PICK_CARD[k]` 的 k 始终和当前 hand/grid slot 对齐。
5. reward、shop、boss relic 等 segment slot 始终和对应 observation token 对齐。
6. mask 不进入网络，因此已选项、选择目的和剩余数量必须同时存在于 observation。

## 5. Phase 路由

动作空间保持统一，但模型按 phase 使用不同 actor-critic：

- `phase=combat`：只执行战斗 specialist；
- 其他 phase：只执行战斗外 specialist，并只 token 化当前页面候选；
- mixed PPO batch：按 phase 拆分，分别 forward 后恢复原 batch 顺序。

因此统一 `Discrete(181)` 只是环境协议，不意味着战斗和战斗外共享 policy/value 模型。
