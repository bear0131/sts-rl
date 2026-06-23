# StS 整局流程 / CommunicationMod 屏幕与命令

记录"游戏本身长什么样":有哪些 `screen_type`、每种屏幕合法的命令、选择把状态带到哪。
来源 = 观察 `orchestrator_new.log`(seed ABC=12647, A0, 打到 act1 boss)整局。
用途:写 sim 回放/差分时照此映射,不必再回去翻日志。

## 一局的宏观循环

```
START → Neow(floor0) → [MAP → 房间 → MAP → 房间 → ...] → boss(act 末) → 下一 act
```

地图每个节点是一种房间;选节点(MAP 的 CHOOSE)进入对应房间,房间结束后回到 MAP 选下一个。
节点类型 → 房间屏幕:

| 节点 | 进入后 | 结束 |
|------|--------|------|
| 怪 M | COMBAT → COMBAT_REWARD(金币/药水/卡) | PROCEED 回 MAP |
| 精英 E | COMBAT → COMBAT_REWARD(+遗物) | PROCEED 回 MAP |
| 事件 ? | EVENT(可能套 CARD_REWARD/GRID) | 选完回 MAP |
| 休息 R | REST(休息/打铁) | PROCEED 回 MAP |
| 商店 $ | SHOP_ROOM → SHOP_SCREEN | LEAVE→SHOP_ROOM→PROCEED 回 MAP |
| 宝箱 T | CHEST → COMBAT_REWARD(遗物) | PROCEED 回 MAP |
| boss | COMBAT → ... | act 结束 |

## screen_type 清单 + 合法命令

战斗内 `game_state.combat_state` 非空;`screen_type` 此时常为 `NONE`(战斗主界面)或 `HAND_SELECT`(选牌子流程)。

| screen_type | 含义 | 合法命令 | 语义 |
|-------------|------|----------|------|
| `NONE`+combat | 战斗主界面 | PLAY n [t] / END / POTION | 出牌(手牌 n 是 **1-based**, 目标 t **0-based**)/ 结束回合 |
| `HAND_SELECT` | 战斗内选牌(战吼/猛击头槌等) | CHOOSE i / CONFIRM / POTION | 勾选第 i 张(0-based),CONFIRM 提交 |
| `EVENT` | 事件(含 Neow) | CHOOSE i | 选第 i 个选项;Neow 是多帧:先寒暄(1 选项)再 4 选项奖励 |
| `CARD_REWARD` | 选牌加入卡组 | CHOOSE i / SKIP | 选第 i 张 / 跳过 |
| `COMBAT_REWARD` | 战后结算(金币/药水/卡/遗物多项) | CHOOSE i / PROCEED | 领第 i 项(选卡会跳 CARD_REWARD)/ 全领完 PROCEED |
| `MAP` | 选下一节点 | CHOOSE i | 选下一行第 i 个可达节点 |
| `GRID` | 网格选牌(打铁升级/商店删牌/转化) | CHOOSE i / CONFIRM | 勾第 i 张,CONFIRM 提交 |
| `REST` | 篝火 | CHOOSE i | 选项:0=休息回血 1=打铁升级(→GRID) 等 |
| `SHOP_ROOM` | 商店入口(商人) | CHOOSE 0 / PROCEED | CHOOSE 进购买界面 / PROCEED 离开 |
| `SHOP_SCREEN` | 购买界面 | CHOOSE i / LEAVE | 买第 i 项(删牌项→GRID)/ 离开 |
| `CHEST` | 宝箱 | CHOOSE 0 / PROCEED | 开箱(→COMBAT_REWARD 给遗物) |
| `GAME_OVER` | 死亡或通关 | PROCEED | 结束 |

## 命令格式速记

- `START IRONCLAD <ascension> <seed>`(种子是第 3 个参数;`START IRONCLAD ABC` 会把 ABC 当 ascension 报错)
- `CHOOSE i` — 在当前选择列表选第 i 项(下标见各屏幕)
- `PLAY n [t]` — 手牌 1-based,目标怪 0-based
- `END` / `PROCEED` / `CONFIRM` / `LEAVE` / `SKIP`
- `POTION <use|discard> <slot> [t]` — 药水(本局没摸清参数,暂不管)

## 易错点(回放时要处理)

- **报错命令不改状态**:一条 SEND 后紧跟 `RECV {"ready_for_command":true,"error":...}`(无 game_state)说明该命令被拒,要丢弃。手滑/试参数(gogogo、裸 2、错 target 的 play、potion 试参)都在此列。
- 同一逻辑操作真机可能拆成多帧(Neow 寒暄+选择、确认弹窗),sim 侧往往一个 API 调用搞定 → 不是 1:1 映射。
- 战斗内 `screen_type` 在出完牌结算时会短暂变 `NONE`/空且无 combat_state(瞬态帧),不是战斗结束。
