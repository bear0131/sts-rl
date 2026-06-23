# bac2 修复报告（2026-06-23 夜）

seed "ABC"=12647, ascension 0。重放命令：
`cpp/.venv/Scripts/python.exe bridge/replay.py bridge/client/logs/orchestrator_bac2.log --natural`

## 一、基线（开工前）

bac2 自然驱动重放卡在 **floor 11**，24 处 diff，最终 `floor=11 hp=39 outcome=UNDECIDED`：

- floor 10 金币：`real=91 sim=61`，Δ30（Looter 偷走的量没归还）。
- floor 11 史莱姆分裂后：`monsters[1].base_damage real=0 sim=8`、`hits real=0 sim=1`（子 slime 立刻拿到攻击意图），随后 RNG 整体错位，到 floor 11 战斗奖励处 `DESYNC`（PROCEED 无法重放）。

## 二、最终结果

bac2 现在能**跑完整段录像（floor 1→18），打穿 act-1 boss（The Guardian），进入 act 2**。
最终 `floor=18 hp=85 outcome=UNDECIDED`（录像本身在 act-2 第一场战斗刚开打时就结束了）。

bac1 回归无影响：仍是原来的 2 处 transient diff，跑到 f16。

剩余 diff（25 处，均为以下三类）：
- **f2 / f7 / f11 / f17 的 transient 时序差异**：和 bac1 同源，单帧不一致、下一帧自动收敛，不影响推进。其中 f11 的 2 处正是“分裂子 slime 意图显示早了一回合”的纯显示差异（见下文 bug 2，战斗 HP/奖励完全对上）。
- **f18 第二场战斗（act-2 遭遇）整体错位**：这是一个**新 bug**（act-2 遭遇生成 RNG 不对），不在本次两个指派范围内，详见“四、新发现但未修”。

## 三、已修 bug

### Bug 1：Looter 偷的金币没归还（floor 10）

**真机行为**：Looter 被击杀（非逃跑）时，偷走的金币作为一个独立的 `STOLEN_GOLD` 奖励项**排在战斗奖励列表最前面**，玩家拾取后金币回来。
真机 floor10 奖励列表：`[STOLEN_GOLD=30, GOLD=20, POTION, CARD]`，且 floor 10 是普通 **MONSTER** 房（不是精英）。

**根因（两个独立的小问题）**：

1. `bridge/replay.py` COMBAT_REWARD 分支只认 `label == "gold"`，**没认 `"stolen_gold"`**。真机 floor10 第一个奖励的 `choice_list[0]` 是 `"stolen_gold"`，replay 直接当成没有匹配的分支跳过，sim 永远没领到那 30 金币。
   - sim 侧其实是对的：`src/combat/BattleContext.cpp:495-508` 的 `exitBattle` 正确算出 `g.info.stolenGold=30`，`src/game/GameContext.cpp:1131` 的 `afterBattle`（curRoom=MONSTER）也把它加进了 reward。问题纯在 replay 没领。
2. sim 的奖励里 stolen gold 被 `addGold` **追加在末尾**（gold[1]），而真机是排在最前（gold[0]）。这会让 `CHOOSE 0` 命中的金币项对不上。

**改法**：
- `bridge/replay.py`：`if label in ("gold", "stolen_gold")` 都走 `claim_gold(0)`。
- `src/game/GameContext.cpp` `afterBattle`：MONSTER（:1131）和 ELITE（:1144，顺手对齐 Mugger 精英场景）两个分支里，把 stolen gold 插到 `reward.gold[0]`、原基础金币挪到 `gold[1]`、`++goldRewardCount`，匹配真机“stolen 在前”的顺序。

**验证**：floor 10 金币 diff 全部消失（41→71→91 完全对上）。

### Bug 2：史莱姆分裂（SPIKE_SLIME_L → 2×SPIKE_SLIME_M）行为错（floor 11）

诊断时发现这其实是**两个**问题叠加。先用真机录像确定了 ground truth（每回合 monster intent）：
- 分裂发生在 turn 4（L 残血变 0 死亡），真机布局是 `[child_M, deadL, child_M]`（死的大 slime 留在 idx1）。
- 两个子 slime 出生时都 roll 到 **LICK**（DEBUFF，dmg=-1），turn5 还是 LICK，turn6 才有一个开始 ATTACK(8)。整场玩家 HP 不掉（39 全程）。

**根因 A — 布局：死的大 slime 被覆盖了。**
`src/combat/MonsterSpecific.cpp:3342` 的 `largeSlimeSplit` 用 `idx1=placeIdx, idx2=placeIdx+1`，直接把原大 slime 那格覆盖成子 slime，monsterCount 只 +1。结果 sim 数组是 `[M, M]`（2 个），真机是 `[M, deadL, M]`（3 格，死的在中间）。bridge 的 `diff()` 会先把 `gone=True` 的怪过滤掉再按下标对齐——sim 因为没保留死 slime，存活子怪的下标和真机错开，全程 diff。
对照参考：同文件 `slimeBossSplit`（:3369）就是把 boss 留在 idx1、子怪放 idx0/idx2、`monsterCount=3`、`monsterTurnIdx=3`——这才是真机布局。

**根因 B — 多消耗了 RNG / 让子怪当回合就行动。**
原 `largeSlimeSplit` 末尾额外调了 `bc.noOpRollMove()`（多推进 1 次 aiRng）并 `extraRollMoveOnTurn.set(idx2)`，且只把 `monsterTurnIdx++`（指到 idx1，导致子怪当回合就 takeTurn）。`slimeBossSplit` 没有这些。这套多余动作让分裂后 aiRng 错位、子怪意图错乱。

**改法**（`MonsterSpecific.cpp:3342` 重写 `largeSlimeSplit`，对齐 `slimeBossSplit`）：
- 把原大 slime 拷到 `placeIdx+1` 并 `curHp=0`（标记死亡，使 `isDeadOrEscaped()`→`gone=True` 被 diff 过滤）；
- 子怪放 `placeIdx` 和 `placeIdx+2`；
- `monsterCount += 2`（min 5），`monsterTurnIdx = idx2 + 1`；
- **删掉** `noOpRollMove()` 和 `extraRollMoveOnTurn.set()`。

**验证**：floor 11 战斗整场 OK，战斗奖励对上，能继续往后跑。剩下的 f11 2 处 diff 是“子 slime 攻击意图比真机早显示一回合”的纯显示差异（base_damage 0 vs 8），属于 transient——战斗 HP、Frail 层数、奖励都完全一致，下一帧收敛。

## 四、修完 floor 11 后陆续暴露并修掉的连锁问题

打通 floor 11 后 bac2 一路往后跑，又暴露出几个独立 bug，按 root-cause→改→重放 的循环逐个修掉：

### Bug 3：Armaments+（全手牌升级）会升级诅咒牌（floor 16 boss 战）
- 现象：f16 起 `CurseOfTheBell` 在 sim 里是 `up:1`、真机 `up:0`，连锁导致整场 boss 战 HP 错位、sim 最终 hp=0 输掉（真机赢了进 f18）。
- 根因：`src/combat/Actions.cpp:901` `UpgradeAllCardsInHand`（=Blessing of the Forge 药水 / Armaments+ 的全升级）对手牌**无条件** `upgrade()`，没查 `canUpgrade()`；而单张 Armaments（:683）是查的。`canUpgrade()`（`CardInstance.cpp:55`）会排除 CURSE/STATUS。
- 改法：循环里加 `if (bc.cards.hand[i].canUpgrade())` 再升级。

### Bug 4：药水 drink 没有立刻 pump action 队列（floor 16 boss 战末尾）
- 现象：Swift Potion（抽 3 张）用了之后 sim 手牌没变（draw 没发生），boss 战末尾错位。
- 根因：binding 的 `drink_potion`（`bindings/slaythespire.cpp`）只调 `bc.drinkPotion()` 把 `DrawCards(3)` 压进队列，**不排空队列**；而 `search::Action::execute`（`Action.cpp:421`）末尾会 `bc.executeActions()`。
- 改法：`bridge/replay.py` 战斗内药水改走 `bc.execute(sts.Action(sts.ActionType.POTION, slot, target))`（discard 用 target=6），和打牌/结束回合一样会立刻 pump 队列。

### 几个 replay 工具侧的缺口（不是 sim bug，但同样卡住推进）
均在 `bridge/replay.py` / `bridge/schemas/combat_state.py` / `bindings/slaythespire.cpp`：

- **事件「离开」帧被当成又选了一次选项**（floor 13 Big Fish）：真机吃完甜甜圈后多一个「离开」帧，sim 选项已 `regainControl()` 退出事件。改：EVENT 分支里若 sim 已不在 `EVENT_SCREEN` 则视为 no-op。
- **战斗奖励界面的「丢药水」没处理**（floor 11）：真机 `POTION DISCARD 1` 丢掉 Blessing of the Forge 腾位给 LiquidMemories，replay 的 overworld 分支没认 `potion`。改：新增 overworld `potion discard` 处理 + 给 GameContext 加 `discard_potion` / `overworld_potions` binding。
- **越界药水槽**（floor 16 真机发了 `POTION USE 3`，是 mod error）：replay 加了 `slot >= len(potions)` 的 no-op 守卫，对齐真机拒绝行为。
- **boss 宝箱选 relic 没处理**（floor 17 BOSS_REWARD）：replay 缺 `BOSS_REWARD` 分支，导致 Slaver's Collar 没拿到。改：新增分支调 `choose_boss_relic(idx)`。
- **怪物 id 映射缺失**：`_MONSTER_SIM2REAL` 补了 Mugger 及一批 act-2 怪（Spiker/Repulsor/Centurion/Gremlin 系等），让 diff 能渲染。
- **录像尾部 CommunicationMod 调试命令**（help/click/menu）：`run_natural` 里跳过，避免驱动幽灵战斗。

## 五、新发现但未修（留给后续）

**act-2 遭遇生成 RNG 不对（floor 18 第二场战斗）**：
- 真机这场是 `[Looter, Mugger]`，sim 生成了 `[Spiker, Repulsor, ...]`，完全不同的遭遇——sim 选错了 act-2 的 monster encounter。
- 这是录像里最后一场战斗，且 act-2 遭遇池/选择逻辑是全新的一块，超出本次 act-1 两个指派 bug 的范围，没动。f18 这 ~20 处 diff 全部来自此处。
- 下一步方向：对照 `MonsterGroup::createMonsters` / act-2 遭遇池的 RNG 调用，看 sim 进 act 2 时 encounter 选择是否多/少调了 RNG 或用错了池。

## 六、改动文件清单

sim（C++，需重编 `cmake --build sts_lightspeed/build --target slaythespire`）：
- `src/combat/MonsterSpecific.cpp` — 重写 `largeSlimeSplit`（bug 2）
- `src/combat/Actions.cpp` — `UpgradeAllCardsInHand` 加 `canUpgrade()`（bug 3）
- `src/game/GameContext.cpp` — `afterBattle` MONSTER/ELITE 的 stolen gold 排到 gold[0]（bug 1）
- `bindings/slaythespire.cpp` — GameContext 加 `discard_potion` / `overworld_potions`

replay 工具（Python）：
- `bridge/replay.py` — stolen_gold 领取、事件离开帧、overworld 丢药水、越界药水守卫、战斗药水走 execute、BOSS_REWARD relic、尾部调试命令跳过
- `bridge/schemas/combat_state.py` — 补 `_MONSTER_SIM2REAL` 映射

（按要求未 git commit，改动留在工作区。）
