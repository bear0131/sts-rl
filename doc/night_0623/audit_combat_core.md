# 战斗核心 / 回合流程 子系统 静态审计报告

审计日期：2026-06-23
审计范围：`cpp/sts_lightspeed/` 下的 BattleContext、CardQueue/ActionQueue、CardManager（洗牌/抽牌部分）。
参考真机：`reference/cardcrawl/`（反编译 Java）。
参考上游：`reference/sts_lightspeed/`（未修改的 gamerpuppy/sts_lightspeed）。

---

## 0. 最重要的结论（先看这条）

**我们 vendored 的整个战斗核心与上游 sts_lightspeed 逐字节一致，没有本地改动。**

逐文件 diff 结果（`reference/sts_lightspeed` vs `cpp/sts_lightspeed`）：

| 文件 | 结果 |
|---|---|
| `src/combat/BattleContext.cpp` | **完全相同** |
| `src/combat/CardManager.cpp` | **完全相同** |
| `src/combat/CardQueue.cpp` | **完全相同** |
| `src/combat/Actions.cpp` | **完全相同** |
| `include/combat/*`（全部头文件） | **完全相同** |
| `include/game/Random.h` | **完全相同** |
| `src/combat/MonsterGroup.cpp` | 有差异（怪物相关，**不在本子系统范围**，归另一个 agent） |

含义：

1. 本子系统**没有引入任何本地回归**。已知的 4 个已修 bug 全部落在怪物/encounter/奖励路径（MonsterGroup、GameContext、商店），不在这里。
2. 因此本报告里的"divergence"全部是 **上游 sim 本身相对真机 Java 的固有差异**，不是我们改坏的。这类差异要么上游早已验证过 RNG 对齐（只是注释里 todo 没删），要么是上游遗留的真实问题。
3. **洗牌/抽牌的 RNG 调用全部正确**（详见下文），这是本子系统最关键的对齐点，结论是 PASS。

下面按"对 RNG 状态的影响"和严重程度排序列出我能从静态对比中确认的点。

---

## 1. 洗牌算法与 RNG 调用次数 —— 全部正确（PASS）

这是审计重点 1/2/5，结论是**完全对齐**，先把验证链条写清楚，方便后续别人复核。

### 1.1 底层 RNG / 洗牌原语

- `java::Random`（`include/game/Random.h:10-47`）精确复刻 `java.util.Random`：`initialScramble`、`next(bits)`、`nextInt(bound)`（含 power-of-2 分支和 rejection 循环）都一字不差。
- `java::Collections::shuffle`（`Random.h:52-58`）：
  ```cpp
  for (int i=size; i>1; i--)
      std::swap(*(begin+i-1), *(begin + rnd.nextInt(i)));
  ```
  这正是 `java.util.Collections.shuffle(List, Random)` 的 Fisher-Yates **倒序**实现（从末尾往前，`nextInt(i)`）。方向、边界、调用次数都对。
- 真机侧 `CardGroup.shuffle(rng)`（`reference/cardcrawl/cards/CardGroup.java:548-550`）：
  ```java
  Collections.shuffle(this.group, new java.util.Random(rng.randomLong()));
  ```
  即"**消耗 1 次 shuffleRng.randomLong() 作为种子，再用 java.util.Random 跑 Collections.shuffle**"。

### 1.2 sim 各处洗牌都遵循"1 次 randomLong + java shuffle"

| 场景 | sim 位置 | 是否对齐 |
|---|---|---|
| 战斗开始初始洗牌 | `CardManager.cpp:27` `java::Collections::shuffle(idxs, java::Random(bc.shuffleRng.randomLong()))` | ✅ 对应 Java `CardGroup.initializeDeck` → `copy.shuffle(shuffleRng)`（`CardGroup.java:914`），各消耗 shuffleRng 1 步 |
| 抽空牌堆触发的重洗（EmptyDeckShuffle） | `Actions.cpp:181-191` 对 discardPile 做 `java::Collections::shuffle(..., java::Random(shuffleRng.randomLong()))` 再 `moveDiscardPileIntoToDrawPile()` | ✅ 对应 `EmptyDeckShuffleAction.update`（`EmptyDeckShuffleAction.java:42`）`discardPile.shuffle(shuffleRng)`，1 步 |
| Deep Breath 的 ShuffleDrawPile | `Actions.cpp:193-201` | ✅ 1 步 shuffleRng |
| Violence 抽牌 | `Actions.cpp:642` `shuffle(attackIdxList...)` | ✅ 1 步 shuffleRng/次 |
| UpgradeRandomCard（Warped Tongs） | `Actions.cpp:953-957` | ✅ 1 步 shuffleRng |

**shuffleRng 种子**：`init` 里 `Random(gc.seed + gc.floorNum)`（`BattleContext.cpp:28-31`），真机 `AbstractDungeon.java:1729` `shuffleRng = new Random(Settings.seed + floorNum)`，一致。

### 1.3 抽牌触发重洗的时机/顺序 —— 对齐

sim `BattleContext::drawCards`（`BattleContext.cpp:2439-2463`）在 `drawPile.size() < amountToDraw` 时：

```cpp
addToTop( DrawCards(temp) );      // 重洗后再抽剩余的
onShuffle();                       // 触发 abacus/sundial 等 onShuffle relic
addToTop( EmptyDeckShuffle() );
if (!drawPile.empty())
    drawCards(drawPile.size());    // 先把当前牌堆抽干（这里是立即递归，不入队）
```

对应 Java `DrawCardAction.update`（`DrawCardAction.java:93-103`）：
```java
addToTop(new DrawCardAction(tmp, ...));   // 最后执行
addToTop(new EmptyDeckShuffleAction());
if (deckSize != 0)
    addToTop(new DrawCardAction(deckSize, false));  // 最先执行：先抽干当前牌堆
```

执行顺序（栈顶优先）两边都是：**先抽干当前 drawPile → EmptyDeckShuffle（消耗 1 次 shuffleRng）→ 抽剩余张数**。sim 把"先抽干当前 drawPile"写成了立即递归 `drawCards()` 而不是入队，但它发生在 `EmptyDeckShuffle` 真正执行之前，与 Java 把 `DrawCardAction(deckSize)` 压在 `EmptyDeckShuffle` 之上、先弹出执行，**效果与 RNG 消耗顺序完全相同**。

`onShuffle` relic 触发（Abacus 加 block、Sundial 计数到 2 给能量）：sim 在"入队 EmptyDeckShuffle 的同时"调用 `onShuffle()`（`BattleContext.cpp:2452`、`2795-2812`）；真机在 `EmptyDeckShuffleAction` **构造函数**里跑 `r.onShuffle()`（`EmptyDeckShuffleAction.java:33-35`），也是入队时机触发。这些 relic 的 onShuffle **都不消耗 RNG**，所以即便细微时机有别也不影响 RNG 对齐。结论：对齐。

> 备注（低）：`onShuffle` 里 Melange relic 的 Scry 被注释掉了（`BattleContext.cpp:2800-2802`），未实现。Melange 的 Scry 不消耗 shuffleRng（它本身不洗牌，只看牌顶），但若 Scry 之后玩家把牌丢回触发其它逻辑，理论上可能影响后续。仅 Melange 持有时才会偏，无该遗物则无影响。

### 1.4 抽牌时消耗 cardRandomRng 的点 —— 对齐

- **Confused（Snecko Eye）**：`CardManager::draw`（`CardManager.cpp:403-411`）对每张 cost>=0 的牌调用 `cardRandomRng.random(3)`，每张恰好 1 次。与真机 SneckoPower.onCardDraw 一致。注意它对 cost<0 的牌（X 费？不，X 费 cost==-1）**不调用**，这点和真机一致（真机也只在 `card.cost >= 0` 时 randomize）。✅
- **Tantrum 重洗回抽牌堆**：`onAfterUseCard`（`BattleContext.cpp:2002`）→ `shuffleIntoDrawPile(cardRandomRng, c)`，内部 `cardRandomRng.random(drawPile.size()-1)`（`CardManager.cpp:215-222`），1 次。对应真机 Tantrum 用 `shuffleCardIntoDrawPile`。✅

---

## 2. 回合开始 / 结束顺序 —— 与真机一致（PASS，含少量已知未实现项）

审计重点 3。逐条核对 Java `AbstractRoom`：

### 2.1 战斗开始顺序

Java `AbstractRoom.update` COMBAT 分支（`AbstractRoom.java:230-247`）：
1. GainEnergy
2. `applyStartOfCombatPreDrawLogic()`（pre-draw relics：Holy Water/Ninja Scroll/Pure Water/Toolbox）
3. `DrawCardAction(gameHandSize)`
4. `applyStartOfCombatLogic()`（at-battle-start relics + powers）
5. `applyStartOfTurnRelics / PostDrawRelics / Powers / Orbs`

sim `initRelics`（`BattleContext.cpp:80-457`）：先收集 `atBattleStartPreDraw` → 执行 pre-draw → `addToBot(DrawCards(cardDrawPerTurn))`（line 391）→ `atBattleStart` → Mercury/RedSkull → `atTurnStartPostDraw`（Gambling Chip / Warped Tongs）。顺序与真机一致。

**与 RNG 相关的关键点**：Toolbox（pre-draw，`ToolboxAction` 用 cardRng/colorless 池）、Gambling Chip（post-draw，`GambleAction`）、Warped Tongs（`UpgradeRandomCardAction` 用 shuffleRng）都在与真机相同的相对位置入队，RNG 消耗顺序对齐。✅

### 2.2 回合结束顺序

Java `AbstractRoom.endTurn`（`AbstractRoom.java:383-413`）：`applyEndOfTurnTriggers()`（玩家 power atEndOfTurn）→ `ClearCardQueueAction` → `DiscardAtEndOfTurnAction` → resetAttributes → (EndTurnAction → MonsterStartTurn)。

sim `onTurnEnding`（`BattleContext.cpp:2107-2119`）：`player.applyEndOfTurnPowers()` → `ClearCardQueue` → `DiscardAtEndOfTurn` → `resetAttributesAtEndOfTurn()` → `UnnamedEndOfTurnAction`。顺序一致。✅

`callEndOfTurnActions`（`BattleContext.cpp:2032-2105`）里 relic OnPlayerEndTurn（CloakClasp/FrozenCore/Nilry/Orichalcum/StoneCalendar）和 power（Metallicize/PlatedArmor/LikeWater/Orbs）均不消耗 RNG，且与真机 `applyEndOfTurnRelics`/`AtEndOfTurnPower` 顺序对应。✅

### 2.3 怪物回合后 / 回合切换

`afterMonsterTurns`（`BattleContext.cpp:2152-2215`）：applyEndOfRoundPowers → ++turn → 玩家 stance/start-of-turn relics/powers → block 清算 → `DrawCards(cardDrawPerTurn)` → DrawReduction 处理 → postDraw relics/powers → 重置 perTurn 计数 → rechargeEnergy。与上游一致，且抽牌动作位置正确（DrawReduction 在抽牌后递减，对应真机）。这里玩家回合开始的抽牌会按 §1.3 的逻辑触发重洗，RNG 正确。✅

---

## 3. 出牌流水线（能量 / X 费 / retain / exhaust）—— 基本对齐，列出可疑点

审计重点 4。主线 `playCardQueueItem → useCard → use{Attack,Skill,Power}Card → onAfterUseCard` 与上游一致。下面是和 RNG 沾边、值得 diff 时盯一眼的点：

### 3.1 【低】X 费 / Whirlwind 能量结算 —— 与 Java 行为对齐但写法绕

`useAttackCard` WHIRLWIND（`BattleContext.cpp:1178-1185`）和 `useSkillCard` TRANSMUTATION（`1463-1472`）的 `energyOnUse`/`ignoreEnergyTotal` 处理逻辑与上游一致。这部分不消耗 RNG，divergence 风险只在"消耗多少能量"的数值上，不影响 RNG 状态。仅在录像 diff 出现能量不符时再查。

### 3.2 【低】Strange Spoon 的 cardRandomRng.randomBoolean —— 时机需关注

`onAfterUseCard`（`BattleContext.cpp:1985-1990`）：
```cpp
if (item.exhaustOnUse && player.hasRelic<R::STRANGE_SPOON>())
    spoonProc = cardRandomRng.randomBoolean();
```
**会消耗 1 次 cardRandomRng**。真机 StrangeSpoon 在 `onUseCard` 时对 exhaust 牌 `cardRandomRng.randomBoolean()`。仅持有 Strange Spoon 且打出 exhaust 牌时触发；调用次数=每张 exhaust 牌 1 次，与真机一致。⚠️ 仅在持有该遗物的录像里需要核对触发条件（注意 `item.exhaustOnUse` 是否在此刻已正确置位——`useCard` line 875 `item.exhaustOnUse |= c.doesExhaust()`，Corruption 把 skill 置 exhaust 在 line 889，都在此之前，OK）。

### 3.3 【低】Mummified Hand 选牌用 cardRandomRng —— 对齐

`mummifiedHandOnUsePower`（`BattleContext.cpp:2847-2876`）：`cardRandomRng.random(0, matchingIdxList.size()-1)`，仅在出 power 牌且手牌有合法目标时 1 次。与真机 MummifiedHand.onUseCard 一致。✅

### 3.4 【低】Dead Branch / Enchiridion 用 cardRandomRng —— 对齐

`triggerAndMoveToExhaustPile`（`2823-2826`）Dead Branch `getTrulyRandomCardInCombat(cardRandomRng, ...)`；`initRelics` Enchiridion（`141`）同。与真机一致。✅

### 3.5 【信息】随机目标用 cardRandomRng —— 对齐

`playCardQueueItem`（`844`）`getRandomMonsterIdx(cardRandomRng)`；Havoc（`1338`）、Distilled Chaos（`2297`）同。真机随机目标走的也是 cardRandomRng（AbstractDungeon.cardRandomRng）。✅

---

## 4. 我没发现"sim 多调/少调一次 RNG"的洗牌/抽牌缺陷

在本子系统范围内（BattleContext / CardManager 洗牌抽牌 / CardQueue / ActionQueue / Actions 洗牌部分），**没有发现**任何"调用次数或顺序与 Java 不同"的 RNG 缺陷。所有 shuffleRng / cardRandomRng 的调用点、次数、相对顺序都能和真机对上。

需要提醒后续维护者的非 RNG 小瑕疵（上游遗留，**不影响 RNG，不属于本审计的对齐目标**，仅记录）：

- 【低·非 RNG】`IMPATIENCE`（`BattleContext.cpp:1341-1352`）：找到 attack 后赋值 `hasAttack = false` 应为 `true`，逻辑反了；导致有 attack 时也会抽牌。上游 bug，但不碰 RNG。
- 【低·非 RNG】`exhaustSpecificCardInHand`（`BattleContext.cpp:2549-2551`）：循环里用 `cards.hand[idx]` 应为 `cards.hand[i]`（idx 写死）。上游 bug，影响 ethereal 牌按 uniqueId 兜底查找，但不碰 RNG。
- 【低·非 RNG】`init` 中 `BLOOD_VIAL`（`237`）标了 `todo not correct`，`Hypnotizing` 等已知项不在本范围。

这些建议丢给"卡牌内容效果"agent 或单列一个非对齐 bug 清单，本报告不深入。

---

## 5. 建议

1. **本子系统无需改动 RNG 相关代码**。洗牌/抽牌/回合流程已与真机对齐，且与上游一致。
2. 录像逐帧 diff 时，如果发现战斗内 RNG 错位，**优先怀疑范围外的文件**：`MonsterGroup.cpp`（本地已改过）、怪物 AI（aiRng）、encounter 初始化（monsterHpRng）、奖励/商店（cardRng/treasureRng）、potion（potionRng）。战斗核心这条链路（shuffleRng / 抽牌 cardRandomRng）基本可以排除。
3. §3.2 Strange Spoon、§1.3 onShuffle(Melange) 这两处虽判定为对齐/无影响，但属于"少见遗物 + 边界条件"，若将来有对应录像，值得专门跑一遍确认。
4. §4 列出的非 RNG 上游 bug（Impatience / exhaustSpecificCardInHand）建议记录到单独的 correctness issue，不要和 RNG 对齐混在一起处理。

---

## 附：审计方法

- 对 `reference/sts_lightspeed`（上游纯净 sim）与 `cpp/sts_lightspeed`（vendored）整目录 `diff -rq`，确认本地改动只有 `MonsterGroup.cpp`。
- 对 `reference/cardcrawl`（真机 Java）逐方法核对：`CardGroup.shuffle/initializeDeck`、`Random.java`、`EmptyDeckShuffleAction`、`ShuffleAction`、`DrawCardAction`、`AbstractRoom.update/endTurn`、`AbstractDungeon.shuffleRng` 初始化。
- 重点验证 Fisher-Yates 方向、`randomLong` 种子消耗次数、抽牌触发重洗的执行顺序。
