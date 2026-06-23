# 卡牌 / Power / 玩家 / 牌组 子系统静态审计 (0623 夜)

审计员：sim 静态审计员（卡牌 / 能力 / 玩家 / 牌组）
范围：`cpp/sts_lightspeed/src/combat/{Actions,Player,CardInstance,CardManager}.cpp`、`src/game/{Card,Deck}.cpp` 及 `BattleContext.cpp` 中的卡牌效果 / power 数值。
方法：只读对照 `reference/cardcrawl/`（真机 Java）与 `reference/sts_lightspeed/`（上游未改 sim）。

> 重要前提说明：本仓库的参考资料在 **workbench 根目录** `../reference/`，不在 `sts-rl/` 内。审计已用该路径完成。
> 几条发现是上游 sim 就有的 bug（local 未改动），但仍与真机 Java 有差异，按要求一并列出并标注来源。

---

## 高严重度

### H1. Iron Wave 的格挡被 `calculateCardBlock` 套了两层

- **sim**：`BattleContext.cpp:1076`
  ```cpp
  addToBot( Actions::GainBlock(calculateCardBlock(calculateCardBlock(up ? 7 : 5))) );
  ```
- **Java**：`reference/cardcrawl/cards/red/IronWave.java:33`（`new GainBlockAction(p, p, this.block)`，`this.block` 由 `applyPowersToBlock()` 算一次）；block 修正链见 `AbstractCard.java:2278`（`modifyBlock` / Dexterity / Frail 只过一遍）。
- **后果**：`calculateCardBlock` 内含 Dexterity 加成与 Frail 的 `*3/4` 取整。套两层会：
  - Dexterity 被加两次（敏捷 +3 时 base5 → 应得 8，实得 11）；
  - Frail 的 `*3/4` floor 被应用两次（base5、Frail → 应得 3，实得 `(5*3/4=3)*3/4=2`）。
  这是直接的格挡数值 diff，且 Iron Wave 很常见。
- **来源**：上游 sim 同样写法（`reference/sts_lightspeed/.../BattleContext.cpp:1076`），属上游 bug，但与真机不一致。
- **建议**：去掉外层调用，改为 `calculateCardBlock(up ? 7 : 5)`。

---

## 中严重度

### M1. Perfected Strike 的 Strike 计数把"消耗堆(exhaust pile)"里的 Strike 也算进去

- **sim**：`BattleContext.cpp:1087-1092` 用 `cards.strikeCount`；该计数在 `CardManager.cpp:258-267` 维护——卡进战斗时 `++`，被移除/消耗(`notifyRemoveFromCombat`)时 `--`。也就是说 **手牌 + 抽牌堆 + 弃牌堆 + 消耗堆** 中的 Strike 都计入（exhaust 之前才会 `--`，留在 exhaust 堆里的不减——需结合 combat-core 确认 exhaust 是否调用 `notifyRemoveFromCombat`；从 `triggerAndMoveToExhaustPile` 看通常**不**减计数）。
- **Java**：`reference/cardcrawl/cards/red/PerfectedStrike.java:29-43` `countCards()` 只遍历 `hand + drawPile + discardPile`，**不含 exhaustPile**。
- **后果**：若有 Strike 卡进了消耗堆（如先用 Strike 再被 Fiend Fire / Second Wind / 其它 exhaust 效果送入消耗堆），sim 的 Perfected Strike 伤害会比真机高，造成伤害 diff。
- **建议**：Perfected Strike 取计数时应排除消耗堆（要么单独遍历三堆，要么维护一个不含 exhaust 的计数）。需与 combat-core agent 协同确认 `strikeCount` 在 exhaust 时的增减语义。

### M2. Perfected Strike 计数包含"正在打出的这张卡自身"的时机风险

- **sim**：`BattleContext.cpp:1088` 注释自承"hack because we calculate strikeCount while non purge cards are still in hand"。
- **Java**：Perfected Strike 自身带 `STRIKE` tag，但 `applyPowers/calculateCardDamage` 执行时卡已离开 hand 进入 limbo（cardInUse），故 **不计自身**。
- **后果**：若 sim 在 `strikeCount` 尚未把当前 Perfected Strike 扣除时计算 base（`6 + strikeCount*(up?3:2)`），会比真机多算 1 张 Strike（多 2/3 伤害）。当前注释暗示作者已用 hack 规避，但属时机相关，**请 combat-core agent 复核**打出顺序里 `notifyRemoveFromHand`/`--strikeCount` 与 `useAttackCard` 的先后。
- **建议**：确认计算 base 前当前卡已不在 `strikeCount` 内。

---

## 低严重度

### L1. `attacked()` 里 Buffer 与 Torii/Tungsten 的相对顺序与 Java 略有出入

- **sim**：`Player.cpp:210-259`，顺序为 block → **Buffer** → Thorns → FlameBarrier → **Torii** → **Tungsten Rod**。
- **Java**：`AbstractPlayer.java:1369-1418`，顺序为 `decrementBlock` → relic `onAttackedToChangeDamage`(**Torii**) → power `onAttackedToChangeDamage`(**Buffer**) → power `onAttacked`(Thorns/FlameBarrier) → relic `onLoseHpLast`(**Tungsten**)。
- **差异**：Java 是 Torii 先于 Buffer；sim 是 Buffer 先于 Torii。仅在"同时持有 Torii + Buffer 且这一击会被消减到 ≤5"这种极端组合下结果可能不同（Buffer 直接吞掉整次伤害，Torii 不触发——两边都让伤害归 0，实际差异通常被掩盖）。Tungsten 相对位置（最后）两边一致。
- **后果**：极罕见 build 才触发，正常对局几乎无 diff。
- **建议**：低优先；若要严格对齐，把 Torii 判定挪到 Buffer 之前。

### L2. AOE + Vigor 卡的 Vigor 是否双重计入（需 combat-core 确认时机，**疑似已被设计规避**）

- **sim**：Cleave/Immolate/Thunderclap/Whirlwind/Reaper/Dramatic Entrance 等在 `BattleContext.cpp:1012/1023/1069/1122/1159/1182` 把 `+ Vigor` **预先加进 baseDamage**；随后 `AttackAllEnemy`/`ReaperAction` 会调用 `calculateCardDamage`，而该函数 `2690-2693` **又加一次 Vigor**。
- **Java**：Vigor 仅在 `atDamageGive` 加一次。
- **关键点**：`onAfterUseCard`(`BattleContext.cpp:1649`) 在卡效果入队后**移除 Vigor**。若被加到 bot 的 `AttackAllEnemy` 在 Vigor 被移除**之后**才执行 `calculateCardDamage`，则计算时 `getStatus<VIGOR>()==0`，只剩预加的那一份 → **正确**。`SwordBoomerang` 行（:1156）的注释明说"vigor is removed afterwards so this is a necessary hack"，强烈暗示这是有意设计。
- **后果**：如果时机如注释所述则无 diff；否则 AOE Vigor 伤害翻倍。
- **建议**：低优先，但请 combat-core agent 确认"`addToBot(AttackAllEnemy)` 的执行 vs `onAfterUseCard` 移除 Vigor"的先后顺序，确保 Vigor 在 AOE 攻击执行时已为 0。

### L3. `Deck::obtainRaw` 对非诅咒卡重复累加 `transformableCount` / `cardTypeCounts`

- **sim**：`Deck.cpp:192-221`。开头 194-195 行先 `++transformableCount; ++cardTypeCounts[type]`，进入 ATTACK/SKILL/POWER 分支后 201-202 行**又各 ++ 一次**。结果非诅咒卡每张被计 2 次。
- **Java**：`AbstractPlayer` 的牌组计数是按实际卡表实时统计，不会重复。
- **后果**：`getTransformableCount()`、按类型计数偏大；影响商店/事件里"可转化/可移除卡数量"的判定路径。**不影响战斗 RNG 与伤害**，replay diff 里大概率不显现。
- **来源**：上游 sim 同样写法（`reference/sts_lightspeed/.../Deck.cpp:192-221`），上游 bug。
- **建议**：低优先；删去 switch 内 ATTACK/SKILL/POWER 分支的重复 `++`（开头已计过一次）。

### L4. `Deck::removeAllMatching` 起始下标越界

- **sim**：`Deck.cpp:318` `for (int i = cards.size(); i >= 0; --i)`，第一次迭代访问 `cards[cards.size()]`（越界读）。
- **后果**：UB；正常情况下若该越界元素恰好不匹配谓词则无害，但属于明显越界。该函数在录像重放主路径中是否被调用需确认，影响面取决于调用点。
- **来源**：上游 sim 同样（注释掉的正确实现就在上方），上游 bug。
- **建议**：低优先；改为 `for (int i = cards.size()-1; i >= 0; --i)`。

---

## 已核对且确认无误（不必再查）

- **damage 计算链顺序**（`BattleContext.cpp:2671-2742`）：Strength/Vigor 加法 → DoubleDamage/PenNib/Weak 乘法 → Stance → Slow → Vulnerable(1.5/Paper Phrog 1.75) → Flight(0.5) → Intangible(max 1)，与 Java `AbstractCard.calculateCardDamage`(`:2298-2330`) 的 atDamageGive→stance→atDamageReceive→atDamageFinal 一致；乘法项彼此可交换，加法项在乘法前，等价。取整 `static_cast<int>`（截断）对已 clamp≥0 的正值等于 `MathUtils.floor`，无 diff。
- **Weak 0.75**：Java `WeakPower.java:58-65` 的 0.6 分支只在"owner 是怪 + 玩家有 Paper Crane"时触发，玩家出牌时 owner 是玩家，恒 0.75，sim `:2702-2704` 正确。
- **Heavy Blade**（`:1054-1058`）：sim 用 `14 + (up?4:2)*Str`，再经 `calculateCardDamage` 内 `+Str` 一次 → 总 `14 + (up?5:3)*Str`，与 Java `HeavyBlade.applyPowers`（临时把 Strength.amount ×magicNumber(3/5)）等价。
- **Searing Blow**（`:1136-1140`）：`n*(n+7)/2 + 12` == Java 累加 `12 + Σ_{k=0}^{n-1}(4+k)`。
- **calculateCardBlock**（`:2744-2759`）：NoBlock→0、Dexterity 加、Frail `*3/4`，与 Java `applyPowersToBlock`(`:2278-2289`) 一致（除 L1 的 Iron Wave 误用外，函数本身正确）。
- **Reaper 回血**（`Actions.cpp:1130-1152`）：`preDamageHp - curHp` 累加实际掉血再 `HealPlayer`，符合真机"按实际造成伤害回血"。
- **player Ritual 时机**：sim `applyEndOfTurnPowers`(`:427-429`) 在回合末加 Strength，与 Java `RitualPower.atEndOfTurn(isPlayer=true)`(`:38-42`) 一致；怪物 Ritual 的 skipFirst（atEndOfRound）属 monster 域。
- **Weak/Vulnerable/Frail 回合末 -1**：`Player::applyAtEndOfRoundPowers`(`:442-488`) 用 `decrementIfNotJustApplied`，对应 Java `justApplied` 跳过首回合。
