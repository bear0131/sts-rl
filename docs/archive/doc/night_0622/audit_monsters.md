# 怪物 / AI / 意图子系统静态审计报告（0623 夜）

审计员：sim 静态审计员（怪物 / AI / 意图）
范围（sim 文件，均在 `cpp/sts_lightspeed/`，路径下文简写）：
- `src/combat/MonsterSpecific.cpp`（initHp / preBattleAction / takeTurn / getMoveForRoll / split / 召唤 / 偷窃等）
- `src/combat/Monster.cpp`（基类、死亡、buff、rollMove 框架）
- `src/combat/MonsterGroup.cpp`（怪物组初始化、HP roll、出场顺序、加怪）
- `src/combat/MonsterMoveDamage.cpp`

真机参考（只读）：`reference/cardcrawl/`（注意实际在仓库**上一级**目录 `D:/zxb/my/sts-rl-workbench/reference/`，不是任务描述里的 `sts-rl/reference/`）。

## 重要前置事实（已核对）

1. **本地 sim 与上游 sts_lightspeed 的 diff**：`MonsterSpecific.cpp` / `Monster.cpp` 与上游**逐字相同**，仅 `MonsterGroup.cpp` 有一处本地改动（带火精英 STR 改为 `act+1`，即已知已修 bug #1）。
   → 因此下面所有发现都是**上游 sts_lightspeed 自带的 bug**（即我们要对齐的目标真机 Java 与上游 sim 的差异），不是本地引入的回归。
2. **rollMove 一定消耗 1 次 aiRng**：Java `AbstractMonster.rollMove()` 永远先 `aiRng.random(99)` 再 `getMove(num)`；sim `Monster::rollMove`（Monster.cpp:629）也永远 `bc.aiRng.random(99)`。即便某怪 `getMove` 完全不读 `num`（Cultist / GremlinNob / Hexaghost / Guardian / Lagavulin 等），两边都照样消耗 1 次，**这部分是对齐的**。
3. **怪物 takeTurn 里的 `MathUtils.random(...)`（playSfx / 对话）不是 aiRng**，是未播种的全局 MathUtils，不推进 aiRng/monsterHpRng。sim 对这些怪正确地**没有**额外消耗 aiRng（已逐一核对 GremlinNob.playSfx、Cultist.takeTurn 的 random(1,10)+playSfx、SlimeBoss.playSfx 等）。
4. **monsterData 与 miscInfo 是同一存储**：`rollMove` 把 `miscInfo` 拷进 `miscInfoCopy` 传给 `getMoveForRoll(..., monsterData, ...)` 再写回（Monster.cpp:630-635）。所以 getMoveForRoll 里 `monsterData` 和 `miscInfo` 指向同一个值（TheChamp 用到，不是 bug）。

---

## 发现（按严重程度排序）

### 【高】1. AcidSlime_L 在 A17 的 getMove 写错了 lastTwoMoves 的 move id —— RNG 错位 + 选错意图

- **sim**：`MonsterSpecific.cpp:1982-1992`（`case MonsterId::ACID_SLIME_L` 的 `asc17` 分支，`roll < 40`）
  ```cpp
  if (roll < 40) {
      if (lastTwoMoves(MMID::ACID_SLIME_M_CORROSIVE_SPIT)) {   // ← 写成了 _M_（中史莱姆的 move id）
          if (bc.aiRng.randomBoolean(0.6F)) return ACID_SLIME_L_TACKLE; else return ACID_SLIME_L_LICK;
      } else {
          return ACID_SLIME_L_CORROSIVE_SPIT;
      }
  }
  ```
- **Java**：`reference/cardcrawl/monsters/exordium/AcidSlime_L.java` getMove，A17，`num < 40`：`if (this.lastTwoMoves((byte)1))`（byte 1 = AcidSlime_L 自己的 CORROSIVE_SPIT）→ randomBoolean(0.6f) tackle/lick，否则 CORROSIVE_SPIT。
- **为何 divergence（RNG 状态）**：AcidSlime_L 的 moveHistory 里只会出现 `ACID_SLIME_L_*` 的 move id，永远不会是 `ACID_SLIME_M_CORROSIVE_SPIT`，所以 `lastTwoMoves(ACID_SLIME_M_CORROSIVE_SPIT)` **恒为 false**。后果：A17 下 roll<40 时永远走 else 返回 CORROSIVE_SPIT，**永远不进 randomBoolean(0.6f) 分支** → 该走 randomBoolean 的回合少消耗 1 次 aiRng，且意图被强制成 CORROSIVE_SPIT 而非 tackle/lick。一旦发生即 aiRng 错位，后续全盘错。
- **影响面**：A17 的大酸史莱姆（含 LARGE_SLIME 房、SlimeBoss 分裂出的 ACID_SLIME_L 子代）。bac 若打 A17 act-1 易触发。
- **严重程度：高**（A17 act-1 常见怪 + 直接 RNG 错位）。
- **建议修法**：第 1984 行 `ACID_SLIME_M_CORROSIVE_SPIT` → `ACID_SLIME_L_CORROSIVE_SPIT`。（注意：同一函数 `roll<70`/`lastMove(LICK)` 等其余 L 分支以及非 A17 分支都已正确用 `ACID_SLIME_L_*`，只有 A17 首支错。）

---

### 【中】2. Book of Stabbing 在 A18 漏加 stabCount —— multi-stab 段数偏少（伤害 divergence）

- **sim**：`MonsterSpecific.cpp:2254-2277`（`case MonsterId::BOOK_OF_STABBING`）
  ```cpp
  if (roll < 15) {
      if (lastMove(SINGLE_STAB)) { ++stabCount; return MULTI_STAB; }
      else {
          return SINGLE_STAB;        // ← return 在前
          if (asc18) { ++stabCount; }  // ← 死代码，永远执行不到
      }
  } else if (lastTwoMoves(MULTI_STAB)) {
      return SINGLE_STAB;            // ← return 在前
      if (asc18) { ++stabCount; }     // ← 死代码
  } else { ++stabCount; return MULTI_STAB; }
  ```
- **Java**：`reference/cardcrawl/monsters/city/BookOfStabbing.java` getMove，选 SINGLE_STAB（byte 2）的两个分支里，`if (ascensionLevel >= 18) ++this.stabCount;` 是在 setMove **之后**正常执行的（Java 没有提前 return 截断）。
- **为何 divergence**：`stabCount`（= multi-stab 的攻击次数 / `attackPlayerHelper(..., miscInfo)` 的 times）在 A18 下应在每次出 SINGLE_STAB 时也 +1；sim 因 C++ 把自增写在 `return` 之后变成死代码，A18 时这两处自增丢失 → MULTI_STAB 命中段数比真机少。**非 aiRng 消耗差异**（getMove 在这两支不读多余 rng），但导致**伤害与战斗状态 divergence**。
- **影响面**：A18+ 的 Book of Stabbing（act-2 精英），bac1/bac2 act-1 不涉及。
- **严重程度：中**（伤害错，但仅 A18 且 act-2）。
- **建议修法**：把两处 `if (asc18) ++stabCount;` 移到 `return` 之前（即先自增再 return），与 Java 顺序一致。

---

### 【中】3. Lagavulin 醒来机制建模与真机不同（被打醒/睡眠计数）

- **sim**：`MonsterSpecific.cpp:888-895`（`LAGAVULIN_SLEEP`）+ `Monster.cpp:388-391, 448-452`（被攻击/受伤时 `ASLEEP` → 立即清醒、扣 8 metallicize）。
  - SLEEP 的 takeTurn：`if (bc.turn == 2 || !hasStatus<ASLEEP>()) setMove(ATTACK) else setMove(SLEEP)`。
- **Java**：`reference/cardcrawl/monsters/exordium/Lagavulin.java`
  - 自然清醒：`takeTurn` case 5 用 `idleCount`，`idleCount >= 3` 才 OPEN 并把 nextMove 设成 ATTACK（即第 3 个怪物回合设意图、第 4 回合攻击）。
  - 被打醒：`damage()` 里若 `!isOutTriggered` 且掉血，立刻 `setMove((byte)4, STUN)`（先 STUN 一回合）再 OPEN。
- **核对结论（部分一致）**：纯睡到底不被打的情况，sim `bc.turn==2`（= 第 3 个怪物回合 getMonsterTurnNumber()==3，见 BattleContext.cpp:643 `turn+1`）设 ATTACK、第 4 回合攻击，**与 Java idleCount>=3 的时序一致**。
- **为何仍可能 divergence**：**被提前打醒**的路径，Java 会多插一个 STUN 回合（move 4，跳过攻击），sim 直接清醒后按 SLEEP→ATTACK 调度走、没有等价 STUN 回合。这会让被打醒后的"第一回合是否攻击"对不上（行为差异）。两边在这些回合都走 `noOpRollMove`/不消耗 aiRng，所以**主要是行为/伤害 divergence，不是 aiRng 计数差**，但若 BattleContext 对 STUN 回合有别的 rng 调用仍需注意。
- **严重程度：中**（act-1 精英，bac 打到过；但仅"打醒"分支偏差，纯睡场景对齐）。
- **建议修法**：在受伤清醒分支（Monster.cpp attackedUnblockedHelper / damageUnblockedHelper 的 `ASLEEP` 处理）区分"被打醒"，置一个 STUN 等价的过渡 move，使其下一回合才攻击，对齐 Java `setMove(4, STUN)`。需先确认 sim 是否有 STUN move id 与之对应。

---

### 【低】4. TheChamp turn-4 Taunt 触发条件与真机的 numTurns 语义不同

- **sim**：`MonsterSpecific.cpp:2864`：`else if ((bc.getMonsterTurnNumber()+1) % 4 == 0) return THE_CHAMP_TAUNT;`
- **Java**：`reference/cardcrawl/monsters/city/Champ.java` getMove：用私有 `numTurns`，每次 getMove `++numTurns`；`if (numTurns == 4 && !thresholdReached) { taunt; numTurns = 0; }`。即 Taunt 是"距上次 Taunt/进入二阶段后第 4 次 getMove"，且会把计数**清零**重新计。
- **为何 divergence**：sim 用全局 `getMonsterTurnNumber()` 的 `%4`，不会因 Taunt 而重置，也不随阶段切换重置；当战斗回合数与 Champ 自己的 numTurns 不同步（例如二阶段重置后）时会在不同回合出 Taunt → 选错意图 + aiRng 错位。
- **影响面**：act-3 boss TheChamp，bac1/bac2 act-1 不涉及。
- **严重程度：低**（远离当前 bac 范围；且需特定回合序列才偏差）。
- **建议修法**：用 miscInfo 里额外的位/计数模拟 Champ 自己的 numTurns（getMove 每次自增、出 Taunt 与进二阶段时清零），不要用全局回合数取模。

---

## 已逐项核对、确认对齐（无需改动）的部分

为避免后续重复审计，记录已比对通过的关键点：

- **rollMove 框架 / aiRng 每次 +1**：对齐（见前置事实 2）。
- **怪物 HP roll 顺序与次数（monsterHpRng）**：
  - `Monster::construct`（Monster.cpp:109-135）先 `initHp`（setRandomHp 1 次），再对 GREEN/RED_LOUSE、DARKLING 额外 roll（biteDamage / nip 伤害）——与 Java 构造函数里 `setHp(...)` 后紧跟 `monsterHpRng.random(...)` 的顺序一致。
  - Louse 的 CurlUp（`preBattleAction` 里 `bc.monsterHpRng.random(...)`，MonsterSpecific.cpp:293-309）与 Java `usePreBattleAction` 一致；且 sim init 顺序为"全部 construct → 全部 rollMove → 全部 preBattleAction"，由于 curlUp 用 monsterHpRng、rollMove 用 aiRng（两条独立序列），多 Louse 时 monsterHpRng 序列 = 各怪 (HP,bite) 交替后再各怪 curlUp，与真机一致。
  - 各怪 ascension HP 阈值（>=7 / >=8 / >=9）对照 initHp（MonsterSpecific.cpp:26-128）与 Java 各构造函数 setHp，抽查 GremlinNob(>=8)、Lagavulin(>=8)、Sentry(>=8)、JawWorm(>=7)、AcidSlime_M(>=7)、SlimeBoss(>=9) 全部一致。
  - 带"丢弃首次 roll"的怪（ORB_WALKER / REPTOMANCER / BRONZE_ORB / TASKMASTER，initHp 里先 `hpRng.random(a,b)` 再 setRandomHp）与真机的 setHp 调两次的行为一致。
- **JawWorm getMove**：firstMove→CHOMP，之后三段 + randomBoolean(0.5625 / 0.357 / 0.416)，sim（MonsterSpecific.cpp:2450-2490）与 Java 完全一致。
- **AcidSlime_M getMove**：A17 与非 A17 的 40/80、30/70 阈值与 lastMove/lastTwoMoves 选择、randomBoolean 概率，sim（1903-1973）与 Java 全一致。
- **SpikeSlime_L / SpikeSlime_M getMove**：不消耗额外 aiRng，30 阈值 + lastTwoMoves/lastMove 结构与 Java 一致。
- **Cultist**：firstMove→INCANTATION 否则 DARK_STRIKE；takeTurn 的 random(1,10)/playSfx 非 aiRng，sim 不多耗 rng——一致。
- **GremlinNob**：getMove 不读 num（A18 走 lastTwoMoves 链、非 A18 走 `num<33`）；BELLOW 的 playSfx 非 aiRng。sim（2402-2434）与 Java 一致。
- **Sentry**：首回合 `idx % 2` 决定 BOLT/BEAM（sim 2642-2651 用 `idx % 2 == 0`，Java 用 `monsters.lastIndexOf(this) % 2 == 0`，THREE_SENTRIES 三只 idx=0,1,2 与真机出场顺序一致）。
- **Hexaghost / TheGuardian**：getMove 固定返回 ACTIVATE / CHARGING_UP，不读 num；首回合时序一致。
- **GremlinLeader 出场**：`createMonsters`（MonsterGroup.cpp:248-259）用 `getGremlin(miscRng)` 构造 idx1/idx2 两 minion + idx3 leader，idx0 留空（default idx=-1，init 的 rollMove 循环会跳过，不多耗 aiRng）；init 时 rollMove 顺序 idx1,idx2,idx3 = 3 次 aiRng，与真机一致。
- **GremlinLeader RALLY 召唤**（`Actions::SummonGremlins`，Actions.cpp:459-498）：aiRng 顺序 = sel0, sel1, roll0, roll1，再 leader 自己的 RollMove → leaderRoll。Java `SummonGremlinAction` 在**构造时**就 `aiRng.random(0,7)` 选怪（两次连续，slot 顺序），`update()` 时才 `init()`(rollMove)，两个召唤 action 先构造后执行 → 序列同为 sel0,sel1,roll0,roll1,leaderRoll。**一致**。pool 大小 8，`getGremlin` 的 `random(7)`==`random(0,7)`==nextInt(8) 一致。
- **SlimeBoss split**：`slimeBossSplit`（3369-3391）spawn SpikeSlime_L(idx0) 再 AcidSlime_L(idx2)，各自 `initSpawnedMonster`→rollMove（2 次 aiRng），与 Java SpawnMonsterAction(SpikeSlime_L) → SpawnMonsterAction(AcidSlime_L) 各自 init() 的顺序一致。子代 HP = 当前血量。
- **Reptomancer summon / Bronze Automaton spawnOrbs / TorchHeads**：搜索顺序与 rollMove 次数已粗查，未见与真机调用次数不符（reptoSummonHelper 搜索序 {4,1,3,0}）。
- **偷窃（Looter / Mugger）**：`stealGoldFromPlayer`（3333-3340）累加到 miscInfo 并扣玩家金币；Looter/Mugger 的 MUG/LUNGE 里的 `aiRng.randomBoolean(...)` 与 `aiRng.random(2)`（对话用）调用位置与 Java 一致。Looter 被杀不归还金币的 bug 由另一 agent 处理，本审计未重复展开。

## 备注

- 史莱姆 split 子代当回合是否立刻拿到攻击意图（largeSlimeSplit）由另一 agent 修复中，本报告未重复分析其 root cause；但已确认 SlimeBoss 的 split（slimeBossSplit）与真机一致地会给子代 rollMove。
- 发现 1（AcidSlime_L A17）与发现 3（Lagavulin 打醒）最贴近当前 bac act-1 目标，建议优先处理发现 1（纯一行 typo，直接造成 A17 RNG 错位）。
