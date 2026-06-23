# 夜间工作总览（2026-06-23）

5 个 subagent 并行:1 个修 bac2（写代码），4 个按 sim 子系统静态查 bug（只读），外加 1 个对"流程/RNG"报告的复核员。本文件是去重排序后的晨读版,细节见各分报告。

---

## 一、最终状态

| 录像 | 之前 | 现在 | 说明 |
|---|---|---|---|
| bac1 | floor16 PLAYER_LOSS, 2 diffs | **不变(2 diffs)** | 无回归;但 2 个 diff 里有 1 个是真 mismatch,见 §四 |
| bac2 | 卡在 floor11 | **跑完整段录像 floor1→18,打穿 act-1 boss(The Guardian)进 act2** | 录像在 act-2 第一场战斗开打即结束;f18 还有 25 diffs,根因是 act-2 遭遇池(超范围) |

两局都已用 `--natural` 独立复跑确认。

---

## 二、今晚改了什么(已落工作区,**未 commit**,待你 review)

`git diff --stat`:replay.py / combat_state.py / slaythespire.cpp(binding) / Actions.cpp / MonsterSpecific.cpp / GameContext.cpp,共 6 文件。详见 `bac2_fix_report.md`。

**sim 真 bug(4 个)**
1. **Looter 偷的金币没归还** — 两处:replay 的 COMBAT_REWARD 只认 `"gold"` label,漏了真机 floor10 用的 `"stolen_gold"`;且 sim 把 stolen gold 排到 gold[1],真机排 gold[0](`GameContext.cpp:1131/1144`)。
2. **史莱姆 split 子代立刻拿攻击意图** — `largeSlimeSplit`(`MonsterSpecific.cpp:3342`)直接覆盖死 slime 那格→数组下标与真机错位,且多余的 `noOpRollMove`/`extraRollMoveOnTurn` 让 aiRng 多走。照正确的 `slimeBossSplit` 重写(保留死 slime 居中、子怪两侧、删多余 RNG)。
3. **Armaments+ 全升级没查 `canUpgrade()`**(`Actions.cpp:901`)→ 把诅咒 CurseOfTheBell 也升级,连锁输掉 boss 战。
4. **binding `drink_potion` 不 pump action 队列** → Swift Potion 抽牌不生效;replay 改走 `execute(Action.POTION,...)`。

**replay/schema 侧缺口**:事件"离开"帧重复触发选项、奖励里丢药水、越界药水槽、boss 宝箱选 relic、怪物 id 映射、录像尾部调试命令。

---

## 三、审计发现:确认为真、值得修但**今晚未动**(按本项目 asc0/act-1 + RNG 对齐 的实际优先级排)

> 重要框架:`BattleContext.cpp`/`CardManager`/`Actions`/`Random.h` 以及 `MonsterSpecific.cpp`/`Monster.cpp` 都与上游 sts_lightspeed **逐字一致**,无本地回归。下列全是"上游 sim vs 真机 Java"的固有差异——正是 RNG 对齐要消灭的目标。

### 高优先(普通对局就会命中)
- **Iron Wave 格挡双重计算** `BattleContext.cpp:1076` — `calculateCardBlock(calculateCardBlock(...))` 套两层,Dexterity 加两次、Frail 的 ×3/4 floor 应用两次。真机 `IronWave.java:33` 只算一遍。常见牌,直接致格挡数值 diff。【cards 报告 H1】
- **Shop `getNewPrice` 数值崩坏** `Shop.cpp:207-216` — basePrice 被 `(int)round(merchantRng.random(0.95,1.05))` 覆盖恒为 1,两个 `std::round(...)` 返回值还被丢弃。真机 `ShopScreen.java:371-383` 应为 `round(base*random(0.95,1.05))` 再串 Courier(×0.8)/Membership(×0.5)。**RNG 步数仍对**(只消耗 1 次 merchantRng),纯数值错,但影响商店定价→affordability→choice_list。【gameflow 报告 H3,已复核】

### 中优先(条件触发)
- **`getEvent` 空池回退误用成员 `eventRng`** `GameContext.cpp:~2048` — `tempLength==0` 时 `getShrine(eventRng)` 应传 `eventRngCopy`(真机 `AbstractDungeon.java:1963-1964`)。只在事件池抽空时触发,但触发后 eventRng **永久错位**。一行改。【gameflow H4,已复核】
- **Perfected Strike 把消耗堆的 Strike 也算进去** — sim 用 `cards.strikeCount`,真机 `countCards()` 只数 hand+draw+discard。Strike 进 exhaust 后伤害偏高。【cards M1】

### 低优先(asc0 本局不触发 / 极边界)
- **AcidSlime_L A17 move-id typo** `MonsterSpecific.cpp:1984` — 用了 `ACID_SLIME_M_CORROSIVE_SPIT` 查 `lastTwoMoves`(应是 `_L_`),恒 false → A17 下 roll<40 永不进 `randomBoolean(0.6)` 分支 → aiRng 错位 + 选错意图。**仅 A17 触发,asc0 无碍**,但单行修,建议顺手改。【monsters #1】
- **`transitionToAct` cardRng setCounter 边界** — 用 `<250/<500/<750`,真机 `dungeonTransitionSetup`(`AbstractDungeon.java:2544-2550`)用 `>0&&<250` 等严格区间。仅 counter∈{0,250,500} 时多推进 cardRng。【gameflow L4,已复核】
- **Book of Stabbing A18 `++stabCount` 写在 return 后成死代码** `MonsterSpecific.cpp:2264/2270`(act-2 精英,A18);**Lagavulin 被提前打醒**真机多插 STUN 回合 sim 直接攻击;**TheChamp turn-4 Taunt** sim 用全局 `%4` 真机用会重置的 `numTurns`(act-3 boss)。【monsters #2/#3/#4】

### 非 RNG、纯 correctness(与对齐无关,单列)
- Impatience `hasAttack` 写反、`exhaustSpecificCardInHand` 用错下标 `hand[idx]`→`hand[i]`【combat-core】;Buffer/Torii 顺序、AOE+Vigor 双计、`Deck::obtainRaw` 重复计数、`removeAllMatching` 起始下标越界【cards L1-L4】。

---

## 四、需要你看一眼的悬案:bac1 的 Parasite

bac1 终局 sim 牌组比真机**多一张 Parasite**(f12 Hypnotizing Mushrooms 起,持续到结束,非 transient)。

- sim 的 Mushrooms `idx==1` = 吃蘑菇 = heal 25% + 获得 Parasite(上 session 据 diff test "真机 +20hp +Parasite" 加的,见 `GameContext.cpp:3110` 注释)。
- 但现在真机终局牌组(16 张)**没有 Parasite**。两者矛盾。
- **最可能的解释**:真机吃了蘑菇(+20hp 对得上)拿到 Parasite,但在 f12~f16 间又**删掉了它**(商店/事件删牌),而 sim 因牌序与真机不同**删错了卡**——这正是你担心的"真机桥不暴露真实牌序"导致的对齐难点的一个具体实例。
- 建议下一步:在真机日志里找 f12~f16 之间的删牌动作,核对 sim 是否删在了同一张牌上。**今晚未动,留给你定夺。**

---

## 五、bac2 唯一剩余 blocker
floor18 第二场战斗:真机 `[Looter,Mugger]`,sim 生成 `[Spiker,Repulsor,...]` —— **act-2 遭遇池生成逻辑**,超出今晚 act-1 范围。f18 那 ~25 处 diff 全来自此。

---

## 六、复核员纠正的一个危险误判
"流程/RNG"原报告的 **H2**(指控 `transitionToMapNode` 每房 reset `shuffleRng`/`cardRandomRng` 是 bug)是**假阳性**:真机 `nextRoomTransition`(`AbstractDungeon.java:1727-1731`)每房确实重 seed 这五个 rng,sim 是**对的**。**切勿按原建议"只重置 miscRng"改**,否则引入回归。原报告里另有多条假阳性(H1/M3/M5/M9/L6 等),均已就地批注 `> 复核:` 行。

---

### 分报告索引
- `bac2_fix_report.md` — 今晚的修复细节
- `audit_combat_core.md` — 战斗核心(PASS,洗牌/抽牌 RNG 无缺陷)
- `audit_monsters.md` — 怪物/AI/意图
- `audit_cards_powers.md` — 卡牌/能力/玩家
- `audit_gameflow_rng.md` — 流程/RNG/商店/事件(开头有"复核结论"小节,逐条 `> 复核:` 批注)
