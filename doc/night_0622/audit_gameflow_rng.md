# 游戏流程 / RNG 子系统静态审计（night_0623）

## 复核结论（优先级排序）

> 复核员补充（2026-06-23）：已用**真机 Java**（`../reference/cardcrawl/`）逐条复核。下表只列最终**确认为真、值得修**的条目。
> 真机参考关键文件：`../reference/cardcrawl/dungeons/AbstractDungeon.java`、`shop/ShopScreen.java`、`neow/NeowReward.java` + `neow/NeowEvent.java`、`rooms/AbstractRoom.java`、`rooms/MonsterRoomElite.java`、`helpers/CardLibrary.java`、`helpers/EventHelper.java`、`cards/CardGroup.java`、`random/Random.java`。

| 优先级 | 条目 | 真机依据 | 性质 |
|---|---|---|---|
| **P0** | **H3** Shop::getNewPrice 整体写坏 | `ShopScreen.java:371-383` `retVal=round(retVal*merchantRng.random(0.95,1.05))` + `applyDiscountToRelic`(399-401) | 数值彻底错（恒≈1），但**只消耗 1 次 merchantRng，RNG 步数对**。只在重定价（持 Courier/Membership 复购）触发。 |
| **P0** | **H4** getEvent 空池回退误用成员 eventRng | `AbstractDungeon.java:1963-1964` `if(tmp.isEmpty()) return getShrine(rng)`（rng=传入的 copy） | 误用成员 `eventRng` 会**额外推进成员 1 步**，触发后 eventRng 永久错位。事件池抽空时触发（act 末期）。改 `getShrine(eventRngCopy)`。 |
| **P3** | **L4** transitionToAct 的 cardRng setCounter 边界条件错 | `AbstractDungeon.java:2544-2550` 真机用 `>0 && <250` / `>250 && <500` / `>500 && <750`（严格、排除 0 和正好对齐值） | sim 用 `<250 / <500 / <750`，在 counter ∈ {0,250,500} 时**多推进** cardRng（如 counter==250 被 sim 顶到 500，真机不动）。边界场景，但确为 bug。 |

**已确认为假阳性 / 无 bug（不必修）**：H1、H2、M1、M2、M3、M4、M5、M6、M9、L1、L2、L3、L5、L6。其中 **H2 是原报告中误判最严重的一条**（见下）。**M8** 为极低风险、可不修（见下）。

> H2 复核小结：真机 `AbstractDungeon.nextRoomTransition`（`AbstractDungeon.java:1727-1731`）**每进一个房间**都把 `monsterHpRng / aiRng / shuffleRng / cardRandomRng / miscRng` **五个**全部重 seed 成 `new Random(Settings.seed + floorNum)`。sim 每层 reset `miscRng/shuffleRng/cardRandomRng` 完全符合真机，原报告"只该重置 miscRng"的断言是凭记忆的误判。

---

审计员：sim 静态审计员（gameflow / RNG / 奖励 / 商店 / 事件 / 地图 / Neow）
审计范围：`cpp/sts_lightspeed/` 下的
`src/game/GameContext.cpp`、`src/game/Game.cpp`、`include/game/Random.h`、
`src/game/Neow.cpp`、`src/game/Shop.cpp`、`src/game/Map.cpp`、`src/game/CombatReward.cpp`(=`Rewards.cpp`)。

> **重要前置说明（工具/环境）**
> 任务书里说的参考资料目录 `reference/cardcrawl/`、`reference/sts_lightspeed/` **在当前工作区里并不存在**
> （`D:\zxb\my\sts-rl-workbench\sts-rl` 下只有 `bridge/`、`cpp/`，git 里也没有 reference 文件）。
> 因此本报告对真机 Java 的引用是基于审计员对反编译 STS 源码
> （`AbstractDungeon` / `NeowReward` / `AbstractRoom` / `ShopScreen` / `MapGenerator` / `EventHelper` 等）的既有知识，
> **无法给出 reference 文件的精确行号**。凡涉及 Java 行为的断言都标注了来源类名，
> 建议在拿到 reference 后逐条复核（尤其标“中/低”且依赖记忆的条目）。
> 已知且已修复的 4 个 bug（带火精英 buff act、getShrine MATCH_AND_KEEP、Hypnotizing Mushrooms idx==1、Calling Bell 的 9 次 cardRng）不再重复报告。

---

## 高严重度

### H1. `mathUtilRng` 用 `seed-897897` 派生，且 game start 漏初始化 `monsterHpRng` / `aiRng`，破坏 seed 派生模型
- **sim**：`GameContext.cpp:37-53` 构造函数初始化列表只 seed 了
  `neowRng/treasureRng/eventRng/relicRng/potionRng/cardRng/cardRandomRng/merchantRng/monsterRng/shuffleRng/miscRng`
  和 `mathUtilRng(seed-897897)`。`GameContext.h:160-173` 声明了 `aiRng`、`monsterHpRng`，
  但构造函数**完全没初始化它们**（默认构造 = `Random(0)`）。
- **Java 对应**（`AbstractDungeon.generateSeeds` / 各 `*Rng` 字段）：真机用同一个 `Settings.seed`
  逐个 `new Random(Settings.seed)` 派生 `monsterRng, monsterHpRng, aiRng, shuffleRng, cardRng,
  treasureRng, relicRng, potionRng, monsterListRng…, eventRng, merchantRng, cardRandomRng, miscRng`。
  `mathUtilRng` 在真机里是 `MathUtils`（libgdx）用**基于时间**的种子，本来就不可对齐——sim 的 `seed-897897`
  注释也写了 “uses a time based seed”，这是个有意的占位，问题不在这。
- **为何 divergence**：`aiRng` / `monsterHpRng` 这两个实例在战斗里（怪物 AI、HP roll）会被消耗。
  若它们没用 `seed` 正确派生，任何读取这两个 rng 的战斗都会立刻偏。虽然战斗子系统不在本审计范围，
  但**派生不完整**属于本子系统（“game start 各 Random 实例派生”）的硬伤。**严重度：高**。
- **建议**：在构造函数初始化列表补 `monsterHpRng(seed)`、`aiRng(seed)`（按真机顺序，二者都直接 `Random(seed)`）。
  注意：派生顺序不影响 seed（每个实例独立 `new Random(seed)`），但必须保证存在且 seed 正确。

> 复核：FALSE-POSITIVE — `../reference/cardcrawl/dungeons/AbstractDungeon.java:386,388`（generateSeeds 确实 `monsterHpRng/aiRng = new Random(Settings.seed)`），但 `:1727-1728` 真机**每进一房间**就把这俩重 seed 成 `new Random(Settings.seed+floorNum)`；sim 靠 `BattleContext::init` 重 seed，且 `src/game/` 内无任何代码读取 `aiRng/monsterHpRng`。构造函数留默认 `Random(0)` 无害（采纳主审计员结论）。

### H2. `transitionToMapNode` / `enterBossTreasureRoom` 把 `shuffleRng` 和 `cardRandomRng` 一起 reset 成 `Random(seed+floorNum)`
- **sim**：`GameContext.cpp:761-765`（`transitionToMapNode`）、`1087-1090`（`enterBossTreasureRoom`）、
  `1157-1161`（act3 第二 boss）三处都做
  ```
  const auto r = Random(seed + floorNum);
  miscRng = r; shuffleRng = r; cardRandomRng = r;
  ```
- **Java 对应**（`AbstractDungeon.nextRoomTransition` / `AbstractRoom`）：真机进房间时只重置
  `AbstractDungeon.miscRng = new Random(Settings.seed + (long)floorNum)`。
  `cardRandomRng` 和 `shuffleRng` **不是**每层重置的——它们在 act 切换/特定时机才 reseed，
  逐房间随 floorNum 重置 `shuffleRng`/`cardRandomRng` 不符合真机。
- **为何 divergence**：`shuffleRng` 用于 `returnColorlessCard`（洗 colorless 池）等，
  `cardRandomRng` 用于 Pandora's Box / 战斗内随机牌 / Falling 之类。每层把它们强行拉回 `seed+floorNum`
  会让“跨房间累积消耗”的语义消失：真机里这俩 rng 的 counter 是连续累加的，sim 每层归零，
  一旦同一 act 内多次使用就会错位。
- **严重度：高**（只要一局里用到 shuffleRng/cardRandomRng ≥2 次且中间过了房间就偏）。
- **建议**：核对真机 `nextRoomTransition`，**只**重置 `miscRng`。`shuffleRng`/`cardRandomRng` 的 reseed 时机
  另行对齐（通常在 `generateMap`/act 转换处）。需结合 reference 确认，但当前“三个一起每层 reset”几乎肯定过度。

> 复核：**FALSE-POSITIVE（原报告最严重误判）** — `../reference/cardcrawl/dungeons/AbstractDungeon.java:1727-1731`：真机 `nextRoomTransition` **每进一个房间**就把 `monsterHpRng / aiRng / shuffleRng / cardRandomRng / miscRng` 五个**全部**重 seed 成 `new Random(Settings.seed + (long)floorNum)`。sim 每层 reset `miscRng/shuffleRng/cardRandomRng` 完全正确（缺的 `monsterHpRng/aiRng` 由 BattleContext 处理）。原报告"只该重置 miscRng"是记忆错误，**不要改**。

### H3. `Shop::getNewPrice` 逻辑整体写坏（删牌价/二次刷新价错乱）
- **sim**：`Shop.cpp:207-216`
  ```cpp
  int Shop::getNewPrice(GameContext &gc, int basePrice) {
      basePrice = static_cast<int>(std::round(gc.merchantRng.random(0.95f, 1.05f)));  // 把 basePrice 直接覆盖成 round(0.95~1.05) → 恒为 1
      if (gc.hasRelic(RelicId::THE_COURIER)) {
          std::round(basePrice * COURIER_FACTOR);   // 返回值被丢弃，无副作用
      }
      if (gc.hasRelic(RelicId::MEMBERSHIP_CARD)) {
          std::round(basePrice * MEMBERSHIP_CARD_FACTOR);  // 同样被丢弃
      }
      return basePrice;   // 恒返回 1
  }
  ```
- **Java 对应**（`ShopScreen` 价格逻辑）：新价应为 `round(basePrice * merchantRng.random(0.95,1.05)) * 折扣`，
  而不是把 basePrice 丢掉、只留一个 `round(0.95~1.05)`（恒等于 1，因为 0.95~1.05 round 到 1）。
- **为何 divergence**：① 价格数值直接错（永远是 1 块 / 折扣后仍约 1）；
  ② **RNG 消耗仍发生**（`merchantRng.random(0.95,1.05)` 推进了一次），所以这条主要是**数值 bug**，
  RNG 步数恰好碰巧对（一次 merchantRng）。但 The Courier 复购物品时定价完全错。
- **严重度：高**（数值层面彻底错；只在持 Courier 复购时触发，但触发即错）。
- **建议**：改为 `basePrice = round(basePrice * gc.merchantRng.random(0.95f,1.05f));` 然后真正
  `basePrice = round(basePrice * factor)` 赋值回去。对照真机确认折扣是否在这里叠加。

> 复核：**CONFIRMED** — `../reference/cardcrawl/shop/ShopScreen.java:371-383`。真机 `getNewPrice`：`retVal = MathUtils.round(retVal * merchantRng.random(0.95f,1.05f))`（373-374，`!isDailyRun` 时），再 Courier `applyDiscountToRelic(retVal,0.8f)`、Membership `(retVal,0.5f)`（376-381），`applyDiscountToRelic = round(price*multiplier)`（399-401），最后写回 `r.price`。sim 把 basePrice 覆盖成 `round(0.95~1.05)`≈1 且丢弃折扣结果——数值彻底错。**RNG 步数仍对**：真机此处只消耗 1 次 merchantRng（那个 0.95~1.05 roll），sim 也只消耗 1 次（`Random::random(float,float)` 推进 1 步），故不影响后续 merchantRng 对齐，纯数值 bug。注意真机折扣是**串行 round**（先 ×0.8 round 再 ×0.5 round），不是连乘。正确改法：`int p=round(basePrice*merchantRng.random(0.95f,1.05f)); if(Courier) p=round(p*0.8f); if(Membership) p=round(p*0.5f); return p;`。

### H4. `getEvent` 在 “tempLength==0 回退到 shrine” 时用了 `eventRng` 而非传入的 copy
- **sim**：`GameContext.cpp:2037-2039`
  ```cpp
  if (tempLength == 0) {
      return getShrine(eventRng);   // 用了成员 eventRng
  }
  ```
  而正常路径 `getShrine(eventRngCopy)`（`generateEvent` 走 copy 语义，见 `2049-2063`）。
- **Java 对应**（`AbstractDungeon.getEvent` / `generateEvent`）：`generateEvent` 拿的是 `eventRng` 的
  **copy**，整条决策（SHRINE_CHANCE roll + getShrine/getEvent 的索引 roll）都在 copy 上跑，
  成员 `eventRng` 只在 `getEventRoomOutcomeHelper` 里被推进 1 次。回退到 shrine 也应在 copy 上 roll。
- **为何 divergence**：在 copy 上 roll 不会推进成员 `eventRng`；这里误用成员 `eventRng` 会**额外推进成员 eventRng 一步**，
  导致后续所有 event 房间的 outcome/index 全部错位。触发条件是“event 池被抽空，本层落到 event 房”——
  Act 末期常见。
- **严重度：高**（一旦触发，eventRng 永久错位一步）。
- **建议**：`return getShrine(eventRngCopy);`。

> 复核：**CONFIRMED** — `../reference/cardcrawl/dungeons/AbstractDungeon.java:1924-1969`。真机 `getEvent(Random rng)`：`if (tmp.isEmpty()) return AbstractDungeon.getShrine(rng);`（1963-1964），用的是**传入的 rng**（即 `nextRoomTransition` 的 `eventRngDuplicate` / generateEvent 的 copy，见 :1746-1751），**不是**成员 `eventRng`。sim `GameContext.cpp` `getEvent` 里的 `return getShrine(eventRng);` 应为 `getShrine(eventRngCopy)`。误用成员会多推进成员 eventRng 1 步，触发后永久错位。低频（事件池抽空才走）。改法如建议。

---

## 中严重度

### M1. `getEventRoomOutcomeHelper` 的 roll 取自 `eventRng.random()`，再被 `generateEvent(eventRng)` 拷贝——需确认拷贝时机
- **sim**：`GameContext.cpp:779`（`curRoom = getEventRoomOutcomeHelper(...)`）→ `2066-2067`
  `float roll = eventRng.random();`（推进成员 eventRng 1 次，符合任务书“helper 消耗 1 roll”）。
  随后 `782` `curEvent = generateEvent(eventRng);`，`generateEvent` **按值传参**（`2049` `Random eventRngCopy`），
  即拷贝当前 eventRng 状态。
- **Java 对应**：`getEventRoomOutcome` 先 `eventRng.random(...)` 决定房间类型（消耗成员 1 次），
  再 `generateEvent` 用 `new Random` copy。**顺序与拷贝点**：sim 这里是对的（helper 先消耗，再拷贝）。
- **结论**：**这条是正确的**，列出来是为了排除——`generateEvent` 按值传参实现 copy 语义没问题。
  唯一隐患见 H4（回退路径误用成员）。**严重度：中→实际无 bug**，仅作确认。

> 复核：CONFIRMED（无 bug）— `../reference/cardcrawl/helpers/EventHelper.java:94-96`（`roll(Random eventRng)` 先 `eventRng.random()` 决定房间类型，消耗 1 次）+ `../reference/cardcrawl/dungeons/AbstractDungeon.java:1746-1751`（真机在 duplicate 上 roll 后 `eventRng = eventRngDuplicate` 写回，再 `generateEvent`）。sim "helper 先消耗成员 1 次 → generateEvent 按值拷贝" 与真机等价。

### M2. `Neow::getOptions` 末尾 `r.random(0,0)` 与 reward[3] 的对齐——确认是否多/漏 roll
- **sim**：`Neow.cpp:70-72`
  ```cpp
  rewards[3].r = Bonus::BOSS_RELIC;
  rewards[3].d = Drawback::LOSE_STARTER_RELIC;
  r.random(0, 0);   // 推进 neowRng 一次（random(0,0)=start+nextInt(1)=start，恒0，但 counter+1 且消耗一次 nextLong）
  ```
- **Java 对应**（`NeowReward` 构造，`AbstractDungeon.neowRng`）：第 4 个选项（BOSS_RELIC + LOSE_STARTER_RELIC）
  在真机里也走了一次 rng（`blessing`/`drawback` 固定，但仍 `neowRng.random(...)` 走一步占位）。
  sim 用 `random(0,0)` 复刻这一步是**对的**（`Random::random(int,int)` 在 `start==end` 时
  `nextInt(1)` 仍调用一次 `nextLong`，`Random.h:159-162`，counter+1）。
- **结论**：**正确**，列出排除。但建议在拿到 reference 后确认真机第 4 选项确实只消耗 1 次 neowRng。
  **严重度：中→实际无 bug**。

> 复核：CONFIRMED（无 bug）— `../reference/cardcrawl/neow/NeowEvent.java:355-363`（blessing 重 seed `rng=new Random(seed)` 后造 4 个 `NeowReward(0..3)`）+ `neow/NeowReward.java:50-58,108-112`。第 4 个 `NeowReward(3)`：`getRewardOptions(3)` 只放 1 个 BOSS_RELIC，构造函数 `NeowEvent.rng.random(0, possibleRewards.size()-1)` = `random(0,0)` 仍消耗 1 次。sim `Neow.cpp` 末尾 `r.random(0,0)` 完全对应。前三项 neowRng 消耗（cat0=`random(0,5)`、cat1=`random(0,4)`、cat2=drawback `random(0,3)`+reward `random(0,N)` 共 2 次）也逐一吻合。

### M3. `Neow::getCardReward` 的 rare 路径重复去重逻辑与真机可能不同
- **sim**：`Neow.cpp:109-136`。三张卡，每张 `rng.randomBoolean(0.33F)` 决定 UNCOMMON/COMMON，
  `rareOnly` 时强制 RARE。去重时**重 roll `getRandomClassCardOfRarity(rng, cc, rarity)`**（不重 roll 稀有度）。
  这里 `rng` 是 `neowRng`（调用点 `GameContext.cpp:2293` `Neow::getCardReward(neowRng, cc, false)`）。
- **Java 对应**（`NeowReward` THREE_CARDS / THREE_RARE_CARDS，`AbstractDungeon.getRewardCards` 风格）：
  真机 Neow 三选一卡同样用 `neowRng`，每张先 roll 稀有度再 roll 卡，重复时**只重 roll 卡**。
  **需注意**：真机 `getRewardCards` 里 `rng.randomBoolean(0.33f)` 之后，UNCOMMON/COMMON 的处理
  ——sim `getCardReward`（非 colorless）在非 rareOnly 时 **保留 COMMON**（`113-116` 没有把 COMMON 提升为 UNCOMMON），
  而 `getColorlessCardReward`（`81-86`）则把 COMMON 提升为 UNCOMMON。普通 Neow 三卡奖励是否允许 COMMON
  取决于真机：标准 Neow “3 cards to choose” 用的是和普通战斗奖励一致的稀有度分布（含 common），看起来 OK。
- **为何可能 divergence**：若真机 Neow 三卡其实走的是 `getCardRarity`(room based) 而非这里的
  `randomBoolean(0.33)`，则稀有度阈值与 rng 消耗次数都会不同。**严重度：中**（依赖记忆，需 reference 核实
  `NeowReward.getCard`/`getColorlessCard` 的确切 rng 调用序列）。
- **建议**：拿 reference 对 `NeowReward` 逐行核对每张卡的 neowRng/cardRng 调用顺序与次数。

> 复核：FALSE-POSITIVE（无 bug）— `../reference/cardcrawl/neow/NeowReward.java:317-376`。真机 `getRewardCards`（非 colorless）：每张 `rollRarity()`（355-360：`randomBoolean(0.33f)`→UNCOMMON else COMMON，**不提升 COMMON**），`getCard(rarity)` 用 `XxxCardPool.getRandomCard(NeowEvent.rng)`，去重时**只重 roll 卡不重 roll 稀有度**（344）。sim `Neow.cpp:109-136` 完全一致。`getColorlessRewardCards`（294-309）则 COMMON→UNCOMMON 提升（301-303），sim `getColorlessCardReward:81-86` 也一致。colorless 选卡用 `cardRng`：真机 `getColorlessCardFromPool`→`getRandomCard(true,rarity)`→`AbstractDungeon.cardRng`（`cards/CardGroup.java:503-504`），sim 用 cardRng 正确。稀有度阈值用 `randomBoolean(0.33)` 而非 room-based `getCardRarity`，与真机相符。

### M4. `createCardReward` 升级 roll 顺序：稀有度全部 roll 完之后再统一 roll 升级，需确认与真机交错顺序一致
- **sim**：`GameContext.cpp:1803-1851`。先一个循环对 `numCards` 张**全部** roll 稀有度 + `getRandomClassCardOfRarity`
  （含去重重 roll）；**之后**再开一个循环对每张 `cardRng.randomBoolean(cardUpgradedChance)`（`1845`）。
  即：所有稀有度/选卡 roll → 所有升级 roll。
- **Java 对应**（`AbstractDungeon.getRewardCards`）：真机也是**先生成全部卡再统一判定升级**
  （`getRewardCards` 先 for 循环填 `retVal`，再单独 for 循环 `if (!rare && cardRng.randomBoolean(chance)) upgrade`）。
  **sim 顺序与真机一致**（这正是任务书说的“每张约 3 次 cardRng：稀有度 + getCard + 升级 roll”，
  但升级 roll 是后置批量的）。
- **结论**：**正确**，列出排除。注意 RARE 卡不 roll 升级（`1845` `rewardRarities[i] != RARE`），与真机一致。
  **严重度：中→实际无 bug**。

> 复核：CONFIRMED（无 bug）— `../reference/cardcrawl/dungeons/AbstractDungeon.java:1403-1458`。真机 `getRewardCards()` 先 for 循环（1412-1444）逐张 `rollRarity()`+`getCard`/`getAnyColorCard`+去重重 roll 填 retVal，**之后**单独 for 循环（1449-1457）`if(c.rarity != RARE && cardRng.randomBoolean(cardUpgradedChance) && canUpgrade()) upgrade`。顺序、RARE 跳过升级 roll 均与 sim 一致。

### M5. `getAnyColorCard`（Prismatic Shard）每次多消耗一次 `cardRng.randomLong()`，需确认真机是否如此
- **sim**：`Game.cpp:133-156`。每个稀有度分支里都先 `cardRng.randomLong();` 再 `cardRng.random(size-1)`。
  即 Prismatic Shard 选卡消耗 **2 次** cardRng（一次 randomLong 占位 + 一次选 idx）。
- **Java 对应**（`AbstractDungeon.getCard` / Prismatic Shard 的 `getColorCardForType`-ish 路径）：
  真机 Prismatic 选色逻辑确实会先用一次 rng 选“颜色”再选卡。sim 用 `randomLong()` 占位选色这步。
  **需 reference 确认**该 `randomLong()` 是否对应真机的“roll color”那次调用且 rng 实例正确（应是 cardRng）。
- **严重度：中**（仅 Prismatic Shard 时触发，但触发即影响后续所有 cardRng 对齐；依赖记忆）。
- **建议**：reference 核对 `getAnyColorCard` 对应真机方法的 rng 调用序列。

> 复核：FALSE-POSITIVE（无 bug）— `../reference/cardcrawl/helpers/CardLibrary.java:1060-1068`。真机 `getAnyColorCard(rarity)`：建池后 `anyCard.shuffle(AbstractDungeon.cardRng)` 再 `anyCard.getRandomCard(true, rarity)`。`shuffle(rng)` = `Collections.shuffle(group, new java.util.Random(rng.randomLong()))`（`cards/CardGroup.java:548-550`，消耗 1 次 `cardRng.randomLong()`），`getRandomCard(true,rarity)` 消耗 1 次 `cardRng.random(size-1)`（CardGroup.java:503-504）。**正好 2 次 cardRng**，与 sim `Game.cpp:133-156` 的 `randomLong()`+`random(size-1)` 完全对应。调用点见真机 `AbstractDungeon.java:1435`（持 PrismaticShard 时走此路）。

### M6. `returnRandomRelicTierElite` 阈值用 `>` 把 roll==82 划进 UNCOMMON，需确认边界
- **sim**：`Game.cpp:283-292`
  ```cpp
  int roll = relicRng.random(99);
  if (roll < 50) COMMON;
  else if (roll > 82) RARE;   // 注意是 > 不是 >=
  else UNCOMMON;              // roll==82 → UNCOMMON
  ```
- **Java 对应**（`AbstractDungeon.returnRandomRelicTier` elite 变体 / `AbstractRoom.getReward`）：
  真机精英遗物 tier 阈值通常是 `roll < 50 → COMMON`、`roll < 83 → UNCOMMON`、`else RARE`，
  等价于 `roll > 82 → RARE`。sim 的 `> 82` 与 `< 83` 在整数上**等价**（83/84/.../99 是 RARE，82 是 UNCOMMON）。
- **结论**：**正确**（边界等价），列出排除。对比 `Shop::rollRelicTier`（`Shop.cpp:256-264`）用
  `<48 / <82 / else`（COMMON<48, UNCOMMON<82, RARE>=82）——这是商店遗物，阈值不同属正常。
  **严重度：中→实际无 bug**。

> 复核：CONFIRMED（无 bug）— 精英遗物 `../reference/cardcrawl/rooms/MonsterRoomElite.java:95-106`：`roll = relicRng.random(0,99); if(roll<50)COMMON; if(roll>82)RARE; else UNCOMMON`，与 sim `Game.cpp:283-292` 的 `<50 / >82 / else` 逐字相同。商店遗物 `../reference/cardcrawl/shop/ShopScreen.java:403-413`：`merchantRng.random(99)` 的 `<48/<82/else`，与 sim `Shop.cpp:256-264` 一致。步数：`random(99)` 与 `random(0,99)` 都只 1 次 `nextInt`（`../reference/cardcrawl/random/Random.java:49-57`），等价。

### M7. `rollElite` 写法冗余但结果正确（一次 monsterRng.random()）
- **sim**：`GameContext.cpp:554-568`。`float roll = monsterRng.random();`（消耗 1 次），
  `currentWeight += 1/3` 那行是死代码（算了没用），但分支 `<1/3 / <2/3 / else` 正确给出 0/1/2 均分。
- **Java 对应**（`MonsterHelper`/`AbstractDungeon` 精英三选一）：一次 rng，1/3 均分。**消耗次数一致**。
- **结论**：**正确**（死代码不影响 rng）。**严重度：低/无**，仅记录。

> 复核：CONFIRMED（无 bug）— sim `generateElites`（`GameContext.cpp:570-583`）每张走 `rollElite(monsterRng)`，与真机 `populateMonsterList(elites=true)` 每张一次 `monsterRng.random()`（`../reference/cardcrawl/dungeons/AbstractDungeon.java:1048,1051`）一致。单次 monsterRng 消耗对齐。

### M8. `generateElites` / `populateMonsterList` 的“重复则 --i 重 roll”消耗模型需确认 monsterRng 推进次数
- **sim**：`GameContext.cpp:570-583`（elites，连续重复重 roll）、`614-695`（weak/strong）。
  每次重 roll 都 `monsterRng.random()` 推进一次，与真机 `MonsterHelper.populateMonsterList` 的
  while/for + 重 roll 语义一致（真机也是重复就重 roll，rng 照样推进）。
- **风险点**：`getMonsterForRoomCreation`（`1742-1751`）/`getEliteForRoomCreation`（`1753-1762`）
  在 list 耗尽时**重新 generateStrong/generateElites**，会**额外消耗 monsterRng**。注释自己也写
  “the number generated here won't be accurate but it should not matter”——
  在长局（同 act 内房间数超过预生成的 12/10）里这会让 monsterRng 偏。
- **严重度：中**（长 act 才触发；真机预生成的 list 大小与重生成时机需 reference 精确核对）。
- **建议**：核对真机 monster/elite list 的预生成数量（sim 注释说 12/10）和耗尽后的重生成逻辑是否消耗等量 rng。

> 复核：NEEDS-MORE（倾向极低风险，可不修）— `../reference/cardcrawl/dungeons/AbstractDungeon.java:1828,1837`（generateMap 时 `generateStrongEnemies(12)`/`generateElites(10)`）+ `:1674-1687`（`nextRoomTransition` 进房时若 list 空则 `generateElites(10)`/`generateStrongEnemies(12)`）+ `:1044-1075`（`populateMonsterList` 重复则 `--i` 重 roll，每次消耗 1 `monsterRng.random()`）。预生成数 12/10 与 sim 完全一致，重 roll 消耗模型一致。**唯一差异**：真机在 `nextRoomTransition`（离开房间、`remove(0)`）时检查耗尽并重生成；sim 用 offset 在 `getMonsterForRoomCreation` 时（`GameContext.cpp:1752-1772`）惰性重生成。act 内 monsterRng 的另一消耗者只有 `generateBoss`（建图时一次），二者顺序不冲突，故重生成那段 monsterRng 突发的相对时机一致——实际不偏。sim 注释自承"won't be accurate but should not matter"成立。

### M9. `addPotionRewards` 的 `rewardsSize>=4 → chance=0` 仍会 roll potionRng
- **sim**：`GameContext.cpp:1764-1784`。即使 `chance=0`，仍执行 `potionRng.random(99) >= chance`
  （`1777`，恒 true 因为 0..99 >= 0），推进 potionRng 一次，然后 `potionChance += 10`。
- **Java 对应**（`AbstractRoom.addPotionToRewards` / `RewardItem`）：真机药水掉落判定
  `potionRng.random(99) < chance`。当 chance 被压到 0 时**真机是否仍 roll**取决于实现——
  标准 STS 里 `chance` 计算后无条件 `potionRng.random(99)`，所以 sim 这里 roll 一次**大概率正确**。
- **结论**：倾向正确，但 `rewardsSize>=4` 这个短路是 sim 自加的（真机 reward 满时是否还掉药水/还 roll 需确认）。
  **严重度：中**（边界场景，需 reference 核对 `chance=0` 时真机是否仍消耗 potionRng）。

> 复核：FALSE-POSITIVE（无 bug，且原报告"短路是 sim 自加的"判断错误）— `../reference/cardcrawl/rooms/AbstractRoom.java:582-610`。真机 `addPotionToRewards()`：`if(this.rewards.size() >= 4) chance = 0;`（599-600）**正是真机行为**，不是 sim 自加。且无论 chance 多少都执行 `if(potionRng.random(0,99) < chance ...)`（603），chance=0 时仍消耗 potionRng 1 次。`blizzardPotionMod -=10`（命中）/`+=10`（未命中）对应 sim `potionChance -= 10`/`+= 10`。sim `GameContext.cpp:1774-1793` 与之完全一致。

### M10. `Shop::setupCards` 第 5 张（POWER）把 COMMON 提升 UNCOMMON 的时机
- **sim**：`Shop.cpp:40-42`。`rarities[4] = rollCardRarityShop(...)`（消耗 cardRng 一次），
  再 `rarities[4] = (COMMON ? UNCOMMON : rarities[4])`。提升发生在 roll **之后**，不影响 rng。
- **Java 对应**（`ShopScreen.init`）：power slot 同样 roll 稀有度后把 common 视作 uncommon。一致。
- **结论**：**正确**。但定价循环 `for i<5`（`47-51`）只给 5 张 class 卡定价，colorless 两张
  `prices[5]/[6]` 单独定价（`53-54`），`saleIdx = merchantRng.random(4)`（`56`，只在前 5 张里打折）——
  与真机一致。**严重度：低/无**，记录确认。

> 复核：NEEDS-MORE（power slot 提升时机已确认正确）— 真机 power slot 稀有度提升见 `../reference/cardcrawl/shop/ShopScreen.java`（init 处对 power 卡 common→uncommon，在 roll 之后），与 sim `Shop.cpp:40-42` 一致；`saleIdx`/定价循环未在本轮逐行核到具体行号，但与 `merchantRng` 步数无新增疑点。

---

## 低严重度

### L1. `Random::nextLong(uint64_t n)` 的 bits 右移与 Java 实现位级一致性
- **sim**：`Random.h:97-105`。`bits = (uint64_t)nextLong() >> 1; value = bits % n;`
  循环条件 `(int64_t)(bits - value + n - 1) < 0`。这与 Java `com.megacrit...Random`（基于
  `RandomXS128`/自定义）的 `nextLong(n)` 拒绝采样一致（`bits>>1` 取非负 63 位，溢出检测用有符号判负）。
- **结论**：**正确**。`random(int range)=nextInt(range+1)=nextLong(range+1)`、
  `random(99)=bits%100` 等与任务书模型一致（`Random.h:154-157, 201-203`）。**无 bug**，记录。

> 复核：CONFIRMED（无 bug）— 真机 `../reference/cardcrawl/random/Random.java:49-72`：`random(int range)=++counter; nextInt(range+1)`，`random(int,int)=++counter; start+nextInt(end-start+1)`，`randomLong=++counter; nextLong()`。每次调用恰 counter+1，与 sim `Random.h` 各重载语义一致（底层 nextLong 拒绝采样位运算同 RandomXS128）。

### L2. `nextFloat` / `random(float)` 与 Java 一致性
- **sim**：`Random.h:116-120`：`x = nextLong()>>40; d = x * NORM_FLOAT; return (float)d;`，
  `NORM_FLOAT=5.96e-8`。`random(float range)=nextFloat()*range`（`181-184`），
  `randomBoolean(chance)=nextFloat()<chance`（`126-129`）。与真机 `Random.random()` / `randomBoolean(float)` 一致。
- **结论**：**正确**，记录。注意 `random(1.0f)`（如 `generateEvent` 的 `eventRngCopy.random(1.0f)<SHRINE_CHANCE`，
  `GameContext.cpp:2050`）= `nextFloat()*1.0f`，消耗 1 次，符合任务书。

> 复核：CONFIRMED（无 bug）— 真机 `../reference/cardcrawl/random/Random.java:84-97`：`random()=++counter; nextFloat()`，`random(float range)=++counter; nextFloat()*range`，`randomBoolean(float)=++counter; nextFloat()<chance`。与 sim 一致。注意真机 `generateEvent` 的 `rng.random(1.0f) < shrineChance`（`AbstractDungeon.java:1845`），sim `eventRngCopy.random(1.0f) < SHRINE_CHANCE` 对应（都消耗 1 次）。

### L3. `Map::fromSeed` 的 offset 公式与 burningEliteBuff
- **sim**：`Map.cpp:56-69`。`offset = act==1 ? 1 : act*(100*(act-1))`，`Random mapRng(seed+offset)`。
  Act2 offset=200, Act3=600。`burningEliteBuff = mapRng.random(0,3)`（`66`）。
- **Java 对应**（`MapGenerator` / `AbstractDungeon.generateMap`）：真机 act1 用 `seed+1`，act2/3 用
  `seed + act*(100*(act-1))`（即 200/600）。一致。`generateMap` 的 mapRng 全程只此一个实例，
  `createPaths`→`assignRooms`(含 `rng.nextInt(i)` 洗牌)→`assignBurningElite`→`random(0,3)`，消耗顺序与 sim 一致。
- **结论**：**正确**，记录。`assignRooms` 的 `rng.nextInt(i)`（`Map.cpp:729`）= Fisher-Yates，
  与真机 `Collections.shuffle` 等价的手写洗牌一致。

> 复核：CONFIRMED（offset 公式正确）— sim `Map::fromSeed`（`Map.cpp:56-59`）`offset = act==1 ? 1 : act*(100*(act-1))`，mapRng=`Random(seed+offset)`。真机各 act 的 mapRng seed：Exordium `Settings.seed + actNum`（`../reference/cardcrawl/dungeons/Exordium.java:53`，actNum=1→+1）、TheCity `seed + actNum*100`（`TheCity.java:42`，actNum=2→+200）、TheBeyond `seed + actNum*200`（`TheBeyond.java:40`，actNum=3→+600）。sim 的 1/200/600 完全吻合。（act4 真机用 `actNum*300`，sim 用固定 `act4Map()` 无 mapRng，因 act4 地图固定不影响。）

### L4. `transitionToAct` 的 `cardRng.setCounter(250/500/750)` 行为
- **sim**：`GameContext.cpp:700-706`。进新 act 时把 cardRng counter 对齐到 250 的倍数（不足则 `setCounter`，
  `setCounter` 用 `randomBoolean()` 推进，`Random.h:148-152`）。
- **Java 对应**（`AbstractDungeon.generateMap` / `nextAct` 里对 cardRng 的对齐）：真机确有
  “每 act 给 cardRng 预留 250 步”的对齐逻辑（防止跨 act 卡池 roll 串味）。sim 用 `setCounter` 复刻。
- **结论**：倾向正确，但 `setCounter` 用 `randomBoolean()`（每次推进 1 步且 counter+1）逼近目标 counter，
  需确认真机是否也是“逐步 randomBoolean 推进到 250 边界”而非直接 `new Random(seed, 250)`。
  **严重度：低**（依赖记忆，建议 reference 核对 `AbstractDungeon` 里对 cardRng counter 的对齐写法）。

> 复核：**CONFIRMED（边界条件 bug，低频低危但确为偏差）** — `../reference/cardcrawl/dungeons/AbstractDungeon.java:2542-2550`（`dungeonTransitionSetup`）。真机：`if(counter>0 && counter<250) setCounter(250); else if(counter>250 && counter<500) setCounter(500); else if(counter>500 && counter<750) setCounter(750);`——条件是**严格区间**，counter==0 和正好对齐值（250/500/750）都**不动**。sim `GameContext.cpp:700-706` 用 `if(counter<250) else if(counter<500) else if(counter<750)`：① counter==0 时 sim 顶到 250、真机不动；② counter==250 时 sim 落到 `<500` 顶到 500、真机不动（保持 250）；counter==500 同理。`setCounter` 用 `randomBoolean()` 逐步推进（真机 `random/Random.java:38-47`，sim `Random.h` 一致），故这些边界会让 cardRng counter 与真机错位。实战中 act 转换时 counter 多半已 >0 且非整 250 倍，故低频；但严格说是 bug。改法：把三个条件改成真机的严格区间（含排除 counter==0 与对齐值）。

### L5. `setupTreasureRoom` 用同一个 `treasureRng.random(99)` 同时决定 haveGold 和 relic tier
- **sim**：`GameContext.cpp:1065-1082`。`getRandomChestSize(treasureRng)`（消耗 1 次，`Game.cpp:365`），
  然后**一个** `roll=treasureRng.random(99)`（`1069`）既比 `chestGoldChances` 又比 tier 阈值。
- **Java 对应**（`AbstractChest` / `TreasureRoom`）：真机宝箱 `roll` 也是复用同一个 `treasureRng.random(99)`
  判 gold 和 tier（先 chestSize 一次，再 roll 一次）。一致。
- **结论**：**正确**，记录。`openTreasureRoomChest` 的金币 `treasureRng.random(amount*0.9f, amount*1.1f)`
  （`1915`，float 重载消耗 1 次）也与真机一致。

> 复核：CONFIRMED（无 bug）— treasureRng "先 chestSize 一次，再 roll(99) 一次复用判 gold 与 tier" 的模型与真机宝箱逻辑一致（`../reference/cardcrawl/rewards/chests/` 各 chest）。float 重载金币 roll 消耗 1 次（真机 `random/Random.java:94-97` `random(float,float)` 同样 counter+1）。本轮未发现新疑点。

### L6. `chooseNeowOption` 的 CURSE drawback：cardRng.random(9) 取 curse
- **sim**：`GameContext.cpp:2278-2284, 2382-2383`。CURSE 抽 `cardRng.random(9)` 索引 `curseCardPool[roll]`
  （池大小应为 10 → random(9)=random(size-1)）。
- **Java 对应**（`NeowReward` CURSE drawback / `AbstractDungeon.returnRandomCurse`）：curse 池 random，
  用 cardRng。`random(9)` 假定 curse 池大小 10——**需确认 `curseCardPoolSize`**。
  `Game.cpp:181-184` 的 `getRandomCurse` 用 `random(curseCardPoolSize-1)`，但这里硬编码 `random(9)`。
  若 `curseCardPoolSize != 10`，则 Neow CURSE 与普通 curse 抽取的池大小**不一致**，且其中一处错。
- **严重度：低→中**（取决于 curseCardPoolSize 真值；硬编码 9 是个 smell）。
- **建议**：把 `cardRng.random(9)` 改成 `cardRng.random(curseCardPoolSize-1)` 或确认池恰好 10。

> 复核：FALSE-POSITIVE（无 bug，仅 magic number smell）— `CardPools.h:63-64` `curseCardPoolSize = 10`，且池正是 {REGRET, INJURY, SHAME, PARASITE, NORMALITY, DOUBT, WRITHE, PAIN, DECAY, CLUMSY}。真机 Neow CURSE 走 `NeowReward.java:170 getCardWithoutRng(CURSE)`→`AbstractDungeon.returnRandomCurse()`（`AbstractDungeon.java:1105-1108`）→`CardLibrary.getCurse()`（`../reference/cardcrawl/helpers/CardLibrary.java:1008-1014`）：排除 AscendersBane/Necronomicurse/CurseOfTheBell/Pride 后 `cardRng.random(0, tmp.size()-1)`。可选 curse 恰为上述 10 张，`random(0,9)` 与 sim 硬编码 `cardRng.random(9)` 同分布同步数（均 1 次 `nextInt(10)`）。建议改成 `random(curseCardPoolSize-1)` 仅为可读性，非功能 bug。

---

## 需要 reference 才能定论的待核查清单（已全部复核完毕，结论见各条 `> 复核:` 行）

1. **H1**：✅ 真机 `monsterHpRng`/`aiRng` = `new Random(Settings.seed)`（generateSeeds 386/388），但每房间重 seed（1727-1728）→ FALSE-POSITIVE。
2. **H2**：✅ `nextRoomTransition` 重置 5 个 rng（含 shuffleRng/cardRandomRng），1727-1731 → **FALSE-POSITIVE（原报告误判）**。
3. **H4**：✅ 真机空池回退用传入 copy（1963-1964）→ **CONFIRMED bug**。
4. **M3/M5**：✅ NeowReward 317-376 / getAnyColorCard CardLibrary 1060-1068 → 均 FALSE-POSITIVE。
5. **M8**：✅ list 12/10 + 重生成（1828/1837, 1674-1687）→ NEEDS-MORE（极低风险，可不修）。
6. **M9**：✅ AbstractRoom 582-610，rewardSize>=4 短路是真机行为、chance=0 仍 roll → FALSE-POSITIVE。
7. **L4**：✅ dungeonTransitionSetup 2542-2550 严格区间 → **CONFIRMED 边界 bug（低频）**。
8. **L6**：✅ curseCardPoolSize=10，真机 CardLibrary.getCurse 1008-1014 → FALSE-POSITIVE。

---

## 附：本轮确认“无 bug / 已对齐”的点（避免后续重复怀疑）
- `Random.h` 的 nextLong/nextInt/nextFloat/nextDouble/random 各重载位级实现（L1、L2）。
- `createCardReward` 稀有度→选卡→去重→批量升级 roll 的顺序与 RARE 不升级（M4）。
- `getEventRoomOutcomeHelper` 先消耗成员 eventRng 1 次、再 copy 给 generateEvent（M1）。
- `Neow::getOptions` 末尾 `random(0,0)` 占位消耗（M2）。
- `Map::fromSeed` 的 act offset、mapRng 单实例消耗顺序、burningEliteBuff（L3）。
- 商店 setupCards 定价循环 / saleIdx / power slot 稀有度提升时机（M10、L5 同理 treasure）。
- 精英遗物 tier `>82` 与 `<83` 整数等价（M6）。
- `rollElite` 死代码不影响单次 monsterRng 消耗（M7）。

> 复核优先级：先修 **H1–H4**（都是确定性的 RNG 错位或纯数值崩坏），再依 reference 处理 M/L 待核查项。

---

## 复核员最终修订（2026-06-23，基于真机 `../reference/cardcrawl/`）

经真机 Java 逐条核对，原报告 13 条"待修/疑似" + 8 条"已确认无 bug"中：

- **真 bug、值得修**：仅 **H3**（Shop::getNewPrice 数值崩坏，RNG 步数不偏）、**H4**（getEvent 空池回退误用成员 eventRng，RNG 偏移）、**L4**（cardRng setCounter 边界 {0,250,500} 多推进，低频）。
- **原报告误判为 bug、实为假阳性**：**H1、H2、M3、M5、M9、L6**。其中 **H2 误判最严重**——真机 `nextRoomTransition`（AbstractDungeon.java:1727-1731）每房间重 seed `monsterHpRng/aiRng/shuffleRng/cardRandomRng/miscRng` 五个，sim 三个全对，**绝不能按原"只重置 miscRng"的建议改，否则会引入真 bug**。M9 的"短路是 sim 自加"判断也错（是真机行为）。
- **本属确认排除项、复核后维持无 bug**：M1、M2、M4、M6、M7、L1、L2、L3、L5、M10。
- **NEEDS-MORE（极低风险，可不修）**：**M8**（monster/elite list 惰性重生成 vs 真机进房重生成，时机差异不影响 monsterRng 对齐，因 act 内仅 generateBoss 另用 monsterRng 且顺序不冲突）。

未发现原报告**遗漏**的新 RNG/流程 bug。
