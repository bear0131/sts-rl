# Ironclad Card Effect Language / 战士卡牌效果语言

This document is generated from `env/core/card_language.py`. Each card is compiled independently for its base and upgraded versions.

本文由 `env/core/card_language.py` 生成。每张牌的基础版和升级版分别编译，不存在依赖自然语言解析的隐式差值。

## Grammar / 语法

```text
card      := CARD_BEGIN TYPE_* RARITY_* COST value statement* CARD_END
statement := DO operation argument* STMT_END
           | TRIGGER event statement* BLOCK_END
           | IF condition statement* BLOCK_END
           | TRAIT property STMT_END
value     := NUM_BEGIN DIGIT_0...DIGIT_9+ NUM_END
           | NUM_BEGIN VALUE_X NUM_END
```

## Model Encoding / 模型编码

Enable with `--card-language`. The model keeps the learned card-ID embedding and adds an effect-language vector produced by token embedding, a local order-sensitive 1-D encoder, and masked pooling. The same effective card vectors are used for hand, reward, selection, shop, draw, discard, exhaust, and full-deck observations.

使用 `--card-language` 开启。模型保留原有卡牌 ID embedding，并叠加由 token embedding、保持局部顺序的一维编码器和掩码池化得到的效果语言向量。手牌、奖励、选牌、商店、抽牌堆、弃牌堆、消耗堆和完整牌组共用同一语义卡表。

Language version / 语言版本: `2`; vocabulary / 词表大小: `125`; maximum combined program length / 最长组合程序: `68`.

## Token Dictionary / Token 词典

| Token | English | 中文 |
|---|---|---|
| `PAD` | Padding; ignored by the model. | 填充；模型忽略。 |
| `ALL` | All matching objects. | 所有匹配对象。 |
| `ALL_COMBAT_PILES` | All combat card zones. | 所有战斗牌区。 |
| `ALL_ENEMIES` | All enemies. | 所有敌人。 |
| `ALL_NON_ATTACK` | All non-Attacks. | 所有非攻击牌。 |
| `ALL_SKILLS` | All Skill cards. | 所有技能牌。 |
| `APPLY` | Apply a status or debuff. | 施加状态或减益。 |
| `ATTACKER` | Enemy that attacked. | 攻击者。 |
| `BASE_VERSION` | The following program is the base card. | 以下程序描述基础版。 |
| `BLOCK_END` | End of a trigger or condition body. | 触发器或条件作用域结束。 |
| `BURN` | Burn status card. | 灼伤。 |
| `CARD_BEGIN` | Start of one card program. | 一张卡牌程序的开始。 |
| `CARD_END` | End of one card program. | 一张卡牌程序的结束。 |
| `CARD_WITH_STRIKE` | Card whose name contains Strike. | 名称含“打击”的牌。 |
| `CHOOSE` | Choose matching objects. | 选择匹配对象。 |
| `CHOOSE_ATTACK_OR_POWER` | Choose an Attack or Power. | 选择攻击或能力牌。 |
| `CHOOSE_NON_EXHUME` | Choose a card other than Exhume. | 选择非发掘牌。 |
| `COST` | Printed energy cost; VALUE_X denotes X-cost. | 印刷能量费用；VALUE_X 表示 X 费。 |
| `CREATED_CARD` | The card just created. | 刚生成的牌。 |
| `CREATE_CARD` | Create a named card. | 生成指定卡。 |
| `CREATE_COPY` | Create copies. | 生成复制。 |
| `CREATE_RANDOM_CARD` | Create a random matching card. | 生成随机匹配卡。 |
| `CURRENT_BLOCK` | Current Block. | 当前格挡。 |
| `DAZED` | Dazed status card. | 晕眩。 |
| `DEAL_DAMAGE` | Deal the stated damage. | 造成给定伤害。 |
| `DEAL_DAMAGE_FORMULA` | Damage uses a named formula. | 伤害使用指定公式。 |
| `DEAL_DAMAGE_FROM` | Damage equals a referenced value. | 伤害等于引用值。 |
| `DIGIT_0` | Decimal digit 0. | 十进制数字 0。 |
| `DIGIT_1` | Decimal digit 1. | 十进制数字 1。 |
| `DIGIT_2` | Decimal digit 2. | 十进制数字 2。 |
| `DIGIT_3` | Decimal digit 3. | 十进制数字 3。 |
| `DIGIT_4` | Decimal digit 4. | 十进制数字 4。 |
| `DIGIT_5` | Decimal digit 5. | 十进制数字 5。 |
| `DIGIT_6` | Decimal digit 6. | 十进制数字 6。 |
| `DIGIT_7` | Decimal digit 7. | 十进制数字 7。 |
| `DIGIT_8` | Decimal digit 8. | 十进制数字 8。 |
| `DIGIT_9` | Decimal digit 9. | 十进制数字 9。 |
| `DISCARD_PILE` | Discard pile. | 弃牌堆。 |
| `DO` | Execute an effect operation. | 执行一个效果操作。 |
| `DOUBLE` | Double the referenced value. | 将引用值翻倍。 |
| `DRAW` | Draw cards. | 抽牌。 |
| `DRAW_PILE` | Draw pile. | 抽牌堆。 |
| `END_TURN` | At end of turn. | 回合结束。 |
| `ETHEREAL` | Ethereal. | 虚无。 |
| `EXHAUST` | Exhaust after use. | 使用后消耗。 |
| `EXHAUSTED_CARD` | Each card exhausted by this effect. | 被本效果消耗的每张牌。 |
| `EXHAUST_CARD` | Exhaust matching cards. | 消耗匹配卡牌。 |
| `EXHAUST_PILE` | Exhaust pile. | 消耗堆。 |
| `FATAL` | If this damage kills the target. | 若此次伤害击杀目标。 |
| `GAIN` | Gain a stat. | 获得属性。 |
| `GAIN_BLOCK` | Gain Block. | 获得格挡。 |
| `GAIN_ENERGY` | Gain energy. | 获得能量。 |
| `GAIN_TEMPORARY` | Gain a temporary stat. | 获得临时属性。 |
| `HAND` | Hand. | 手牌。 |
| `HEAL_FROM` | Heal from a referenced amount. | 按引用数值回复生命。 |
| `IF` | Execute the body if the following condition is true. | 若随后条件成立则执行作用域。 |
| `IGNORE_COST` | Do not pay printed cost. | 不支付印刷费用。 |
| `INCREASE_BASE_DAMAGE` | Permanently increase this instance's combat damage. | 本场战斗增加此实例伤害。 |
| `INNATE` | Innate. | 固有。 |
| `KEEP_BLOCK_BETWEEN_TURNS` | Block is retained between turns. | 回合间保留格挡。 |
| `LOSE` | Lose a stat. | 失去属性。 |
| `LOSE_HP` | Lose HP directly. | 直接失去生命。 |
| `MAX_HP` | Maximum HP. | 最大生命。 |
| `MOVE_CARD` | Move a card between zones. | 在区域间移动卡牌。 |
| `NEXT_ATTACKS` | Upcoming Attack cards. | 接下来攻击牌。 |
| `NO_DRAW` | No Draw. | 无法抽牌。 |
| `NUM_BEGIN` | Start of a numeric literal. | 一个数值字面量的开始。 |
| `NUM_END` | End of a numeric literal. | 一个数值字面量的结束。 |
| `ON_ANY_CARD_EXHAUSTED` | Whenever any card is exhausted. | 任意牌被消耗时。 |
| `ON_ATTACKED_THIS_TURN` | Whenever attacked this turn. | 本回合每次受到攻击。 |
| `ON_CARD_HP_LOSS` | When a card causes HP loss. | 卡牌导致生命损失时。 |
| `ON_DRAW_STATUS` | When a Status is drawn. | 抽到状态牌时。 |
| `ON_DRAW_STATUS_OR_CURSE` | When a Status or Curse is drawn. | 抽到状态或诅咒时。 |
| `ON_GAIN_BLOCK` | Whenever Block is gained. | 每次获得格挡。 |
| `ON_HP_LOSS` | Whenever HP is lost. | 每次失去生命。 |
| `ON_PLAY_ATTACK_THIS_TURN` | Whenever an Attack is played this turn. | 本回合每次打出攻击牌。 |
| `ON_PLAY_SKILL` | Whenever a Skill is played. | 每次打出技能牌。 |
| `ON_THIS_CARD_EXHAUSTED` | When this card is exhausted. | 本牌被消耗时。 |
| `PER` | For each. | 每个。 |
| `PLAYABLE_IF_ALL_HAND_ATTACK` | Playable only if all hand cards are Attacks. | 仅当全部手牌为攻击牌时可打出。 |
| `PLAYED_CARD` | The card just played. | 刚打出的牌。 |
| `PLAY_CARD` | Play a card from a zone. | 从区域打出卡牌。 |
| `PLAY_TWICE` | Play upcoming cards twice. | 重复打出后续卡牌。 |
| `PLUS` | Add the following term. | 加上后续项。 |
| `RANDOM` | Choose randomly. | 随机选择。 |
| `RANDOM_ENEMY` | Random enemy. | 随机敌人。 |
| `RARITY_BASIC` | Basic rarity. | 基础。 |
| `RARITY_COMMON` | Common rarity. | 普通。 |
| `RARITY_RARE` | Rare rarity. | 稀有。 |
| `RARITY_UNCOMMON` | Uncommon rarity. | 罕见。 |
| `REDUCE_COST` | Reduce card cost. | 降低卡牌费用。 |
| `REPEATABLE_UPGRADE` | May be upgraded repeatedly. | 可重复升级。 |
| `REST_OF_HAND` | Every other card in hand. | 其余手牌。 |
| `SEARING_BLOW_UPGRADES` | 12 + n(n+7)/2 for upgrade count n. | 升级 n 次时为 12+n(n+7)/2。 |
| `SELF` | Player. | 玩家。 |
| `SET_COST` | Set card cost. | 设置卡牌费用。 |
| `SET_STRENGTH_MULTIPLIER` | Set Strength scaling multiplier. | 设置力量加成倍率。 |
| `SHUFFLE` | Shuffle into destination. | 洗入目标牌堆。 |
| `START_TURN` | At start of turn. | 回合开始。 |
| `STMT_END` | End of one statement. | 一条语句的结束。 |
| `STRENGTH` | Strength. | 力量。 |
| `TARGET_ENEMY` | Chosen enemy. | 选定敌人。 |
| `TARGET_HAS_VULNERABLE` | Target currently has Vulnerable. | 目标当前有易伤。 |
| `TARGET_INTENDS_ATTACK` | Target intends to attack. | 目标意图攻击。 |
| `THEN_EXHAUST` | Exhaust after playing. | 打出后消耗。 |
| `THIS_CARD` | This card instance. | 本牌实例。 |
| `THIS_COMBAT` | For this combat. | 本场战斗。 |
| `TIMES` | Repeat count. | 重复次数。 |
| `TOP` | Top of pile. | 牌堆顶。 |
| `TRAIT` | Declare an intrinsic card property. | 声明卡牌固有属性。 |
| `TRIGGER` | Execute the body when the following event occurs. | 当随后事件发生时执行作用域。 |
| `TYPE_ATTACK` | Attack card. | 攻击牌。 |
| `TYPE_POWER` | Power card. | 能力牌。 |
| `TYPE_SKILL` | Skill card. | 技能牌。 |
| `UNBLOCKED_DAMAGE_DEALT` | Total unblocked damage dealt. | 实际造成的未格挡伤害。 |
| `UNTIL_END_TURN` | Until turn end. | 直到回合结束。 |
| `UPGRADED_VERSION` | The following program is upgraded. | 以下程序描述升级版。 |
| `UPGRADE_CARD` | Upgrade matching cards. | 升级匹配卡牌。 |
| `UPGRADE_COUNT` | Number of upgrades applied to this card. | 本牌的升级次数。 |
| `VALUE_X` | Energy spent on an X-cost card. | X 费牌实际消耗的能量。 |
| `VERSIONS_BEGIN` | Start of the base/upgraded pair. | 基础版/升级版组合开始。 |
| `VERSIONS_END` | End of the base/upgraded pair. | 基础版/升级版组合结束。 |
| `VULNERABLE` | Vulnerable. | 易伤。 |
| `WEAK` | Weak. | 虚弱。 |
| `WOUND` | Wound status card. | 伤口。 |

## Cards / 卡牌翻译

### 痛击 / Bash

- Type / 类型: `attack`
- Rarity / 稀有度: `basic`
- Cost / 费用: `2` -> `2`
- Damage / 伤害: `8` -> `10`
- Block / 格挡: `0` -> `0`
- English: Deal 8/10 damage. Apply 2/3 Vulnerable.
- 中文: 造成8/10点伤害。施加2/3层易伤。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_BASIC COST NUM_BEGIN DIGIT_2 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_8 NUM_END TARGET_ENEMY STMT_END DO APPLY VULNERABLE NUM_BEGIN DIGIT_2 NUM_END TARGET_ENEMY STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_BASIC COST NUM_BEGIN DIGIT_2 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_1 DIGIT_0 NUM_END TARGET_ENEMY STMT_END DO APPLY VULNERABLE NUM_BEGIN DIGIT_3 NUM_END TARGET_ENEMY STMT_END CARD_END
```

### 防御 / Defend_R

- Type / 类型: `skill`
- Rarity / 稀有度: `basic`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `5` -> `8`
- English: Gain 5/8 Block.
- 中文: 获得5/8点格挡。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_BASIC COST NUM_BEGIN DIGIT_1 NUM_END DO GAIN_BLOCK NUM_BEGIN DIGIT_5 NUM_END SELF STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_BASIC COST NUM_BEGIN DIGIT_1 NUM_END DO GAIN_BLOCK NUM_BEGIN DIGIT_8 NUM_END SELF STMT_END CARD_END
```

### 打击 / Strike_R

- Type / 类型: `attack`
- Rarity / 稀有度: `basic`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `6` -> `9`
- Block / 格挡: `0` -> `0`
- English: Deal 6/9 damage.
- 中文: 造成6/9点伤害。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_BASIC COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_6 NUM_END TARGET_ENEMY STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_BASIC COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_9 NUM_END TARGET_ENEMY STMT_END CARD_END
```

### 愤怒 / Anger

- Type / 类型: `attack`
- Rarity / 稀有度: `common`
- Cost / 费用: `0` -> `0`
- Damage / 伤害: `6` -> `8`
- Block / 格挡: `0` -> `0`
- English: Deal 6/8 damage. Add a copy to the discard pile.
- 中文: 造成6/8点伤害。将一张本牌的复制加入弃牌堆。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_0 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_6 NUM_END TARGET_ENEMY STMT_END DO CREATE_COPY THIS_CARD NUM_BEGIN DIGIT_1 NUM_END DISCARD_PILE STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_0 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_8 NUM_END TARGET_ENEMY STMT_END DO CREATE_COPY THIS_CARD NUM_BEGIN DIGIT_1 NUM_END DISCARD_PILE STMT_END CARD_END
```

### 武装 / Armaments

- Type / 类型: `skill`
- Rarity / 稀有度: `common`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `5` -> `5`
- English: Gain 5 Block. Upgrade one/all cards in hand for this combat.
- 中文: 获得5点格挡。本场战斗中升级手牌中的一张/所有牌。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_COMMON COST NUM_BEGIN DIGIT_1 NUM_END DO GAIN_BLOCK NUM_BEGIN DIGIT_5 NUM_END SELF STMT_END DO UPGRADE_CARD CHOOSE NUM_BEGIN DIGIT_1 NUM_END HAND THIS_COMBAT STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_COMMON COST NUM_BEGIN DIGIT_1 NUM_END DO GAIN_BLOCK NUM_BEGIN DIGIT_5 NUM_END SELF STMT_END DO UPGRADE_CARD ALL HAND THIS_COMBAT STMT_END CARD_END
```

### 全身撞击 / Body Slam

- Type / 类型: `attack`
- Rarity / 稀有度: `common`
- Cost / 费用: `1` -> `0`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Deal damage equal to current Block.
- 中文: 造成等同于当前格挡值的伤害。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE_FROM CURRENT_BLOCK TARGET_ENEMY STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_0 NUM_END DO DEAL_DAMAGE_FROM CURRENT_BLOCK TARGET_ENEMY STMT_END CARD_END
```

### 交锋 / Clash

- Type / 类型: `attack`
- Rarity / 稀有度: `common`
- Cost / 费用: `0` -> `0`
- Damage / 伤害: `14` -> `18`
- Block / 格挡: `0` -> `0`
- English: Playable only if every card in hand is an Attack. Deal 14/18 damage.
- 中文: 仅当所有手牌均为攻击牌时可打出。造成14/18点伤害。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_0 NUM_END TRAIT PLAYABLE_IF_ALL_HAND_ATTACK STMT_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_1 DIGIT_4 NUM_END TARGET_ENEMY STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_0 NUM_END TRAIT PLAYABLE_IF_ALL_HAND_ATTACK STMT_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_1 DIGIT_8 NUM_END TARGET_ENEMY STMT_END CARD_END
```

### 顺劈斩 / Cleave

- Type / 类型: `attack`
- Rarity / 稀有度: `common`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `8` -> `11`
- Block / 格挡: `0` -> `0`
- English: Deal 8/11 damage to all enemies.
- 中文: 对所有敌人造成8/11点伤害。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_8 NUM_END ALL_ENEMIES STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_1 DIGIT_1 NUM_END ALL_ENEMIES STMT_END CARD_END
```

### 金刚臂 / Clothesline

- Type / 类型: `attack`
- Rarity / 稀有度: `common`
- Cost / 费用: `2` -> `2`
- Damage / 伤害: `12` -> `14`
- Block / 格挡: `0` -> `0`
- English: Deal 12/14 damage. Apply 2/3 Weak.
- 中文: 造成12/14点伤害。施加2/3层虚弱。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_2 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_1 DIGIT_2 NUM_END TARGET_ENEMY STMT_END DO APPLY WEAK NUM_BEGIN DIGIT_2 NUM_END TARGET_ENEMY STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_2 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_1 DIGIT_4 NUM_END TARGET_ENEMY STMT_END DO APPLY WEAK NUM_BEGIN DIGIT_3 NUM_END TARGET_ENEMY STMT_END CARD_END
```

### 活动肌肉 / Flex

- Type / 类型: `skill`
- Rarity / 稀有度: `common`
- Cost / 费用: `0` -> `0`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Gain 2/4 Strength; lose it at end of turn.
- 中文: 获得2/4点力量，并在回合结束时失去这些力量。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_COMMON COST NUM_BEGIN DIGIT_0 NUM_END DO GAIN_TEMPORARY STRENGTH NUM_BEGIN DIGIT_2 NUM_END SELF UNTIL_END_TURN STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_COMMON COST NUM_BEGIN DIGIT_0 NUM_END DO GAIN_TEMPORARY STRENGTH NUM_BEGIN DIGIT_4 NUM_END SELF UNTIL_END_TURN STMT_END CARD_END
```

### 破灭 / Havoc

- Type / 类型: `skill`
- Rarity / 稀有度: `common`
- Cost / 费用: `1` -> `0`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Play the top draw-pile card and Exhaust it.
- 中文: 打出抽牌堆顶牌并将其消耗。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_COMMON COST NUM_BEGIN DIGIT_1 NUM_END DO PLAY_CARD TOP DRAW_PILE THEN_EXHAUST IGNORE_COST STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_COMMON COST NUM_BEGIN DIGIT_0 NUM_END DO PLAY_CARD TOP DRAW_PILE THEN_EXHAUST IGNORE_COST STMT_END CARD_END
```

### 头槌 / Headbutt

- Type / 类型: `attack`
- Rarity / 稀有度: `common`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `9` -> `12`
- Block / 格挡: `0` -> `0`
- English: Deal 9/12 damage. Put a discard-pile card on top of draw pile.
- 中文: 造成9/12点伤害。将弃牌堆中的一张牌置于抽牌堆顶。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_9 NUM_END TARGET_ENEMY STMT_END DO MOVE_CARD CHOOSE NUM_BEGIN DIGIT_1 NUM_END DISCARD_PILE TOP DRAW_PILE STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_1 DIGIT_2 NUM_END TARGET_ENEMY STMT_END DO MOVE_CARD CHOOSE NUM_BEGIN DIGIT_1 NUM_END DISCARD_PILE TOP DRAW_PILE STMT_END CARD_END
```

### 重刃 / Heavy Blade

- Type / 类型: `attack`
- Rarity / 稀有度: `common`
- Cost / 费用: `2` -> `2`
- Damage / 伤害: `14` -> `14`
- Block / 格挡: `0` -> `0`
- English: Deal 14 damage. Strength affects it 3/5 times.
- 中文: 造成14点伤害。力量对本牌生效3/5倍。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_2 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_1 DIGIT_4 NUM_END TARGET_ENEMY STMT_END DO SET_STRENGTH_MULTIPLIER NUM_BEGIN DIGIT_3 NUM_END THIS_CARD STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_2 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_1 DIGIT_4 NUM_END TARGET_ENEMY STMT_END DO SET_STRENGTH_MULTIPLIER NUM_BEGIN DIGIT_5 NUM_END THIS_CARD STMT_END CARD_END
```

### 铁斩波 / Iron Wave

- Type / 类型: `attack`
- Rarity / 稀有度: `common`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `5` -> `7`
- Block / 格挡: `5` -> `7`
- English: Gain 5/7 Block. Deal 5/7 damage.
- 中文: 获得5/7点格挡。造成5/7点伤害。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_1 NUM_END DO GAIN_BLOCK NUM_BEGIN DIGIT_5 NUM_END SELF STMT_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_5 NUM_END TARGET_ENEMY STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_1 NUM_END DO GAIN_BLOCK NUM_BEGIN DIGIT_7 NUM_END SELF STMT_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_7 NUM_END TARGET_ENEMY STMT_END CARD_END
```

### 完美打击 / Perfected Strike

- Type / 类型: `attack`
- Rarity / 稀有度: `common`
- Cost / 费用: `2` -> `2`
- Damage / 伤害: `6` -> `6`
- Block / 格挡: `0` -> `0`
- English: Deal 6 damage, plus 2/3 per card containing 'Strike'.
- 中文: 造成6点伤害。每张名称中含“打击”的牌使伤害增加2/3点。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_2 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_6 NUM_END PLUS NUM_BEGIN DIGIT_2 NUM_END PER CARD_WITH_STRIKE ALL_COMBAT_PILES TARGET_ENEMY STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_2 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_6 NUM_END PLUS NUM_BEGIN DIGIT_3 NUM_END PER CARD_WITH_STRIKE ALL_COMBAT_PILES TARGET_ENEMY STMT_END CARD_END
```

### 剑柄打击 / Pommel Strike

- Type / 类型: `attack`
- Rarity / 稀有度: `common`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `9` -> `10`
- Block / 格挡: `0` -> `0`
- English: Deal 9/10 damage. Draw 1/2 cards.
- 中文: 造成9/10点伤害。抽1/2张牌。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_9 NUM_END TARGET_ENEMY STMT_END DO DRAW NUM_BEGIN DIGIT_1 NUM_END SELF STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_1 DIGIT_0 NUM_END TARGET_ENEMY STMT_END DO DRAW NUM_BEGIN DIGIT_2 NUM_END SELF STMT_END CARD_END
```

### 耸肩无视 / Shrug It Off

- Type / 类型: `skill`
- Rarity / 稀有度: `common`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `8` -> `11`
- English: Gain 8/11 Block. Draw 1 card.
- 中文: 获得8/11点格挡。抽1张牌。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_COMMON COST NUM_BEGIN DIGIT_1 NUM_END DO GAIN_BLOCK NUM_BEGIN DIGIT_8 NUM_END SELF STMT_END DO DRAW NUM_BEGIN DIGIT_1 NUM_END SELF STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_COMMON COST NUM_BEGIN DIGIT_1 NUM_END DO GAIN_BLOCK NUM_BEGIN DIGIT_1 DIGIT_1 NUM_END SELF STMT_END DO DRAW NUM_BEGIN DIGIT_1 NUM_END SELF STMT_END CARD_END
```

### 飞剑回旋镖 / Sword Boomerang

- Type / 类型: `attack`
- Rarity / 稀有度: `common`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `3` -> `3`
- Block / 格挡: `0` -> `0`
- English: Deal 3 damage to a random enemy 3/4 times.
- 中文: 对随机敌人造成3点伤害，共3/4次。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_3 NUM_END RANDOM_ENEMY TIMES NUM_BEGIN DIGIT_3 NUM_END STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_3 NUM_END RANDOM_ENEMY TIMES NUM_BEGIN DIGIT_4 NUM_END STMT_END CARD_END
```

### 闪电霹雳 / Thunderclap

- Type / 类型: `attack`
- Rarity / 稀有度: `common`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `4` -> `7`
- Block / 格挡: `0` -> `0`
- English: Deal 4/7 damage and apply 1 Vulnerable to all enemies.
- 中文: 对所有敌人造成4/7点伤害并施加1层易伤。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_4 NUM_END ALL_ENEMIES STMT_END DO APPLY VULNERABLE NUM_BEGIN DIGIT_1 NUM_END ALL_ENEMIES STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_7 NUM_END ALL_ENEMIES STMT_END DO APPLY VULNERABLE NUM_BEGIN DIGIT_1 NUM_END ALL_ENEMIES STMT_END CARD_END
```

### 坚毅 / True Grit

- Type / 类型: `skill`
- Rarity / 稀有度: `common`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `7` -> `9`
- English: Gain 7/9 Block. Exhaust one random/chosen card.
- 中文: 获得7/9点格挡。随机/选择消耗一张牌。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_COMMON COST NUM_BEGIN DIGIT_1 NUM_END DO GAIN_BLOCK NUM_BEGIN DIGIT_7 NUM_END SELF STMT_END DO EXHAUST_CARD RANDOM NUM_BEGIN DIGIT_1 NUM_END HAND STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_COMMON COST NUM_BEGIN DIGIT_1 NUM_END DO GAIN_BLOCK NUM_BEGIN DIGIT_9 NUM_END SELF STMT_END DO EXHAUST_CARD CHOOSE NUM_BEGIN DIGIT_1 NUM_END HAND STMT_END CARD_END
```

### 双重打击 / Twin Strike

- Type / 类型: `attack`
- Rarity / 稀有度: `common`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `5` -> `7`
- Block / 格挡: `0` -> `0`
- English: Deal 5/7 damage twice.
- 中文: 造成2次5/7点伤害。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_5 NUM_END TARGET_ENEMY TIMES NUM_BEGIN DIGIT_2 NUM_END STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_7 NUM_END TARGET_ENEMY TIMES NUM_BEGIN DIGIT_2 NUM_END STMT_END CARD_END
```

### 战吼 / Warcry

- Type / 类型: `skill`
- Rarity / 稀有度: `common`
- Cost / 费用: `0` -> `0`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Draw 1/2 cards. Put a hand card on top of draw pile. Exhaust.
- 中文: 抽1/2张牌。将一张手牌置于抽牌堆顶。消耗。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_COMMON COST NUM_BEGIN DIGIT_0 NUM_END DO DRAW NUM_BEGIN DIGIT_1 NUM_END SELF STMT_END DO MOVE_CARD CHOOSE NUM_BEGIN DIGIT_1 NUM_END HAND TOP DRAW_PILE STMT_END TRAIT EXHAUST STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_COMMON COST NUM_BEGIN DIGIT_0 NUM_END DO DRAW NUM_BEGIN DIGIT_2 NUM_END SELF STMT_END DO MOVE_CARD CHOOSE NUM_BEGIN DIGIT_1 NUM_END HAND TOP DRAW_PILE STMT_END TRAIT EXHAUST STMT_END CARD_END
```

### 狂野打击 / Wild Strike

- Type / 类型: `attack`
- Rarity / 稀有度: `common`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `12` -> `17`
- Block / 格挡: `0` -> `0`
- English: Deal 12/17 damage. Shuffle a Wound into draw pile.
- 中文: 造成12/17点伤害。将一张伤口洗入抽牌堆。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_1 DIGIT_2 NUM_END TARGET_ENEMY STMT_END DO CREATE_CARD WOUND NUM_BEGIN DIGIT_1 NUM_END DRAW_PILE SHUFFLE STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_COMMON COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_1 DIGIT_7 NUM_END TARGET_ENEMY STMT_END DO CREATE_CARD WOUND NUM_BEGIN DIGIT_1 NUM_END DRAW_PILE SHUFFLE STMT_END CARD_END
```

### 战斗专注 / Battle Trance

- Type / 类型: `skill`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `0` -> `0`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Draw 3/4 cards. Cannot draw more cards this turn.
- 中文: 抽3/4张牌。本回合不能再抽牌。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_0 NUM_END DO DRAW NUM_BEGIN DIGIT_3 NUM_END SELF STMT_END DO APPLY NO_DRAW NUM_BEGIN DIGIT_1 NUM_END SELF UNTIL_END_TURN STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_0 NUM_END DO DRAW NUM_BEGIN DIGIT_4 NUM_END SELF STMT_END DO APPLY NO_DRAW NUM_BEGIN DIGIT_1 NUM_END SELF UNTIL_END_TURN STMT_END CARD_END
```

### 以血还血 / Blood For Blood

- Type / 类型: `attack`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `4` -> `3`
- Damage / 伤害: `18` -> `22`
- Block / 格挡: `0` -> `0`
- English: Costs 1 less each time HP is lost this combat. Deal 18/22 damage.
- 中文: 本场战斗中每次失去生命，费用降低1。造成18/22点伤害。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_UNCOMMON COST NUM_BEGIN DIGIT_4 NUM_END TRIGGER ON_HP_LOSS DO REDUCE_COST NUM_BEGIN DIGIT_1 NUM_END THIS_CARD THIS_COMBAT STMT_END BLOCK_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_1 DIGIT_8 NUM_END TARGET_ENEMY STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_UNCOMMON COST NUM_BEGIN DIGIT_3 NUM_END TRIGGER ON_HP_LOSS DO REDUCE_COST NUM_BEGIN DIGIT_1 NUM_END THIS_CARD THIS_COMBAT STMT_END BLOCK_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_2 DIGIT_2 NUM_END TARGET_ENEMY STMT_END CARD_END
```

### 放血 / Bloodletting

- Type / 类型: `skill`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `0` -> `0`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Lose 3 HP. Gain 2/3 Energy.
- 中文: 失去3点生命。获得2/3点能量。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_0 NUM_END DO LOSE_HP NUM_BEGIN DIGIT_3 NUM_END SELF STMT_END DO GAIN_ENERGY NUM_BEGIN DIGIT_2 NUM_END SELF STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_0 NUM_END DO LOSE_HP NUM_BEGIN DIGIT_3 NUM_END SELF STMT_END DO GAIN_ENERGY NUM_BEGIN DIGIT_3 NUM_END SELF STMT_END CARD_END
```

### 燃烧契约 / Burning Pact

- Type / 类型: `skill`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Exhaust 1 card. Draw 2/3 cards.
- 中文: 消耗1张牌。抽2/3张牌。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END DO EXHAUST_CARD CHOOSE NUM_BEGIN DIGIT_1 NUM_END HAND STMT_END DO DRAW NUM_BEGIN DIGIT_2 NUM_END SELF STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END DO EXHAUST_CARD CHOOSE NUM_BEGIN DIGIT_1 NUM_END HAND STMT_END DO DRAW NUM_BEGIN DIGIT_3 NUM_END SELF STMT_END CARD_END
```

### 残杀 / Carnage

- Type / 类型: `attack`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `2` -> `2`
- Damage / 伤害: `20` -> `28`
- Block / 格挡: `0` -> `0`
- English: Ethereal. Deal 20/28 damage.
- 中文: 虚无。造成20/28点伤害。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_UNCOMMON COST NUM_BEGIN DIGIT_2 NUM_END TRAIT ETHEREAL STMT_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_2 DIGIT_0 NUM_END TARGET_ENEMY STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_UNCOMMON COST NUM_BEGIN DIGIT_2 NUM_END TRAIT ETHEREAL STMT_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_2 DIGIT_8 NUM_END TARGET_ENEMY STMT_END CARD_END
```

### 燃烧 / Combust

- Type / 类型: `power`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: At end of turn lose 1 HP and deal 5/7 damage to all enemies.
- 中文: 回合结束时失去1点生命，并对所有敌人造成5/7点伤害。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_POWER RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END TRIGGER END_TURN DO LOSE_HP NUM_BEGIN DIGIT_1 NUM_END SELF STMT_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_5 NUM_END ALL_ENEMIES STMT_END BLOCK_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_POWER RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END TRIGGER END_TURN DO LOSE_HP NUM_BEGIN DIGIT_1 NUM_END SELF STMT_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_7 NUM_END ALL_ENEMIES STMT_END BLOCK_END CARD_END
```

### 黑暗之拥 / Dark Embrace

- Type / 类型: `power`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `2` -> `1`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Whenever a card is Exhausted, draw 1 card.
- 中文: 每当一张牌被消耗时，抽1张牌。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_POWER RARITY_UNCOMMON COST NUM_BEGIN DIGIT_2 NUM_END TRIGGER ON_ANY_CARD_EXHAUSTED DO DRAW NUM_BEGIN DIGIT_1 NUM_END SELF STMT_END BLOCK_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_POWER RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END TRIGGER ON_ANY_CARD_EXHAUSTED DO DRAW NUM_BEGIN DIGIT_1 NUM_END SELF STMT_END BLOCK_END CARD_END
```

### 缴械 / Disarm

- Type / 类型: `skill`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Enemy loses 2/3 Strength. Exhaust.
- 中文: 敌人失去2/3点力量。消耗。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END DO LOSE STRENGTH NUM_BEGIN DIGIT_2 NUM_END TARGET_ENEMY STMT_END TRAIT EXHAUST STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END DO LOSE STRENGTH NUM_BEGIN DIGIT_3 NUM_END TARGET_ENEMY STMT_END TRAIT EXHAUST STMT_END CARD_END
```

### 飞身踢 / Dropkick

- Type / 类型: `attack`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `5` -> `8`
- Block / 格挡: `0` -> `0`
- English: Deal 5/8 damage. If target is Vulnerable, gain 1 Energy and draw 1.
- 中文: 造成5/8点伤害。若目标有易伤，获得1点能量并抽1张牌。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_5 NUM_END TARGET_ENEMY STMT_END IF TARGET_HAS_VULNERABLE DO GAIN_ENERGY NUM_BEGIN DIGIT_1 NUM_END SELF STMT_END DO DRAW NUM_BEGIN DIGIT_1 NUM_END SELF STMT_END BLOCK_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_8 NUM_END TARGET_ENEMY STMT_END IF TARGET_HAS_VULNERABLE DO GAIN_ENERGY NUM_BEGIN DIGIT_1 NUM_END SELF STMT_END DO DRAW NUM_BEGIN DIGIT_1 NUM_END SELF STMT_END BLOCK_END CARD_END
```

### 双持 / Dual Wield

- Type / 类型: `skill`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Choose an Attack or Power; add 1/2 copies to hand.
- 中文: 选择一张攻击牌或能力牌，将1/2张复制加入手牌。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END DO CREATE_COPY CHOOSE_ATTACK_OR_POWER NUM_BEGIN DIGIT_1 NUM_END HAND STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END DO CREATE_COPY CHOOSE_ATTACK_OR_POWER NUM_BEGIN DIGIT_2 NUM_END HAND STMT_END CARD_END
```

### 巩固 / Entrench

- Type / 类型: `skill`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `2` -> `1`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Double current Block.
- 中文: 使当前格挡翻倍。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_2 NUM_END DO DOUBLE CURRENT_BLOCK SELF STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END DO DOUBLE CURRENT_BLOCK SELF STMT_END CARD_END
```

### 进化 / Evolve

- Type / 类型: `power`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Whenever a Status is drawn, draw 1/2 additional cards.
- 中文: 每当抽到状态牌时，额外抽1/2张牌。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_POWER RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END TRIGGER ON_DRAW_STATUS DO DRAW NUM_BEGIN DIGIT_1 NUM_END SELF STMT_END BLOCK_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_POWER RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END TRIGGER ON_DRAW_STATUS DO DRAW NUM_BEGIN DIGIT_2 NUM_END SELF STMT_END BLOCK_END CARD_END
```

### 无惧疼痛 / Feel No Pain

- Type / 类型: `power`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Whenever a card is Exhausted, gain 3/4 Block.
- 中文: 每当一张牌被消耗时，获得3/4点格挡。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_POWER RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END TRIGGER ON_ANY_CARD_EXHAUSTED DO GAIN_BLOCK NUM_BEGIN DIGIT_3 NUM_END SELF STMT_END BLOCK_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_POWER RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END TRIGGER ON_ANY_CARD_EXHAUSTED DO GAIN_BLOCK NUM_BEGIN DIGIT_4 NUM_END SELF STMT_END BLOCK_END CARD_END
```

### 火焰吐息 / Fire Breathing

- Type / 类型: `power`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: When a Status or Curse is drawn, deal 6/10 damage to all enemies.
- 中文: 每当抽到状态牌或诅咒牌时，对所有敌人造成6/10点伤害。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_POWER RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END TRIGGER ON_DRAW_STATUS_OR_CURSE DO DEAL_DAMAGE NUM_BEGIN DIGIT_6 NUM_END ALL_ENEMIES STMT_END BLOCK_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_POWER RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END TRIGGER ON_DRAW_STATUS_OR_CURSE DO DEAL_DAMAGE NUM_BEGIN DIGIT_1 DIGIT_0 NUM_END ALL_ENEMIES STMT_END BLOCK_END CARD_END
```

### 火焰屏障 / Flame Barrier

- Type / 类型: `skill`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `2` -> `2`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `12` -> `16`
- English: Gain 12/16 Block. This turn, attackers take 4/6 damage.
- 中文: 获得12/16点格挡。本回合每次受到攻击时，对攻击者造成4/6点伤害。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_2 NUM_END DO GAIN_BLOCK NUM_BEGIN DIGIT_1 DIGIT_2 NUM_END SELF STMT_END TRIGGER ON_ATTACKED_THIS_TURN DO DEAL_DAMAGE NUM_BEGIN DIGIT_4 NUM_END ATTACKER STMT_END BLOCK_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_2 NUM_END DO GAIN_BLOCK NUM_BEGIN DIGIT_1 DIGIT_6 NUM_END SELF STMT_END TRIGGER ON_ATTACKED_THIS_TURN DO DEAL_DAMAGE NUM_BEGIN DIGIT_6 NUM_END ATTACKER STMT_END BLOCK_END CARD_END
```

### 幽灵铠甲 / Ghostly Armor

- Type / 类型: `skill`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `10` -> `13`
- English: Ethereal. Gain 10/13 Block.
- 中文: 虚无。获得10/13点格挡。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END TRAIT ETHEREAL STMT_END DO GAIN_BLOCK NUM_BEGIN DIGIT_1 DIGIT_0 NUM_END SELF STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END TRAIT ETHEREAL STMT_END DO GAIN_BLOCK NUM_BEGIN DIGIT_1 DIGIT_3 NUM_END SELF STMT_END CARD_END
```

### 御血术 / Hemokinesis

- Type / 类型: `attack`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `15` -> `20`
- Block / 格挡: `0` -> `0`
- English: Lose 2 HP. Deal 15/20 damage.
- 中文: 失去2点生命。造成15/20点伤害。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END DO LOSE_HP NUM_BEGIN DIGIT_2 NUM_END SELF STMT_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_1 DIGIT_5 NUM_END TARGET_ENEMY STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END DO LOSE_HP NUM_BEGIN DIGIT_2 NUM_END SELF STMT_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_2 DIGIT_0 NUM_END TARGET_ENEMY STMT_END CARD_END
```

### 地狱之刃 / Infernal Blade

- Type / 类型: `skill`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `1` -> `0`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Add a random Attack to hand; it costs 0 this turn. Exhaust.
- 中文: 将一张随机攻击牌加入手牌；其本回合费用为0。消耗。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END DO CREATE_RANDOM_CARD TYPE_ATTACK NUM_BEGIN DIGIT_1 NUM_END HAND STMT_END DO SET_COST NUM_BEGIN DIGIT_0 NUM_END CREATED_CARD UNTIL_END_TURN STMT_END TRAIT EXHAUST STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_0 NUM_END DO CREATE_RANDOM_CARD TYPE_ATTACK NUM_BEGIN DIGIT_1 NUM_END HAND STMT_END DO SET_COST NUM_BEGIN DIGIT_0 NUM_END CREATED_CARD UNTIL_END_TURN STMT_END TRAIT EXHAUST STMT_END CARD_END
```

### 燃烧 / Inflame

- Type / 类型: `power`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Gain 2/3 Strength.
- 中文: 获得2/3点力量。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_POWER RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END DO GAIN STRENGTH NUM_BEGIN DIGIT_2 NUM_END SELF STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_POWER RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END DO GAIN STRENGTH NUM_BEGIN DIGIT_3 NUM_END SELF STMT_END CARD_END
```

### 威吓 / Intimidate

- Type / 类型: `skill`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `0` -> `0`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Apply 1/2 Weak to all enemies. Exhaust.
- 中文: 对所有敌人施加1/2层虚弱。消耗。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_0 NUM_END DO APPLY WEAK NUM_BEGIN DIGIT_1 NUM_END ALL_ENEMIES STMT_END TRAIT EXHAUST STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_0 NUM_END DO APPLY WEAK NUM_BEGIN DIGIT_2 NUM_END ALL_ENEMIES STMT_END TRAIT EXHAUST STMT_END CARD_END
```

### 金属化 / Metallicize

- Type / 类型: `power`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: At end of turn, gain 3/4 Block.
- 中文: 回合结束时获得3/4点格挡。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_POWER RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END TRIGGER END_TURN DO GAIN_BLOCK NUM_BEGIN DIGIT_3 NUM_END SELF STMT_END BLOCK_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_POWER RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END TRIGGER END_TURN DO GAIN_BLOCK NUM_BEGIN DIGIT_4 NUM_END SELF STMT_END BLOCK_END CARD_END
```

### 硬撑 / Power Through

- Type / 类型: `skill`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `15` -> `20`
- English: Add 2 Wounds to hand. Gain 15/20 Block.
- 中文: 将2张伤口加入手牌。获得15/20点格挡。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END DO CREATE_CARD WOUND NUM_BEGIN DIGIT_2 NUM_END HAND STMT_END DO GAIN_BLOCK NUM_BEGIN DIGIT_1 DIGIT_5 NUM_END SELF STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END DO CREATE_CARD WOUND NUM_BEGIN DIGIT_2 NUM_END HAND STMT_END DO GAIN_BLOCK NUM_BEGIN DIGIT_2 DIGIT_0 NUM_END SELF STMT_END CARD_END
```

### 连续拳 / Pummel

- Type / 类型: `attack`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `2` -> `2`
- Block / 格挡: `0` -> `0`
- English: Deal 2 damage 4/5 times. Exhaust.
- 中文: 造成4/5次2点伤害。消耗。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_2 NUM_END TARGET_ENEMY TIMES NUM_BEGIN DIGIT_4 NUM_END STMT_END TRAIT EXHAUST STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_2 NUM_END TARGET_ENEMY TIMES NUM_BEGIN DIGIT_5 NUM_END STMT_END TRAIT EXHAUST STMT_END CARD_END
```

### 狂怒 / Rage

- Type / 类型: `skill`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `0` -> `0`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: This turn, gain 3/5 Block whenever an Attack is played.
- 中文: 本回合每打出一张攻击牌，获得3/5点格挡。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_0 NUM_END TRIGGER ON_PLAY_ATTACK_THIS_TURN DO GAIN_BLOCK NUM_BEGIN DIGIT_3 NUM_END SELF STMT_END BLOCK_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_0 NUM_END TRIGGER ON_PLAY_ATTACK_THIS_TURN DO GAIN_BLOCK NUM_BEGIN DIGIT_5 NUM_END SELF STMT_END BLOCK_END CARD_END
```

### 暴走 / Rampage

- Type / 类型: `attack`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `8` -> `8`
- Block / 格挡: `0` -> `0`
- English: Deal 8 damage. Increase this card's damage by 5/8 this combat.
- 中文: 造成8点伤害。本场战斗中本牌每次打出后伤害增加5/8点。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_8 NUM_END TARGET_ENEMY STMT_END DO INCREASE_BASE_DAMAGE NUM_BEGIN DIGIT_5 NUM_END THIS_CARD THIS_COMBAT STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_8 NUM_END TARGET_ENEMY STMT_END DO INCREASE_BASE_DAMAGE NUM_BEGIN DIGIT_8 NUM_END THIS_CARD THIS_COMBAT STMT_END CARD_END
```

### 鲁莽冲锋 / Reckless Charge

- Type / 类型: `attack`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `0` -> `0`
- Damage / 伤害: `7` -> `10`
- Block / 格挡: `0` -> `0`
- English: Deal 7/10 damage. Shuffle a Dazed into draw pile.
- 中文: 造成7/10点伤害。将一张晕眩洗入抽牌堆。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_UNCOMMON COST NUM_BEGIN DIGIT_0 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_7 NUM_END TARGET_ENEMY STMT_END DO CREATE_CARD DAZED NUM_BEGIN DIGIT_1 NUM_END DRAW_PILE SHUFFLE STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_UNCOMMON COST NUM_BEGIN DIGIT_0 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_1 DIGIT_0 NUM_END TARGET_ENEMY STMT_END DO CREATE_CARD DAZED NUM_BEGIN DIGIT_1 NUM_END DRAW_PILE SHUFFLE STMT_END CARD_END
```

### 撕裂 / Rupture

- Type / 类型: `power`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Whenever a card causes HP loss, gain 1/2 Strength.
- 中文: 每当因卡牌效果失去生命时，获得1/2点力量。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_POWER RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END TRIGGER ON_CARD_HP_LOSS DO GAIN STRENGTH NUM_BEGIN DIGIT_1 NUM_END SELF STMT_END BLOCK_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_POWER RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END TRIGGER ON_CARD_HP_LOSS DO GAIN STRENGTH NUM_BEGIN DIGIT_2 NUM_END SELF STMT_END BLOCK_END CARD_END
```

### 灼热攻击 / Searing Blow

- Type / 类型: `attack`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `2` -> `2`
- Damage / 伤害: `12` -> `16`
- Block / 格挡: `0` -> `0`
- English: Deal 12/16 damage. Can be upgraded repeatedly.
- 中文: 造成12/16点伤害。可重复升级。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_UNCOMMON COST NUM_BEGIN DIGIT_2 NUM_END DO DEAL_DAMAGE_FORMULA SEARING_BLOW_UPGRADES UPGRADE_COUNT NUM_BEGIN DIGIT_0 NUM_END TARGET_ENEMY STMT_END TRAIT REPEATABLE_UPGRADE STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_UNCOMMON COST NUM_BEGIN DIGIT_2 NUM_END DO DEAL_DAMAGE_FORMULA SEARING_BLOW_UPGRADES UPGRADE_COUNT NUM_BEGIN DIGIT_1 NUM_END TARGET_ENEMY STMT_END TRAIT REPEATABLE_UPGRADE STMT_END CARD_END
```

### 重振精神 / Second Wind

- Type / 类型: `skill`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `5` -> `7`
- English: Exhaust all non-Attacks in hand; gain 5/7 Block per card.
- 中文: 消耗手牌中所有非攻击牌；每张获得5/7点格挡。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END DO EXHAUST_CARD ALL_NON_ATTACK HAND STMT_END DO GAIN_BLOCK NUM_BEGIN DIGIT_5 NUM_END PER EXHAUSTED_CARD SELF STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END DO EXHAUST_CARD ALL_NON_ATTACK HAND STMT_END DO GAIN_BLOCK NUM_BEGIN DIGIT_7 NUM_END PER EXHAUSTED_CARD SELF STMT_END CARD_END
```

### 盛怒 / Seeing Red

- Type / 类型: `skill`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `1` -> `0`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Gain 2 Energy. Exhaust.
- 中文: 获得2点能量。消耗。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END DO GAIN_ENERGY NUM_BEGIN DIGIT_2 NUM_END SELF STMT_END TRAIT EXHAUST STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_0 NUM_END DO GAIN_ENERGY NUM_BEGIN DIGIT_2 NUM_END SELF STMT_END TRAIT EXHAUST STMT_END CARD_END
```

### 哨卫 / Sentinel

- Type / 类型: `skill`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `5` -> `8`
- English: Gain 5/8 Block. If Exhausted, gain 2/3 Energy.
- 中文: 获得5/8点格挡。若本牌被消耗，获得2/3点能量。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END DO GAIN_BLOCK NUM_BEGIN DIGIT_5 NUM_END SELF STMT_END TRIGGER ON_THIS_CARD_EXHAUSTED DO GAIN_ENERGY NUM_BEGIN DIGIT_2 NUM_END SELF STMT_END BLOCK_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END DO GAIN_BLOCK NUM_BEGIN DIGIT_8 NUM_END SELF STMT_END TRIGGER ON_THIS_CARD_EXHAUSTED DO GAIN_ENERGY NUM_BEGIN DIGIT_3 NUM_END SELF STMT_END BLOCK_END CARD_END
```

### 断魂斩 / Sever Soul

- Type / 类型: `attack`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `2` -> `2`
- Damage / 伤害: `16` -> `22`
- Block / 格挡: `0` -> `0`
- English: Exhaust all non-Attacks in hand. Deal 16/22 damage.
- 中文: 消耗手牌中所有非攻击牌。造成16/22点伤害。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_UNCOMMON COST NUM_BEGIN DIGIT_2 NUM_END DO EXHAUST_CARD ALL_NON_ATTACK HAND STMT_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_1 DIGIT_6 NUM_END TARGET_ENEMY STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_UNCOMMON COST NUM_BEGIN DIGIT_2 NUM_END DO EXHAUST_CARD ALL_NON_ATTACK HAND STMT_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_2 DIGIT_2 NUM_END TARGET_ENEMY STMT_END CARD_END
```

### 震荡波 / Shockwave

- Type / 类型: `skill`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `2` -> `2`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Apply 3/5 Weak and Vulnerable to all enemies. Exhaust.
- 中文: 对所有敌人施加3/5层虚弱和易伤。消耗。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_2 NUM_END DO APPLY WEAK NUM_BEGIN DIGIT_3 NUM_END ALL_ENEMIES STMT_END DO APPLY VULNERABLE NUM_BEGIN DIGIT_3 NUM_END ALL_ENEMIES STMT_END TRAIT EXHAUST STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_2 NUM_END DO APPLY WEAK NUM_BEGIN DIGIT_5 NUM_END ALL_ENEMIES STMT_END DO APPLY VULNERABLE NUM_BEGIN DIGIT_5 NUM_END ALL_ENEMIES STMT_END TRAIT EXHAUST STMT_END CARD_END
```

### 观察弱点 / Spot Weakness

- Type / 类型: `skill`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: If target intends to attack, gain 3/4 Strength.
- 中文: 若目标意图攻击，获得3/4点力量。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END IF TARGET_INTENDS_ATTACK DO GAIN STRENGTH NUM_BEGIN DIGIT_3 NUM_END SELF STMT_END BLOCK_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_UNCOMMON COST NUM_BEGIN DIGIT_1 NUM_END IF TARGET_INTENDS_ATTACK DO GAIN STRENGTH NUM_BEGIN DIGIT_4 NUM_END SELF STMT_END BLOCK_END CARD_END
```

### 上勾拳 / Uppercut

- Type / 类型: `attack`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `2` -> `2`
- Damage / 伤害: `13` -> `13`
- Block / 格挡: `0` -> `0`
- English: Deal 13 damage. Apply 1/2 Weak and 1/2 Vulnerable.
- 中文: 造成13点伤害。施加1/2层虚弱和1/2层易伤。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_UNCOMMON COST NUM_BEGIN DIGIT_2 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_1 DIGIT_3 NUM_END TARGET_ENEMY STMT_END DO APPLY WEAK NUM_BEGIN DIGIT_1 NUM_END TARGET_ENEMY STMT_END DO APPLY VULNERABLE NUM_BEGIN DIGIT_1 NUM_END TARGET_ENEMY STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_UNCOMMON COST NUM_BEGIN DIGIT_2 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_1 DIGIT_3 NUM_END TARGET_ENEMY STMT_END DO APPLY WEAK NUM_BEGIN DIGIT_2 NUM_END TARGET_ENEMY STMT_END DO APPLY VULNERABLE NUM_BEGIN DIGIT_2 NUM_END TARGET_ENEMY STMT_END CARD_END
```

### 旋风斩 / Whirlwind

- Type / 类型: `attack`
- Rarity / 稀有度: `uncommon`
- Cost / 费用: `-1` -> `-1`
- Damage / 伤害: `5` -> `8`
- Block / 格挡: `0` -> `0`
- English: Deal 5/8 damage to all enemies X times.
- 中文: 对所有敌人造成X次5/8点伤害。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_UNCOMMON COST NUM_BEGIN VALUE_X NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_5 NUM_END ALL_ENEMIES TIMES NUM_BEGIN VALUE_X NUM_END STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_UNCOMMON COST NUM_BEGIN VALUE_X NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_8 NUM_END ALL_ENEMIES TIMES NUM_BEGIN VALUE_X NUM_END STMT_END CARD_END
```

### 壁垒 / Barricade

- Type / 类型: `power`
- Rarity / 稀有度: `rare`
- Cost / 费用: `3` -> `2`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Block is not removed at start of turn.
- 中文: 回合开始时不再失去格挡。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_POWER RARITY_RARE COST NUM_BEGIN DIGIT_3 NUM_END TRAIT KEEP_BLOCK_BETWEEN_TURNS STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_POWER RARITY_RARE COST NUM_BEGIN DIGIT_2 NUM_END TRAIT KEEP_BLOCK_BETWEEN_TURNS STMT_END CARD_END
```

### 狂暴 / Berserk

- Type / 类型: `power`
- Rarity / 稀有度: `rare`
- Cost / 费用: `0` -> `0`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Gain 2/1 Vulnerable. At start of turn, gain 1 Energy.
- 中文: 获得2/1层易伤。每回合开始时获得1点能量。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_POWER RARITY_RARE COST NUM_BEGIN DIGIT_0 NUM_END DO APPLY VULNERABLE NUM_BEGIN DIGIT_2 NUM_END SELF STMT_END TRIGGER START_TURN DO GAIN_ENERGY NUM_BEGIN DIGIT_1 NUM_END SELF STMT_END BLOCK_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_POWER RARITY_RARE COST NUM_BEGIN DIGIT_0 NUM_END DO APPLY VULNERABLE NUM_BEGIN DIGIT_1 NUM_END SELF STMT_END TRIGGER START_TURN DO GAIN_ENERGY NUM_BEGIN DIGIT_1 NUM_END SELF STMT_END BLOCK_END CARD_END
```

### 重锤 / Bludgeon

- Type / 类型: `attack`
- Rarity / 稀有度: `rare`
- Cost / 费用: `3` -> `3`
- Damage / 伤害: `32` -> `42`
- Block / 格挡: `0` -> `0`
- English: Deal 32/42 damage.
- 中文: 造成32/42点伤害。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_RARE COST NUM_BEGIN DIGIT_3 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_3 DIGIT_2 NUM_END TARGET_ENEMY STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_RARE COST NUM_BEGIN DIGIT_3 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_4 DIGIT_2 NUM_END TARGET_ENEMY STMT_END CARD_END
```

### 残暴 / Brutality

- Type / 类型: `power`
- Rarity / 稀有度: `rare`
- Cost / 费用: `0` -> `0`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Upgraded is Innate. At start of turn lose 1 HP and draw 1.
- 中文: 升级后固有。每回合开始时失去1点生命并抽1张牌。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_POWER RARITY_RARE COST NUM_BEGIN DIGIT_0 NUM_END TRIGGER START_TURN DO LOSE_HP NUM_BEGIN DIGIT_1 NUM_END SELF STMT_END DO DRAW NUM_BEGIN DIGIT_1 NUM_END SELF STMT_END BLOCK_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_POWER RARITY_RARE COST NUM_BEGIN DIGIT_0 NUM_END TRAIT INNATE STMT_END TRIGGER START_TURN DO LOSE_HP NUM_BEGIN DIGIT_1 NUM_END SELF STMT_END DO DRAW NUM_BEGIN DIGIT_1 NUM_END SELF STMT_END BLOCK_END CARD_END
```

### 腐化 / Corruption

- Type / 类型: `power`
- Rarity / 稀有度: `rare`
- Cost / 费用: `3` -> `2`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Skills cost 0 and are Exhausted when played.
- 中文: 技能牌费用变为0，但打出后会被消耗。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_POWER RARITY_RARE COST NUM_BEGIN DIGIT_3 NUM_END DO SET_COST NUM_BEGIN DIGIT_0 NUM_END ALL_SKILLS THIS_COMBAT STMT_END TRIGGER ON_PLAY_SKILL DO EXHAUST_CARD PLAYED_CARD STMT_END BLOCK_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_POWER RARITY_RARE COST NUM_BEGIN DIGIT_2 NUM_END DO SET_COST NUM_BEGIN DIGIT_0 NUM_END ALL_SKILLS THIS_COMBAT STMT_END TRIGGER ON_PLAY_SKILL DO EXHAUST_CARD PLAYED_CARD STMT_END BLOCK_END CARD_END
```

### 恶魔形态 / Demon Form

- Type / 类型: `power`
- Rarity / 稀有度: `rare`
- Cost / 费用: `3` -> `3`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: At start of turn, gain 2/3 Strength.
- 中文: 每回合开始时获得2/3点力量。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_POWER RARITY_RARE COST NUM_BEGIN DIGIT_3 NUM_END TRIGGER START_TURN DO GAIN STRENGTH NUM_BEGIN DIGIT_2 NUM_END SELF STMT_END BLOCK_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_POWER RARITY_RARE COST NUM_BEGIN DIGIT_3 NUM_END TRIGGER START_TURN DO GAIN STRENGTH NUM_BEGIN DIGIT_3 NUM_END SELF STMT_END BLOCK_END CARD_END
```

### 双发 / Double Tap

- Type / 类型: `skill`
- Rarity / 稀有度: `rare`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: This turn, the next 1/2 Attacks are played twice.
- 中文: 本回合接下来1/2张攻击牌会打出两次。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_RARE COST NUM_BEGIN DIGIT_1 NUM_END DO PLAY_TWICE NUM_BEGIN DIGIT_1 NUM_END NEXT_ATTACKS UNTIL_END_TURN STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_RARE COST NUM_BEGIN DIGIT_1 NUM_END DO PLAY_TWICE NUM_BEGIN DIGIT_2 NUM_END NEXT_ATTACKS UNTIL_END_TURN STMT_END CARD_END
```

### 发掘 / Exhume

- Type / 类型: `skill`
- Rarity / 稀有度: `rare`
- Cost / 费用: `1` -> `0`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Put a non-Exhume card from Exhaust pile into hand. Exhaust.
- 中文: 从消耗堆选择一张非发掘牌加入手牌。消耗。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_RARE COST NUM_BEGIN DIGIT_1 NUM_END DO MOVE_CARD CHOOSE_NON_EXHUME EXHAUST_PILE HAND STMT_END TRAIT EXHAUST STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_RARE COST NUM_BEGIN DIGIT_0 NUM_END DO MOVE_CARD CHOOSE_NON_EXHUME EXHAUST_PILE HAND STMT_END TRAIT EXHAUST STMT_END CARD_END
```

### 吞噬 / Feed

- Type / 类型: `attack`
- Rarity / 稀有度: `rare`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `10` -> `12`
- Block / 格挡: `0` -> `0`
- English: Deal 10/12 damage. Fatal: gain 3/4 Max HP. Exhaust.
- 中文: 造成10/12点伤害。致命：最大生命增加3/4。消耗。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_RARE COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_1 DIGIT_0 NUM_END TARGET_ENEMY STMT_END TRIGGER FATAL DO GAIN MAX_HP NUM_BEGIN DIGIT_3 NUM_END SELF STMT_END BLOCK_END TRAIT EXHAUST STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_RARE COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_1 DIGIT_2 NUM_END TARGET_ENEMY STMT_END TRIGGER FATAL DO GAIN MAX_HP NUM_BEGIN DIGIT_4 NUM_END SELF STMT_END BLOCK_END TRAIT EXHAUST STMT_END CARD_END
```

### 恶魔之焰 / Fiend Fire

- Type / 类型: `attack`
- Rarity / 稀有度: `rare`
- Cost / 费用: `2` -> `2`
- Damage / 伤害: `7` -> `10`
- Block / 格挡: `0` -> `0`
- English: Exhaust the rest of hand; deal 7/10 damage per card. Exhaust.
- 中文: 消耗其余所有手牌；每张造成7/10点伤害。消耗。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_RARE COST NUM_BEGIN DIGIT_2 NUM_END DO EXHAUST_CARD REST_OF_HAND STMT_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_7 NUM_END PER EXHAUSTED_CARD TARGET_ENEMY STMT_END TRAIT EXHAUST STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_RARE COST NUM_BEGIN DIGIT_2 NUM_END DO EXHAUST_CARD REST_OF_HAND STMT_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_1 DIGIT_0 NUM_END PER EXHAUSTED_CARD TARGET_ENEMY STMT_END TRAIT EXHAUST STMT_END CARD_END
```

### 燔祭 / Immolate

- Type / 类型: `attack`
- Rarity / 稀有度: `rare`
- Cost / 费用: `2` -> `2`
- Damage / 伤害: `21` -> `28`
- Block / 格挡: `0` -> `0`
- English: Deal 21/28 damage to all enemies. Add a Burn to discard pile.
- 中文: 对所有敌人造成21/28点伤害。将一张灼伤加入弃牌堆。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_RARE COST NUM_BEGIN DIGIT_2 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_2 DIGIT_1 NUM_END ALL_ENEMIES STMT_END DO CREATE_CARD BURN NUM_BEGIN DIGIT_1 NUM_END DISCARD_PILE STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_RARE COST NUM_BEGIN DIGIT_2 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_2 DIGIT_8 NUM_END ALL_ENEMIES STMT_END DO CREATE_CARD BURN NUM_BEGIN DIGIT_1 NUM_END DISCARD_PILE STMT_END CARD_END
```

### 岿然不动 / Impervious

- Type / 类型: `skill`
- Rarity / 稀有度: `rare`
- Cost / 费用: `2` -> `2`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `30` -> `40`
- English: Gain 30/40 Block. Exhaust.
- 中文: 获得30/40点格挡。消耗。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_RARE COST NUM_BEGIN DIGIT_2 NUM_END DO GAIN_BLOCK NUM_BEGIN DIGIT_3 DIGIT_0 NUM_END SELF STMT_END TRAIT EXHAUST STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_RARE COST NUM_BEGIN DIGIT_2 NUM_END DO GAIN_BLOCK NUM_BEGIN DIGIT_4 DIGIT_0 NUM_END SELF STMT_END TRAIT EXHAUST STMT_END CARD_END
```

### 势不可挡 / Juggernaut

- Type / 类型: `power`
- Rarity / 稀有度: `rare`
- Cost / 费用: `2` -> `2`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Whenever Block is gained, deal 5/7 damage to a random enemy.
- 中文: 每当获得格挡时，对随机敌人造成5/7点伤害。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_POWER RARITY_RARE COST NUM_BEGIN DIGIT_2 NUM_END TRIGGER ON_GAIN_BLOCK DO DEAL_DAMAGE NUM_BEGIN DIGIT_5 NUM_END RANDOM_ENEMY STMT_END BLOCK_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_POWER RARITY_RARE COST NUM_BEGIN DIGIT_2 NUM_END TRIGGER ON_GAIN_BLOCK DO DEAL_DAMAGE NUM_BEGIN DIGIT_7 NUM_END RANDOM_ENEMY STMT_END BLOCK_END CARD_END
```

### 突破极限 / Limit Break

- Type / 类型: `skill`
- Rarity / 稀有度: `rare`
- Cost / 费用: `1` -> `1`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Double Strength. Base Exhausts; upgraded does not.
- 中文: 使力量翻倍。未升级时消耗；升级后不消耗。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_RARE COST NUM_BEGIN DIGIT_1 NUM_END DO DOUBLE STRENGTH SELF STMT_END TRAIT EXHAUST STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_RARE COST NUM_BEGIN DIGIT_1 NUM_END DO DOUBLE STRENGTH SELF STMT_END CARD_END
```

### 祭品 / Offering

- Type / 类型: `skill`
- Rarity / 稀有度: `rare`
- Cost / 费用: `0` -> `0`
- Damage / 伤害: `0` -> `0`
- Block / 格挡: `0` -> `0`
- English: Lose 6 HP. Gain 2 Energy. Draw 3/5 cards. Exhaust.
- 中文: 失去6点生命。获得2点能量。抽3/5张牌。消耗。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_SKILL RARITY_RARE COST NUM_BEGIN DIGIT_0 NUM_END DO LOSE_HP NUM_BEGIN DIGIT_6 NUM_END SELF STMT_END DO GAIN_ENERGY NUM_BEGIN DIGIT_2 NUM_END SELF STMT_END DO DRAW NUM_BEGIN DIGIT_3 NUM_END SELF STMT_END TRAIT EXHAUST STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_SKILL RARITY_RARE COST NUM_BEGIN DIGIT_0 NUM_END DO LOSE_HP NUM_BEGIN DIGIT_6 NUM_END SELF STMT_END DO GAIN_ENERGY NUM_BEGIN DIGIT_2 NUM_END SELF STMT_END DO DRAW NUM_BEGIN DIGIT_5 NUM_END SELF STMT_END TRAIT EXHAUST STMT_END CARD_END
```

### 死神 / Reaper

- Type / 类型: `attack`
- Rarity / 稀有度: `rare`
- Cost / 费用: `2` -> `2`
- Damage / 伤害: `4` -> `5`
- Block / 格挡: `0` -> `0`
- English: Deal 4/5 damage to all enemies; heal unblocked damage. Exhaust.
- 中文: 对所有敌人造成4/5点伤害；回复等同于未被格挡伤害的生命。消耗。
- Base DSL / 基础版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_RARE COST NUM_BEGIN DIGIT_2 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_4 NUM_END ALL_ENEMIES STMT_END DO HEAL_FROM UNBLOCKED_DAMAGE_DEALT SELF STMT_END TRAIT EXHAUST STMT_END CARD_END
```

- Upgraded DSL / 升级版:

```text
CARD_BEGIN TYPE_ATTACK RARITY_RARE COST NUM_BEGIN DIGIT_2 NUM_END DO DEAL_DAMAGE NUM_BEGIN DIGIT_5 NUM_END ALL_ENEMIES STMT_END DO HEAL_FROM UNBLOCKED_DAMAGE_DEALT SELF STMT_END TRAIT EXHAUST STMT_END CARD_END
```

