# v7 step 3000 固定与随机自回归评测

- 参数：`.tmp-vector-rule-v7-contract-v4-independent-damage-resume-20260831/eval-snapshot.pt`
- checkpoint step：`3000`
- schema / executor contract：`7 / 4`
- 随机种子：`20260831`
- 设备：`cpu`
- outer：从 `<OUTPUT_BEGIN>` 开始完全自由贪心生成 effect-tree header。
- local：真实 NodeSpec 前缀由环境给出；模型 token 自回归，错误 token 保留在上下文；只有 `<READ_RESPONSE_*>` 和 `<OBS_*>` 等环境 token 按真实 transcript 注入。
- 牌堆：模型只接收抽牌堆、弃牌堆、消耗堆数量，不接收各堆具体卡牌组成。

## 总体结果

- outer 完全正确：8/12；token 准确率 90.55%（794 个 token）。
- local 完全正确节点：71/76；模型 token 准确率 99.15%（2126 个 token）。
- delta：41 个向量，活跃位 MAE `0.036056`，全向量 MAE `0.001822`。
- 模型输出 `<UNK>`：0 个。

| 案例 | 打出的牌 | outer | local 节点 | local token | delta 活跃位 MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| 固定例1：飞身踢，敌人有2层易伤 | 飞身踢（Dropkick） | 100.00% | 7/8 | 97.73% | 0.027648 |
| 固定例2：剑柄打击，洗牌并触发日晷 | 剑柄打击（Pommel Strike） | 36.28% | 10/11 | 98.83% | 0.059855 |
| 随机例1 | 重刃+（Heavy Blade+） | 98.36% | 5/6 | 99.48% | 0.040562 |
| 随机例2 | 打击（Strike_R） | 98.33% | 5/6 | 96.39% | 0.026995 |
| 随机例3 | 震荡波+（Shockwave+） | 100.00% | 8/8 | 100.00% | 0.069650 |
| 随机例4 | 缴械（Disarm） | 100.00% | 4/4 | 100.00% | 0.014583 |
| 随机例5 | 狂暴+（Berserk+） | 100.00% | 4/4 | 100.00% | 0.007571 |
| 随机例6 | 燃烧（Combust） | 100.00% | 4/4 | 100.00% | 0.020342 |
| 随机例7 | 飞剑回旋镖+（Sword Boomerang+） | 100.00% | 9/10 | 99.59% | 0.026400 |
| 随机例8 | 灼热攻击（Searing Blow） | 98.36% | 6/6 | 100.00% | 0.024995 |
| 随机例9 | 破灭（Havoc） | 100.00% | 4/4 | 100.00% | 0.020998 |
| 随机例10 | 硬撑（Power Through） | 100.00% | 5/5 | 100.00% | 0.018312 |

## 1. 固定例1：飞身踢，敌人有2层易伤

沿用既有固定回归例；敌人没有格挡，易伤参与伤害计算。

- 场景：第 0 层，第 0 幕，固定回归遭遇 `固定测试敌人`，第 1 回合。
- 打出：飞身踢（Dropkick）。卡面规则：造成5/8点伤害。若目标有易伤，获得1点能量并抽1张牌。
- 手牌（3 张）：飞身踢（Dropkick）、打击（Strike_R）、防御（Defend_R）。
- 遗物：无。
- 战士：生命 80/80，格挡 0，能量 3，金币 99；状态：无；计数器：无。
- 牌堆计数：抽牌堆 5，弃牌堆 1，消耗堆 0；具体组成不提供给模型。
- 敌人 slot 0（`ExampleEnemy`）：生命 50/60，格挡 0，意图伤害 0→0，段数 0，状态：易伤=2。

### Outer 结果

准确率 100.00%，完全一致=True，正常结束=True，首次错误：无。

目标 header：

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_ATTACK
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_ATTACK
          ENTITY_REF
          ENTITY_ENEMY_MASK
          <REGION_BEGIN>
            REGION_KIND
            REGION_DAMAGE
            ENTITY_REF
            ENTITY_ENEMY_SLOT_0
            BASE_DAMAGE
            NUM_LSD_BEGIN
            DIGIT_5
            NUM_END
          <REGION_END>
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_ENERGY
          ENTITY_REF
          ENTITY_SELF
          ENERGY_OP
          VALUE_GAIN
          ENERGY_AMOUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_DRAW
          ENTITY_REF
          ENTITY_SELF
          DRAW_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          VALUE_DISCARD
          CARD_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

自回归生成 header：

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_ATTACK
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_ATTACK
          ENTITY_REF
          ENTITY_ENEMY_MASK
          <REGION_BEGIN>
            REGION_KIND
            REGION_DAMAGE
            ENTITY_REF
            ENTITY_ENEMY_SLOT_0
            BASE_DAMAGE
            NUM_LSD_BEGIN
            DIGIT_5
            NUM_END
          <REGION_END>
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_ENERGY
          ENTITY_REF
          ENTITY_SELF
          ENERGY_OP
          VALUE_GAIN
          ENERGY_AMOUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_DRAW
          ENTITY_REF
          ENTITY_SELF
          DRAW_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          VALUE_DISCARD
          CARD_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

### Local 结果

| path | 节点 | 完全正确 | token 准确率 | 首次错误 | delta 活跃位 MAE |
| --- | --- | ---: | ---: | --- | ---: |
| `root` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.036711 |
| `root/1` | `ATTACK` | True | 100.00% | 无 | - |
| `root/1/0` | `DAMAGE` | False | 90.74% | 位置43：期望 `VALUE_enemy_0_hp`，预测 `VALUE_enemy_0_block` | 0.029808 |
| `root/2` | `ENERGY` | True | 100.00% | 无 | 0.050669 |
| `root/3` | `DRAW` | True | 100.00% | 无 | 0.001013 |
| `root/4` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.001912 |
| `root/5` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.100000、预测=-0.147841、绝对误差=0.047841；`player.counter.cards_played` 目标=+0.100000、预测=+0.106125、绝对误差=0.006125；`player.counter.attacks_played` 目标=+0.100000、预测=+0.043833、绝对误差=0.056167。
- `root/1/0` / `DAMAGE`：`enemy[0].hp` 目标=-0.035000、预测=-0.005192、绝对误差=0.029808。
- `root/2` / `ENERGY`：`player.energy` 目标=+0.100000、预测=+0.150669、绝对误差=0.050669。
- `root/3` / `DRAW`：`piles.draw_count` 目标=-0.010000、预测=-0.008987、绝对误差=0.001013。
- `root/4` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.011912、绝对误差=0.001912。

local 出错节点的完整自回归对照：

#### `root/1/0` / `DAMAGE`

目标：

```text
<REGION_BEGIN>
  REGION_KIND
  REGION_DAMAGE
  ENTITY_REF
  ENTITY_ENEMY_SLOT_0
  BASE_DAMAGE
  NUM_LSD_BEGIN
  DIGIT_5
  NUM_END
  <NODE_EXECUTE>
  <CALCULATIONS_BEGIN>
    CALCULATION_BEGIN
      CALCULATION_CHECK_PRESENCE
      CALC_FIELD_field
      VALUE_enemy_0_buff_Vulnerable
      CALC_FIELD_result
      VALUE_reg_has_vulnerable
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_MULTIPLY_VULNERABLE
      CALC_FIELD_left
      VALUE_header_BASE_DAMAGE
      CALC_FIELD_result
      VALUE_reg_damage_vulnerable_numerator
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_5
      DIGIT_1
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_FLOOR_DIVIDE_VULNERABLE
      CALC_FIELD_left
      VALUE_reg_damage_vulnerable_numerator
      CALC_FIELD_result
      VALUE_reg_damage_after_vulnerable
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_7
      NUM_END
    CALCULATION_END
    <READ_REQUEST_BEGIN>
      READ_SOURCE
      VALUE_enemy_0_hp
      READ_BIND
      VALUE_reg_target_hp
    <READ_REQUEST_END>
    <READ_RESPONSE_BEGIN>
      READ_BIND
      VALUE_reg_target_hp
      READ_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      DIGIT_5
      NUM_END
    <READ_RESPONSE_END>
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT_HP
      CALC_FIELD_left
      VALUE_reg_target_hp
      CALC_FIELD_right
      VALUE_reg_damage_after_vulnerable
      CALC_FIELD_result
      VALUE_reg_target_hp_after
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_3
      DIGIT_4
      NUM_END
    CALCULATION_END
  <CALCULATIONS_END>
  DELTA_SLOT
<REGION_END>
```

预测（环境 token 已按真实 transcript 注入）：

```text
<REGION_BEGIN>
  REGION_KIND
  REGION_DAMAGE
  ENTITY_REF
  ENTITY_ENEMY_SLOT_0
  BASE_DAMAGE
  NUM_LSD_BEGIN
  DIGIT_5
  NUM_END
  <NODE_EXECUTE>
  <CALCULATIONS_BEGIN>
    CALCULATION_BEGIN
      CALCULATION_CHECK_PRESENCE
      CALC_FIELD_field
      VALUE_enemy_0_buff_Vulnerable
      CALC_FIELD_result
      VALUE_reg_has_vulnerable
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_MULTIPLY_VULNERABLE
      CALC_FIELD_left
      VALUE_header_BASE_DAMAGE
      CALC_FIELD_result
      VALUE_reg_damage_vulnerable_numerator
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_5
      DIGIT_1
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_FLOOR_DIVIDE_VULNERABLE
      CALC_FIELD_left
      VALUE_reg_damage_vulnerable_numerator
      CALC_FIELD_result
      VALUE_reg_damage_after_vulnerable
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_7
      NUM_END
    CALCULATION_END
    <READ_REQUEST_BEGIN>
      READ_SOURCE
      VALUE_enemy_0_block
      READ_BIND
      VALUE_reg_target_block
    <READ_REQUEST_END>
    <READ_RESPONSE_BEGIN>
      READ_BIND
      VALUE_reg_target_hp
      READ_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      DIGIT_5
      NUM_END
    <READ_RESPONSE_END>
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT_HP
      CALC_FIELD_left
      VALUE_reg_target_block
      CALC_FIELD_right
      VALUE_reg_damage_after_vulnerable
      CALC_FIELD_result
      VALUE_reg_target_hp_after
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_5
      DIGIT_1
      NUM_END
    CALCULATION_END
  <CALCULATIONS_END>
  DELTA_SLOT
<REGION_END>
```


## 2. 固定例2：剑柄打击，洗牌并触发日晷

抽牌堆为0、弃牌堆为3；日晷计数器为2。

- 场景：第 0 层，第 0 幕，固定回归遭遇 `固定测试敌人`，第 1 回合。
- 打出：剑柄打击（Pommel Strike）。卡面规则：造成9/10点伤害。抽1/2张牌。
- 手牌（3 张）：剑柄打击（Pommel Strike）、打击（Strike_R）、防御（Defend_R）。
- 遗物：日晷（Sundial，计数器=2）。
- 战士：生命 80/80，格挡 0，能量 3，金币 99；状态：无；计数器：日晷=2。
- 牌堆计数：抽牌堆 0，弃牌堆 3，消耗堆 0；具体组成不提供给模型。
- 敌人 slot 0（`ExampleEnemy`）：生命 60/60，格挡 0，意图伤害 0→0，段数 0，状态：无。

### Outer 结果

准确率 36.28%，完全一致=False，正常结束=True，首次错误：位置40：期望 `REGION_PILE_SHUFFLE`，预测 `REGION_DRAW`。

目标 header：

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_ATTACK
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_ATTACK
          ENTITY_REF
          ENTITY_ENEMY_MASK
          <REGION_BEGIN>
            REGION_KIND
            REGION_DAMAGE
            ENTITY_REF
            ENTITY_ENEMY_SLOT_0
            BASE_DAMAGE
            NUM_LSD_BEGIN
            DIGIT_9
            NUM_END
          <REGION_END>
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_PILE_SHUFFLE
          ENTITY_REF
          ENTITY_PILES
          PILE_OPERATION
          VALUE_SHUFFLE_DISCARD_INTO_DRAW
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_DRAW
          ENTITY_REF
          ENTITY_SELF
          DRAW_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          VALUE_DISCARD
          CARD_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_RELIC_COUNTER
          ENTITY_REF
          ENTITY_SELF
          COUNTER_REF
          VALUE_player_relic_counter_0
          CHANGE_AMOUNT
          NUM_LSD_BEGIN
          SIGN_NEG
          DIGIT_2
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_ENERGY
          ENTITY_REF
          ENTITY_SELF
          ENERGY_OP
          VALUE_GAIN
          ENERGY_AMOUNT
          NUM_LSD_BEGIN
          DIGIT_2
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_DRAW
          ENTITY_REF
          ENTITY_SELF
          DRAW_COUNT
          NUM_LSD_BEGIN
          DIGIT_2
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

自回归生成 header：

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_ATTACK
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_ATTACK
          ENTITY_REF
          ENTITY_ENEMY_MASK
          <REGION_BEGIN>
            REGION_KIND
            REGION_DAMAGE
            ENTITY_REF
            ENTITY_ENEMY_SLOT_0
            BASE_DAMAGE
            NUM_LSD_BEGIN
            DIGIT_9
            NUM_END
          <REGION_END>
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_DRAW
          ENTITY_REF
          ENTITY_SELF
          DRAW_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          VALUE_DISCARD
          CARD_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_RELIC_COUNTER
          ENTITY_REF
          ENTITY_SELF
          COUNTER_REF
          VALUE_player_relic_counter_0
          CHANGE_AMOUNT
          NUM_LSD_BEGIN
          SIGN_NEG
          DIGIT_2
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_ENERGY
          ENTITY_REF
          ENTITY_SELF
          ENERGY_OP
          VALUE_GAIN
          ENERGY_AMOUNT
          NUM_LSD_BEGIN
          DIGIT_2
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_DRAW
          ENTITY_REF
          ENTITY_SELF
          DRAW_COUNT
          NUM_LSD_BEGIN
          DIGIT_2
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

### Local 结果

| path | 节点 | 完全正确 | token 准确率 | 首次错误 | delta 活跃位 MAE |
| --- | --- | ---: | ---: | --- | ---: |
| `root` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.037128 |
| `root/1` | `ATTACK` | True | 100.00% | 无 | - |
| `root/1/0` | `DAMAGE` | False | 83.33% | 位置13：期望 `VALUE_enemy_0_hp`，预测 `VALUE_enemy_0_block` | 0.046040 |
| `root/2` | `PILE_SHUFFLE` | True | 100.00% | 无 | 0.014640 |
| `root/3` | `DRAW` | True | 100.00% | 无 | 0.001120 |
| `root/4` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.001060 |
| `root/5` | `RELIC_COUNTER` | True | 100.00% | 无 | 0.147625 |
| `root/6` | `ENERGY` | True | 100.00% | 无 | 0.049278 |
| `root/7` | `DRAW` | True | 100.00% | 无 | 0.009308 |
| `root/8` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.100000、预测=-0.148787、绝对误差=0.048787；`player.counter.cards_played` 目标=+0.100000、预测=+0.106525、绝对误差=0.006525；`player.counter.attacks_played` 目标=+0.100000、预测=+0.043928、绝对误差=0.056072。
- `root/1/0` / `DAMAGE`：`enemy[0].hp` 目标=-0.045000、预测=+0.001040、绝对误差=0.046040。
- `root/2` / `PILE_SHUFFLE`：`piles.draw_count` 目标=+0.030000、预测=+0.018621、绝对误差=0.011379；`piles.discard_count` 目标=-0.030000、预测=-0.012099、绝对误差=0.017901。
- `root/3` / `DRAW`：`piles.draw_count` 目标=-0.010000、预测=-0.008880、绝对误差=0.001120。
- `root/4` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.011060、绝对误差=0.001060。
- `root/5` / `RELIC_COUNTER`：`player.relic.counter_sum` 目标=-0.200000、预测=-0.056255、绝对误差=0.143745；`player.relic.active_counter_count` 目标=-0.100000、预测=-0.031748、绝对误差=0.068252；`player.relic.max_counter` 目标=-0.200000、预测=-0.014483、绝对误差=0.185517；`player.relic_counter[0]` 目标=-0.200000、预测=-0.007014、绝对误差=0.192986。
- `root/6` / `ENERGY`：`player.energy` 目标=+0.200000、预测=+0.150722、绝对误差=0.049278。
- `root/7` / `DRAW`：`piles.draw_count` 目标=-0.020000、预测=-0.010692、绝对误差=0.009308。

local 出错节点的完整自回归对照：

#### `root/1/0` / `DAMAGE`

目标：

```text
<REGION_BEGIN>
  REGION_KIND
  REGION_DAMAGE
  ENTITY_REF
  ENTITY_ENEMY_SLOT_0
  BASE_DAMAGE
  NUM_LSD_BEGIN
  DIGIT_9
  NUM_END
  <NODE_EXECUTE>
  <CALCULATIONS_BEGIN>
    <READ_REQUEST_BEGIN>
      READ_SOURCE
      VALUE_enemy_0_hp
      READ_BIND
      VALUE_reg_target_hp
    <READ_REQUEST_END>
    <READ_RESPONSE_BEGIN>
      READ_BIND
      VALUE_reg_target_hp
      READ_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      DIGIT_6
      NUM_END
    <READ_RESPONSE_END>
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT_HP
      CALC_FIELD_left
      VALUE_reg_target_hp
      CALC_FIELD_right
      VALUE_header_BASE_DAMAGE
      CALC_FIELD_result
      VALUE_reg_target_hp_after
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_1
      DIGIT_5
      NUM_END
    CALCULATION_END
  <CALCULATIONS_END>
  DELTA_SLOT
<REGION_END>
```

预测（环境 token 已按真实 transcript 注入）：

```text
<REGION_BEGIN>
  REGION_KIND
  REGION_DAMAGE
  ENTITY_REF
  ENTITY_ENEMY_SLOT_0
  BASE_DAMAGE
  NUM_LSD_BEGIN
  DIGIT_9
  NUM_END
  <NODE_EXECUTE>
  <CALCULATIONS_BEGIN>
    <READ_REQUEST_BEGIN>
      READ_SOURCE
      VALUE_enemy_0_block
      READ_BIND
      VALUE_reg_target_block
    <READ_REQUEST_END>
    <READ_RESPONSE_BEGIN>
      READ_BIND
      VALUE_reg_target_hp
      READ_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      DIGIT_6
      NUM_END
    <READ_RESPONSE_END>
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT_HP
      CALC_FIELD_left
      VALUE_reg_target_hp
      CALC_FIELD_right
      VALUE_header_BASE_DAMAGE
      CALC_FIELD_result
      VALUE_reg_target_hp_after
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_3
      DIGIT_6
      NUM_END
    CALCULATION_END
  <CALCULATIONS_END>
  DELTA_SLOT
<REGION_END>
```


## 3. 随机例1

由训练分布直接生成，随机种子=20260831，流内序号=0；未筛选。

- 场景：第 13 层，第 1 幕，普通遭遇 `Lots Of Slimes`，第 6 回合。
- 打出：重刃+（Heavy Blade+）。卡面规则：造成14点伤害。力量对本牌生效3/5倍。
- 手牌（1 张）：重刃+（Heavy Blade+）。
- 遗物：燃烧之血（Burning Blood，计数器=1）、纸蛙（Paper Frog，计数器=5）、灯笼（Lantern，计数器=5）。
- 战士：生命 49/102，格挡 48，能量 3，金币 268；状态：火焰屏障=1、脆弱=2、双发=1、金属化=5、恶魔形态=2；计数器：已打出攻击牌数=3、已打出技能牌数=9、日晷=5、双节棍=1、钢笔尖=7。
- 牌堆计数：抽牌堆 2，弃牌堆 4，消耗堆 4；具体组成不提供给模型。
- 敌人 slot 0（`Lots Of Slimes#0`）：生命 33/57，格挡 24，意图伤害 34→21，段数 4，状态：人工制品=4。
- 敌人 slot 1（`Lots Of Slimes#1`）：生命 61/86，格挡 15，意图伤害 39→38，段数 4，状态：力量=3、多层护甲=3。

### Outer 结果

准确率 98.36%，完全一致=False，正常结束=True，首次错误：位置31：期望 `ENTITY_ENEMY_SLOT_1`，预测 `ENTITY_ENEMY_SLOT_0`。

目标 header：

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_ATTACK
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_2
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_ATTACK
          ENTITY_REF
          ENTITY_ENEMY_MASK
          <REGION_BEGIN>
            REGION_KIND
            REGION_DAMAGE
            ENTITY_REF
            ENTITY_ENEMY_SLOT_1
            BASE_DAMAGE
            NUM_LSD_BEGIN
            DIGIT_4
            DIGIT_1
            NUM_END
          <REGION_END>
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          VALUE_DISCARD
          CARD_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

自回归生成 header：

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_ATTACK
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_2
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_ATTACK
          ENTITY_REF
          ENTITY_ENEMY_MASK
          <REGION_BEGIN>
            REGION_KIND
            REGION_DAMAGE
            ENTITY_REF
            ENTITY_ENEMY_SLOT_0
            BASE_DAMAGE
            NUM_LSD_BEGIN
            DIGIT_4
            DIGIT_1
            NUM_END
          <REGION_END>
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          VALUE_DISCARD
          CARD_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

### Local 结果

| path | 节点 | 完全正确 | token 准确率 | 首次错误 | delta 活跃位 MAE |
| --- | --- | ---: | ---: | --- | ---: |
| `root` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.037580 |
| `root/1` | `ATTACK` | True | 100.00% | 无 | - |
| `root/1/0` | `DAMAGE` | False | 98.70% | 位置58：期望 `DIGIT_1`，预测 `DIGIT_3` | 0.089041 |
| `root/2` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.001028 |
| `root/3` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.200000、预测=-0.150541、绝对误差=0.049459；`player.counter.cards_played` 目标=+0.100000、预测=+0.108765、绝对误差=0.008765；`player.counter.attacks_played` 目标=+0.100000、预测=+0.045484、绝对误差=0.054516。
- `root/1/0` / `DAMAGE`：`enemy[1].block` 目标=-0.140000、预测=-0.050959、绝对误差=0.089041。
- `root/2` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.011028、绝对误差=0.001028。

local 出错节点的完整自回归对照：

#### `root/1/0` / `DAMAGE`

目标：

```text
<REGION_BEGIN>
  REGION_KIND
  REGION_DAMAGE
  ENTITY_REF
  ENTITY_ENEMY_SLOT_1
  BASE_DAMAGE
  NUM_LSD_BEGIN
  DIGIT_4
  DIGIT_1
  NUM_END
  <NODE_EXECUTE>
  <CALCULATIONS_BEGIN>
    <READ_REQUEST_BEGIN>
      READ_SOURCE
      VALUE_enemy_1_block
      READ_BIND
      VALUE_reg_target_block
    <READ_REQUEST_END>
    <READ_RESPONSE_BEGIN>
      READ_BIND
      VALUE_reg_target_block
      READ_VALUE
      NUM_LSD_BEGIN
      DIGIT_5
      DIGIT_1
      NUM_END
    <READ_RESPONSE_END>
    CALCULATION_BEGIN
      CALCULATION_CHECK_PRESENCE
      CALC_FIELD_field
      VALUE_enemy_1_block_present
      CALC_FIELD_result
      VALUE_reg_has_block
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_BLOCK_ABSORB
      CALC_FIELD_damage
      VALUE_header_BASE_DAMAGE
      CALC_FIELD_block
      VALUE_reg_target_block
      CALC_FIELD_absorbed
      VALUE_reg_absorbed_damage
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_4
      DIGIT_1
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT_BLOCK
      CALC_FIELD_left
      VALUE_reg_target_block
      CALC_FIELD_right
      VALUE_reg_absorbed_damage
      CALC_FIELD_result
      VALUE_reg_target_block_after
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_1
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT_BLOCK_DAMAGE
      CALC_FIELD_left
      VALUE_header_BASE_DAMAGE
      CALC_FIELD_right
      VALUE_reg_absorbed_damage
      CALC_FIELD_result
      VALUE_reg_unblocked_damage
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
    CALCULATION_END
    <READ_REQUEST_BEGIN>
      READ_SOURCE
      VALUE_enemy_1_hp
      READ_BIND
      VALUE_reg_target_hp
    <READ_REQUEST_END>
    <READ_RESPONSE_BEGIN>
      READ_BIND
      VALUE_reg_target_hp
      READ_VALUE
      NUM_LSD_BEGIN
      DIGIT_1
      DIGIT_6
      NUM_END
    <READ_RESPONSE_END>
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT_HP
      CALC_FIELD_left
      VALUE_reg_target_hp
      CALC_FIELD_right
      VALUE_reg_unblocked_damage
      CALC_FIELD_result
      VALUE_reg_target_hp_after
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_1
      DIGIT_6
      NUM_END
    CALCULATION_END
  <CALCULATIONS_END>
  DELTA_SLOT
<REGION_END>
```

预测（环境 token 已按真实 transcript 注入）：

```text
<REGION_BEGIN>
  REGION_KIND
  REGION_DAMAGE
  ENTITY_REF
  ENTITY_ENEMY_SLOT_1
  BASE_DAMAGE
  NUM_LSD_BEGIN
  DIGIT_4
  DIGIT_1
  NUM_END
  <NODE_EXECUTE>
  <CALCULATIONS_BEGIN>
    <READ_REQUEST_BEGIN>
      READ_SOURCE
      VALUE_enemy_1_block
      READ_BIND
      VALUE_reg_target_block
    <READ_REQUEST_END>
    <READ_RESPONSE_BEGIN>
      READ_BIND
      VALUE_reg_target_block
      READ_VALUE
      NUM_LSD_BEGIN
      DIGIT_5
      DIGIT_1
      NUM_END
    <READ_RESPONSE_END>
    CALCULATION_BEGIN
      CALCULATION_CHECK_PRESENCE
      CALC_FIELD_field
      VALUE_enemy_1_block_present
      CALC_FIELD_result
      VALUE_reg_has_block
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_BLOCK_ABSORB
      CALC_FIELD_damage
      VALUE_header_BASE_DAMAGE
      CALC_FIELD_block
      VALUE_reg_target_block
      CALC_FIELD_absorbed
      VALUE_reg_absorbed_damage
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_4
      DIGIT_1
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT_BLOCK
      CALC_FIELD_left
      VALUE_reg_target_block
      CALC_FIELD_right
      VALUE_reg_absorbed_damage
      CALC_FIELD_result
      VALUE_reg_target_block_after
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_3
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT_BLOCK_DAMAGE
      CALC_FIELD_left
      VALUE_header_BASE_DAMAGE
      CALC_FIELD_right
      VALUE_reg_absorbed_damage
      CALC_FIELD_result
      VALUE_reg_unblocked_damage
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
    CALCULATION_END
    <READ_REQUEST_BEGIN>
      READ_SOURCE
      VALUE_enemy_1_hp
      READ_BIND
      VALUE_reg_target_hp
    <READ_REQUEST_END>
    <READ_RESPONSE_BEGIN>
      READ_BIND
      VALUE_reg_target_hp
      READ_VALUE
      NUM_LSD_BEGIN
      DIGIT_1
      DIGIT_6
      NUM_END
    <READ_RESPONSE_END>
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT_HP
      CALC_FIELD_left
      VALUE_reg_target_hp
      CALC_FIELD_right
      VALUE_reg_unblocked_damage
      CALC_FIELD_result
      VALUE_reg_target_hp_after
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_1
      DIGIT_6
      NUM_END
    CALCULATION_END
  <CALCULATIONS_END>
  DELTA_SLOT
<REGION_END>
```


## 4. 随机例2

由训练分布直接生成，随机种子=20260831，流内序号=1；未筛选。

- 场景：第 45 层，第 3 幕，精英遭遇 `Giant Head`，第 9 回合。
- 打出：打击（Strike_R）。卡面规则：造成6/9点伤害。
- 手牌（4 张）：痛击（Bash）、打击+（Strike_R+）、顺劈斩（Cleave）、打击（Strike_R）。
- 遗物：燃烧之血（Burning Blood，计数器=7）、红头骨（Red Skull，计数器=9）、钢笔尖（Pen Nib，计数器=8）、纸蛙（Paper Frog，计数器=4）、锚（Anchor，计数器=1）、开心小花（Happy Flower，计数器=9）、小血瓶（Blood Vial，计数器=3）、符文金字塔（Runic Pyramid，计数器=0）、符文立方体（Runic Cube，计数器=5）。
- 战士：生命 35/75，格挡 49，能量 5，金币 982；状态：易伤=2；计数器：已打出卡牌数=9、已打出攻击牌数=7、已打出技能牌数=8、日晷=9、双节棍=6、钢笔尖=4。
- 牌堆计数：抽牌堆 2，弃牌堆 2，消耗堆 4；具体组成不提供给模型。
- 敌人 slot 0（`Giant Head#0`）：生命 98/138，格挡 28，意图伤害 28→57，段数 4，状态：易伤=3、虚弱=5、人工制品=3。
- 敌人 slot 1（`Giant Head#1`）：生命 86/103，格挡 30，意图伤害 24→11，段数 1，状态：虚弱=4、力量=2、人工制品=5。

### Outer 结果

准确率 98.33%，完全一致=False，正常结束=True，首次错误：位置31：期望 `ENTITY_ENEMY_SLOT_1`，预测 `ENTITY_ENEMY_SLOT_0`。

目标 header：

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_ATTACK
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_ATTACK
          ENTITY_REF
          ENTITY_ENEMY_MASK
          <REGION_BEGIN>
            REGION_KIND
            REGION_DAMAGE
            ENTITY_REF
            ENTITY_ENEMY_SLOT_1
            BASE_DAMAGE
            NUM_LSD_BEGIN
            DIGIT_6
            NUM_END
          <REGION_END>
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          VALUE_DISCARD
          CARD_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

自回归生成 header：

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_ATTACK
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_ATTACK
          ENTITY_REF
          ENTITY_ENEMY_MASK
          <REGION_BEGIN>
            REGION_KIND
            REGION_DAMAGE
            ENTITY_REF
            ENTITY_ENEMY_SLOT_0
            BASE_DAMAGE
            NUM_LSD_BEGIN
            DIGIT_6
            NUM_END
          <REGION_END>
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          VALUE_DISCARD
          CARD_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

### Local 结果

| path | 节点 | 完全正确 | token 准确率 | 首次错误 | delta 活跃位 MAE |
| --- | --- | ---: | ---: | --- | ---: |
| `root` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.034363 |
| `root/1` | `ATTACK` | True | 100.00% | 无 | - |
| `root/1/0` | `DAMAGE` | False | 90.91% | 位置11：期望 `<READ_REQUEST_BEGIN>`，预测 `CALCULATION_BEGIN` | 0.031602 |
| `root/2` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.000281 |
| `root/3` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.100000、预测=-0.142256、绝对误差=0.042256；`player.counter.cards_played` 目标=+0.100000、预测=+0.106271、绝对误差=0.006271；`player.counter.attacks_played` 目标=+0.100000、预测=+0.045437、绝对误差=0.054563。
- `root/1/0` / `DAMAGE`：`enemy[1].block` 目标=-0.060000、预测=-0.028398、绝对误差=0.031602。
- `root/2` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.009719、绝对误差=0.000281。

local 出错节点的完整自回归对照：

#### `root/1/0` / `DAMAGE`

目标：

```text
<REGION_BEGIN>
  REGION_KIND
  REGION_DAMAGE
  ENTITY_REF
  ENTITY_ENEMY_SLOT_1
  BASE_DAMAGE
  NUM_LSD_BEGIN
  DIGIT_6
  NUM_END
  <NODE_EXECUTE>
  <CALCULATIONS_BEGIN>
    <READ_REQUEST_BEGIN>
      READ_SOURCE
      VALUE_enemy_1_block
      READ_BIND
      VALUE_reg_target_block
    <READ_REQUEST_END>
    <READ_RESPONSE_BEGIN>
      READ_BIND
      VALUE_reg_target_block
      READ_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      DIGIT_3
      NUM_END
    <READ_RESPONSE_END>
    CALCULATION_BEGIN
      CALCULATION_CHECK_PRESENCE
      CALC_FIELD_field
      VALUE_enemy_1_block_present
      CALC_FIELD_result
      VALUE_reg_has_block
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_BLOCK_ABSORB
      CALC_FIELD_damage
      VALUE_header_BASE_DAMAGE
      CALC_FIELD_block
      VALUE_reg_target_block
      CALC_FIELD_absorbed
      VALUE_reg_absorbed_damage
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_6
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT_BLOCK
      CALC_FIELD_left
      VALUE_reg_target_block
      CALC_FIELD_right
      VALUE_reg_absorbed_damage
      CALC_FIELD_result
      VALUE_reg_target_block_after
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_4
      DIGIT_2
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT_BLOCK_DAMAGE
      CALC_FIELD_left
      VALUE_header_BASE_DAMAGE
      CALC_FIELD_right
      VALUE_reg_absorbed_damage
      CALC_FIELD_result
      VALUE_reg_unblocked_damage
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
    CALCULATION_END
    <READ_REQUEST_BEGIN>
      READ_SOURCE
      VALUE_enemy_1_hp
      READ_BIND
      VALUE_reg_target_hp
    <READ_REQUEST_END>
    <READ_RESPONSE_BEGIN>
      READ_BIND
      VALUE_reg_target_hp
      READ_VALUE
      NUM_LSD_BEGIN
      DIGIT_6
      DIGIT_8
      NUM_END
    <READ_RESPONSE_END>
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT_HP
      CALC_FIELD_left
      VALUE_reg_target_hp
      CALC_FIELD_right
      VALUE_reg_unblocked_damage
      CALC_FIELD_result
      VALUE_reg_target_hp_after
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_6
      DIGIT_8
      NUM_END
    CALCULATION_END
  <CALCULATIONS_END>
  DELTA_SLOT
<REGION_END>
```

预测（环境 token 已按真实 transcript 注入）：

```text
<REGION_BEGIN>
  REGION_KIND
  REGION_DAMAGE
  ENTITY_REF
  ENTITY_ENEMY_SLOT_1
  BASE_DAMAGE
  NUM_LSD_BEGIN
  DIGIT_6
  NUM_END
  <NODE_EXECUTE>
  <CALCULATIONS_BEGIN>
    CALCULATION_BEGIN
      CALCULATION_CHECK_PRESENCE
      CALC_FIELD_field
      VALUE_enemy_1_buff_Vulnerable
      CALC_FIELD_result
      VALUE_reg_has_vulnerable
      <READ_RESPONSE_BEGIN>
        READ_BIND
        VALUE_reg_target_block
        READ_VALUE
        NUM_LSD_BEGIN
        DIGIT_0
        DIGIT_3
        NUM_END
      <READ_RESPONSE_END>
      CALCULATION_BEGIN
        CALCULATION_CHECK_PRESENCE
        CALC_FIELD_field
        VALUE_enemy_1_block_present
        CALC_FIELD_result
        VALUE_reg_has_block
      CALCULATION_END
      CALCULATION_BEGIN
        CALCULATION_BLOCK_ABSORB
        CALC_FIELD_damage
        VALUE_header_BASE_DAMAGE
        CALC_FIELD_block
        VALUE_reg_target_block
        CALC_FIELD_absorbed
        VALUE_reg_absorbed_damage
        CALC_VALUE
        NUM_LSD_BEGIN
        DIGIT_6
        NUM_END
      CALCULATION_END
      CALCULATION_BEGIN
        CALCULATION_SUBTRACT_BLOCK
        CALC_FIELD_left
        VALUE_reg_target_block
        CALC_FIELD_right
        VALUE_reg_absorbed_damage
        CALC_FIELD_result
        VALUE_reg_target_block_after
        CALC_VALUE
        NUM_LSD_BEGIN
        DIGIT_6
        DIGIT_2
        NUM_END
      CALCULATION_END
      CALCULATION_BEGIN
        CALCULATION_SUBTRACT_BLOCK_DAMAGE
        CALC_FIELD_left
        VALUE_header_BASE_DAMAGE
        CALC_FIELD_right
        VALUE_reg_absorbed_damage
        CALC_FIELD_result
        VALUE_reg_unblocked_damage
        CALC_VALUE
        NUM_LSD_BEGIN
        DIGIT_0
        NUM_END
      CALCULATION_END
      <READ_REQUEST_BEGIN>
        READ_SOURCE
        VALUE_enemy_1_hp
        READ_BIND
        VALUE_reg_target_hp
      <READ_REQUEST_END>
      <READ_RESPONSE_BEGIN>
        READ_BIND
        VALUE_reg_target_hp
        READ_VALUE
        NUM_LSD_BEGIN
        DIGIT_6
        DIGIT_8
        NUM_END
      <READ_RESPONSE_END>
      CALCULATION_BEGIN
        CALCULATION_SUBTRACT_HP
        CALC_FIELD_left
        VALUE_reg_target_hp
        CALC_FIELD_right
        VALUE_reg_unblocked_damage
        CALC_FIELD_result
        VALUE_reg_target_hp_after
        CALC_VALUE
        NUM_LSD_BEGIN
        DIGIT_6
        DIGIT_8
        NUM_END
      CALCULATION_END
    <CALCULATIONS_END>
    DELTA_SLOT
  <REGION_END>
```


## 5. 随机例3

由训练分布直接生成，随机种子=20260831，流内序号=2；未筛选。

- 场景：第 47 层，第 3 幕，普通遭遇 `Spire Growth`，第 5 回合。
- 打出：震荡波+（Shockwave+）。卡面规则：对所有敌人施加3/5层虚弱和易伤。消耗。
- 手牌（3 张）：撕裂（Rupture）、震荡波+（Shockwave+）、旋风斩+（Whirlwind+）。
- 遗物：燃烧之血（Burning Blood，计数器=8）、红头骨（Red Skull，计数器=7）、准备背包（Bag of Preparation，计数器=6）、钢笔尖（Pen Nib，计数器=2）、铜制鳞片（Bronze Scales，计数器=1）、锚（Anchor，计数器=8）、痛楚印记（Mark of Pain，计数器=2）、奴隶贩子颈环（SlaversCollar，计数器=4）。
- 战士：生命 53/84，格挡 54，能量 3，金币 641；状态：腐化=5；计数器：已打出卡牌数=5、已打出攻击牌数=5、已打出技能牌数=2、日晷=6、双节棍=6、钢笔尖=1。
- 牌堆计数：抽牌堆 0，弃牌堆 0，消耗堆 3；具体组成不提供给模型。
- 敌人 slot 0（`Spire Growth#0`）：生命 150/168，格挡 56，意图伤害 42→6，段数 3，状态：无。
- 敌人 slot 1（`Spire Growth#1`）：生命 29/116，格挡 3，意图伤害 0→50，段数 3，状态：易伤=4、多层护甲=1。

### Outer 结果

准确率 100.00%，完全一致=True，正常结束=True，首次错误：无。

目标 header：

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_SKILL
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_2
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_STATE_CHANGE
          ENTITY_REF
          ENTITY_ENEMY_SLOT_0
          FIELD_OR_STATUS
          VALUE_enemy_0_buff_Weakened
          CHANGE_AMOUNT
          NUM_LSD_BEGIN
          DIGIT_5
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_STATE_CHANGE
          ENTITY_REF
          ENTITY_ENEMY_SLOT_0
          FIELD_OR_STATUS
          VALUE_enemy_0_buff_Vulnerable
          CHANGE_AMOUNT
          NUM_LSD_BEGIN
          DIGIT_5
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_STATE_CHANGE
          ENTITY_REF
          ENTITY_ENEMY_SLOT_1
          FIELD_OR_STATUS
          VALUE_enemy_1_buff_Weakened
          CHANGE_AMOUNT
          NUM_LSD_BEGIN
          DIGIT_5
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_STATE_CHANGE
          ENTITY_REF
          ENTITY_ENEMY_SLOT_1
          FIELD_OR_STATUS
          VALUE_enemy_1_buff_Vulnerable
          CHANGE_AMOUNT
          NUM_LSD_BEGIN
          DIGIT_5
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          VALUE_EXHAUST
          CARD_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

自回归生成 header：

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_SKILL
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_2
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_STATE_CHANGE
          ENTITY_REF
          ENTITY_ENEMY_SLOT_0
          FIELD_OR_STATUS
          VALUE_enemy_0_buff_Weakened
          CHANGE_AMOUNT
          NUM_LSD_BEGIN
          DIGIT_5
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_STATE_CHANGE
          ENTITY_REF
          ENTITY_ENEMY_SLOT_0
          FIELD_OR_STATUS
          VALUE_enemy_0_buff_Vulnerable
          CHANGE_AMOUNT
          NUM_LSD_BEGIN
          DIGIT_5
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_STATE_CHANGE
          ENTITY_REF
          ENTITY_ENEMY_SLOT_1
          FIELD_OR_STATUS
          VALUE_enemy_1_buff_Weakened
          CHANGE_AMOUNT
          NUM_LSD_BEGIN
          DIGIT_5
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_STATE_CHANGE
          ENTITY_REF
          ENTITY_ENEMY_SLOT_1
          FIELD_OR_STATUS
          VALUE_enemy_1_buff_Vulnerable
          CHANGE_AMOUNT
          NUM_LSD_BEGIN
          DIGIT_5
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          VALUE_EXHAUST
          CARD_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

### Local 结果

| path | 节点 | 完全正确 | token 准确率 | 首次错误 | delta 活跃位 MAE |
| --- | --- | ---: | ---: | --- | ---: |
| `root` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.033510 |
| `root/1` | `STATE_CHANGE` | True | 100.00% | 无 | 0.197866 |
| `root/2` | `STATE_CHANGE` | True | 100.00% | 无 | 0.142209 |
| `root/3` | `STATE_CHANGE` | True | 100.00% | 无 | 0.044895 |
| `root/4` | `STATE_CHANGE` | True | 100.00% | 无 | - |
| `root/5` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.002050 |
| `root/6` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.200000、预测=-0.143681、绝对误差=0.056319；`player.counter.cards_played` 目标=+0.100000、预测=+0.107767、绝对误差=0.007767；`player.counter.skills_played` 目标=+0.100000、预测=+0.063554、绝对误差=0.036446。
- `root/1` / `STATE_CHANGE`：`enemy[0].buff.Weakened` 目标=+1.000000、预测=+1.197866、绝对误差=0.197866。
- `root/2` / `STATE_CHANGE`：`enemy[0].buff.Vulnerable` 目标=+1.000000、预测=+1.142209、绝对误差=0.142209。
- `root/3` / `STATE_CHANGE`：`enemy[1].buff.Weakened` 目标=+1.000000、预测=+0.955105、绝对误差=0.044895。
- `root/5` / `CARD_DESTINATION`：`piles.exhaust_count` 目标=+0.010000、预测=+0.007950、绝对误差=0.002050。

## 6. 随机例4

由训练分布直接生成，随机种子=20260831，流内序号=3；未筛选。

- 场景：第 2 层，第 1 幕，普通遭遇 `Three Louse`，第 8 回合。
- 打出：缴械（Disarm）。卡面规则：敌人失去2/3点力量。消耗。
- 手牌（4 张）：头槌+（Headbutt+）、打击（Strike_R）、武装（Armaments）、缴械（Disarm）。
- 遗物：燃烧之血（Burning Blood，计数器=9）、锚（Anchor，计数器=7）。
- 战士：生命 31/84，格挡 25，能量 1，金币 277；状态：撕裂=1、燃烧=1、脆弱=4；计数器：已打出卡牌数=8、已打出攻击牌数=3、已打出技能牌数=4、日晷=6、双节棍=7、钢笔尖=2。
- 牌堆计数：抽牌堆 1，弃牌堆 1，消耗堆 2；具体组成不提供给模型。
- 敌人 slot 0（`Three Louse#0`）：生命 30/46，格挡 58，意图伤害 16→45，段数 3，状态：多层护甲=5。
- 敌人 slot 1（`Three Louse#1`）：生命 58/72，格挡 25，意图伤害 7→10，段数 4，状态：无。
- 敌人 slot 2（`Three Louse#2`）：生命 34/121，格挡 42，意图伤害 4→10，段数 4，状态：虚弱=4。

### Outer 结果

准确率 100.00%，完全一致=True，正常结束=True，首次错误：无。

目标 header：

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_SKILL
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          VALUE_DISCARD
          CARD_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

自回归生成 header：

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_SKILL
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          VALUE_DISCARD
          CARD_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

### Local 结果

| path | 节点 | 完全正确 | token 准确率 | 首次错误 | delta 活跃位 MAE |
| --- | --- | ---: | ---: | --- | ---: |
| `root` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.018926 |
| `root/1` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.001551 |
| `root/2` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.100000、预测=-0.123130、绝对误差=0.023130；`player.counter.cards_played` 目标=+0.100000、预测=+0.104992、绝对误差=0.004992；`player.counter.skills_played` 目标=+0.100000、预测=+0.071343、绝对误差=0.028657。
- `root/1` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.011551、绝对误差=0.001551。

## 7. 随机例5

由训练分布直接生成，随机种子=20260831，流内序号=4；未筛选。

- 场景：第 6 层，第 1 幕，普通遭遇 `Blue Slaver`，第 5 回合。
- 打出：狂暴+（Berserk+）。卡面规则：获得2/1层易伤。每回合开始时获得1点能量。
- 手牌（2 张）：狂暴+（Berserk+）、震荡波+（Shockwave+）。
- 遗物：打击木偶（StrikeDummy，计数器=3）。
- 战士：生命 54/83，格挡 22，能量 3，金币 922；状态：力量=7、狂怒=5、撕裂=2；计数器：已打出卡牌数=7、已打出攻击牌数=7、已打出技能牌数=5、日晷=6、双节棍=6、钢笔尖=7。
- 牌堆计数：抽牌堆 0，弃牌堆 5，消耗堆 1；具体组成不提供给模型。
- 敌人 slot 0（`Blue Slaver#0`）：生命 46/66，格挡 40，意图伤害 32→35，段数 4，状态：虚弱=4、多层护甲=2。

### Outer 结果

准确率 100.00%，完全一致=True，正常结束=True，首次错误：无。

目标 header：

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_SKILL
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_0
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          VALUE_DISCARD
          CARD_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

自回归生成 header：

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_SKILL
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_0
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          VALUE_DISCARD
          CARD_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

### Local 结果

| path | 节点 | 完全正确 | token 准确率 | 首次错误 | delta 活跃位 MAE |
| --- | --- | ---: | ---: | --- | ---: |
| `root` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.011064 |
| `root/1` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.000584 |
| `root/2` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.counter.cards_played` 目标=+0.100000、预测=+0.113245、绝对误差=0.013245；`player.counter.skills_played` 目标=+0.100000、预测=+0.091116、绝对误差=0.008884。
- `root/1` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.010584、绝对误差=0.000584。

## 8. 随机例6

由训练分布直接生成，随机种子=20260831，流内序号=5；未筛选。

- 场景：第 1 层，第 1 幕，精英遭遇 `Gremlin Nob`，第 3 回合。
- 打出：燃烧（Combust）。卡面规则：回合结束时失去1点生命，并对所有敌人造成5/7点伤害。
- 手牌（4 张）：吞噬（Feed）、灼热攻击+（Searing Blow+）、突破极限（Limit Break）、燃烧（Combust）。
- 遗物：锚（Anchor，计数器=8）。
- 战士：生命 40/107，格挡 29，能量 2，金币 742；状态：力量=6；计数器：已打出卡牌数=4、已打出攻击牌数=7、已打出技能牌数=6、日晷=2、双节棍=9、钢笔尖=3。
- 牌堆计数：抽牌堆 1，弃牌堆 4，消耗堆 4；具体组成不提供给模型。
- 敌人 slot 0（`Gremlin Nob#0`）：生命 41/52，格挡 37，意图伤害 35→41，段数 4，状态：力量=5、易伤=3、虚弱=4。

### Outer 结果

准确率 100.00%，完全一致=True，正常结束=True，首次错误：无。

目标 header：

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_SKILL
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          VALUE_DISCARD
          CARD_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

自回归生成 header：

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_SKILL
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          VALUE_DISCARD
          CARD_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

### Local 结果

| path | 节点 | 完全正确 | token 准确率 | 首次错误 | delta 活跃位 MAE |
| --- | --- | ---: | ---: | --- | ---: |
| `root` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.026478 |
| `root/1` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.001937 |
| `root/2` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.100000、预测=-0.137310、绝对误差=0.037310；`player.counter.cards_played` 目标=+0.100000、预测=+0.107838、绝对误差=0.007838；`player.counter.skills_played` 目标=+0.100000、预测=+0.065716、绝对误差=0.034284。
- `root/1` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.011937、绝对误差=0.001937。

## 9. 随机例7

由训练分布直接生成，随机种子=20260831，流内序号=6；未筛选。

- 场景：第 46 层，第 3 幕，普通遭遇 `Four Shapes`，第 9 回合。
- 打出：飞剑回旋镖+（Sword Boomerang+）。卡面规则：对随机敌人造成3点伤害，共3/4次。
- 手牌（1 张）：飞剑回旋镖+（Sword Boomerang+）。
- 遗物：燃烧之血（Burning Blood，计数器=2）、纸蛙（Paper Frog，计数器=7）、精致折扇（Ornamental Fan，计数器=8）、赤牛（Akabeko，计数器=7）、钢笔尖（Pen Nib，计数器=0）、锚（Anchor，计数器=9）、手里剑（Shuriken，计数器=5）、异蛇之眼（Snecko Eye，计数器=0）、奴隶贩子颈环（SlaversCollar，计数器=7）。
- 战士：生命 37/82，格挡 11，能量 2，金币 968；状态：双发=1、进化=3、黑暗之拥=4、勇猛=2、虚弱=4；计数器：已打出卡牌数=7、已打出攻击牌数=6、已打出技能牌数=7、日晷=4、双节棍=3、钢笔尖=4。
- 牌堆计数：抽牌堆 5，弃牌堆 4，消耗堆 5；具体组成不提供给模型。
- 敌人 slot 0（`Four Shapes#0`）：生命 25/75，格挡 56，意图伤害 0→62，段数 2，状态：无。
- 敌人 slot 1（`Four Shapes#1`）：生命 74/101，格挡 35，意图伤害 35→33，段数 4，状态：金属化=4、力量=7。
- 敌人 slot 2（`Four Shapes#2`）：生命 83/148，格挡 14，意图伤害 14→48，段数 4，状态：多层护甲=4。

### Outer 结果

准确率 100.00%，完全一致=True，正常结束=True，首次错误：无。

目标 header：

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_ATTACK
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_ATTACK
          ENTITY_REF
          ENTITY_ENEMY_MASK
          <REGION_BEGIN>
            REGION_KIND
            REGION_REPEAT
            REPEAT_COUNT
            NUM_LSD_BEGIN
            DIGIT_4
            NUM_END
            <REGION_BEGIN>
              REGION_KIND
              REGION_RANDOM_HIT
              ENTITY_REF
              ENTITY_QUERY_TARGET
              <REGION_BEGIN>
                REGION_KIND
                REGION_QUERY
                ENTITY_REF
                ENTITY_ENEMY_MASK
                QUERY_KIND
                VALUE_RANDOM_TARGET
              <REGION_END>
              <REGION_BEGIN>
                REGION_KIND
                REGION_DAMAGE
                ENTITY_REF
                ENTITY_QUERY_TARGET
                BASE_DAMAGE
                NUM_LSD_BEGIN
                DIGIT_3
                NUM_END
              <REGION_END>
            <REGION_END>
            <REGION_BEGIN>
              REGION_KIND
              REGION_CALL
              CALL_TARGET
              VALUE_SELF
              ARG_DECREMENT
              NUM_LSD_BEGIN
              DIGIT_1
              NUM_END
            <REGION_END>
          <REGION_END>
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          VALUE_DISCARD
          CARD_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

自回归生成 header：

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_ATTACK
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_ATTACK
          ENTITY_REF
          ENTITY_ENEMY_MASK
          <REGION_BEGIN>
            REGION_KIND
            REGION_REPEAT
            REPEAT_COUNT
            NUM_LSD_BEGIN
            DIGIT_4
            NUM_END
            <REGION_BEGIN>
              REGION_KIND
              REGION_RANDOM_HIT
              ENTITY_REF
              ENTITY_QUERY_TARGET
              <REGION_BEGIN>
                REGION_KIND
                REGION_QUERY
                ENTITY_REF
                ENTITY_ENEMY_MASK
                QUERY_KIND
                VALUE_RANDOM_TARGET
              <REGION_END>
              <REGION_BEGIN>
                REGION_KIND
                REGION_DAMAGE
                ENTITY_REF
                ENTITY_QUERY_TARGET
                BASE_DAMAGE
                NUM_LSD_BEGIN
                DIGIT_3
                NUM_END
              <REGION_END>
            <REGION_END>
            <REGION_BEGIN>
              REGION_KIND
              REGION_CALL
              CALL_TARGET
              VALUE_SELF
              ARG_DECREMENT
              NUM_LSD_BEGIN
              DIGIT_1
              NUM_END
            <REGION_END>
          <REGION_END>
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          VALUE_DISCARD
          CARD_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

### Local 结果

| path | 节点 | 完全正确 | token 准确率 | 首次错误 | delta 活跃位 MAE |
| --- | --- | ---: | ---: | --- | ---: |
| `root` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.036243 |
| `root/1` | `ATTACK` | True | 100.00% | 无 | - |
| `root/1/0` | `REPEAT` | True | 100.00% | 无 | - |
| `root/1/0/0` | `RANDOM_HIT` | True | 100.00% | 无 | - |
| `root/1/0/0/0` | `QUERY` | True | 100.00% | 无 | - |
| `root/1/0/0/1` | `DAMAGE` | False | 98.70% | 位置60：期望 `DIGIT_3`，预测 `DIGIT_1` | 0.022361 |
| `root/1/0/1` | `CALL` | True | 100.00% | 无 | - |
| `root/2` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.000911 |
| `root/3` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.100000、预测=-0.148053、绝对误差=0.048053；`player.counter.cards_played` 目标=+0.100000、预测=+0.108339、绝对误差=0.008339；`player.counter.attacks_played` 目标=+0.100000、预测=+0.047662、绝对误差=0.052338。
- `root/1/0/0/1` / `DAMAGE`：`enemy[0].block` 目标=-0.030000、预测=-0.052361、绝对误差=0.022361。
- `root/2` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.010911、绝对误差=0.000911。

local 出错节点的完整自回归对照：

#### `root/1/0/0/1` / `DAMAGE`

目标：

```text
<REGION_BEGIN>
  REGION_KIND
  REGION_DAMAGE
  ENTITY_REF
  ENTITY_QUERY_TARGET
  BASE_DAMAGE
  NUM_LSD_BEGIN
  DIGIT_3
  NUM_END
  <NODE_EXECUTE>
  <OBS_BEGIN>
    OBS_RANDOM_TARGET
    ENTITY_ENEMY_SLOT_0
  <OBS_END>
  <CALCULATIONS_BEGIN>
    <READ_REQUEST_BEGIN>
      READ_SOURCE
      VALUE_enemy_0_block
      READ_BIND
      VALUE_reg_target_block
    <READ_REQUEST_END>
    <READ_RESPONSE_BEGIN>
      READ_BIND
      VALUE_reg_target_block
      READ_VALUE
      NUM_LSD_BEGIN
      DIGIT_6
      DIGIT_5
      NUM_END
    <READ_RESPONSE_END>
    CALCULATION_BEGIN
      CALCULATION_CHECK_PRESENCE
      CALC_FIELD_field
      VALUE_enemy_0_block_present
      CALC_FIELD_result
      VALUE_reg_has_block
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_BLOCK_ABSORB
      CALC_FIELD_damage
      VALUE_header_BASE_DAMAGE
      CALC_FIELD_block
      VALUE_reg_target_block
      CALC_FIELD_absorbed
      VALUE_reg_absorbed_damage
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_3
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT_BLOCK
      CALC_FIELD_left
      VALUE_reg_target_block
      CALC_FIELD_right
      VALUE_reg_absorbed_damage
      CALC_FIELD_result
      VALUE_reg_target_block_after
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_3
      DIGIT_5
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT_BLOCK_DAMAGE
      CALC_FIELD_left
      VALUE_header_BASE_DAMAGE
      CALC_FIELD_right
      VALUE_reg_absorbed_damage
      CALC_FIELD_result
      VALUE_reg_unblocked_damage
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
    CALCULATION_END
    <READ_REQUEST_BEGIN>
      READ_SOURCE
      VALUE_enemy_0_hp
      READ_BIND
      VALUE_reg_target_hp
    <READ_REQUEST_END>
    <READ_RESPONSE_BEGIN>
      READ_BIND
      VALUE_reg_target_hp
      READ_VALUE
      NUM_LSD_BEGIN
      DIGIT_5
      DIGIT_2
      NUM_END
    <READ_RESPONSE_END>
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT_HP
      CALC_FIELD_left
      VALUE_reg_target_hp
      CALC_FIELD_right
      VALUE_reg_unblocked_damage
      CALC_FIELD_result
      VALUE_reg_target_hp_after
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_5
      DIGIT_2
      NUM_END
    CALCULATION_END
  <CALCULATIONS_END>
  DELTA_SLOT
<REGION_END>
```

预测（环境 token 已按真实 transcript 注入）：

```text
<REGION_BEGIN>
  REGION_KIND
  REGION_DAMAGE
  ENTITY_REF
  ENTITY_QUERY_TARGET
  BASE_DAMAGE
  NUM_LSD_BEGIN
  DIGIT_3
  NUM_END
  <NODE_EXECUTE>
  <OBS_BEGIN>
    OBS_RANDOM_TARGET
    ENTITY_ENEMY_SLOT_0
  <OBS_END>
  <CALCULATIONS_BEGIN>
    <READ_REQUEST_BEGIN>
      READ_SOURCE
      VALUE_enemy_0_block
      READ_BIND
      VALUE_reg_target_block
    <READ_REQUEST_END>
    <READ_RESPONSE_BEGIN>
      READ_BIND
      VALUE_reg_target_block
      READ_VALUE
      NUM_LSD_BEGIN
      DIGIT_6
      DIGIT_5
      NUM_END
    <READ_RESPONSE_END>
    CALCULATION_BEGIN
      CALCULATION_CHECK_PRESENCE
      CALC_FIELD_field
      VALUE_enemy_0_block_present
      CALC_FIELD_result
      VALUE_reg_has_block
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_BLOCK_ABSORB
      CALC_FIELD_damage
      VALUE_header_BASE_DAMAGE
      CALC_FIELD_block
      VALUE_reg_target_block
      CALC_FIELD_absorbed
      VALUE_reg_absorbed_damage
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_3
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT_BLOCK
      CALC_FIELD_left
      VALUE_reg_target_block
      CALC_FIELD_right
      VALUE_reg_absorbed_damage
      CALC_FIELD_result
      VALUE_reg_target_block_after
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_1
      DIGIT_5
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT_BLOCK_DAMAGE
      CALC_FIELD_left
      VALUE_header_BASE_DAMAGE
      CALC_FIELD_right
      VALUE_reg_absorbed_damage
      CALC_FIELD_result
      VALUE_reg_unblocked_damage
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
    CALCULATION_END
    <READ_REQUEST_BEGIN>
      READ_SOURCE
      VALUE_enemy_0_hp
      READ_BIND
      VALUE_reg_target_hp
    <READ_REQUEST_END>
    <READ_RESPONSE_BEGIN>
      READ_BIND
      VALUE_reg_target_hp
      READ_VALUE
      NUM_LSD_BEGIN
      DIGIT_5
      DIGIT_2
      NUM_END
    <READ_RESPONSE_END>
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT_HP
      CALC_FIELD_left
      VALUE_reg_target_hp
      CALC_FIELD_right
      VALUE_reg_unblocked_damage
      CALC_FIELD_result
      VALUE_reg_target_hp_after
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_5
      DIGIT_2
      NUM_END
    CALCULATION_END
  <CALCULATIONS_END>
  DELTA_SLOT
<REGION_END>
```


## 10. 随机例8

由训练分布直接生成，随机种子=20260831，流内序号=7；未筛选。

- 场景：第 19 层，第 2 幕，普通遭遇 `Two Thieves`，第 6 回合。
- 打出：灼热攻击（Searing Blow）。卡面规则：造成12/16点伤害。可重复升级。
- 手牌（4 张）：灼热攻击（Searing Blow）、重振精神+（Second Wind+）、灼热攻击（Searing Blow）、愤怒（Anger）。
- 遗物：燃烧之血（Burning Blood，计数器=3）、赌博筹码（Gambling Chip，计数器=0）。
- 战士：生命 30/102，格挡 53，能量 4，金币 224；状态：燃烧=5、虚弱=1、壁垒=1、腐化=2、双发=3；计数器：已打出卡牌数=4、已打出攻击牌数=9、已打出技能牌数=9、日晷=9、双节棍=4、钢笔尖=4。
- 牌堆计数：抽牌堆 5，弃牌堆 2，消耗堆 4；具体组成不提供给模型。
- 敌人 slot 0（`Two Thieves#0`）：生命 87/130，格挡 19，意图伤害 4→32，段数 1，状态：无。
- 敌人 slot 1（`Two Thieves#1`）：生命 32/59，格挡 53，意图伤害 49→8，段数 2，状态：虚弱=1、易伤=2。
- 敌人 slot 2（`Two Thieves#2`）：生命 43/146，格挡 36，意图伤害 37→16，段数 4，状态：力量=5。

### Outer 结果

准确率 98.36%，完全一致=False，正常结束=True，首次错误：位置31：期望 `ENTITY_ENEMY_SLOT_0`，预测 `ENTITY_ENEMY_SLOT_1`。

目标 header：

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_ATTACK
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_2
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_ATTACK
          ENTITY_REF
          ENTITY_ENEMY_MASK
          <REGION_BEGIN>
            REGION_KIND
            REGION_DAMAGE
            ENTITY_REF
            ENTITY_ENEMY_SLOT_0
            BASE_DAMAGE
            NUM_LSD_BEGIN
            DIGIT_2
            DIGIT_1
            NUM_END
          <REGION_END>
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          VALUE_DISCARD
          CARD_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

自回归生成 header：

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_ATTACK
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_2
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_ATTACK
          ENTITY_REF
          ENTITY_ENEMY_MASK
          <REGION_BEGIN>
            REGION_KIND
            REGION_DAMAGE
            ENTITY_REF
            ENTITY_ENEMY_SLOT_1
            BASE_DAMAGE
            NUM_LSD_BEGIN
            DIGIT_2
            DIGIT_1
            NUM_END
          <REGION_END>
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          VALUE_DISCARD
          CARD_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

### Local 结果

| path | 节点 | 完全正确 | token 准确率 | 首次错误 | delta 活跃位 MAE |
| --- | --- | ---: | ---: | --- | ---: |
| `root` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.029528 |
| `root/1` | `ATTACK` | True | 100.00% | 无 | - |
| `root/1/0` | `DAMAGE` | True | 100.00% | 无 | 0.036039 |
| `root/2` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.000348 |
| `root/3` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.200000、预测=-0.165377、绝对误差=0.034623；`player.counter.cards_played` 目标=+0.100000、预测=+0.106452、绝对误差=0.006452；`player.counter.attacks_played` 目标=+0.100000、预测=+0.052489、绝对误差=0.047511。
- `root/1/0` / `DAMAGE`：`enemy[0].block` 目标=-0.120000、预测=-0.083961、绝对误差=0.036039。
- `root/2` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.009652、绝对误差=0.000348。

## 11. 随机例9

由训练分布直接生成，随机种子=20260831，流内序号=8；未筛选。

- 场景：第 25 层，第 2 幕，普通遭遇 `Shelled Parasite And Fungi`，第 9 回合。
- 打出：破灭（Havoc）。卡面规则：打出抽牌堆顶牌并将其消耗。
- 手牌（3 张）：残暴+（Brutality+）、破灭（Havoc）、耸肩无视（Shrug It Off）。
- 遗物：燃烧之血（Burning Blood，计数器=6）、水银沙漏（Mercury Hourglass，计数器=8）、精致折扇（Ornamental Fan，计数器=1）、灯笼（Lantern，计数器=4）、痛楚印记（Mark of Pain，计数器=7）。
- 战士：生命 54/100，格挡 39，能量 2，金币 667；状态：双发=3、壁垒=4、脆弱=2、火焰吐息=5、腐化=5；计数器：已打出卡牌数=7、已打出技能牌数=1、日晷=8、双节棍=4、钢笔尖=4。
- 牌堆计数：抽牌堆 4，弃牌堆 2，消耗堆 5；具体组成不提供给模型。
- 敌人 slot 0（`Shelled Parasite And Fungi#0`）：生命 126/137，格挡 59，意图伤害 10→19，段数 3，状态：中毒=1、金属化=4。

### Outer 结果

准确率 100.00%，完全一致=True，正常结束=True，首次错误：无。

目标 header：

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_SKILL
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          VALUE_DISCARD
          CARD_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

自回归生成 header：

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_SKILL
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          VALUE_DISCARD
          CARD_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

### Local 结果

| path | 节点 | 完全正确 | token 准确率 | 首次错误 | delta 活跃位 MAE |
| --- | --- | ---: | ---: | --- | ---: |
| `root` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.027893 |
| `root/1` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.000314 |
| `root/2` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.100000、预测=-0.142669、绝对误差=0.042668；`player.counter.cards_played` 目标=+0.100000、预测=+0.105563、绝对误差=0.005563；`player.counter.skills_played` 目标=+0.100000、预测=+0.064553、绝对误差=0.035447。
- `root/1` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.009686、绝对误差=0.000314。

## 12. 随机例10

由训练分布直接生成，随机种子=20260831，流内序号=9；未筛选。

- 场景：第 13 层，第 1 幕，精英遭遇 `Three Sentries`，第 6 回合。
- 打出：硬撑（Power Through）。卡面规则：将2张伤口加入手牌。获得15/20点格挡。
- 手牌（2 张）：双发+（Double Tap+）、硬撑（Power Through）。
- 遗物：燃烧之血（Burning Blood，计数器=3）。
- 战士：生命 20/82，格挡 13，能量 2，金币 989；状态：恶魔形态=1、脆弱=3、火焰屏障=3、虚弱=2；计数器：已打出卡牌数=4、已打出攻击牌数=2、已打出技能牌数=8、双节棍=9。
- 牌堆计数：抽牌堆 0，弃牌堆 3，消耗堆 3；具体组成不提供给模型。
- 敌人 slot 0（`Three Sentries#0`）：生命 59/92，格挡 21，意图伤害 18→0，段数 2，状态：无。

### Outer 结果

准确率 100.00%，完全一致=True，正常结束=True，首次错误：无。

目标 header：

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_SKILL
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_STATE_CHANGE
          ENTITY_REF
          ENTITY_SELF
          FIELD_OR_STATUS
          VALUE_player_block
          CHANGE_AMOUNT
          NUM_LSD_BEGIN
          DIGIT_5
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          VALUE_DISCARD
          CARD_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

自回归生成 header：

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_SKILL
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_STATE_CHANGE
          ENTITY_REF
          ENTITY_SELF
          FIELD_OR_STATUS
          VALUE_player_block
          CHANGE_AMOUNT
          NUM_LSD_BEGIN
          DIGIT_5
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          VALUE_DISCARD
          CARD_COUNT
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

### Local 结果

| path | 节点 | 完全正确 | token 准确率 | 首次错误 | delta 活跃位 MAE |
| --- | --- | ---: | ---: | --- | ---: |
| `root` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.026250 |
| `root/1` | `STATE_CHANGE` | True | 100.00% | 无 | 0.012788 |
| `root/2` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.000024 |
| `root/3` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.100000、预测=-0.136115、绝对误差=0.036115；`player.counter.cards_played` 目标=+0.100000、预测=+0.107564、绝对误差=0.007564；`player.counter.skills_played` 目标=+0.100000、预测=+0.064930、绝对误差=0.035070。
- `root/1` / `STATE_CHANGE`：`player.block` 目标=+0.150000、预测=+0.162788、绝对误差=0.012788。
- `root/2` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.009976、绝对误差=0.000024。
