# v7 step 3500 固定与随机自回归评测

- 参数：`.tmp-vector-rule-v7-contract-v10-register-values-20260901/last.pt`
- checkpoint step：`3500`
- schema / executor contract：`7 / 10`
- 随机种子：`20260831`
- 设备：`cuda`
- outer：从 `<OUTPUT_BEGIN>` 开始完全自由贪心生成 effect-tree header。
- local：真实 NodeSpec 前缀由环境给出；模型 token 自回归，错误 token 保留在上下文；运行时数值通过环境在寄存器操作数后插入被 mask 的 `CONST_VALUE`，随机目标仍通过 `<OBS_*>` 交互。
- 牌堆：模型只接收抽牌堆、弃牌堆、消耗堆数量，不接收各堆具体卡牌组成。

## 总体结果

- outer 完全正确：1/2；token 准确率 63.21%（212 个 token）。
- local 完全正确节点：18/20；模型 token 准确率 99.05%（524 个 token）。
- delta：12 个向量，活跃位 MAE `0.025509`，全向量 MAE `0.001408`。
- 模型输出 `<UNK>`：0 个。

| 案例 | 打出的牌 | outer | local 节点 | local token | delta 活跃位 MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| 固定例1：飞身踢，敌人有2层易伤 | 飞身踢（Dropkick） | 100.00% | 8/9 | 98.71% | 0.035688 |
| 固定例2：剑柄打击，洗牌并触发日晷 | 剑柄打击（Pommel Strike） | 37.10% | 10/11 | 99.32% | 0.018238 |

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
        REGION_CARD_COMMIT
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PLAY
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_ATTACK
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
        REGION_CARD_COMMIT
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PLAY
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_ATTACK
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
| `root` | `CARD_COMMIT` | True | 100.00% | 无 | - |
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.026056 |
| `root/1` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/1/0` | `ATTACK` | True | 100.00% | 无 | - |
| `root/1/0/0` | `DAMAGE` | False | 97.32% | 位置147：期望 `DIGIT_3`，预测 `DIGIT_5` | 0.031694 |
| `root/1/1` | `ENERGY` | True | 100.00% | 无 | 0.111385 |
| `root/1/2` | `DRAW` | True | 100.00% | 无 | 0.007069 |
| `root/2` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.002238 |
| `root/3` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.100000、预测=-0.126056、绝对误差=0.026056。
- `root/1/0/0` / `DAMAGE`：`enemy[0].hp` 目标=-0.035000、预测=-0.003306、绝对误差=0.031694。
- `root/1/1` / `ENERGY`：`player.energy` 目标=+0.100000、预测=+0.211385、绝对误差=0.111385。
- `root/1/2` / `DRAW`：`piles.draw_count` 目标=-0.010000、预测=-0.002931、绝对误差=0.007069。
- `root/2` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.007762、绝对误差=0.002238。

local 出错节点的完整自回归对照：

#### `root/1/0/0` / `DAMAGE`

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
  <READ_REQUEST_BEGIN>
    READ_SOURCE
    VALUE_enemy_0_block
    READ_BIND
    VALUE_reg_target_block
  <READ_REQUEST_END>
  <READ_REQUEST_BEGIN>
    READ_SOURCE
    VALUE_enemy_0_hp
    READ_BIND
    VALUE_reg_target_hp
  <READ_REQUEST_END>
  <CALCULATIONS_BEGIN>
    CALCULATION_BEGIN
      CALCULATION_CHECK_PRESENCE
      CALC_FIELD_field
      VALUE_enemy_0_buff_Vulnerable
      CALC_FIELD_result
      VALUE_reg_has_vulnerable
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_MULTIPLY
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
      CALCULATION_FLOOR_DIVIDE
      CALC_FIELD_left
      VALUE_reg_damage_vulnerable_numerator
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_5
      DIGIT_1
      NUM_END
      CALC_FIELD_result
      VALUE_reg_damage_after_vulnerable
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_7
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_CHECK_PRESENCE
      CALC_FIELD_field
      VALUE_enemy_0_block_present
      CALC_FIELD_result
      VALUE_reg_has_block
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_MIN
      CALC_FIELD_left
      VALUE_reg_damage_after_vulnerable
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_7
      NUM_END
      CALC_FIELD_right
      VALUE_reg_target_block
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
      CALC_FIELD_result
      VALUE_reg_absorbed_damage
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT
      CALC_FIELD_left
      VALUE_reg_target_block
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
      CALC_FIELD_right
      VALUE_reg_absorbed_damage
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
      CALC_FIELD_result
      VALUE_reg_target_block_after
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT
      CALC_FIELD_left
      VALUE_reg_damage_after_vulnerable
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_7
      NUM_END
      CALC_FIELD_right
      VALUE_reg_absorbed_damage
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
      CALC_FIELD_result
      VALUE_reg_unblocked_damage
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_7
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT
      CALC_FIELD_left
      VALUE_reg_target_hp
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      DIGIT_5
      NUM_END
      CALC_FIELD_right
      VALUE_reg_unblocked_damage
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_7
      NUM_END
      CALC_FIELD_result
      VALUE_reg_target_hp_after_subtract
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_3
      DIGIT_4
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_RELU
      CALC_FIELD_input
      VALUE_reg_target_hp_after_subtract
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_3
      DIGIT_4
      NUM_END
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
  <READ_REQUEST_BEGIN>
    READ_SOURCE
    VALUE_enemy_0_block
    READ_BIND
    VALUE_reg_target_block
  <READ_REQUEST_END>
  <READ_REQUEST_BEGIN>
    READ_SOURCE
    VALUE_enemy_0_hp
    READ_BIND
    VALUE_reg_target_hp
  <READ_REQUEST_END>
  <CALCULATIONS_BEGIN>
    CALCULATION_BEGIN
      CALCULATION_CHECK_PRESENCE
      CALC_FIELD_field
      VALUE_enemy_0_buff_Vulnerable
      CALC_FIELD_result
      VALUE_reg_has_vulnerable
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_MULTIPLY
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
      CALCULATION_FLOOR_DIVIDE
      CALC_FIELD_left
      VALUE_reg_damage_vulnerable_numerator
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_5
      DIGIT_1
      NUM_END
      CALC_FIELD_result
      VALUE_reg_damage_after_vulnerable
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_7
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_CHECK_PRESENCE
      CALC_FIELD_field
      VALUE_enemy_0_block_present
      CALC_FIELD_result
      VALUE_reg_has_block
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_MIN
      CALC_FIELD_left
      VALUE_reg_damage_after_vulnerable
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_7
      NUM_END
      CALC_FIELD_right
      VALUE_reg_target_block
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
      CALC_FIELD_result
      VALUE_reg_absorbed_damage
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT
      CALC_FIELD_left
      VALUE_reg_target_block
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
      CALC_FIELD_right
      VALUE_reg_absorbed_damage
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
      CALC_FIELD_result
      VALUE_reg_target_block_after
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT
      CALC_FIELD_left
      VALUE_reg_damage_after_vulnerable
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_7
      NUM_END
      CALC_FIELD_right
      VALUE_reg_absorbed_damage
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
      CALC_FIELD_result
      VALUE_reg_unblocked_damage
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_7
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT
      CALC_FIELD_left
      VALUE_reg_target_hp
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      DIGIT_5
      NUM_END
      CALC_FIELD_right
      VALUE_reg_unblocked_damage
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_7
      NUM_END
      CALC_FIELD_result
      VALUE_reg_target_hp_after_subtract
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_5
      DIGIT_5
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_RELU
      CALC_FIELD_input
      VALUE_reg_target_hp_after_subtract
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_3
      DIGIT_4
      NUM_END
      CALC_FIELD_result
      VALUE_reg_target_hp_after
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_3
      DIGIT_5
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

准确率 37.10%，完全一致=False，正常结束=True，首次错误：位置45：期望 `REGION_PILE_SHUFFLE`，预测 `REGION_DRAW`。

目标 header：

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_COMMIT
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PLAY
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_ATTACK
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
            REGION_RELIC_COUNTER
            ENTITY_REF
            ENTITY_RELIC_SLOT_0
            COUNTER_REF
            VALUE_relic_0_counter
            COUNTER_EVENT
            VALUE_PILE_SHUFFLE
            COUNTER_INCREMENT
            NUM_LSD_BEGIN
            DIGIT_1
            NUM_END
            COUNTER_THRESHOLD
            NUM_LSD_BEGIN
            DIGIT_3
            NUM_END
            COUNTER_RESET
            NUM_LSD_BEGIN
            DIGIT_1
            NUM_END
            TRIGGER_EFFECT
            VALUE_GAIN_ENERGY
            TRIGGER_AMOUNT
            NUM_LSD_BEGIN
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
        REGION_CARD_COMMIT
        ENTITY_REF
        ENTITY_SELF
        CARD_ACTION
        VALUE_ANONYMOUS_RULE_VECTOR
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          ENTITY_REF
          ENTITY_SELF
          PAYMENT_COST
          NUM_LSD_BEGIN
          DIGIT_1
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PLAY
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_ATTACK
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
| `root` | `CARD_COMMIT` | True | 100.00% | 无 | - |
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.027836 |
| `root/1` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/1/0` | `ATTACK` | True | 100.00% | 无 | - |
| `root/1/0/0` | `DAMAGE` | False | 97.56% | 位置104：期望 `DIGIT_1`，预测 `DIGIT_9` | 0.042335 |
| `root/1/1` | `PILE_SHUFFLE` | True | 100.00% | 无 | 0.017242 |
| `root/1/2` | `RELIC_COUNTER` | True | 100.00% | 无 | - |
| `root/1/3` | `ENERGY` | True | 100.00% | 无 | 0.012573 |
| `root/1/4` | `DRAW` | True | 100.00% | 无 | 0.006536 |
| `root/2` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.003900 |
| `root/3` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.100000、预测=-0.127836、绝对误差=0.027836。
- `root/1/0/0` / `DAMAGE`：`enemy[0].hp` 目标=-0.045000、预测=-0.002665、绝对误差=0.042335。
- `root/1/1` / `PILE_SHUFFLE`：`piles.draw_count` 目标=+0.030000、预测=+0.017411、绝对误差=0.012589。
- `root/1/1` / `PILE_SHUFFLE`：`piles.discard_count` 目标=-0.030000、预测=-0.008104、绝对误差=0.021896。
- `root/1/3` / `ENERGY`：`player.energy` 目标=+0.200000、预测=+0.212573、绝对误差=0.012573。
- `root/1/4` / `DRAW`：`piles.draw_count` 目标=-0.010000、预测=-0.003464、绝对误差=0.006536。
- `root/2` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.006100、绝对误差=0.003900。

local 出错节点的完整自回归对照：

#### `root/1/0/0` / `DAMAGE`

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
  <READ_REQUEST_BEGIN>
    READ_SOURCE
    VALUE_enemy_0_block
    READ_BIND
    VALUE_reg_target_block
  <READ_REQUEST_END>
  <READ_REQUEST_BEGIN>
    READ_SOURCE
    VALUE_enemy_0_hp
    READ_BIND
    VALUE_reg_target_hp
  <READ_REQUEST_END>
  <CALCULATIONS_BEGIN>
    CALCULATION_BEGIN
      CALCULATION_CHECK_PRESENCE
      CALC_FIELD_field
      VALUE_enemy_0_block_present
      CALC_FIELD_result
      VALUE_reg_has_block
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_MIN
      CALC_FIELD_left
      VALUE_header_BASE_DAMAGE
      CALC_FIELD_right
      VALUE_reg_target_block
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
      CALC_FIELD_result
      VALUE_reg_absorbed_damage
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT
      CALC_FIELD_left
      VALUE_reg_target_block
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
      CALC_FIELD_right
      VALUE_reg_absorbed_damage
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
      CALC_FIELD_result
      VALUE_reg_target_block_after
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT
      CALC_FIELD_left
      VALUE_header_BASE_DAMAGE
      CALC_FIELD_right
      VALUE_reg_absorbed_damage
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
      CALC_FIELD_result
      VALUE_reg_unblocked_damage
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_9
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT
      CALC_FIELD_left
      VALUE_reg_target_hp
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      DIGIT_6
      NUM_END
      CALC_FIELD_right
      VALUE_reg_unblocked_damage
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_9
      NUM_END
      CALC_FIELD_result
      VALUE_reg_target_hp_after_subtract
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_1
      DIGIT_5
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_RELU
      CALC_FIELD_input
      VALUE_reg_target_hp_after_subtract
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_1
      DIGIT_5
      NUM_END
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
  <READ_REQUEST_BEGIN>
    READ_SOURCE
    VALUE_enemy_0_block
    READ_BIND
    VALUE_reg_target_block
  <READ_REQUEST_END>
  <READ_REQUEST_BEGIN>
    READ_SOURCE
    VALUE_enemy_0_hp
    READ_BIND
    VALUE_reg_target_hp
  <READ_REQUEST_END>
  <CALCULATIONS_BEGIN>
    CALCULATION_BEGIN
      CALCULATION_CHECK_PRESENCE
      CALC_FIELD_field
      VALUE_enemy_0_block_present
      CALC_FIELD_result
      VALUE_reg_has_block
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_MIN
      CALC_FIELD_left
      VALUE_header_BASE_DAMAGE
      CALC_FIELD_right
      VALUE_reg_target_block
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
      CALC_FIELD_result
      VALUE_reg_absorbed_damage
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT
      CALC_FIELD_left
      VALUE_reg_target_block
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
      CALC_FIELD_right
      VALUE_reg_absorbed_damage
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
      CALC_FIELD_result
      VALUE_reg_target_block_after
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT
      CALC_FIELD_left
      VALUE_header_BASE_DAMAGE
      CALC_FIELD_right
      VALUE_reg_absorbed_damage
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
      CALC_FIELD_result
      VALUE_reg_unblocked_damage
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_9
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT
      CALC_FIELD_left
      VALUE_reg_target_hp
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      DIGIT_6
      NUM_END
      CALC_FIELD_right
      VALUE_reg_unblocked_damage
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_9
      NUM_END
      CALC_FIELD_result
      VALUE_reg_target_hp_after_subtract
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_9
      DIGIT_5
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_RELU
      CALC_FIELD_input
      VALUE_reg_target_hp_after_subtract
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_1
      DIGIT_5
      NUM_END
      CALC_FIELD_result
      VALUE_reg_target_hp_after
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_9
      DIGIT_5
      NUM_END
    CALCULATION_END
  <CALCULATIONS_END>
  DELTA_SLOT
<REGION_END>
```

