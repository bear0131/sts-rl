# v7 step 10000 固定与随机自回归评测

- 参数：`.tmp-vector-rule-v7-contract-v10-step10000\last.pt`
- checkpoint step：`10000`
- schema / executor contract：`7 / 10`
- 随机种子：`20260831`
- 设备：`cuda`
- outer：从 `<OUTPUT_BEGIN>` 开始完全自由贪心生成 effect-tree header。
- local：真实 NodeSpec 前缀由环境给出；模型 token 自回归，错误 token 保留在上下文；运行时数值通过环境在寄存器操作数后插入被 mask 的 `CONST_VALUE`，随机目标仍通过 `<OBS_*>` 交互。
- 牌堆：模型只接收抽牌堆、弃牌堆、消耗堆数量，不接收各堆具体卡牌组成。

## 总体结果

- outer 完全正确：1/2；token 准确率 63.21%（212 个 token）。
- local 完全正确节点：20/20；模型 token 准确率 100.00%（524 个 token）。
- delta：12 个向量，活跃位 MAE `0.025597`，全向量 MAE `0.000974`。
- 模型输出 `<UNK>`：0 个。

| 案例 | 打出的牌 | outer | local 节点 | local token | delta 活跃位 MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| 固定例1：飞身踢，敌人有2层易伤 | 飞身踢（Dropkick） | 100.00% | 9/9 | 100.00% | 0.035395 |
| 固定例2：剑柄打击，洗牌并触发日晷 | 剑柄打击（Pommel Strike） | 37.10% | 11/11 | 100.00% | 0.018597 |

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
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.018513 |
| `root/1` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/1/0` | `ATTACK` | True | 100.00% | 无 | - |
| `root/1/0/0` | `DAMAGE` | True | 100.00% | 无 | 0.022121 |
| `root/1/1` | `ENERGY` | True | 100.00% | 无 | 0.133974 |
| `root/1/2` | `DRAW` | True | 100.00% | 无 | 0.001170 |
| `root/2` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.001199 |
| `root/3` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.100000、预测=-0.118513、绝对误差=0.018513。
- `root/1/0/0` / `DAMAGE`：`enemy[0].hp` 目标=-0.035000、预测=-0.012879、绝对误差=0.022121。
- `root/1/1` / `ENERGY`：`player.energy` 目标=+0.100000、预测=+0.233974、绝对误差=0.133974。
- `root/1/2` / `DRAW`：`piles.draw_count` 目标=-0.010000、预测=-0.008830、绝对误差=0.001170。
- `root/2` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.008801、绝对误差=0.001199。

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
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.018876 |
| `root/1` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/1/0` | `ATTACK` | True | 100.00% | 无 | - |
| `root/1/0/0` | `DAMAGE` | True | 100.00% | 无 | 0.036570 |
| `root/1/1` | `PILE_SHUFFLE` | True | 100.00% | 无 | 0.006175 |
| `root/1/2` | `RELIC_COUNTER` | True | 100.00% | 无 | - |
| `root/1/3` | `ENERGY` | True | 100.00% | 无 | 0.060873 |
| `root/1/4` | `DRAW` | True | 100.00% | 无 | 0.000573 |
| `root/2` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.000939 |
| `root/3` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.100000、预测=-0.118876、绝对误差=0.018876。
- `root/1/0/0` / `DAMAGE`：`enemy[0].hp` 目标=-0.045000、预测=-0.008430、绝对误差=0.036570。
- `root/1/1` / `PILE_SHUFFLE`：`piles.draw_count` 目标=+0.030000、预测=+0.019180、绝对误差=0.010820。
- `root/1/1` / `PILE_SHUFFLE`：`piles.discard_count` 目标=-0.030000、预测=-0.028469、绝对误差=0.001531。
- `root/1/3` / `ENERGY`：`player.energy` 目标=+0.200000、预测=+0.260873、绝对误差=0.060873。
- `root/1/4` / `DRAW`：`piles.draw_count` 目标=-0.010000、预测=-0.009427、绝对误差=0.000573。
- `root/2` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.009061、绝对误差=0.000939。
