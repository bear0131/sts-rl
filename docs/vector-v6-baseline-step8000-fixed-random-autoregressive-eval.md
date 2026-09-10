# v7 step 8000 固定与随机自回归评测

- 参数：`.eval-v6-snapshots/baseline-last.pt`
- checkpoint step：`8000`
- schema / executor contract：`7 / 5`
- 随机种子：`20260831`
- 设备：`cuda`
- outer：从 `<OUTPUT_BEGIN>` 开始完全自由贪心生成 effect-tree header。
- local：真实 NodeSpec 前缀由环境给出；模型 token 自回归，错误 token 保留在上下文；只有 `<READ_RESPONSE_*>` 和 `<OBS_*>` 等环境 token 按真实 transcript 注入。
- 牌堆：模型只接收抽牌堆、弃牌堆、消耗堆数量，不接收各堆具体卡牌组成。

## 总体结果

- outer 完全正确：5/12；token 准确率 33.07%（1161 个 token）。
- local 完全正确节点：82/82；模型 token 准确率 100.00%（1615 个 token）。
- delta：37 个向量，活跃位 MAE `0.019202`，全向量 MAE `0.000935`。
- 模型输出 `<UNK>`：0 个。

| 案例 | 打出的牌 | outer | local 节点 | local token | delta 活跃位 MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| 固定例1：飞身踢，敌人有2层易伤 | 飞身踢（Dropkick） | 100.00% | 9/9 | 100.00% | 0.027553 |
| 固定例2：剑柄打击，洗牌并触发日晷 | 剑柄打击（Pommel Strike） | 3.17% | 11/11 | 100.00% | 0.017649 |
| 随机例1 | 重刃+（Heavy Blade+） | 100.00% | 7/7 | 100.00% | 0.011995 |
| 随机例2 | 观察弱点（Spot Weakness） | 8.33% | 7/7 | 100.00% | 0.044955 |
| 随机例3 | 耸肩无视+（Shrug It Off+） | 7.59% | 8/8 | 100.00% | 0.006591 |
| 随机例4 | 吞噬+（Feed+） | 5.52% | 8/8 | 100.00% | 0.015143 |
| 随机例5 | 腐化+（Corruption+） | 7.69% | 5/5 | 100.00% | 0.002017 |
| 随机例6 | 无惧疼痛（Feel No Pain） | 100.00% | 6/6 | 100.00% | 0.066810 |
| 随机例7 | 燃烧（Combust） | 100.00% | 5/5 | 100.00% | 0.006828 |
| 随机例8 | 战斗专注（Battle Trance） | 7.77% | 6/6 | 100.00% | 0.008139 |
| 随机例9 | 双发+（Double Tap+） | 8.79% | 5/5 | 100.00% | 0.002304 |
| 随机例10 | 腐化（Corruption） | 100.00% | 5/5 | 100.00% | 0.000563 |

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
  <RUNTIME_CONTEXT_BEGIN>
  <RUNTIME_CONTEXT_END>
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
  <RUNTIME_CONTEXT_BEGIN>
  <RUNTIME_CONTEXT_END>
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
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.017575 |
| `root/1` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/1/0` | `ATTACK` | True | 100.00% | 无 | - |
| `root/1/0/0` | `DAMAGE` | True | 100.00% | 无 | 0.003445 |
| `root/1/1` | `ENERGY` | True | 100.00% | 无 | 0.114630 |
| `root/1/2` | `DRAW` | True | 100.00% | 无 | 0.001818 |
| `root/2` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.000295 |
| `root/3` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.100000、预测=-0.117575、绝对误差=0.017575。
- `root/1/0/0` / `DAMAGE`：`enemy[0].hp` 目标=-0.035000、预测=-0.031555、绝对误差=0.003445。
- `root/1/1` / `ENERGY`：`player.energy` 目标=+0.100000、预测=+0.214630、绝对误差=0.114630。
- `root/1/2` / `DRAW`：`piles.draw_count` 目标=-0.010000、预测=-0.011818、绝对误差=0.001818。
- `root/2` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.009705、绝对误差=0.000295。

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

准确率 3.17%，完全一致=False，正常结束=True，首次错误：位置2：期望 `<RUNTIME_CONTEXT_END>`，预测 `<READ_REQUEST_BEGIN>`。

目标 header：

```text
<OUTPUT_BEGIN>
  <RUNTIME_CONTEXT_BEGIN>
  <RUNTIME_CONTEXT_END>
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
  <RUNTIME_CONTEXT_BEGIN>
  <READ_REQUEST_BEGIN>
    READ_SOURCE
    VALUE_relic_0_counter
    READ_BIND
    VALUE_runtime_relic_counter_0
  <READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<EFFECT_TREE_BEGIN>
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
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.013979 |
| `root/1` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/1/0` | `ATTACK` | True | 100.00% | 无 | - |
| `root/1/0/0` | `DAMAGE` | True | 100.00% | 无 | 0.028279 |
| `root/1/1` | `PILE_SHUFFLE` | True | 100.00% | 无 | 0.007490 |
| `root/1/2` | `RELIC_COUNTER` | True | 100.00% | 无 | - |
| `root/1/3` | `ENERGY` | True | 100.00% | 无 | 0.063779 |
| `root/1/4` | `DRAW` | True | 100.00% | 无 | 0.001845 |
| `root/2` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.000678 |
| `root/3` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.100000、预测=-0.113979、绝对误差=0.013979。
- `root/1/0/0` / `DAMAGE`：`enemy[0].hp` 目标=-0.045000、预测=-0.016721、绝对误差=0.028279。
- `root/1/1` / `PILE_SHUFFLE`：`piles.draw_count` 目标=+0.030000、预测=+0.023375、绝对误差=0.006625；`piles.discard_count` 目标=-0.030000、预测=-0.021644、绝对误差=0.008356。
- `root/1/3` / `ENERGY`：`player.energy` 目标=+0.200000、预测=+0.263779、绝对误差=0.063779。
- `root/1/4` / `DRAW`：`piles.draw_count` 目标=-0.010000、预测=-0.011845、绝对误差=0.001845。
- `root/2` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.010678、绝对误差=0.000678。

## 3. 随机例1

由训练分布直接生成，随机种子=20260831，流内序号=0；未筛选。

- 场景：第 13 层，第 1 幕，普通遭遇 `Lots Of Slimes`，第 9 回合。
- 打出：重刃+（Heavy Blade+）。卡面规则：造成14点伤害。力量对本牌生效3/5倍。
- 手牌（1 张）：重刃+（Heavy Blade+）。
- 遗物：燃烧之血（Burning Blood，计数器=0）、纸蛙（Paper Frog，计数器=0）、灯笼（Lantern，计数器=0）。
- 战士：生命 32/110，格挡 57，能量 2，金币 848；状态：活力=3、燃烧=5、火焰屏障=2、腐化=3、进化=5；计数器：无。
- 牌堆计数：抽牌堆 1，弃牌堆 5，消耗堆 0；具体组成不提供给模型。
- 敌人 slot 0（`Lots Of Slimes#0`）：生命 20/51，格挡 6，意图伤害 39→14，段数 1，状态：金属化=3。
- 敌人 slot 1（`Lots Of Slimes#1`）：生命 43/59，格挡 49，意图伤害 25→47，段数 1，状态：人工制品=3、金属化=4。

### Outer 结果

准确率 100.00%，完全一致=True，正常结束=True，首次错误：无。

目标 header：

```text
<OUTPUT_BEGIN>
  <RUNTIME_CONTEXT_BEGIN>
  <RUNTIME_CONTEXT_END>
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
          DIGIT_2
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
              DIGIT_4
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
  <RUNTIME_CONTEXT_BEGIN>
  <RUNTIME_CONTEXT_END>
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
          DIGIT_2
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
              DIGIT_4
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
| `root` | `CARD_COMMIT` | True | 100.00% | 无 | - |
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.012467 |
| `root/1` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/1/0` | `ATTACK` | True | 100.00% | 无 | - |
| `root/1/0/0` | `DAMAGE` | True | 100.00% | 无 | 0.015485 |
| `root/2` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.001050 |
| `root/3` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.200000、预测=-0.212467、绝对误差=0.012467。
- `root/1/0/0` / `DAMAGE`：`enemy[0].hp` 目标=-0.040000、预测=-0.057404、绝对误差=0.017404；`enemy[0].block` 目标=-0.060000、预测=-0.059706、绝对误差=0.000294；`enemy[0].block_present` 目标=-1.000000、预测=-0.971242、绝对误差=0.028758。
- `root/2` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.011050、绝对误差=0.001050。

## 4. 随机例2

由训练分布直接生成，随机种子=20260831，流内序号=1；未筛选。

- 场景：第 33 层，第 2 幕，首领遭遇 `Automaton`，第 10 回合。
- 打出：观察弱点（Spot Weakness）。卡面规则：若目标意图攻击，获得3/4点力量。
- 手牌（2 张）：观察弱点（Spot Weakness）、缴械+（Disarm+）。
- 遗物：燃烧之血（Burning Blood，计数器=0）、红头骨（Red Skull，计数器=0）、灯笼（Lantern，计数器=0）、天鹅绒颈圈（Velvet Choker，计数器=2）。
- 战士：生命 96/99，格挡 4，能量 2，金币 115；状态：撕裂=1、狂怒=2、勇猛=1；计数器：无。
- 牌堆计数：抽牌堆 1，弃牌堆 0，消耗堆 2；具体组成不提供给模型。
- 敌人 slot 0（`Automaton#0`）：生命 111/139，格挡 1，意图伤害 25→1，段数 1，状态：无。
- 敌人 slot 1（`Automaton#1`）：生命 81/83，格挡 26，意图伤害 30→16，段数 1，状态：金属化=4、力量=4。
- 敌人 slot 2（`Automaton#2`）：生命 66/69，格挡 31，意图伤害 36→33，段数 3，状态：无。

### Outer 结果

准确率 8.33%，完全一致=False，正常结束=True，首次错误：位置8：期望 `<READ_RESPONSE_BEGIN>`，预测 `<READ_REQUEST_END>`。

目标 header：

```text
<OUTPUT_BEGIN>
  <RUNTIME_CONTEXT_BEGIN>
  <READ_REQUEST_BEGIN>
    READ_SOURCE
    VALUE_relic_3_counter
    READ_BIND
    VALUE_runtime_relic_counter_3
  <READ_REQUEST_END>
  <READ_RESPONSE_BEGIN>
    READ_BIND
    VALUE_runtime_relic_counter_3
    READ_VALUE
    NUM_LSD_BEGIN
    DIGIT_2
    NUM_END
  <READ_RESPONSE_END>
  <RUNTIME_CONTEXT_END>
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
          VALUE_SKILL
          <REGION_BEGIN>
            REGION_KIND
            REGION_RELIC_COUNTER
            ENTITY_REF
            ENTITY_RELIC_SLOT_3
            COUNTER_REF
            VALUE_relic_3_counter
            COUNTER_EVENT
            VALUE_CARD_PLAY
            COUNTER_INCREMENT
            NUM_LSD_BEGIN
            DIGIT_1
            NUM_END
            COUNTER_THRESHOLD
            NUM_LSD_BEGIN
            DIGIT_6
            NUM_END
            COUNTER_RESET
            NUM_LSD_BEGIN
            DIGIT_0
            NUM_END
            TRIGGER_EFFECT
            VALUE_NONE
            TRIGGER_AMOUNT
            NUM_LSD_BEGIN
            DIGIT_0
            NUM_END
          <REGION_END>
          <REGION_BEGIN>
            REGION_KIND
            REGION_STATE_CHANGE
            ENTITY_REF
            ENTITY_SELF
            FIELD_OR_STATUS
            VALUE_player_buff_Strength
            CHANGE_AMOUNT
            NUM_LSD_BEGIN
            DIGIT_3
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
  <RUNTIME_CONTEXT_BEGIN>
  <READ_REQUEST_BEGIN>
    READ_SOURCE
    VALUE_relic_3_counter
    READ_BIND
    VALUE_runtime_relic_counter_3
  <READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<REGION_BEGIN>
  REGION_KIND
  REGION_RELIC_COUNTER
  ENTITY_REF
  ENTITY_RELIC_SLOT_3
  COUNTER_REF
  VALUE_relic_3_counter
  COUNTER_EVENT
  VALUE_CARD_PLAY
  COUNTER_INCREMENT
  NUM_LSD_BEGIN
  DIGIT_1
  NUM_END
  COUNTER_THRESHOLD
  NUM_LSD_BEGIN
  DIGIT_6
  NUM_END
  COUNTER_RESET
  NUM_LSD_BEGIN
  DIGIT_0
  NUM_END
  TRIGGER_EFFECT
  VALUE_NONE
  TRIGGER_AMOUNT
  NUM_LSD_BEGIN
  DIGIT_0
  NUM_END
<REGION_END>
<REGION_BEGIN>
  REGION_KIND
  REGION_STATE_CHANGE
  ENTITY_REF
  ENTITY_SELF
  FIELD_OR_STATUS
  VALUE_player_buff_Strength
  CHANGE_AMOUNT
  NUM_LSD_BEGIN
  DIGIT_3
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
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.000901 |
| `root/1` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/1/0` | `RELIC_COUNTER` | True | 100.00% | 无 | - |
| `root/1/1` | `STATE_CHANGE` | True | 100.00% | 无 | 0.132779 |
| `root/2` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.001185 |
| `root/3` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.100000、预测=-0.099099、绝对误差=0.000901。
- `root/1/1` / `STATE_CHANGE`：`player.buff.Strength` 目标=+1.000000、预测=+1.132779、绝对误差=0.132779。
- `root/2` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.011185、绝对误差=0.001185。

## 5. 随机例3

由训练分布直接生成，随机种子=20260831，流内序号=2；未筛选。

- 场景：第 41 层，第 3 幕，普通遭遇 `Transient`，第 10 回合。
- 打出：耸肩无视+（Shrug It Off+）。卡面规则：获得8/11点格挡。抽1张牌。
- 手牌（4 张）：死神+（Reaper+）、完美打击（Perfected Strike）、耸肩无视+（Shrug It Off+）、幽灵铠甲+（Ghostly Armor+）。
- 遗物：燃烧之血（Burning Blood，计数器=0）、开信刀（Letter Opener，计数器=0）、锚（Anchor，计数器=0）、金刚杵（Vajra，计数器=0）、添水（Sozu，计数器=0）、咖啡滤杯（Coffee Dripper，计数器=0）。
- 战士：生命 76/100，格挡 5，能量 4，金币 532；状态：残暴=3、敏捷=3；计数器：无。
- 牌堆计数：抽牌堆 2，弃牌堆 3，消耗堆 1；具体组成不提供给模型。
- 敌人 slot 0（`Transient#0`）：生命 48/195，格挡 6，意图伤害 7→6，段数 4，状态：中毒=4。
- 敌人 slot 1（`Transient#1`）：生命 54/163，格挡 39，意图伤害 9→52，段数 2，状态：多层护甲=5、易伤=4。

### Outer 结果

准确率 7.59%，完全一致=False，正常结束=True，首次错误：位置8：期望 `<READ_RESPONSE_BEGIN>`，预测 `<READ_REQUEST_END>`。

目标 header：

```text
<OUTPUT_BEGIN>
  <RUNTIME_CONTEXT_BEGIN>
  <READ_REQUEST_BEGIN>
    READ_SOURCE
    VALUE_relic_1_counter
    READ_BIND
    VALUE_runtime_relic_counter_1
  <READ_REQUEST_END>
  <READ_RESPONSE_BEGIN>
    READ_BIND
    VALUE_runtime_relic_counter_1
    READ_VALUE
    NUM_LSD_BEGIN
    DIGIT_0
    NUM_END
  <READ_RESPONSE_END>
  <RUNTIME_CONTEXT_END>
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
          VALUE_SKILL
          <REGION_BEGIN>
            REGION_KIND
            REGION_RELIC_COUNTER
            ENTITY_REF
            ENTITY_RELIC_SLOT_1
            COUNTER_REF
            VALUE_relic_1_counter
            COUNTER_EVENT
            VALUE_SKILL_PLAY
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
            VALUE_DAMAGE_ALL
            TRIGGER_AMOUNT
            NUM_LSD_BEGIN
            DIGIT_5
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
            DIGIT_1
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
  <RUNTIME_CONTEXT_BEGIN>
  <READ_REQUEST_BEGIN>
    READ_SOURCE
    VALUE_relic_1_counter
    READ_BIND
    VALUE_runtime_relic_counter_1
  <READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<EFFECT_TREE_BEGIN>
  <REGION_BEGIN>
    REGION_KIND
    REGION_RELIC_COUNTER
    ENTITY_REF
    ENTITY_RELIC_SLOT_1
    COUNTER_REF
    VALUE_relic_1_counter
    COUNTER_EVENT
    VALUE_SKILL_PLAY
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
    VALUE_DAMAGE_ALL
    TRIGGER_AMOUNT
    NUM_LSD_BEGIN
    DIGIT_1
    NUM_END
  <REGION_END>
  <REGION_BEGIN>
    REGION_KIND
    REGION_DAMAGE
    ENTITY_REF
    ENTITY_ENEMY_SLOT_0
    BASE_DAMAGE
    NUM_LSD_BEGIN
    DIGIT_5
    DIGIT_1
    DIGIT_1
    DIGIT_1
    NUM_END
    DAMAGE_SOURCE_KIND
    VALUE_RELIC_FIXED
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
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.003828 |
| `root/1` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/1/0` | `RELIC_COUNTER` | True | 100.00% | 无 | - |
| `root/1/1` | `STATE_CHANGE` | True | 100.00% | 无 | 0.021952 |
| `root/1/2` | `DRAW` | True | 100.00% | 无 | 0.000536 |
| `root/2` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.000047 |
| `root/3` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.100000、预测=-0.103828、绝对误差=0.003828。
- `root/1/1` / `STATE_CHANGE`：`player.block` 目标=+0.110000、预测=+0.088048、绝对误差=0.021952。
- `root/1/2` / `DRAW`：`piles.draw_count` 目标=-0.010000、预测=-0.010536、绝对误差=0.000536。
- `root/2` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.010047、绝对误差=0.000047。

## 6. 随机例4

由训练分布直接生成，随机种子=20260831，流内序号=3；未筛选。

- 场景：第 6 层，第 1 幕，普通遭遇 `Blue Slaver`，第 2 回合。
- 打出：吞噬+（Feed+）。卡面规则：造成10/12点伤害。致命：最大生命增加3/4。消耗。
- 手牌（4 张）：铁斩波（Iron Wave）、发掘（Exhume）、坚毅+（True Grit+）、吞噬+（Feed+）。
- 遗物：双节棍（Nunchaku，计数器=0）。
- 战士：生命 101/105，格挡 19，能量 1，金币 206；状态：无；计数器：无。
- 牌堆计数：抽牌堆 2，弃牌堆 2，消耗堆 0；具体组成不提供给模型。
- 敌人 slot 0（`Blue Slaver#0`）：生命 36/45，格挡 53，意图伤害 33→22，段数 3，状态：多层护甲=4。

### Outer 结果

准确率 5.52%，完全一致=False，正常结束=True，首次错误：位置8：期望 `<READ_RESPONSE_BEGIN>`，预测 `<READ_REQUEST_END>`。

目标 header：

```text
<OUTPUT_BEGIN>
  <RUNTIME_CONTEXT_BEGIN>
  <READ_REQUEST_BEGIN>
    READ_SOURCE
    VALUE_relic_0_counter
    READ_BIND
    VALUE_runtime_relic_counter_0
  <READ_REQUEST_END>
  <READ_RESPONSE_BEGIN>
    READ_BIND
    VALUE_runtime_relic_counter_0
    READ_VALUE
    NUM_LSD_BEGIN
    DIGIT_0
    NUM_END
  <READ_RESPONSE_END>
  <RUNTIME_CONTEXT_END>
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
            REGION_RELIC_COUNTER
            ENTITY_REF
            ENTITY_RELIC_SLOT_0
            COUNTER_REF
            VALUE_relic_0_counter
            COUNTER_EVENT
            VALUE_ATTACK_PLAY
            COUNTER_INCREMENT
            NUM_LSD_BEGIN
            DIGIT_1
            NUM_END
            COUNTER_THRESHOLD
            NUM_LSD_BEGIN
            DIGIT_0
            DIGIT_1
            NUM_END
            COUNTER_RESET
            NUM_LSD_BEGIN
            DIGIT_1
            NUM_END
            TRIGGER_EFFECT
            VALUE_GAIN_ENERGY
            TRIGGER_AMOUNT
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
              DIGIT_2
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
  <RUNTIME_CONTEXT_BEGIN>
  <READ_REQUEST_BEGIN>
    READ_SOURCE
    VALUE_relic_0_counter
    READ_BIND
    VALUE_runtime_relic_counter_0
  <READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<EFFECT_TREE_BEGIN>
  <REGION_BEGIN>
    REGION_KIND
    REGION_CARD_PLAY
    ENTITY_REF
    ENTITY_SELF
    PLAY_KIND
    VALUE_ATTACK
    <REGION_BEGIN>
      REGION_KIND
      REGION_RELIC_COUNTER
      ENTITY_REF
      ENTITY_RELIC_SLOT_0
      COUNTER_REF
      VALUE_relic_0_counter
      COUNTER_EVENT
      VALUE_ATTACK_PLAY
      COUNTER_INCREMENT
      NUM_LSD_BEGIN
      DIGIT_1
      NUM_END
      COUNTER_THRESHOLD
      NUM_LSD_BEGIN
      DIGIT_0
      DIGIT_1
      NUM_END
      COUNTER_RESET
      NUM_LSD_BEGIN
      DIGIT_1
      NUM_END
      TRIGGER_EFFECT
      VALUE_GAIN_ENERGY
      TRIGGER_AMOUNT
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
        DIGIT_2
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
| `root` | `CARD_COMMIT` | True | 100.00% | 无 | - |
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.002186 |
| `root/1` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/1/0` | `RELIC_COUNTER` | True | 100.00% | 无 | - |
| `root/1/1` | `ATTACK` | True | 100.00% | 无 | - |
| `root/1/1/0` | `DAMAGE` | True | 100.00% | 无 | 0.042150 |
| `root/2` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.001093 |
| `root/3` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.100000、预测=-0.102186、绝对误差=0.002186。
- `root/1/1/0` / `DAMAGE`：`enemy[0].block` 目标=-0.120000、预测=-0.162150、绝对误差=0.042150。
- `root/2` / `CARD_DESTINATION`：`piles.exhaust_count` 目标=+0.010000、预测=+0.008907、绝对误差=0.001093。

## 7. 随机例5

由训练分布直接生成，随机种子=20260831，流内序号=4；未筛选。

- 场景：第 43 层，第 3 幕，普通遭遇 `Three Shapes`，第 1 回合。
- 打出：腐化+（Corruption+）。卡面规则：技能牌费用变为0，但打出后会被消耗。
- 手牌（3 张）：势不可挡（Juggernaut）、腐化+（Corruption+）、武装（Armaments）。
- 遗物：燃烧之血（Burning Blood，计数器=0）、纸蛙（Paper Frog，计数器=0）、精致折扇（Ornamental Fan，计数器=0）、锚（Anchor，计数器=0）、水银沙漏（Mercury Hourglass，计数器=0）、小血瓶（Blood Vial，计数器=0）、符文金字塔（Runic Pyramid，计数器=0）、符文圆顶（Runic Dome，计数器=0）。
- 战士：生命 73/87，格挡 41，能量 2，金币 748；状态：脆弱=5、力量=4、撕裂=3、敏捷=7、恶魔形态=4；计数器：无。
- 牌堆计数：抽牌堆 1，弃牌堆 2，消耗堆 2；具体组成不提供给模型。
- 敌人 slot 0（`Three Shapes#0`）：生命 26/102，格挡 59，意图伤害 19→33，段数 3，状态：多层护甲=5。
- 敌人 slot 1（`Three Shapes#1`）：生命 110/120，格挡 25，意图伤害 7→20，段数 4，状态：无。
- 敌人 slot 2（`Three Shapes#2`）：生命 60/74，格挡 4，意图伤害 10→53，段数 2，状态：人工制品=2、中毒=1。

### Outer 结果

准确率 7.69%，完全一致=False，正常结束=True，首次错误：位置8：期望 `<READ_RESPONSE_BEGIN>`，预测 `<READ_REQUEST_END>`。

目标 header：

```text
<OUTPUT_BEGIN>
  <RUNTIME_CONTEXT_BEGIN>
  <READ_REQUEST_BEGIN>
    READ_SOURCE
    VALUE_relic_2_counter
    READ_BIND
    VALUE_runtime_relic_counter_2
  <READ_REQUEST_END>
  <READ_RESPONSE_BEGIN>
    READ_BIND
    VALUE_runtime_relic_counter_2
    READ_VALUE
    NUM_LSD_BEGIN
    DIGIT_0
    NUM_END
  <READ_RESPONSE_END>
  <RUNTIME_CONTEXT_END>
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
          DIGIT_2
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PLAY
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_POWER
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
  <RUNTIME_CONTEXT_BEGIN>
  <READ_REQUEST_BEGIN>
    READ_SOURCE
    VALUE_relic_2_counter
    READ_BIND
    VALUE_runtime_relic_counter_2
  <READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<EFFECT_TREE_BEGIN>
  <REGION_BEGIN>
    REGION_KIND
    REGION_STATE_CHANGE
    ENTITY_REF
    ENTITY_SELF
    FIELD_OR_STATUS
    VALUE_player_buff_Double_Tap
    CHANGE_AMOUNT
    NUM_LSD_BEGIN
    SIGN_NEG
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
| `root` | `CARD_COMMIT` | True | 100.00% | 无 | - |
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.002554 |
| `root/1` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/2` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.001481 |
| `root/3` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.200000、预测=-0.202554、绝对误差=0.002554。
- `root/2` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.011481、绝对误差=0.001481。

## 8. 随机例6

由训练分布直接生成，随机种子=20260831，流内序号=5；未筛选。

- 场景：第 33 层，第 2 幕，首领遭遇 `Collector`，第 5 回合。
- 打出：无惧疼痛（Feel No Pain）。卡面规则：每当一张牌被消耗时，获得3/4点格挡。
- 手牌（3 张）：战吼（Warcry）、无惧疼痛（Feel No Pain）、双持+（Dual Wield+）。
- 遗物：金刚杵（Vajra，计数器=0）、打击木偶（StrikeDummy，计数器=0）、地精之角（Gremlin Horn，计数器=0）、符文圆顶（Runic Dome，计数器=0）。
- 战士：生命 41/103，格挡 28，能量 3，金币 782；状态：进化=4、恶魔形态=1；计数器：无。
- 牌堆计数：抽牌堆 0，弃牌堆 5，消耗堆 1；具体组成不提供给模型。
- 敌人 slot 0（`Collector#0`）：生命 109/116，格挡 46，意图伤害 41→32，段数 4，状态：虚弱=4、多层护甲=2。

### Outer 结果

准确率 100.00%，完全一致=True，正常结束=True，首次错误：无。

目标 header：

```text
<OUTPUT_BEGIN>
  <RUNTIME_CONTEXT_BEGIN>
  <RUNTIME_CONTEXT_END>
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
          VALUE_POWER
          <REGION_BEGIN>
            REGION_KIND
            REGION_STATE_CHANGE
            ENTITY_REF
            ENTITY_SELF
            FIELD_OR_STATUS
            VALUE_player_buff_Feel_No_Pain
            CHANGE_AMOUNT
            NUM_LSD_BEGIN
            DIGIT_3
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
  <RUNTIME_CONTEXT_BEGIN>
  <RUNTIME_CONTEXT_END>
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
          VALUE_POWER
          <REGION_BEGIN>
            REGION_KIND
            REGION_STATE_CHANGE
            ENTITY_REF
            ENTITY_SELF
            FIELD_OR_STATUS
            VALUE_player_buff_Feel_No_Pain
            CHANGE_AMOUNT
            NUM_LSD_BEGIN
            DIGIT_3
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
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.007226 |
| `root/1` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/1/0` | `STATE_CHANGE` | True | 100.00% | 无 | 0.192208 |
| `root/2` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.000996 |
| `root/3` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.100000、预测=-0.107226、绝对误差=0.007226。
- `root/1/0` / `STATE_CHANGE`：`player.buff.Feel No Pain` 目标=+1.000000、预测=+1.192208、绝对误差=0.192208。
- `root/2` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.010996、绝对误差=0.000996。

## 9. 随机例7

由训练分布直接生成，随机种子=20260831，流内序号=6；未筛选。

- 场景：第 1 层，第 1 幕，精英遭遇 `Gremlin Nob`，第 6 回合。
- 打出：燃烧（Combust）。卡面规则：回合结束时失去1点生命，并对所有敌人造成5/7点伤害。
- 手牌（4 张）：吞噬（Feed）、灼热攻击+（Searing Blow+）、突破极限（Limit Break）、燃烧（Combust）。
- 遗物：锚（Anchor，计数器=0）。
- 战士：生命 94/103，格挡 9，能量 3，金币 312；状态：腐化=4、力量=3、黑暗之拥=5、双发=2、无惧疼痛=1；计数器：无。
- 牌堆计数：抽牌堆 0，弃牌堆 5，消耗堆 2；具体组成不提供给模型。
- 敌人 slot 0（`Gremlin Nob#0`）：生命 41/52，格挡 37，意图伤害 35→41，段数 4，状态：力量=5、易伤=3、虚弱=4。
- 敌人 slot 1（`Gremlin Nob#1`）：生命 27/63，格挡 21，意图伤害 6→33，段数 2，状态：人工制品=2。
- 敌人 slot 2（`Gremlin Nob#2`）：生命 37/62，格挡 19，意图伤害 28→32，段数 4，状态：中毒=1。

### Outer 结果

准确率 100.00%，完全一致=True，正常结束=True，首次错误：无。

目标 header：

```text
<OUTPUT_BEGIN>
  <RUNTIME_CONTEXT_BEGIN>
  <RUNTIME_CONTEXT_END>
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
          VALUE_POWER
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
  <RUNTIME_CONTEXT_BEGIN>
  <RUNTIME_CONTEXT_END>
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
          VALUE_POWER
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
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.012417 |
| `root/1` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/2` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.001239 |
| `root/3` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.100000、预测=-0.112417、绝对误差=0.012417。
- `root/2` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.011239、绝对误差=0.001239。

## 10. 随机例8

由训练分布直接生成，随机种子=20260831，流内序号=7；未筛选。

- 场景：第 9 层，第 1 幕，普通遭遇 `Blue Slaver`，第 3 回合。
- 打出：战斗专注（Battle Trance）。卡面规则：抽3/4张牌。本回合不能再抽牌。
- 手牌（4 张）：战斗专注（Battle Trance）、狂暴+（Berserk+）、耸肩无视+（Shrug It Off+）、狂暴+（Berserk+）。
- 遗物：燃烧之血（Burning Blood，计数器=0）、苦无（Kunai，计数器=1）、灯笼（Lantern，计数器=0）。
- 战士：生命 36/105，格挡 6，能量 5，金币 302；状态：脆弱=4、进化=4、无惧疼痛=3；计数器：无。
- 牌堆计数：抽牌堆 3，弃牌堆 5，消耗堆 5；具体组成不提供给模型。
- 敌人 slot 0（`Blue Slaver#0`）：生命 25/83，格挡 7，意图伤害 0→31，段数 2，状态：无。

### Outer 结果

准确率 7.77%，完全一致=False，正常结束=True，首次错误：位置8：期望 `<READ_RESPONSE_BEGIN>`，预测 `<READ_REQUEST_END>`。

目标 header：

```text
<OUTPUT_BEGIN>
  <RUNTIME_CONTEXT_BEGIN>
  <READ_REQUEST_BEGIN>
    READ_SOURCE
    VALUE_relic_1_counter
    READ_BIND
    VALUE_runtime_relic_counter_1
  <READ_REQUEST_END>
  <READ_RESPONSE_BEGIN>
    READ_BIND
    VALUE_runtime_relic_counter_1
    READ_VALUE
    NUM_LSD_BEGIN
    DIGIT_1
    NUM_END
  <READ_RESPONSE_END>
  <RUNTIME_CONTEXT_END>
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
          DIGIT_0
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PLAY
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_SKILL
          <REGION_BEGIN>
            REGION_KIND
            REGION_DRAW
            ENTITY_REF
            ENTITY_SELF
            DRAW_COUNT
            NUM_LSD_BEGIN
            DIGIT_3
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
  <RUNTIME_CONTEXT_BEGIN>
  <READ_REQUEST_BEGIN>
    READ_SOURCE
    VALUE_relic_1_counter
    READ_BIND
    VALUE_runtime_relic_counter_1
  <READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
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
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | - |
| `root/1` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/1/0` | `DRAW` | True | 100.00% | 无 | 0.015330 |
| `root/2` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.000947 |
| `root/3` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/1/0` / `DRAW`：`piles.draw_count` 目标=-0.030000、预测=-0.014670、绝对误差=0.015330。
- `root/2` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.010947、绝对误差=0.000947。

## 11. 随机例9

由训练分布直接生成，随机种子=20260831，流内序号=8；未筛选。

- 场景：第 44 层，第 3 幕，普通遭遇 `Transient`，第 2 回合。
- 打出：双发+（Double Tap+）。卡面规则：本回合接下来1/2张攻击牌会打出两次。
- 手牌（2 张）：双发+（Double Tap+）、观察弱点+（Spot Weakness+）。
- 遗物：燃烧之血（Burning Blood，计数器=0）、水银沙漏（Mercury Hourglass，计数器=0）、日晷（Sundial，计数器=0）、灯笼（Lantern，计数器=0）、符文立方体（Runic Cube，计数器=0）、符文圆顶（Runic Dome，计数器=0）。
- 战士：生命 65/101，格挡 40，能量 1，金币 171；状态：无；计数器：无。
- 牌堆计数：抽牌堆 1，弃牌堆 4，消耗堆 5；具体组成不提供给模型。
- 敌人 slot 0（`Transient#0`）：生命 140/142，格挡 33，意图伤害 22→48，段数 3，状态：多层护甲=4、虚弱=1。
- 敌人 slot 1（`Transient#1`）：生命 166/178，格挡 37，意图伤害 45→48，段数 4，状态：无。
- 敌人 slot 2（`Transient#2`）：生命 66/66，格挡 57，意图伤害 10→37，段数 2，状态：无。

### Outer 结果

准确率 8.79%，完全一致=False，正常结束=True，首次错误：位置8：期望 `<READ_RESPONSE_BEGIN>`，预测 `<READ_REQUEST_END>`。

目标 header：

```text
<OUTPUT_BEGIN>
  <RUNTIME_CONTEXT_BEGIN>
  <READ_REQUEST_BEGIN>
    READ_SOURCE
    VALUE_relic_2_counter
    READ_BIND
    VALUE_runtime_relic_counter_2
  <READ_REQUEST_END>
  <READ_RESPONSE_BEGIN>
    READ_BIND
    VALUE_runtime_relic_counter_2
    READ_VALUE
    NUM_LSD_BEGIN
    DIGIT_0
    NUM_END
  <READ_RESPONSE_END>
  <RUNTIME_CONTEXT_END>
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
          VALUE_SKILL
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
  <RUNTIME_CONTEXT_BEGIN>
  <READ_REQUEST_BEGIN>
    READ_SOURCE
    VALUE_relic_2_counter
    READ_BIND
    VALUE_runtime_relic_counter_2
  <READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<READ_REQUEST_END>
<EFFECT_TREE_BEGIN>
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
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.004144 |
| `root/1` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/2` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.000464 |
| `root/3` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.100000、预测=-0.104144、绝对误差=0.004144。
- `root/2` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.010464、绝对误差=0.000464。

## 12. 随机例10

由训练分布直接生成，随机种子=20260831，流内序号=9；未筛选。

- 场景：第 3 层，第 1 幕，普通遭遇 `Two Fungi Beasts`，第 5 回合。
- 打出：腐化（Corruption）。卡面规则：技能牌费用变为0，但打出后会被消耗。
- 手牌（5 张）：吞噬（Feed）、腐化（Corruption）、火焰屏障+（Flame Barrier+）、双持（Dual Wield）、坚毅+（True Grit+）。
- 遗物：燃烧之血（Burning Blood，计数器=0）、纸鹤（Paper Crane，计数器=0）。
- 战士：生命 37/70，格挡 47，能量 3，金币 760；状态：脆弱=2、敏捷=8、黑暗之拥=3；计数器：无。
- 牌堆计数：抽牌堆 1，弃牌堆 3，消耗堆 0；具体组成不提供给模型。
- 敌人 slot 0（`Two Fungi Beasts#0`）：生命 90/117，格挡 50，意图伤害 26→11，段数 2，状态：虚弱=1、易伤=2、金属化=5。

### Outer 结果

准确率 100.00%，完全一致=True，正常结束=True，首次错误：无。

目标 header：

```text
<OUTPUT_BEGIN>
  <RUNTIME_CONTEXT_BEGIN>
  <RUNTIME_CONTEXT_END>
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
          DIGIT_3
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PLAY
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_POWER
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
  <RUNTIME_CONTEXT_BEGIN>
  <RUNTIME_CONTEXT_END>
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
          DIGIT_3
          NUM_END
        <REGION_END>
        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PLAY
          ENTITY_REF
          ENTITY_SELF
          PLAY_KIND
          VALUE_POWER
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
| `root/0` | `CARD_PAYMENT` | True | 100.00% | 无 | 0.001055 |
| `root/1` | `CARD_PLAY` | True | 100.00% | 无 | - |
| `root/2` | `CARD_DESTINATION` | True | 100.00% | 无 | 0.000072 |
| `root/3` | `COMPLETE` | True | 100.00% | 无 | - |

delta 活跃位明细（数值均为规范化向量空间）：

- `root/0` / `CARD_PAYMENT`：`player.energy` 目标=-0.300000、预测=-0.298945、绝对误差=0.001055。
- `root/2` / `CARD_DESTINATION`：`piles.discard_count` 目标=+0.010000、预测=+0.009928、绝对误差=0.000072。
