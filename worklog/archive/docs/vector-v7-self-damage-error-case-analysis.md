# `SUBTRACT_PLAYER_HP/CALC_VALUE` 具体错误案例

这份记录使用线上 baseline checkpoint 的当前参数做单案例局部自回归诊断。
诊断工具是 [`inspect_vector_rule_v7_self_damage.py`](../rl/inspect_vector_rule_v7_self_damage.py)。
本次只增加观察，不修改自伤计算器的现有语义。

## 案例 A：非致命自伤，纯模型数值错误

输入是战士打出「放血」：

```text
战士 HP（操作前） = 10
放血造成的自伤   = 3
敌人             = 60 HP，0 格挡
```

当前数据生成器给出的目标计算是：

```text
10 - 3 = 7
```

对应的 `SELF_DAMAGE` 节点：

```text
READ_TRIGGER_DAMAGE value=3
SUBTRACT_PLAYER_HP left=10 right=3 result=7
```

### 目标局部序列

环境返回的读取值不属于模型预测目标；以下保留完整结构：

```text
<REGION_BEGIN>
  REGION_KIND
  REGION_SELF_DAMAGE
  ENTITY_REF
  ENTITY_SELF
  <NODE_EXECUTE>
  <CALCULATIONS_BEGIN>
    <READ_REQUEST_BEGIN>
      READ_SOURCE
      VALUE_parent_trigger_damage
      READ_BIND
      VALUE_reg_trigger_damage
    <READ_REQUEST_END>
    <READ_RESPONSE_BEGIN>
      READ_BIND
      VALUE_reg_trigger_damage
      READ_VALUE
      NUM_LSD_BEGIN
      DIGIT_3
      NUM_END
    <READ_RESPONSE_END>
    <READ_REQUEST_BEGIN>
      READ_SOURCE
      VALUE_player_hp
      READ_BIND
      VALUE_reg_player_hp
    <READ_REQUEST_END>
    <READ_RESPONSE_BEGIN>
      READ_BIND
      VALUE_reg_player_hp
      READ_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      DIGIT_1
      NUM_END
    <READ_RESPONSE_END>
    CALCULATION_BEGIN
      CALCULATION_SUBTRACT_PLAYER_HP
      CALC_FIELD_left
      VALUE_reg_player_hp
      CALC_FIELD_right
      VALUE_reg_trigger_damage
      CALC_FIELD_result
      VALUE_reg_player_hp_after
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_7
      NUM_END
    CALCULATION_END
  <CALCULATIONS_END>
  DELTA_SLOT
<REGION_END>
```

### 模型局部序列

模型正确生成了所有结构、读取请求和寄存器绑定，但在结果数字处输出了
`13`（低位优先为 `3, 1`）：

```text
CALCULATION_BEGIN
  CALCULATION_SUBTRACT_PLAYER_HP
  CALC_FIELD_left
  VALUE_reg_player_hp
  CALC_FIELD_right
  VALUE_reg_trigger_damage
  CALC_FIELD_result
  VALUE_reg_player_hp_after
  CALC_VALUE
  NUM_LSD_BEGIN
  DIGIT_3
  DIGIT_1
  NUM_END
CALCULATION_END
```

指标：

```text
局部 token 准确率：23 / 29 = 79.31%
首个错误：目标 DIGIT_7，模型 DIGIT_3
目标 result：7
模型 result：13
```

对应的 delta 头也没有得到正确的状态变化：

```text
槽位             目标             模型
player.hp        -0.015000       +0.020059
player.hp_frac   -0.037500       -0.010982
```

因此这个案例说明，`CALCULATION_SUBTRACT_PLAYER_HP/CALC_VALUE` 的高错误率并不是
因为模型没有读到 HP，也不是因为它选错了 `SUBTRACT_PLAYER_HP`。读取值和运算结构
已经正确，错误集中在“根据两个已知数生成精确结果数字”这一步，同时 delta 回归也
没有学到自伤的负方向。

原始完整诊断：
[`vector-v7-self-damage-case-baseline-step1000-hp10.json`](vector-v7-self-damage-case-baseline-step1000-hp10.json)

注意：该文件记录的实际 checkpoint step 是 1500，文件名沿用了当时评测批次的
step-1000 命名。

## 案例 B：致命自伤，当前数据标签错误

输入改为：

```text
战士 HP（操作前） = 2
放血造成的自伤   = 3
```

当前生成器输出的节点是：

```text
SUBTRACT_PLAYER_HP left=2 right=3 result=1
DELTA_SLOT player.hp=-0.005
```

也就是说，当前代码把结果和状态变更都写成了：

```text
max(1, 2 - 3) = 1
```

这不是模型预测造成的，而是 producer 在生成训练目标时就已经这么写了。没有最低
1 点生命的锁血规则时，实际应先按：

```text
max(0, 2 - 3) = 0
```

当前模型在这个样本上还出现了额外的自回归结构错误（重复输出一次
`CALCULATION_SUBTRACT_PLAYER_HP`，并把结果数字预测为 30），因此这个样本不能直接
用来估计纯模型算术能力；它首先证明了训练标签需要单独核查。

原始完整诊断：
[`vector-v7-self-damage-case-baseline-step1500-hp2.json`](vector-v7-self-damage-case-baseline-step1500-hp2.json)

## 结论

目前应把问题拆成两层：

1. 先确认并修正致命自伤样本的 producer 标签，否则模型会被要求学习错误的
   `result=1` 和错误的 delta。
2. 对 HP 足够高、不触发边界的样本，`CALC_VALUE` 仍有独立的精确算术生成问题；
   案例 A 已经证明这一点与最低 HP 规则无关。

## 同一张牌的 HP 扫描

为检查“模型是否默认做加法”，使用同一个 `Bloodletting` 节点，只改变输入 HP，
并复用同一个 baseline checkpoint（step 1500）做局部解码：

| HP | 目标 `HP-3` | 模型输出 |
|---:|---:|---:|
| 4 | 1 | 30 |
| 5 | 2 | 30 |
| 6 | 3 | 60 |
| 7 | 4 | 70 |
| 8 | 5 | 34 |
| 9 | 6 | 34 |
| 10 | 7 | 13 |
| 11 | 8 | 10 |
| 12 | 9 | 10 |
| 13 | 10 | 10 |
| 14 | 11 | 10 |
| 15 | 12 | 10 |
| 16 | 13 | 10 |
| 17 | 14 | 10 |
| 18 | 15 | 14 |
| 19 | 16 | 14 |
| 20 | 17 | 23 |
| 21 | 18 | 20 |
| 22 | 19 | 20 |
| 23 | 20 | 20 |

只有少数点（例如 HP=10、20）碰巧接近 `HP+3`，整体并不符合固定加法。更符合
“读取到了输入，但 decoder 没有稳定学会精确的减法和逐位进借位”。
