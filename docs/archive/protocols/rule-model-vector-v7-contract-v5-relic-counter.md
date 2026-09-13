# Vector Rule Model v7 Contract 9: Relic Counter Region

v7 relics are locked features. The world vector contains only fixed binary
presence slots; it does not contain relic rules or exact counters:

```text
player.relic_present[0] = 1   # a relic is installed
player.relic_present[1] = 0   # the slot is empty
```

The `1` is presence, not a counter value. Stateless relics never create a
`RELIC_COUNTER` region. A stateful relic creates one only when its event fires.
For example, a counter-bearing relic in slot 0 can use this shape:

```text
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
  DIGIT_3
  NUM_END
  COUNTER_RESET
  NUM_LSD_BEGIN
  DIGIT_1
  NUM_END
  <NODE_EXECUTE>
  <CALCULATIONS_BEGIN>
    <READ_REQUEST_BEGIN>
      READ_SOURCE
      VALUE_relic_0_counter
      READ_BIND
      VALUE_reg_relic_counter
    <READ_REQUEST_END>
    CALCULATION_BEGIN
      CALCULATION_ADD_COUNTER
      CALC_FIELD_left
      VALUE_reg_relic_counter
      CONST_VALUE
      NUM_LSD_BEGIN
      DIGIT_9
      NUM_END
      CALC_FIELD_right
      VALUE_header_COUNTER_INCREMENT
      CALC_FIELD_result
      VALUE_reg_counter_advanced
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_CHECK_COUNTER_THRESHOLD
      CALC_FIELD_left
      VALUE_reg_counter_advanced
      CALC_FIELD_threshold
      VALUE_header_COUNTER_THRESHOLD
      CALC_FIELD_result
      VALUE_reg_counter_triggered
      CALC_VALUE
      BOOL_1
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_SELECT_COUNTER_RESULT
      CALC_FIELD_advanced
      VALUE_reg_counter_advanced
      CALC_FIELD_triggered
      VALUE_reg_counter_triggered
      CALC_FIELD_reset
      VALUE_header_COUNTER_RESET
      CALC_FIELD_result
      VALUE_reg_relic_counter_after
      CALC_VALUE
      NUM_LSD_BEGIN
      DIGIT_0
      NUM_END
    CALCULATION_END
    CALCULATION_BEGIN
      CALCULATION_WRITE_RELIC_COUNTER
      CALC_FIELD_field
      VALUE_relic_0_counter
      CALC_FIELD_value
      VALUE_reg_relic_counter_after
    CALCULATION_END
  <CALCULATIONS_END>
<REGION_END>
```

The environment owns `READ_BIND`, `<READ_REQUEST_END>`, `CONST_VALUE`, and the
inline number (`9` above); those tokens are visible context but masked from
language loss. There is no `READ_RESPONSE` block and no `REGISTER_CONTEXT`.
The model predicts the semantic request, symbolic operations, and each
`CALC_VALUE` result. A counter update remains environment-owned state and does
not require a continuous `DELTA_SLOT` unless it also changes a modeled world
slot.

The record-level `relics` list contains only `present` and `counter_present`
metadata. Exact values are addressed through the counter request and become
available only at the operand site. A stateless relic therefore cannot be
queried as a counter by accident.
