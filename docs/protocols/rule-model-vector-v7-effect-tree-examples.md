# Effect Tree Fixed Examples

> These two blocks preserve the earlier concrete-value rendering used to
> inspect tree causality. They are no longer the executor training stream.
> This file is a historical concrete rendering. Contract 9 trains symbolic
> reads and operations as specified in
> [`rule-model-vector-v7-symbolic-executor.md`](rule-model-vector-v7-symbolic-executor.md);
> exact runtime values are inserted as environment-owned inline constants,
> while every `DELTA_SLOT`
> remains a continuous vector target.

This is the concrete, indented token document for the two fixed rule-model
cases. Every line in a tree block is one serialized token; `key=value` pairs
are attributes of that token. Two spaces represent one active region level.
State changes use raw values in calculation rows and normalized values on
`DELTA_SLOT` rows. Siblings execute from top to bottom. Trigger regions are
nested under the operation that caused them. This is the schema-v7 readable
contract and is separate from the current executable contract-9 output.

## Current Local Node Shape (Contract 9)

The concrete blocks below are retained for causal comparison. The executable
local sequence uses inline runtime constants. For a damage node whose
environment bound target HP `50` and target block `0`, the relevant prefix/body
is:

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
    VALUE_enemy_0_hp
    READ_BIND
    VALUE_reg_target_hp
  <READ_REQUEST_END>
  <CALCULATIONS_BEGIN>
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
    VALUE_reg_damage
    CALC_FIELD_result
    VALUE_reg_target_hp_after_subtract
    CALC_VALUE
    NUM_LSD_BEGIN
    DIGIT_3
    NUM_END
    CALCULATION_END
  <CALCULATIONS_END>
  DELTA_SLOT
<REGION_END>
```

`READ_BIND` and `<READ_REQUEST_END>` are environment-owned. At the operand
site, `CONST_VALUE` and its number tokens are also environment-owned and
masked; `CALC_VALUE` and its result digits remain model-owned. There is no
`<READ_RESPONSE_*>` or `<REGISTER_CONTEXT_*>` block.

## Example 1: Dropkick Into Two Vulnerable

Initial relevant state:

```text
player.energy = 3
player.cards_played = 0
player.attacks_played = 0
enemy[0].hp = 50
enemy[0].block = 0
enemy[0].Vulnerable = 2
piles.draw_count = 5
piles.discard_count = 1
```

Concrete tree:

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF

        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          PAYMENT_COST value=1
          <CALCULATIONS_BEGIN>
            CALCULATION_APPLY_DELTA field=player.energy left=3 delta=-1 result=2
            CALCULATION_APPLY_DELTA field=player.cards_played left=0 delta=1 result=1
            CALCULATION_APPLY_DELTA field=player.attacks_played left=0 delta=1 result=1
          <CALCULATIONS_END>
          DELTA_SLOT player.energy=-0.100000; player.counter.cards_played=+0.100000; player.counter.attacks_played=+0.100000
        <REGION_END>

        <REGION_BEGIN>
          REGION_KIND
          REGION_ATTACK
          TARGET_REF
          ENTITY_ENEMY_SLOT_0

          <REGION_BEGIN>
            REGION_KIND
            REGION_DAMAGE
            TARGET_REF
            ENTITY_ENEMY_SLOT_0
            BASE_DAMAGE
            NUM_LSD_BEGIN
            DIGIT_5
            NUM_END
            <CALCULATIONS_BEGIN>
              CALCULATION_READ_STRENGTH value=0
              CALCULATION_ADD_STRENGTH left=5 right=0 result=5
              CALCULATION_READ_VULNERABLE value=2
              CALCULATION_MULTIPLY_VULNERABLE left=5 right=3 result=15
              CALCULATION_FLOOR_DIVIDE_VULNERABLE left=15 right=2 result=7
              CALCULATION_READ_TARGET_BLOCK value=0
              CALCULATION_BLOCK_ABSORB left=7 right=0 absorbed=0
              CALCULATION_SUBTRACT_BLOCK left=0 right=0 result=0
              CALCULATION_SUBTRACT_BLOCK_DAMAGE left=7 right=0 result=7
              CALCULATION_READ_TARGET_HP value=50
              CALCULATION_SUBTRACT_HP left=50 right=7 result=43
            <CALCULATIONS_END>
            DELTA_SLOT enemy[0].hp=-0.035000
          <REGION_END>

          <REGION_BEGIN>
            REGION_KIND
            REGION_CONDITION
            CONDITION_KIND
            CONDITION_TARGET_VULNERABLE
            CONDITION_VALUE
            BOOL_1

            <REGION_BEGIN>
              REGION_KIND
              REGION_ENERGY_REFUND
              ENTITY_REF
              ENTITY_SELF
              ENERGY_AMOUNT value=1
              <CALCULATIONS_BEGIN>
                CALCULATION_APPLY_DELTA field=player.energy left=2 delta=1 result=3
              <CALCULATIONS_END>
              DELTA_SLOT player.energy=+0.100000
            <REGION_END>

            <REGION_BEGIN>
              REGION_KIND
              REGION_DRAW
              ENTITY_REF
              ENTITY_SELF
              DRAW_COUNT
              VALUE_1
              <CALCULATIONS_BEGIN>
                CALCULATION_APPLY_DELTA field=piles.draw_count left=5 delta=-1 result=4
              <CALCULATIONS_END>
              DELTA_SLOT piles.draw_count=-0.010000
            <REGION_END>
          <REGION_END>
        <REGION_END>

        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          PILE_DISCARD
          <CALCULATIONS_BEGIN>
            CALCULATION_APPLY_DELTA field=piles.discard_count left=1 delta=1 result=2
          <CALCULATIONS_END>
          DELTA_SLOT piles.discard_count=+0.010000
        <REGION_END>

        <REGION_BEGIN>
          REGION_KIND
          REGION_ACTION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
          <CALCULATIONS_BEGIN>
          <CALCULATIONS_END>
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

The condition is a child of the attack region because both the energy refund
and the draw depend on the selected target being Vulnerable. A false condition
would close `REGION_CONDITION` without the refund/draw children.

Final relevant state:

```text
player.energy = 3
player.cards_played = 1
player.attacks_played = 1
enemy[0].hp = 43
enemy[0].block = 0
enemy[0].Vulnerable = 2
piles.draw_count = 4
piles.discard_count = 2
```

## Example 2: Pommel Strike, Shuffle, And Sundial

Initial relevant state:

```text
player.energy = 3
player.cards_played = 0
player.attacks_played = 0
enemy[0].hp = 60
enemy[0].block = 0
piles.draw_count = 0
piles.discard_count = 3
relic[0] = Sundial
relic_counter[0] = 2
```

Concrete tree:

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_CARD_PLAY
        ENTITY_REF
        ENTITY_SELF

        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_PAYMENT
          PAYMENT_COST value=1
          <CALCULATIONS_BEGIN>
            CALCULATION_APPLY_DELTA field=player.energy left=3 delta=-1 result=2
            CALCULATION_APPLY_DELTA field=player.cards_played left=0 delta=1 result=1
            CALCULATION_APPLY_DELTA field=player.attacks_played left=0 delta=1 result=1
          <CALCULATIONS_END>
          DELTA_SLOT player.energy=-0.100000; player.counter.cards_played=+0.100000; player.counter.attacks_played=+0.100000
        <REGION_END>

        <REGION_BEGIN>
          REGION_KIND
          REGION_ATTACK
          TARGET_REF
          ENTITY_ENEMY_SLOT_0

          <REGION_BEGIN>
            REGION_KIND
            REGION_DAMAGE
            TARGET_REF
            ENTITY_ENEMY_SLOT_0
            BASE_DAMAGE
            NUM_LSD_BEGIN
            DIGIT_9
            NUM_END
            <CALCULATIONS_BEGIN>
              CALCULATION_READ_STRENGTH value=0
              CALCULATION_ADD_STRENGTH left=9 right=0 result=9
              CALCULATION_NO_VULNERABLE input=9 result=9
              CALCULATION_READ_TARGET_BLOCK value=0
              CALCULATION_BLOCK_ABSORB left=9 right=0 absorbed=0
              CALCULATION_SUBTRACT_BLOCK left=0 right=0 result=0
              CALCULATION_SUBTRACT_BLOCK_DAMAGE left=9 right=0 result=9
              CALCULATION_READ_TARGET_HP value=60
              CALCULATION_SUBTRACT_HP left=60 right=9 result=51
            <CALCULATIONS_END>
            DELTA_SLOT enemy[0].hp=-0.045000
          <REGION_END>
        <REGION_END>

        <REGION_BEGIN>
          REGION_KIND
          REGION_DRAW
          ENTITY_REF
          ENTITY_SELF
          DRAW_COUNT
          VALUE_1

          <REGION_BEGIN>
            REGION_KIND
            REGION_DRAW_PILE_EMPTY
            CONDITION_VALUE
            BOOL_1

            <REGION_BEGIN>
              REGION_KIND
              REGION_PILE_SHUFFLE
              ENTITY_REF
              ENTITY_PILES
              <CALCULATIONS_BEGIN>
                CALCULATION_APPLY_DELTA field=piles.draw_count left=0 delta=3 result=3
                CALCULATION_APPLY_DELTA field=piles.discard_count left=3 delta=-3 result=0
              <CALCULATIONS_END>
              DELTA_SLOT piles.draw_count=+0.030000; piles.discard_count=-0.030000

              <REGION_BEGIN>
                REGION_KIND
                REGION_TRIGGER_ON_SHUFFLE
                TRIGGER_OWNER
                RELIC_VECTOR_SLOT_0

                <REGION_BEGIN>
                  REGION_KIND
                  REGION_RELIC_COUNTER
                  ENTITY_REF
                  ENTITY_SELF
                  <CALCULATIONS_BEGIN>
                    CALCULATION_APPLY_DELTA field=player.relic.counter_sum left=2 delta=-2 result=0
                    CALCULATION_APPLY_DELTA field=player.relic.active_counter_count left=1 delta=-1 result=0
                    CALCULATION_APPLY_DELTA field=player.relic.max_counter left=2 delta=-2 result=0
                    CALCULATION_APPLY_DELTA field=player.relic_counter[0] left=2 delta=-2 result=0
                  <CALCULATIONS_END>
                  DELTA_SLOT player.relic.counter_sum=-0.200000; player.relic.active_counter_count=-0.100000; player.relic.max_counter=-0.200000; player.relic_counter[0]=-0.200000
                <REGION_END>

                <REGION_BEGIN>
                  REGION_KIND
                  REGION_ENERGY_GAIN
                  ENTITY_REF
                  ENTITY_SELF
                  ENERGY_AMOUNT value=2
                  <CALCULATIONS_BEGIN>
                    CALCULATION_APPLY_DELTA field=player.energy left=2 delta=2 result=4
                  <CALCULATIONS_END>
                  DELTA_SLOT player.energy=+0.200000
                <REGION_END>

                <REGION_BEGIN>
                  REGION_KIND
                  REGION_DRAW
                  ENTITY_REF
                  ENTITY_SELF
                  DRAW_COUNT
                  VALUE_2
                  <CALCULATIONS_BEGIN>
                    CALCULATION_APPLY_DELTA field=piles.draw_count left=3 delta=-2 result=1
                  <CALCULATIONS_END>
                  DELTA_SLOT piles.draw_count=-0.020000
                <REGION_END>
              <REGION_END>
            <REGION_END>
          <REGION_END>

          <REGION_BEGIN>
            REGION_KIND
            REGION_DRAW_CARD
            ENTITY_REF
            ENTITY_SELF
            DRAW_COUNT
            VALUE_1
            <CALCULATIONS_BEGIN>
              CALCULATION_APPLY_DELTA field=piles.draw_count left=1 delta=-1 result=0
            <CALCULATIONS_END>
            DELTA_SLOT piles.draw_count=-0.010000
          <REGION_END>
        <REGION_END>

        <REGION_BEGIN>
          REGION_KIND
          REGION_CARD_DESTINATION
          ENTITY_REF
          ENTITY_PILES
          DESTINATION
          PILE_DISCARD
          <CALCULATIONS_BEGIN>
            CALCULATION_APPLY_DELTA field=piles.discard_count left=0 delta=1 result=1
          <CALCULATIONS_END>
          DELTA_SLOT piles.discard_count=+0.010000
        <REGION_END>

        <REGION_BEGIN>
          REGION_KIND
          REGION_ACTION_COMPLETE
          ENTITY_REF
          ENTITY_SELF
          <CALCULATIONS_BEGIN>
          <CALCULATIONS_END>
        <REGION_END>
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

The shuffle and Sundial subtree belongs to the draw region. Attack-triggered
effects would be children of `REGION_ATTACK`; they cannot leak their full token
history into this draw subtree. Only the ordered parent summary is shared.

Final relevant state:

```text
player.energy = 4
player.cards_played = 1
player.attacks_played = 1
enemy[0].hp = 51
enemy[0].block = 0
piles.draw_count = 0
piles.discard_count = 1
relic[0] = Sundial
relic_counter[0] = 0
```

The existing schema-v5 executable output remains in
`rule-model-vector-v6-examples-schema-v5.txt`. The tree examples above define
the intended schema-v7 result and must be generated by the future tree producer
before schema-v7 training begins.
