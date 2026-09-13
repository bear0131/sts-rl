# Vector Rule Model v7: Interactive Executor Contract 10

## Responsibility Boundary

Contract 10 keeps the v5 numeric interface while making all register reads
explicit and compact.
The model requests a semantic field and the environment immediately binds a
stable register name. The runtime also records every model-owned `CALC_VALUE`.
When either kind of valued register is later used as an operand, the
environment appends its exact number inline as a masked `CONST_VALUE` block.
There is no response block and no separate register-context prelude.

- A Transformer must not reconstruct an exact integer token sequence from a
  normalized continuous state slot.
- A fixed card or relic rule number is already present in its anonymous rule
  token sequence. The outer Transformer copies/derives that exact number into
  the `NodeSpec` header.
- A runtime number is supplied exactly by the environment after the model
  requests its semantic field.
- Relics enter the world vector as fixed-slot `0/1` presence features.  A
  counter-bearing relic never exposes its exact counter in that vector; a
  `RELIC_COUNTER` region requests the exact value and owns its update.
- Cards with no counter state do not create synthetic counter regions.  Card
  payment also does not update generic `cards_played`, `attacks_played`, or
  `skills_played` fields; each relic owns its own counter.
- The local Transformer still predicts one continuous world delta at each
  `DELTA_SLOT` for each independent state-changing impact.

The last direction is intentionally retained. Mapping exact discrete digits
to a normalized continuous change is not an inverse-quantization problem:
each decimal position has a fixed linear contribution, followed by the fixed
scale of its destination slot.

```text
rule Transformer
  -> anonymous rule vector
  -> outer Transformer emits tree structure and exact NodeSpec constants

world Transformer
  + conditioned NodeSpec
  -> local Transformer requests runtime values
  -> environment binds names and expands every valued operand inline
  -> local Transformer emits symbolic operations and CALC_VALUE results
  -> delta head predicts the normalized vector at DELTA_SLOT

environment
  -> resolves requested runtime fields and records completed CALC_VALUE results
  -> appends masked CONST_VALUE numbers at all later operand uses
  -> applies the predicted/teacher delta before the next node snapshot
```

There is no `SOURCE_RULE` in a local node. The original card sequence is not
needed after the outer model has constructed the complete `NodeSpec`.

## Relic Presence And Counter Regions

The model-facing relic slots are deliberately binary:

```text
player.relic_present[0] = 1       # a relic is installed in slot 0
player.relic_present[1] = 0       # slot 1 is empty
```

The `1` does not mean that the relic counter is `1`; it only selects whether a
relic rule may be active.  Exact values are owned by the environment.  When a
matching event occurs, the outer tree contains a leaf such as:

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
TRIGGER_EFFECT
VALUE_GAIN_STRENGTH
TRIGGER_AMOUNT
NUM_LSD_BEGIN
DIGIT_1
NUM_END
<NODE_EXECUTE>
```

Its local executor performs the complete counter transition:

```text
READ_REQUEST relic[0].counter -> reg.relic_counter
ADD_COUNTER reg.relic_counter CONST_VALUE(2) 1 -> reg.counter_advanced
CHECK_COUNTER_THRESHOLD reg.counter_advanced 3 -> reg.counter_triggered
SELECT_COUNTER_RESULT advanced triggered reset -> reg.relic_counter_after
WRITE_RELIC_COUNTER relic[0].counter <- reg.relic_counter_after
DELTA_SLOT
```

The `CONST_VALUE(2)` is shown compactly here; in the serialized stream it is
`CONST_VALUE`, followed by `NUM_LSD_BEGIN`, the least-significant-first digits,
and `NUM_END`. These tokens are inserted by the environment and masked. A
stateless relic has
`counter_present=false` and can appear in the presence vector, but it can
never produce this region or a counter read.

## Two Kinds Of Numbers

### Rule constants

Examples are base damage, a fixed energy cost, draw count and a fixed status
amount. They are part of the anonymous card/relic program and are predicted
by the outer decoder:

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
```

The same header is copied into the local decoder prefix and is masked from
local loss:

```text
...complete NodeSpec header...
<NODE_EXECUTE>
```

The local calculation refers directly to `header.BASE_DAMAGE`. It must not
emit `READ_REQUEST node_param.BASE_DAMAGE` or reread the card rule.

### Runtime values

Examples are current HP, block, energy, Strength magnitude, pile counts,
relic counters and a recursive call-frame value. The continuous world vector
may expose coarse values and presence bits, but it is not an exact integer
store. The local model emits only what it needs:

```text
model:
  <READ_REQUEST_BEGIN>
  READ_SOURCE
  VALUE_enemy_0_hp
  READ_BIND
  VALUE_reg_target_hp
  <READ_REQUEST_END>

environment:
  # emitted only when the bound register is used as an operand
  VALUE_reg_target_hp
  CONST_VALUE
  NUM_LSD_BEGIN
  DIGIT_0
  DIGIT_5
  NUM_END
```

Digits are least-significant first, so the inline number above is `50`. The
environment locates the value; the model still decides which semantic field is
needed, which operator to use, and what numeric result to predict. A result
register is not expanded at its assignment site: it is assigned once by the
model-owned `CALC_VALUE`. The runtime records that value when the calculation
closes and expands it every time the register is later used as an operand.

```text
CALC_FIELD_result
VALUE_reg_unblocked_damage
CALC_VALUE
NUM_LSD_BEGIN
DIGIT_9
NUM_END

# Later use: the environment copies the stored 9 next to the register name.
CALC_FIELD_right
VALUE_reg_unblocked_damage
CONST_VALUE
NUM_LSD_BEGIN
DIGIT_9
NUM_END
```

## State Views

`FeatureView` is the continuous input to the world Transformer. It contains
normalized scalar features and reliable coarse branch features.

`ValueStore` is the environment-side exact state addressed by semantic paths:

```text
player.hp
player.block
player.energy
player.buff.Strength
enemy[0].hp
enemy[0].block
enemy[0].buff.Vulnerable
piles.draw_count
player.relic_counter[2]
action.paid_energy
call_frame.remaining
```

Known buffs use strict `0/1` presence features. Their exact magnitudes remain
in `ValueStore` and are requested only after the corresponding branch is
selected. Block likewise has a derived `block_present` bit in addition to its
coarse scalar feature.

Presence bits answer questions such as "does Strength change this formula?".
They do not answer "how much Strength is present?".

## Local Calculation And Delta

The model owns field selection, operation selection, ordering, and the numeric
result of each operation. `CALCULATION_SUBTRACT`, `CALCULATION_RELU`, and the
other operator tokens are deliberately retained: the operator is stated at
the start of each block, while `left`, `right`, `input`, and `result` identify
the operands and named result registers.

```text
CALCULATION_BEGIN
CALCULATION_ADD
CALC_FIELD_left
VALUE_header_BASE_DAMAGE
CALC_FIELD_right
VALUE_reg_strength
CALC_FIELD_result
VALUE_reg_damage_after_strength
CALCULATION_END

DELTA_SLOT
```

At `DELTA_SLOT`, the existing continuous delta head predicts the complete
normalized world-delta vector for this atomic node. The token is both a
language target and the alignment point for the vector regression target.
There is one `DELTA_SLOT` for every independent vector in
`EffectNode.delta_slots`; pure container/control nodes have none.

The environment-owned inline constant solves the hard direction:

```text
normalized state -> exact integer digits       forbidden model task
```

The retained delta head learns the easy direction:

```text
exact digits + symbolic operation -> normalized continuous delta
```

For example, an HP change of `-7` maps to `-7 / hp_scale` in one known slot.
The tens and ones digits have fixed positional contributions. This does not
require the model to invert a noisy normalized representation.

## Damage Example

Assume base damage 5, Strength 3, target Vulnerable, target block 4 and target
HP 50. The local model receives `BASE_DAMAGE=5` in its prefix and requests the
runtime operands before the calculation blocks. Bound operand values are
inserted inline by the environment:

```text
READ_REQUEST player.buff.Strength -> reg.strength
READ_REQUEST enemy[0].block -> reg.target_block
READ_REQUEST enemy[0].hp -> reg.target_hp
CALCULATION_ADD header.BASE_DAMAGE reg.strength -> reg.damage_after_strength
CHECK_PRESENCE enemy[0].buff.Vulnerable -> reg.has_vulnerable
CALCULATION_MULTIPLY reg.damage_after_strength vulnerable_multiplier -> reg.damage_vulnerable_numerator
CALCULATION_FLOOR_DIVIDE reg.damage_vulnerable_numerator vulnerable_divisor -> reg.damage_after_vulnerable
CHECK_PRESENCE enemy[0].block_present -> reg.has_block
CALCULATION_MIN reg.damage_after_vulnerable reg.target_block -> reg.absorbed_damage
CALCULATION_SUBTRACT reg.target_block reg.absorbed_damage -> reg.target_block_after
CALCULATION_SUBTRACT reg.damage_after_vulnerable reg.absorbed_damage -> reg.unblocked_damage
CALCULATION_SUBTRACT reg.target_hp reg.unblocked_damage -> reg.target_hp_after_subtract
CALCULATION_RELU reg.target_hp_after_subtract -> reg.target_hp_after
DELTA_SLOT
```

The model never requests the card's base damage and emits no synthetic load
operation for it. The masked NodeSpec prefix establishes the header constant.
The model requests exact runtime operands; each use of a bound operand is
followed inline by an environment-owned `CONST_VALUE` number. The model then
predicts the arithmetic operators, intermediate result numbers, and target
block/HP changes in normalized slot space. The operator remains explicit even
though the numeric result is still model-generated.

## Dynamic Card Parameters

Not every realized number in a card event is a rule constant.

Body Slam declares:

```text
DAMAGE_SOURCE
VALUE_PLAYER_BLOCK
```

Its local node requests `player.block`; the realized block value must not be
placed in `BASE_DAMAGE` merely because an event trace calls it base damage.

Whirlwind declares runtime sources instead of realized X values:

```text
CARD_PAYMENT: PAYMENT_SOURCE VALUE_PLAYER_ENERGY
REPEAT:       REPEAT_SOURCE  VALUE_PAID_ENERGY
CALL:         ARG_DECREMENT  1
```

The payment node reads current energy. The repeat node requests
`action.paid_energy`, and the recursive call requests `call_frame.remaining`.
Only the decrement `1` is a fixed header constant.

The same rule applies to future dynamic costs and parent-derived values: the
header specifies a semantic source; the environment returns its current exact
value only when requested.

## Training Ownership

One shared causal Transformer can train outer and local scopes. Ownership is
defined by masks, not by separate model classes:

| Span | Visible as context | Token loss | Delta loss |
| --- | --- | --- | --- |
| Outer NodeSpec, including rule digits | yes | yes | no |
| Local conditioned NodeSpec prefix | yes | no | no |
| Model read request | yes | yes | no |
| Environment inline constant (`CONST_VALUE` and digits) | yes | no | no |
| Model symbolic operation | yes | yes | no |
| `DELTA_SLOT` | yes | yes | yes |

The next local node is conditioned on a world snapshot updated with the
teacher delta during training. Autoregressive evaluation may apply the
predicted delta to measure rollout drift.

## Online Runtime Integration

`VectorRuleWorldModel.generate_interactive()` implements the same ownership
boundary used by training. A local environment supplies a `VectorValueStore`
whose keys are serialized semantic source tokens and whose binding map is
scoped to the active NodeSpec:

```python
value_store = VectorValueStore.from_state(
    raw_combat_state,
    bindings={
        "enemy[0].hp": "reg.target_hp",
        "enemy[0].block": "reg.target_block",
    },
    extra_values={"action.paid_energy": paid_energy},
)

transcript, deltas = model.generate_interactive(
    state,
    output_begin_id=codec.token_to_id["<OUTPUT_BEGIN>"],
    output_end_id=codec.token_to_id["<REGION_END>"],
    token_index=codec.token_to_id,
    value_binder=value_store,
    prefix_ids=node_spec_prefix_ids,  # REGION_BEGIN ... NODE_EXECUTE
)
delta_slots = transcript.outputs_for_model_token(deltas, "DELTA_SLOT")
```

`from_state()` exposes exact player/enemy fields, pile counts, buffs, and relic
counters. `extra_values` carries invocation-local sources such as paid energy,
parent trigger values, and `call_frame.remaining`.

When the model completes `<READ_REQUEST_BEGIN> READ_SOURCE VALUE_*`, the
runtime calls the store, appends `READ_BIND VALUE_reg_* <READ_REQUEST_END>`,
and records the binding in `transcript.value_bindings`. It also parses each
model-owned `CALC_VALUE` when its calculation closes. Later, when either kind
of valued register appears in a non-result `CALC_FIELD_*`, the runtime appends
`CONST_VALUE` and the current exact number. A callback-backed
`VectorValueStore` can read directly from live simulator state; `write()` is
available for mutable mapping-backed stores.

Binding maps are local because the same world field may serve different roles
in different nodes, such as player block being `reg.damage` for Body Slam.
Random target/card selection remains a separate `QUERY/OBS` interaction and
uses the query oracle only when such a query occurs. For contract-10 local
execution, `prefix_ids` is the environment-owned NodeSpec prefix and generation
terminates on `<REGION_END>`; without a prefix the same method retains the
legacy full-transition `<OUTPUT_BEGIN> ... <OUTPUT_END>` mode.

The returned decoder rows align only with model-owned tokens; environment
insertions do not create dummy delta rows. `outputs_for_model_token()` selects
the predicted vectors located at `DELTA_SLOT` for the environment to apply
before the next node snapshot. For stochastic interaction the model owns all
three query tokens, including `<QUERY_END>`; only the following `OBS` span is
inserted by the environment.

## Contract Invariants

- Card/relic names and `SOURCE_RULE` never enter a local sequence.
- Fixed rule numbers are predicted once by outer and copied into local input.
- No `READ_REQUEST node_param.*` is generated for fixed rule constants.
- No exact runtime number is reconstructed from a continuous vector.
- Runtime bindings and inline `CONST_VALUE` digits are environment-owned and
  masked from token loss.
- Every runtime or calculated register is expanded only at later operand
  sites; there is no register-copy prediction target or context block.
- The model cannot request `effect_delta.*` or otherwise ask for the answer.
- Every realized independent delta aligns with exactly one `DELTA_SLOT`.
- Derived features such as `hp_frac` and `block_present` are not independent
  semantic effects, even when their cached continuous slots are refreshed.
