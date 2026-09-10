# Vector Rule Model v7: Effect Tree

> Current executable protocol: executor contract 9 is specified in
> [`rule-model-vector-v7-symbolic-executor.md`](rule-model-vector-v7-symbolic-executor.md).
> The tree topology, node taxonomy, scope isolation and recursive repeat
> design in this document still apply. Exact runtime values now arrive through
> environment-owned inline constants; `DELTA_SLOT` remains the continuous update
> target for each independent state-changing impact.

The current schema-v7 stream is an effect tree.  The legacy schema-v5 stream is
retained only as the producer-side realized event trace.  A multi-effect card
otherwise repeats a large amount of unrelated context:
an attack, its retaliation/kill triggers, a draw, and draw-triggered relics
all share one causal Transformer history.  This proposal introduces an
ordered effect tree for the next data contract.  Schema-v5 remains available
until the tree compiler, trainer, and examples have been migrated together.

The complete concrete serialization for the two fixed cases is in
[`rule-model-vector-v7-effect-tree-examples.md`](rule-model-vector-v7-effect-tree-examples.md).
That document is the readable contract: each line in a tree block is one
token (token attributes are written on the same line), and indentation shows
the active region scope.  The examples are intentionally independent of card
names in the serialized stream; names only identify the environment-side test
fixture.

## Goals

- Keep the exact execution order of a card and all triggered effects.
- Make the ownership of a trigger explicit by nesting it under its cause.
- Keep every state-changing leaf atomic and emit one `DELTA_SLOT` per
  independent impact.
- Execute every `DAMAGE` invocation independently from its NodeSpec and the
  current player/target vectors, including recursive multi-hit calls.
- Prevent sibling regions from attending to each other's complete token
  histories.  A sibling receives only the parent context and the previous
  sibling's compact summary.
- Keep cards, relics, and hand instances anonymous.  Names remain
  environment-side lookup keys and never enter the model stream.

## Tree Data Model

The producer first builds an `EffectNode` tree and only then serializes it.
The minimal node fields are:

```text
EffectNode
  kind: CARD_PLAY | CARD_PAYMENT | CARD_DESTINATION | ATTACK | DAMAGE | SELF_DAMAGE |
        DRAW | DRAW_CARD | PILE_SHUFFLE | ENERGY | RELIC_COUNTER |
        STATE_CHANGE | CONDITION | TRIGGER | REPEAT | CALL | AOE_DAMAGE_ROUND |
        RANDOM_HIT | QUERY | COMPLETE
  entity_ref: ENTITY_SELF | ENTITY_PILES | ENTITY_ENEMY_SLOT_i
  header_params: typed values extracted before entering the node
  children: ordered tuple[EffectNode, ...]
  calculations: symbolic read/operation plan for this node
  realized_delta: continuous regression target aligned with DELTA_SLOT
  condition: optional predicate evaluated by the environment
  repeat_count: optional integer for REPEAT nodes
  call_target: optional SELF or local node definition for CALL nodes
  query/observation: optional environment interaction for QUERY nodes
```

`children` is an execution-order list, not a set.  A node's children are
evaluated left to right.  A trigger is a child of the effect that caused it;
it is not appended as an unrelated top-level event.  A `STATE_CHANGE` leaf
requests exact operands by name and commits a typed operation; the environment
updates the state and derives its vector representation.

The serialized grammar uses explicit scopes.  Every scope has a matching end
marker and the renderer indents begin/end markers:

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EFFECT_TREE_BEGIN>
      <REGION_BEGIN>
        REGION_KIND
        REGION_ATTACK
        ... ordered child regions ...
      <REGION_END>
    <EFFECT_TREE_END>
  <TRANSITION_END>
<OUTPUT_END>
```

The model-owned calculation grammar contains field names, register names and
operations. Runtime reads are requested by semantic source, then each bound
register operand is followed by an environment-owned `CONST_VALUE` and its
least-significant-first number. Those constant tokens are masked from language
loss. `SOURCE_RULE` and card/relic name tokens remain absent.

## Canonical Node Kinds

`EffectNode.kind` uses the following stable vocabulary.  Operation-specific
values such as `ENERGY_OP=GAIN` or `CONDITION_TARGET_VULNERABLE` are attributes
of a kind, not additional kinds.  This keeps the output vocabulary bounded.

| Kind | Category | Meaning | Representative example |
| --- | --- | --- | --- |
| `CARD_PLAY` | container | Root of one played-card transition. | Pommel Strike: payment, attack, draw, destination, complete. |
| `CARD_PAYMENT` | leaf | Pay the typed energy cost or read the X-cost source. It does not update generic play counters. | Energy 3 -> 2. |
| `CARD_DESTINATION` | leaf | Put the played card in discard, exhaust, or another destination. | Pommel Strike enters the discard pile. |
| `ATTACK` | container | Ordered attack operation that owns damage and attack-trigger children. | Dropkick attacks enemy slot 0, then checks Vulnerable. |
| `DAMAGE` | leaf | One target hit; starts with base damage, then computes modifiers, block, HP, and one delta. | Enemy slot 0: base 5 -> Vulnerable-adjusted result 7 -> HP 50 -> 43. |
| `SELF_DAMAGE` | leaf | Apply a parent/card-bound damage value to player HP. | Offering reads its HP-loss binding and player HP, then commits damage. |
| `DRAW` | container | Request a number of cards and own empty-pile, shuffle, and draw-trigger children. | Pommel Strike requests two cards. |
| `DRAW_CARD` | leaf | Consume one card from the draw-count vector after the draw operation succeeds. | Draw count 1 -> 0 for Pommel's second draw. |
| `PILE_SHUFFLE` | effect/container | Move discard-count into draw-count when the draw pile is empty; may own shuffle triggers. | Draw 0, discard 3 -> draw 3, discard 0. |
| `ENERGY` | leaf | Add, refund, or spend energy; the operation is an attribute. | Dropkick `ENERGY_OP=REFUND`, energy 2 -> 3. |
| `RELIC_COUNTER` | leaf | Read one exact counter, add the declared increment, check threshold, select reset/advanced result, and write the new counter. | Sundial counter 2 -> 0 after a shuffle. |
| `STATE_CHANGE` | leaf | Generic atomic vector delta for HP, block, buffs, debuffs, flags, or counters not covered above. | Apply 2 Vulnerable to an enemy power vector. |
| `CONDITION` | branch | Evaluate a predicate and open children only when it is true. | `TARGET_HAS_VULNERABLE=true` opens refund and draw children. |
| `TRIGGER` | container | Causal effects owned by the operation that fired them. | Shuffle owns the Sundial trigger subtree. |
| `REPEAT` | container/template | Define one body and invoke it recursively until the runtime remaining count is zero; the body is not statically unrolled. | Whirlwind requests `action.paid_energy` and uses one all-enemy body. |
| `CALL` | control | Invoke a node body or itself with a runtime call frame and fixed decrement. | `CALL_SELF` requests `call_frame.remaining`, then subtracts one. |
| `AOE_DAMAGE_ROUND` | container | One deterministic repeat iteration over enemy slots. | Whirlwind round: damage slot 0, then slot 1. |
| `RANDOM_HIT` | container | One random-hit iteration containing a query, observation, and damage child. | Sword Boomerang asks for a target, then hits slot 1. |
| `QUERY` | interaction | Pause generation and require an environment observation before continuing. | `QUERY_RANDOM_TARGET` -> environment returns `ENTITY_ENEMY_SLOT_1`. |
| `COMPLETE` | terminal | Close the action after all ordered children finish. | `ACTION_COMPLETE` after Pommel's discard update. |

The readable region aliases used by the fixed examples map to these kinds:
`REGION_CARD_PLAY` -> `CARD_PLAY`, `REGION_CARD_PAYMENT` -> `CARD_PAYMENT`,
`REGION_CARD_DESTINATION` -> `CARD_DESTINATION`, `REGION_ATTACK` -> `ATTACK`,
`REGION_DAMAGE` -> `DAMAGE`, `REGION_DRAW` -> `DRAW`, `REGION_DRAW_CARD` ->
`DRAW_CARD`, `REGION_PILE_SHUFFLE` -> `PILE_SHUFFLE`, `REGION_ENERGY_GAIN` and
`REGION_ENERGY_REFUND` -> `ENERGY`, `REGION_RELIC_COUNTER` -> `RELIC_COUNTER`,
`REGION_DRAW_PILE_EMPTY` -> `CONDITION`, `REGION_TRIGGER_ON_SHUFFLE` ->
`TRIGGER`, and `REGION_ACTION_COMPLETE` -> `COMPLETE`.  These aliases make the
examples easy to scan; the producer should store the canonical kind plus
operation/predicate attributes.

Every container has ordered `children`; every state-changing leaf has one
symbolic calculation plan and one or more `DELTA_SLOT` entries, one per
independent impact. `QUERY` pauses generation for
an environment observation; `READ_REQUEST` declares a semantic runtime field
and the environment binds it immediately. A false
`CONDITION` has no child effects. A `DAMAGE` child does not read a card rule
sequence: it requests the bound base-damage parameter and exact runtime fields
by semantic name.

## Node Parameter Boundary

The card/relic rule vector is converted into a small, typed `NodeSpec` before
the node body is executed:

```text
NodeSpec
  kind
  entity_ref
  header_params       immutable scalars/enums extracted from card/relic vector
  world_reads         declared WorldRead values needed by this node
  children
```

`WorldRead` is a runtime input capability, not a serialized card feature:

```text
PLAYER_VECTOR       current warrior vector, including buffs and relic counts
ENEMY_VECTOR(slot)   current vector for one enemy slot
ENEMY_MASK           active/eligible enemy slot ids, without their full vectors
PILE_VECTOR          draw/discard/exhaust counts
PARENT_SUMMARY       compact result of the parent or previous sibling
CALL_FRAME           remaining count and invocation index for recursive calls
QUERY_RESPONSE       environment answer, such as the selected enemy slot
DRAW_RESULT_FLAGS    optional environment flags for a drawn card/status
```

The outer autoregressive model emits the ordered `NodeSpec` headers.  A node
executor then reads only the header, the selected world entity slice, and the
parent/previous-child summaries.  It never receives the card name or the
original card-rule token sequence.  For example, the payment boundary is:

```text
known NodeSpec prefix (emitted by outer)
  <REGION_BEGIN>
  REGION_KIND
  REGION_CARD_PAYMENT
  PAYMENT_COST
  NUM_LSD_BEGIN
  DIGIT_1
  NUM_END
  PLAY_KIND
  VALUE_ATTACK
  <NODE_EXECUTE>

executor prediction
  <CALCULATIONS_BEGIN>
  ...
  <CALCULATIONS_END>
  DELTA_SLOT
<REGION_END>
```

`<NODE_EXECUTE>` is the hard ownership boundary.  Everything through that
token is copied from the outer result into the executor context.  Those
tokens participate in causal attention but are masked out of the local token
loss.  The first local label is the token after `<NODE_EXECUTE>`.  At
inference the executor is seeded with this exact prefix, so it cannot replace
`REGION_CARD_PAYMENT` with `REGION_CARD_PLAY`, change the payment cost, or
invent a different target mode.

This makes a node-local training record possible.  The same
`NodeExecutorTransformer` can be trained on parameterized payment, damage,
draw, and state-change records without including any card identity.  At
runtime it may be one shared Transformer with per-kind adapters, or separate
small Transformers per node family; the input contract is identical.

The parameter extraction boundary is:

```text
CARD_PAYMENT     PAYMENT_COST or PAYMENT_SOURCE
CARD_PLAY        PLAY_KIND
CARD_DESTINATION DESTINATION, CARD_COUNT
ATTACK           TARGET_MODE, HIT_COUNT
DAMAGE           BASE_DAMAGE or DAMAGE_SOURCE, DAMAGE_FORMULA_KIND
DRAW             DRAW_COUNT
ENERGY           ENERGY_OP, ENERGY_AMOUNT
STATE_CHANGE     FIELD_OR_STATUS, CHANGE_AMOUNT, TARGET_MODE
CONDITION        PREDICATE_KIND, PREDICATE_VALUE
TRIGGER          TRIGGER_EVENT, TRIGGER_OWNER
REPEAT           REPEAT_SOURCE, ORDER_MODE
CALL             CALL_TARGET, ARG_DECREMENT
AOE_DAMAGE_ROUND ENEMY_ORDER
RANDOM_HIT       HIT_INDEX, QUERY_KIND
QUERY            QUERY_KIND, CANDIDATE_SET
```

`DRAW_CARD`, `PILE_SHUFFLE`, `RELIC_COUNTER`, and `COMPLETE` have no card-rule
lookup in their bodies.  They receive only their explicit operation values,
the relevant world vectors, and summaries from their parent. A fixed
`BASE_DAMAGE=5` is an outer header value extracted from anonymous rule tokens,
not a request to reread the card. A dynamic value such as Body Slam's current
block is represented by `DAMAGE_SOURCE` instead.

## Call Boundaries And World Inputs

Each node declares a minimal world-input set.  The executor rejects a call
that does not provide a required read and does not expose undeclared vectors to
the node.  This is the intended boundary for the current count-only pile
contract:

| Kind | Required world reads | Boundary rule |
| --- | --- | --- |
| `CARD_PLAY` | selected card vector, `PLAYER_VECTOR`, target availability mask | The only node allowed to consume the selected card vector; it emits child headers. |
| `CARD_PAYMENT` | `PLAYER_VECTOR` | Uses fixed `PAYMENT_COST`, or requests the runtime value named by `PAYMENT_SOURCE`; no original card rule. |
| `CARD_DESTINATION` | `PILE_VECTOR` | Uses the extracted destination; only pile counts are changed. |
| `ATTACK` | `PLAYER_VECTOR`, target mask or selected target refs, `PARENT_SUMMARY` | Chooses/forwards target scope; damage children receive one enemy vector each. |
| `DAMAGE` | `PLAYER_VECTOR`, `ENEMY_VECTOR(target)`, `PARENT_SUMMARY` | Reads base-damage header and current target state; never reads another enemy or card tokens. |
| `DRAW` | `PLAYER_VECTOR`, `PILE_VECTOR`, `PARENT_SUMMARY` | Uses extracted `DRAW_COUNT`; owns empty-pile, shuffle, and draw-trigger children. |
| `DRAW_CARD` | `PILE_VECTOR`, optional `DRAW_RESULT_FLAGS` | Changes only pile counts; drawn-card identity is unavailable unless the environment supplies flags. |
| `PILE_SHUFFLE` | `PILE_VECTOR`, `PARENT_SUMMARY` | Transfers pile counts and passes a shuffle summary to trigger children; the trigger reads relic counters separately. |
| `ENERGY` | `PLAYER_VECTOR` | Uses extracted `ENERGY_OP` and `ENERGY_AMOUNT`; no enemy vector unless a separate condition selected it. |
| `RELIC_COUNTER` | `PLAYER_VECTOR` (relic slice), `PARENT_SUMMARY` | Updates only declared relic-counter fields and does not inspect card rules. |
| `STATE_CHANGE` | `PLAYER_VECTOR` or `ENEMY_VECTOR(target)` | One target per leaf; an AOE parent spawns one leaf per target. |
| `CONDITION` | only the operand vectors named by its predicate, `PARENT_SUMMARY` | A false predicate closes without exposing or generating child effects. |
| `TRIGGER` | source-event `PARENT_SUMMARY`, trigger-owner vector | Inherits cause context; it cannot read unrelated sibling histories. |
| `REPEAT` | `CALL_FRAME`, inherited target mask, `PARENT_SUMMARY` | Defines one static body and passes the frame to recursive calls. |
| `CALL` | `CALL_FRAME`, `PARENT_SUMMARY` | Control-only node; it does not read world state itself. |
| `AOE_DAMAGE_ROUND` | `PLAYER_VECTOR`, active enemy mask, `CALL_FRAME` | Iterates slot ids; each `DAMAGE` child receives one refreshed enemy vector. |
| `RANDOM_HIT` | eligible enemy mask, `CALL_FRAME`, `PARENT_SUMMARY` | Queries before reading enemy stats; after the observation, passes only the selected enemy vector. |
| `QUERY` | query constraints and candidate ids | Waits for the environment; stats are supplied only if the query predicate requires them. |
| `COMPLETE` | `PARENT_SUMMARY` and final transition summary | Terminal marker; performs no card or enemy lookup. |

`PLAYER_VECTOR` includes warrior scalars, buffs, combat counters, and the
relic-count vector.  `PILE_VECTOR` remains a separate small world vector even
though pile contents are intentionally not exposed.  A node that needs data
from several enemies must receive an explicit `ENEMY_MASK` and delegate the
per-enemy work to child nodes; it never receives the entire enemy matrix by
default.

At call entry the environment gives the executor read-only snapshots of the
declared vectors.  The executor emits calculation results and delta slots; the
environment applies those deltas atomically before the next sibling or
recursive call is entered.  Consequently a recursive body sees the updated
`PLAYER_VECTOR` and `ENEMY_VECTOR(target)` from the previous invocation, not a
stale copy.  A call frame is:

```text
CallFrame
  node_spec_ref       static node header/body
  invocation_index    runtime-supplied ordinal
  remaining           runtime-supplied recursion argument
  inherited_scope     target mask and entity references
  parent_summary      compact causal context
  world_snapshot      declared player/enemy/pile vector views
```

`world_snapshot` is scoped to the node's `WorldRead` declaration and is not a
mutable global pointer.  This lets the same node executor be trained against
standalone vector snapshots while preserving the exact sequential state
updates during a full combat transition.

There is no damage-result cache in the call frame. Every recursive `DAMAGE`
invocation reloads its `BASE_DAMAGE` binding from the static NodeSpec, checks
the current Strength/Vulnerable/block presence features, requests the exact
runtime operands it needs, and expands the full arithmetic chain. This keeps
the body independently trainable and remains correct when an earlier hit or
nested trigger changes combat state.

## Recursive Repeat

`REPEAT` is represented as a small recursive program instead of an expanded
list of rounds.  The node contains one body and one `CALL_SELF` instruction:

```text
<REGION_BEGIN>
  REGION_KIND
  REGION_REPEAT
  REPEAT_SOURCE VALUE_PAID_ENERGY
  <REGION_BEGIN>
    REGION_KIND
    REGION_AOE_DAMAGE_ROUND
    INVOCATION_INDEX value=0
    ... one round body and its per-target DELTA_SLOT rows ...
  <REGION_END>
  <CALL_BEGIN>
    CALL_KIND
    CALL_SELF
    ARG_DECREMENT value=1
  <CALL_END>
<REGION_END>
```

The runtime calls the same body with `remaining=2`, then `remaining=1`, and
stops at `remaining=0`.  The model emits the body once and does not generate
three copies of its tokens.  Each runtime invocation still evaluates its
damage against the current world vectors and records independent deltas and
trigger children; only the static body is shared.  The invocation index and
remaining count are execution context supplied by the runtime, so they do not
carry card identity.

The execution rule is equivalent to:

```text
run_repeat(remaining):
  if remaining == 0:
    return
  run body(invocation_index)
  if remaining > 1:
    CALL_SELF(remaining - 1)
```

The recursive call is bounded by the requested `action.paid_energy`, the
runtime `call_frame.remaining`, and a global maximum call depth.
A failed condition or a dead-target policy may return before issuing the next
`CALL_SELF`.  At the end of the recursive call, the parent receives one compact
repeat summary containing the per-invocation delta summaries and final world
state, rather than the full history of every round.

## Local Attention

Tree serialization carries a scope id for each token (not emitted as a card
name or other semantic feature).  The decoder uses a block-causal mask:

```text
token may attend to:  ancestors, the current region, and prior summary tokens
token may not attend to: complete histories of prior sibling regions
```

At `<REGION_END>`, a small learned `REGION_SUMMARY` is emitted.  It contains
the region kind, changed-state summary, and unresolved trigger/query count.
The next sibling attends to this summary instead of every token in the
previous subtree.  The parent receives one summary per child.  This preserves
execution order while making attention and KV-cache growth proportional to
the active subtree plus summaries.

The implementation can use one causal Transformer with a scope-aware mask, so
the outer architecture remains the same:

```text
rule Transformer(s) -> anonymous card/relic vectors
state/world Transformer -> world memory
outer effect Transformer -> ordered NodeSpec headers and tree structure
NodeExecutorTransformer + local scope mask -> node bodies autoregressively
```

For long regions, the same mask can be implemented as separate local decoder
windows with a parent summary prefix.  Both implementations produce identical
tokens and labels.  `NodeExecutorTransformer` is a logical boundary: it may
share the outer Transformer's weights at runtime, while node-local records can
still be sampled and trained independently.  A separate per-family executor
is an optional optimization, not a required third model.

### Local Sequence Length

For the tree model, sequence length is measured per invocation scope, not by
concatenating every runtime expansion.  The outer model receives the ordered
headers and summaries.  A node executor receives only:

```text
node header + declared world reads + parent/previous-child summary + local body
```

`REPEAT` contributes its static body once plus a `CALL_SELF` frame; an X-cost
Whirlwind with five energy therefore does not multiply the executor input by
five.  `AOE_DAMAGE_ROUND` likewise iterates the active enemy mask at runtime,
while each `DAMAGE` invocation receives only one enemy vector.  The useful
capacity metrics for a future v7 trainer are therefore
`outer_header_length_max`, `node_local_length_max`, and
`repeat_static_body_length_max`; the flat v6 `sequence_length_max` is not a
proxy for any of these local limits.

The schema-v7 trainer is now available as
`rl.train_rule_vectors_v7`.  It keeps the same anonymous card/relic encoders,
world encoder and one shared causal decoder, but trains that decoder on two
views of every generated record: one outer header sequence and one local
sequence for every node.  Local examples do not pass the selected card, hand,
or relic rule programs; all parameters needed by the node are already in its
typed header.  For a local example, that header plus `<NODE_EXECUTE>` is an
input-only prefix and only the calculation/body, `DELTA_SLOT`, and closing
marker are prediction targets.  Each producer record is expanded in memory
and consumed once, so this remains compatible with one-pass online data
generation.

Calculation rows use the expanded v7 grammar: runtime reads are followed by
environment-owned `READ_BIND` names, and every bound register used as an
operand is immediately followed by `CONST_VALUE` plus a masked number. Then
`CALCULATIONS_BEGIN` opens `CALCULATION_BEGIN`, the operator,
`CALC_FIELD_*` entries and model-owned result numbers (`NUM_LSD_BEGIN`, digit
tokens, `NUM_END`), followed by `CALCULATION_END`. Numeric values are therefore
not baked into vocabulary items and a new value does not become an unknown token
merely because it was absent from the bootstrap stream.

The shared causal event decoder is configured with four Transformer layers by
default (`decoder_layers=4`). Outer tree headers and local register/calculation
bodies use this same decoder; the card, relic, hand, and world encoders remain
separate configurable modules.

Each independent state-changing impact emits one `DELTA_SLOT` line.  It is the
alignment point for one complete dense world-delta regression target; sparse
`index=value` delta lines are not part of the language sequence.

The v7 decoder vocabulary is built once from a bootstrap JSONL stream and then
held fixed while fresh records are generated.  Tokens not present in the
bootstrap are mapped to `<UNK>` and counted in the run metrics.  A remote CUDA
run can be started with:

```bash
.venv/bin/python -m rl.train_rule_vectors_v7 \
  --vocab-data .tmp-vector-rule-v7-remote.jsonl \
  --stream-command '.venv/bin/python -m rl.make_vector_rule_v7_data --stdout --stream --seed 1701' \
  --stream-steps 10000 --batch-size 1024 --device cuda \
  --validation-stream-command '.venv/bin/python -m rl.make_vector_rule_v7_data --stdout --stream --seed 1702' \
  --validation-stream-steps 1 --validation-every-steps 100
```

The checkpoint schema is `7` and must not be loaded by the schema-v5 flat
trainer or evaluator.  `sequence_length_max` in v7 training refers to the
short outer/local scope actually sent to attention; the diagnostic flat event
length remains available in the producer statistics only.

## Damage Subregion

Damage is one atomic child region rather than separate resolve/apply regions.
Its header states the semantic operation directly: apply the card's damage
formula to one target.  The card vector is consumed by the enclosing
`CARD_PLAY` region; the damage child does not look up a card name or reread the
card-rule token sequence.

The canonical shape is:

```text
<REGION_BEGIN>
  REGION_KIND
  REGION_DAMAGE
  TARGET_REF
  ENTITY_ENEMY_SLOT_0
  BASE_DAMAGE
  NUM_LSD_BEGIN
  DIGIT_5
  NUM_END
  <NODE_EXECUTE>
  <CALCULATIONS_BEGIN>
    ... a bound runtime operand is followed by CONST_VALUE and its digits ...
    ... operations refer directly to VALUE_header_BASE_DAMAGE ...
    ... strength/vulnerable arithmetic, producing result=7 ...
    ... block absorption and HP subtraction ...
  <CALCULATIONS_END>
  DELTA_SLOT enemy[0].hp=-0.035000
<REGION_END>
```

`BASE_DAMAGE` is the only card-derived scalar at the start of the damage
subregion. Outer predicts it from the anonymous rule tokens and local receives
it in the masked NodeSpec prefix. There is no card lookup or read request. The
calculation chain then derives the actual pre-block damage;
that result is not emitted a second time as a standalone amount token.  The
remaining calculations explain how it changes the target state.

For a multi-hit attack, `REPEAT` invokes this same static DAMAGE NodeSpec
recursively. Every invocation performs the complete calculation against the
refreshed player and target vectors; it never rereads the card rule because
`BASE_DAMAGE` is already in the NodeSpec:

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
  <CALCULATIONS_BEGIN>
    ... use VALUE_header_BASE_DAMAGE directly ...
    ... read current Strength and calculate this hit ...
    ... apply current Vulnerable, block absorption, and HP subtraction ...
  <CALCULATIONS_END>
  DELTA_SLOT
<REGION_END>
```

This is the intended interpretation of:

```text
3 base damage, 3 hits, +3 Strength, Vulnerable
  3 + 3 = 6
  floor(6 * 3 / 2) = 9 per hit
  REPEAT count=3 executes the independent 9-damage body three times
```

There is deliberately no `COUNT_ATTACK_HITS`, `DAMAGE_RESOLUTION`, or
`REUSE_ATTACK_DAMAGE` operation inside the body. The parent repeat controls
the number of calls; each call owns one delta, so block, HP, death, and on-hit
triggers can change after every hit.

## Card Shapes

### Pommel Strike: attack then draw

The top-level card region has two ordered children.  Attack-side consequences
stay below the attack; draw-side relic effects stay below the draw:

```text
<REGION_BEGIN>
  REGION_KIND
  REGION_CARD_PLAY
  <REGION_BEGIN>
    REGION_KIND
    REGION_ATTACK
    <REGION_BEGIN>
      REGION_KIND
      REGION_DAMAGE
      TARGET_REF
      ENTITY_ENEMY_SLOT_0
      BASE_DAMAGE
      NUM_LSD_BEGIN
      DIGIT_9
      NUM_END
      <NODE_EXECUTE>
      <CALCULATIONS_BEGIN>
        CALCULATION_LOAD_HEADER_BASE_DAMAGE
        ... strength, block absorption, and HP subtraction ...
      <CALCULATIONS_END>
      DELTA_SLOT enemy[0].hp=-0.045000
      <REGION_BEGIN>
        REGION_KIND
        REGION_TRIGGER_ON_ATTACK
        ... retaliation, kill-trigger draw, or other attack consequences ...
      <REGION_END>
    <REGION_END>
    <REGION_BEGIN>
      REGION_KIND
      REGION_DRAW
      ... draw one card and update draw_count ...
      <REGION_BEGIN>
        REGION_KIND
        REGION_TRIGGER_ON_DRAW
        ... Sundial/Abacus counter and draw effects ...
      <REGION_END>
    <REGION_END>
  <REGION_END>
<REGION_END>
```

The attack and draw regions are siblings, so the draw decoder does not attend
to every attack-calculation token.  It sees the card parent and the compact
attack summary, which is enough to continue the ordered transition.

### Whirlwind: deterministic all-enemy repeat

Whirlwind has no random query.  Its tree contains one recursive repeat region
with a fixed round order over enemy slots. Each recursive round independently
calculates one hit for every active enemy from the refreshed vectors:

```text
<REGION_BEGIN>
  REGION_KIND
  REGION_REPEAT
  REPEAT_SOURCE VALUE_PAID_ENERGY
  <REGION_BEGIN>
    REGION_KIND
    REGION_AOE_DAMAGE_ROUND
    INVOCATION_INDEX value=0
    <REGION_BEGIN>
      REGION_KIND
      REGION_DAMAGE
      ENTITY_REF
      ENTITY_ENEMY_SLOT_0
      ... independently calculate and apply this slot-0 hit ...
      DELTA_SLOT ...
    <REGION_END>
    <REGION_BEGIN>
      REGION_KIND
      REGION_DAMAGE
      ENTITY_REF
      ENTITY_ENEMY_SLOT_1
      ... independently calculate and apply this slot-1 hit ...
      DELTA_SLOT ...
    <REGION_END>
  <REGION_END>
  <CALL_BEGIN>
    CALL_KIND
    CALL_SELF
    ARG_DECREMENT value=1
  <CALL_END>
<REGION_END>
```

The repeat count is the number of recursive invocations, not a license to
collapse all targets into one delta.  The runtime invokes the same body for
`remaining=2` and `remaining=1`; each target and each hit retains an atomic
delta and can own nested retaliation or kill-trigger regions.

### Sword Boomerang: random target per hit

Sword Boomerang has one query for every runtime invocation.  The environment
observation identifies the enemy slot, and that observation is inserted before
the hit subtree.  The repeat body is emitted once and recursively called for the
remaining hits:

```text
<REGION_BEGIN>
  REGION_KIND
  REGION_REPEAT
  REPEAT_SOURCE VALUE_PAID_ENERGY
  <REGION_BEGIN>
    REGION_KIND
    REGION_RANDOM_HIT
    INVOCATION_INDEX value=0
    <QUERY_BEGIN>
      QUERY_RANDOM_TARGET
    <QUERY_END>
    <OBS_BEGIN>
      ENTITY_ENEMY_SLOT_1
    <OBS_END>
    <REGION_BEGIN>
      REGION_KIND
      REGION_DAMAGE
      ENTITY_REF
      ENTITY_ENEMY_SLOT_1
      ... independently calculate this slot-1 hit from the NodeSpec ...
      DELTA_SLOT ...
    <REGION_END>
  <REGION_END>
  <CALL_BEGIN>
    CALL_KIND
    CALL_SELF
    ARG_DECREMENT value=1
  <CALL_END>
<REGION_END>
```

The runtime repeats the body with `remaining=2` and `remaining=1`, asking the
environment for a new target on every invocation. Every returned target runs
the same complete `DAMAGE` calculation against its current vector. The model
does not emit three copies of the random-hit subtree.

### Dropkick: conditional children of the attack

Dropkick's attack is followed by a condition evaluated against the target's
Vulnerable state.  The refund and draw are children of the successful branch,
not unrelated top-level events:

```text
<REGION_BEGIN>
  REGION_KIND
  REGION_ATTACK
  REGION_DAMAGE ... DELTA_SLOT ...
  <REGION_BEGIN>
    REGION_KIND
    REGION_CONDITION
    CONDITION_TARGET_VULNERABLE
    <REGION_BEGIN>
      REGION_KIND
      REGION_ENERGY_REFUND
      ... +1 player energy ...
    <REGION_END>
    <REGION_BEGIN>
      REGION_KIND
      REGION_DRAW
      ... -1 draw_count ...
    <REGION_END>
  <REGION_END>
<REGION_END>
```

If the condition is false, the condition region closes with no child effects;
the model does not hallucinate the refund or draw.

## Migration Order

1. Add `EffectNode`, deterministic tree validation, and the serializer.  The
   serializer must round-trip every node and preserve one sparse delta target
   per independent state impact.
2. Compile the simulator's flat effect callbacks into regions with explicit
   trigger ownership.  Add fixed tests for Pommel/Sundial, Whirlwind, Sword
   Boomerang queries, and Dropkick's conditional branch.
3. Add scope ids and a block-causal attention mask to the decoder.  Compare
   masked and local-window implementations on identical token logits.
4. Bump the data/checkpoint schema and make the random producer emit only tree
   records.  Generate the documentation examples from that same producer and
   serializer, never by hand.
5. Train and evaluate the tree contract separately from schema-v5.  Do not
   resume a schema-v5 checkpoint: the target vocabulary, mask, and labels are
   different.

## v7 executor ownership updates

The current v7 producer keeps the aggregate `delta` only for sequential world
snapshots. It derives `delta_slots`, one sparse world vector for each changed
impact. A local sequence contains one `DELTA_SLOT` token per sparse vector, so a
damage event can supervise block consumption and HP loss independently. Derived
`block_present` and `hp_frac` components stay with their block/HP source slot.

Damage nodes always use the same causal block path, including a target with zero
block: read target block, calculate `MIN` absorption, calculate remaining block,
calculate unblocked damage, read target HP, subtract, then apply `RELU` to HP.

Runtime reads still have a model-predicted semantic request. The environment
assigns the binding name and expands each later bound operand inline with
`CONST_VALUE` and masked digits. There is no response block or register context;
the model is trained on the request, symbolic calculations, and their numeric
results.

Relics are v7 locked features. The world vector exposes fixed binary relic
presence slots, while v7 records contain no relic rule programs. Exact counters
remain in the environment and are exposed through an explicit `RELIC_COUNTER`
request followed by inline constants when the counter is used; stateless relics
never generate a counter region.
