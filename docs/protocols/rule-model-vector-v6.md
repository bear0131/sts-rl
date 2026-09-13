# Standalone Rule Model v6

This document describes the anonymous vector/event rule model implemented in
`rl/rule_vector_model.py`.  The current model and data contract is schema v5:
`RULE_VECTOR_SCHEMA_VERSION = 5` and `VECTOR_DATA_SCHEMA_VERSION = 5`.
The older token-rule v5 data and checkpoints are a different contract and are
not accepted by this trainer.

The next planned contract groups events into an ordered effect tree with
scope-local attention.  See [`rule-model-vector-v7-effect-tree.md`](rule-model-vector-v7-effect-tree.md)
for the proposed tree grammar, target-specific damage reuse, and separate
formats for Whirlwind and Sword Boomerang.  The runnable v6 trainer remains
flat until that migration is implemented end to end.

## Model Layout

The model has two Transformer levels plus the autoregressive decoder:

1. `CardRuleEncoder` and `RelicRuleEncoder` each read one rule-token program
   and produce one anonymous vector.  There is no card-id or relic-name
   embedding in either encoder; names are environment-side lookup keys only.
   New configurations share the contextual Transformer backbone between the
   two encoders while retaining separate card/relic token embeddings and output
   projections.  Use `share_rule_transformer=False` (or
   `--no-share-rule-transformer`) when loading or reproducing an older
   independent-encoder checkpoint.
2. `HandCompositionEncoder` reads the set of anonymous card vectors and their
   upgrade flags without positional embeddings, producing one permutation-
   invariant hand vector when an operation needs the hand.
3. The world Transformer reads the player vector, fixed enemy-slot vectors,
   three pile-count features, optional action/hand vectors, and relic vectors.
   A causal Transformer decoder then emits the event stream autoregressively.

All module widths, layer counts, attention heads, feed-forward widths, dropout,
and token limits are configurable through `default_vector_model_config` and
the command-line options in `rl/train_rule_vectors.py`.

## State And Deltas

The combat state has a fixed numeric layout:

```text
player: hp, max_hp, block, energy, turn, floor, gold, hp_frac,
        player buffs, combat counters, relic summary, relic counter slots
enemy:  max_enemies fixed slots, each with present, hp, max_hp, block,
        base_damage, adjusted_damage, hits, gone, and monster buffs
piles:  draw_count, discard_count, exhaust_count
```

Each enemy remains in its slot, including an empty/dead slot.  Card identity,
pile order, hand contents, and full-deck contents are not state-vector targets.
Drawing, discarding, exhausting, or shuffling therefore changes only the three
pile counts.  `vector_delta(before, after)` is an element-wise delta over this
layout; the model predicts that full vector sparsely at a `DELTA_SLOT`.

Relic counters are included in the player vector: the summary fields are relic
count, counter sum, active-counter count, and maximum counter, followed by
fixed counter slots.  A counter update consequently has an ordinary state
delta and an arithmetic calculation trace.

## Hand Composition

When the exact hand matters, the environment resolves each hand instance to a
card-rule vector and passes the vectors plus upgrade values to
`HandCompositionEncoder`.  It is a set Transformer with a learned `HAND_CLS`
token and no positional embeddings, so permuting hand slots does not change the
encoding.  The deterministic diagnostic summary is:

```text
[normalized hand_count, normalized total_upgrade_count,
 normalized upgraded_instance_count, rule_sum[D], rule_mean[D], rule_max[D]]
```

By default the decoder does not regress a latent `HandCompositionEncoder`
output.  Hand edits are represented structurally and the environment can
re-encode the resulting hand.  Optional `predict_hand_delta=True` enables an
experimental `HAND_DELTA_SLOT` head.

Base Armaments emits a chosen-hand query followed by a hand-upgrade event;
upgraded Armaments applies the upgrade to all hand instances.  Burning Pact
emits a chosen-hand query, a hand-exhaust event, and a separate hand-draw
event, preserving trigger boundaries.  If drawn-card identity is required,
the environment answers one `QUERY_RANDOM_CARD` per draw with an anonymous
card-rule vector.

## Event Stream

The event stream used by both training records and the fixed examples is
serialized by `serialize_vector_events`.  Every event always has a calculation
section, which may be empty, and an optional normalized delta:

```text
<OUTPUT_BEGIN>
  <TRANSITION_BEGIN>
    <EVENT_BEGIN>
      EVENT_KIND
      EVENT_DAMAGE
      ENTITY_REF
      ENTITY_ENEMY_SLOT_0
      <CALCULATIONS_BEGIN>
        CALCULATION_BEGIN
          CALCULATION_READ_BASE_DAMAGE
          CALC_FIELD_value
          NUM_LSD_BEGIN
          DIGIT_5
          NUM_END
        CALCULATION_END
        ...
      <CALCULATIONS_END>
      DELTA_SLOT enemy[0].hp=-0.035000
    <EVENT_END>
  <TRANSITION_END>
<OUTPUT_END>
```

There is no `SOURCE_RULE`, `ACTION_CARD`, or other source/name token in the
event stream.  The action card program, hand programs, and relic programs are
separate model inputs.  A `DELTA_SLOT` remains one target token; the readable
renderer expands only its non-zero fields on that line.

Calculation blocks expose the arithmetic that produced a transition.  Damage
traces read base damage and Strength, add them, apply Vulnerable with integer
floor division, read target block, account for absorbed damage, and subtract
the unblocked amount from target HP.  Non-damage state changes use
`CALCULATION_APPLY_DELTA` with raw left/delta/result values.  Numeric values use
the v5 numeric grammar (`NUM_LSD_BEGIN` for integers and `NUM_BEGIN` for
decimals), so calculations are available to the decoder rather than hidden in
an opaque source label.

For a multi-hit card, the resolved attack damage is cached per target.  The
first hit emits the complete card-to-target calculation and, when applicable,
`CALCULATION_COUNT_ATTACK_HITS` (for example, 3 base damage with 3 Strength and
Vulnerable resolves to 9 per hit and `9 * 3` total).  Later hits for that target
emit only `CALCULATION_REUSE_ATTACK_DAMAGE value=9`, followed by their own
block/HP calculation and delta.  This keeps the numeric result explicit while
avoiding repeated card-to-target arithmetic.

### Sequence-length policy

The trainer reports two different lengths because batching introduces padding:
`sequence_length_*` counts non-PAD shifted target positions per sample, while
`model_input_length_max` is the largest padded width actually passed to the
decoder Transformer.  Both values are tracked with one helper for fixed and
streaming train/validation, printed in progress logs, written to checkpoint
metrics, and logged to W&B.  Thus `sequence_length_max` is the longest realized
trace and `model_input_length_max` is the true peak decoder input width.

The longest randomized traces are deterministic multi-hit/all-enemy attacks.
For example, upgraded Whirlwind with five energy against three active enemies
produces five rounds, producing fifteen atomic `EVENT_DAMAGE` blocks.  In the
current generator a 2000-sample scan (seed 0) found 770 decoder input
positions at its longest trace: the first hit for each enemy carries the full
damage arithmetic, while the remaining twelve hits use the shorter
cached-damage calculation.  This remains below the default 2048-token limit.

Damage calculations do not append a second generic `APPLY_DELTA` for HP and
block, because `SUBTRACT_BLOCK` and `SUBTRACT_HP` already contain the same
left/result values.  Generic calculations are still retained for additional
changes such as lethal-hit `present/gone` flags.  Per-hit damage events and
their `DELTA_SLOT`s are intentionally not merged: target block, HP, death, and
future on-hit triggers can differ after every hit.  Only the card-to-target
arithmetic is reused.  The regression suite also samples randomized records
and rejects a stream that exceeds the 2048-token training contract.

For random effects the model emits a query and pauses until the environment
answers:

```text
<QUERY_BEGIN>
  QUERY_RANDOM_TARGET
<QUERY_END>
<OBS_BEGIN>
  ENTITY_ENEMY_SLOT_1
<OBS_END>
```

Sword Boomerang uses one random-target query and one damage event per hit.  The
same interaction mechanism handles random cards, created cards, top-draw
cards, and chosen hand cards.  Havoc is represented as a nested scope/call and
the configured event, query, and nesting limits are validated before training.

## Fixed Examples

The executable source for the two required examples is
`rl/render_rule_vector_examples.py`.  It constructs deterministic canonical
states and calls the same `_effect_record` producer used by randomized
training, then passes those events through `serialize_vector_events`.

```powershell
.\.venv\Scripts\python.exe -m rl.render_rule_vector_examples > docs/rule-model-vector-v6-examples-schema-v5.txt
```

The generated document
[`rule-model-vector-v6-examples-schema-v5.txt`](rule-model-vector-v6-examples-schema-v5.txt)
contains the complete one-token-per-line output:

- Example 1: Fly Kick (`Dropkick`) against an enemy with two Vulnerable and no
  block.  Five base damage becomes seven after Strength/Vulnerable arithmetic.
- Example 2: Pommel Strike with an empty draw pile, three-card discard pile,
  and Sundial counter two.  The card enters discard, the discard is shuffled
  into draw, then Sundial resets its counter, grants energy, and draws cards.

The fixed example states deliberately use `draw=0`, `discard=3`, and
`Sundial counter=2` in the second case so the shuffle and relic trigger are
guaranteed.  The evaluator imports the same state builders and expected event
streams from the renderer.

## Model Size Parameters

`default_vector_model_config` and `train_rule_vectors.py` expose independent
parameters for all modules, including:

```text
card/relic: dim, token-dim, heads, layers, feedforward, max-tokens, dropout
hand:       dim, hidden, heads, layers, feedforward, max-cards, dropout
world:      dim, heads, layers, feedforward, dropout, max-enemies, max-relics
decoder:    dim, heads, layers, feedforward, max-tokens, dropout
deltas:     hidden, layers, dropout
```

Widths are checked when constructing `VectorRuleWorldModel`; for example, the
card/relic outputs must match the world input widths and the decoder must match
the delta-head input width.  Shared rule mode requires matching token width,
head count, layer count, feed-forward width, and dropout for the two rule
encoders (their maximum token limits may differ).  A small model can be selected without changing
module code:

```python
cfg = default_vector_model_config(
    card_dim=64, relic_dim=64, hand_dim=64,
    world_dim=128, decoder_dim=128,
    card_layers=1, relic_layers=1, hand_layers=1,
    world_layers=2, decoder_layers=2,
    card_heads=4, relic_heads=4, hand_heads=4,
    world_heads=4, decoder_heads=4,
)
```

## Training Contract

`rl/train_rule_vectors.py` accepts only independent schema-v5 JSONL or
JSONL.GZ records.  It rejects legacy `state/action/effects/next_state` records,
card/relic names in model inputs, and pile contents.  A record has this shape
(vector lengths are abbreviated):

```json
{
  "schema_version": 5,
  "state": {
    "player": [0.0],
    "enemies": [[0.0]],
    "piles": [0.0, 0.0, 0.0],
    "relic_counters": []
  },
  "action": {"card_rule": [12, 7, 19], "upgrade": 0},
  "hand": [{"card_rule": [4, 8], "upgrade": 1}],
  "relics": [{"rule": [3, 5], "counter": 0}],
  "events": [
    {
      "kind": "EVENT_DAMAGE",
      "entity_ref": "ENTITY_ENEMY_SLOT_0",
      "calculations": [
        {"type": "READ_BASE_DAMAGE", "value": 5},
        {"type": "READ_STRENGTH", "value": 0},
        {"type": "ADD_STRENGTH", "left": 5, "right": 0, "result": 5}
      ],
      "delta": [0.0]
    },
    {
      "kind": "EVENT_ACTION_COMPLETE",
      "entity_ref": "ENTITY_SELF",
      "calculations": []
    }
  ]
}
```

The loader validates every fixed state width and token id, serializes the event
calculations, and stores full deltas sparsely for supervision at the matching
`DELTA_SLOT`.  Structure tokens use cross-entropy; continuous world deltas use
Smooth-L1 only where a delta target exists.

The current producer is `rl/make_vector_rule_training_data.py`.  It samples
one to five hand cards, zero to five cards in each pile, random Ironclad
combat values, buffs, counters, floor-conditioned encounters, and relics.  It
emits fresh records and includes the full calculation list on every event.

For one-pass training, use `--stream-command`; the trainer consumes each JSON
record once and terminates the producer at the step budget.  Use a separate
validation stream for fresh validation records.  Fixed files default to one
epoch and require `--allow-data-reuse` for repeated epochs.

```powershell
.\.venv\Scripts\python.exe -m rl.train_rule_vectors `
  --stream-command ".\.venv\Scripts\python.exe -m rl.make_vector_rule_training_data --stdout --stream --seed 17" `
  --stream-steps 100 --batch-size 1024 --max-cards 5 `
  --out .tmp-vector-stream-v6 --device cuda `
  --validation-stream-command ".\.venv\Scripts\python.exe -m rl.make_vector_rule_training_data --stdout --stream --seed 18" `
  --validation-stream-steps 1 --validation-batch-size 128 `
  --validation-every-steps 1 --wandb --wandb-project sts-rl-vector-v6
```

## Evaluation And Verification

`rl/eval_vector_rule_autoregressive.py` loads a schema-v5 checkpoint, rebuilds
its saved model configuration, and evaluates the same two executable examples.
It reports token agreement and aligns continuous delta rows by `DELTA_SLOT`.

```powershell
.\.venv\Scripts\python.exe -m rl.eval_vector_rule_autoregressive `
  --checkpoint .tmp-current-vector-v6-last.pt --device cpu `
  --max-new-tokens 256 --output docs/vector-v6-autoregressive-eval.json
```

Run the focused regression suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  rl/test_rule_vector_model.py rl/test_vector_rule_data.py `
  rl/test_make_vector_rule_training_data.py `
  rl/test_eval_vector_rule_autoregressive.py
```
