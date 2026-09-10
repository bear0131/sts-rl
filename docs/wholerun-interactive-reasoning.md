# Whole-run interactive reasoning

This is a new whole-run policy architecture. By default the actor, critic,
card-program encoder, and reasoning decoder are initialized and trained
together from scratch. The non-reasoning ``split-pointer`` wholerun can
optionally load only the standalone v7 ``card_encoder`` with
``--card-vector-checkpoint`` and freeze it; the actor, critic, and projection
layers remain trainable.

## Data flow

For each environment transition:

1. The policy encodes the state and every available card from its complete
   card-rule DSL.
2. The actor selects a normal thin action.
3. The environment executes that action and records the resulting transition.
4. The reasoning decoder autoregressively replays the executed transition.
5. When the transition contains a stochastic result, the decoder emits a
   query token and the environment inserts the observed result.
6. PPO updates the actor and critic. A separate auxiliary update uses the same
   optimizer to train the reasoning transcript.

The action is therefore committed before the outcome is explained. Reasoning
cannot choose a favorable random result after seeing it.

## Anonymous cards

Model-visible card instances contain:

```text
CARD_INSTANCE_BEGIN CARD_REF <local-number>
RULE_BEGIN <complete card DSL> RUNTIME_COST <number> RULE_END
CARD_INSTANCE_END
```

`CARD_REF` is local to one transition and carries no card identity. Canonical
card ids are used only on the environment side to look up the DSL. The policy's
card-id embedding is frozen and never read. Empty card slots produce an exact
zero vector. The old hand damage/block summaries and per-card pile-count
projection are disabled to prevent identity and precomputed-value shortcuts.

Runtime cost remains visible because cost can change during a run and is part
of the current card surface.

## Numbers

Integers from `-32` through `255` use one token. Non-negative tokens are named
`INT_0` through `INT_255`; negative tokens are named `INT_NEG_1` through
`INT_NEG_32`. Values outside that range retain a digit fallback:

```text
NUM_BEGIN [SIGN_NEG] DIGIT_0...DIGIT_9+ NUM_END
```

The fallback has no positive-sign token; an omitted sign means non-negative.
Fixed numeric literals inside model-visible anonymous card programs use the
same compact representation. Symbolic X-cost remains `VALUE_X`.

## Interactive stochastic results

The model emits one of these queries only when it reaches the corresponding
random operation:

```text
QUERY_RANDOM_TARGET -> OBS_RANDOM_TARGET
QUERY_DRAW_CARD     -> OBS_DRAW_CARD
QUERY_RANDOM_CARD   -> OBS_RANDOM_CARD
QUERY_CREATED_CARD  -> OBS_CREATED_CARD
```

The environment response includes an anonymous target slot or full anonymous
card program. Environment-owned response tokens are present in the decoder
context, but their teacher labels are `-100`, so they receive no language-model
loss. Later model tokens can attend to them.

## Local calculations

A changed numeric value is serialized locally as:

```text
CALC_BEGIN
CALC_INPUT_BEFORE INT_3
CALC_INPUT_DELTA INT_NEG_1
CALC_OP_ADD
CALC_RESULT INT_2
CALC_END
VALUE_AFTER INT_2
```

The calculation immediately precedes the externally visible value. This keeps
the operands and result next to the event that consumes them instead of placing
all calculations in a separate prefix.

## Trace precision

`info["reasoning_trace"]["precision"]` reports the available supervision:

- `atomic`: a trace-enabled simulator captured snapshots after individual
  combat operations.
- `state_diff`: retained only for inspecting legacy traces produced before
  atomic tracing was required. New interactive-reasoning environments reject
  a simulator without `execute_with_trace()` instead of silently generating a
  net-difference target.
- `semantic_only`: a non-combat transition has action-level supervision but no
  fabricated atomic effects.

Non-combat actions do not yet have per-effect simulator hooks. Their current
reasoning target identifies the executed action and transition completion.

## Training

To use a pretrained anonymous v7 card-vector frontend in non-reasoning
wholerun, select ``split-pointer`` and add its standalone checkpoint:

```bash
python -m rl.trainer \
  --task wholerun \
  --arch split-pointer \
  --card-vector-checkpoint rl/runs/rules-vector-v7/last.pt
```

The checkpoint's card vector width becomes the wholerun ``card_dim`` (the v7
default is 128). With the default ``d_model=256``, each card is represented as
``128 + upgrade`` and passed through the trainable ``129 -> 256`` card
projection. The same frozen encoder is shared by combat and overworld card
tokens.

Example single-machine command:

```bash
python -m rl.trainer \
  --task wholerun \
  --arch semantic-reasoning \
  --steps 10000000 \
  --n-envs 8 \
  --n-steps 512 \
  --batch-size 256 \
  --reasoning-coef 0.2 \
  --reasoning-batch-size 16 \
  --reasoning-updates 1
```

The architecture automatically enables interactive trace capture. Relevant
metrics are logged under `reasoning/`:

```text
loss
token_accuracy
supervised_tokens
samples
eligible_samples
dropped_too_long
atomic_trace_rate
environment_tokens
optimizer_steps
```

`--max-reasoning-tokens` is retained as the initial learned-position table size
for checkpoint compatibility. It is no longer a decoder context limit and long
samples are not dropped; `dropped_too_long` is kept as a compatibility metric
and is always zero in the new path. W&B also records
`reasoning/sequence_length_mean` and `reasoning/sequence_length_max`.

## Cached generation

Teacher forcing uses one parallel causal decoder pass. Interactive generation
uses `ReasoningKVCache`: each model token and each environment-injected token is
processed once with `CachedReasoningDecoder.step()`. The cache grows by one
position for every consumed token, so generation does not recompute the full
prefix after an oracle response.

Run the protocol and cache regression tests with:

```bash
python -m pytest rl/test_wholerun_reasoning.py -q
```
