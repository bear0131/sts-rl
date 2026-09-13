# Continuous State To Number Token Experiment

## Question

Can the current Transformer path recover an exact integer from a normalized
continuous state vector and emit the integer with the v7 autoregressive number
grammar?

```text
player.hp = raw_hp / 200
    -> player projection
    -> world Transformer
    -> causal token Transformer
    -> NUM_LSD_BEGIN, DIGIT_n..., NUM_END
```

This isolates the failure seen at
`CALCULATION_SUBTRACT_PLAYER_HP/CALC_FIELD_left`.

## Setup

- Target values: every integer from 0 through 200.
- Input width: the real 153-slot player vector.
- Token format: the same least-significant-first integer format used by v7.
- Model width: 256, 8 heads, 3 world layers, 4 decoder layers, FFN 1024.
- Training: 2,500 steps, batch size 512, or 1.28 million examples per regime.
- Exact metric: greedy autoregressive output must match the complete token
  sequence. Teacher-forced token accuracy is reported separately.
- Full-support regime: all 201 values occur during training.
- Holdout regime: multiples of 7 never occur during training. This leaves 29
  interleaved validation integers while retaining all decimal digits at other
  positions in the training set.
- Each evaluation value is rendered in eight independently generated contexts.

Two input conditions were tested:

- `random`: HP, max HP, and HP fraction are consistent; all other player slots
  are randomized as nuisance state.
- `zero`: all nuisance slots are fixed, max HP is fixed at 200, and only HP and
  HP fraction vary. This is the easy scalar control.

## Results

| Input | Training support | Evaluation support | Teacher-forced token accuracy | Greedy exact accuracy |
| --- | --- | --- | ---: | ---: |
| Random full vector | All 0..200 | All 0..200 | 63.16% | 2.36% |
| Random full vector | Excludes multiples of 7 | Seen values | 68.66% | 4.72% |
| Random full vector | Excludes multiples of 7 | Held-out values | 55.47% | 0.00% |
| Fixed nuisance | All 0..200 | All 0..200 | 79.22% | 10.95% |
| Fixed nuisance | Excludes multiples of 7 | Seen values | 80.05% | 11.05% |
| Fixed nuisance | Excludes multiples of 7 | Held-out values | 78.91% | 6.90% |
| Deterministic `round(hp * 200)` | N/A | All values | 100% | 100% |

The teacher-forced metric is substantially inflated by the fixed grammar and
gold previous digits. Exact autoregressive recovery remains poor even when the
model has repeatedly seen every possible integer and no nuisance state varies.
Typical failures are nearby but incorrect values, for example `0 -> 2`,
`1 -> 3`, and `14 -> 10`.

## Conclusion

The current continuous-vector-to-decimal-token path is not a reliable exact
state-reading mechanism. It learns the number grammar and coarse value
regions, but does not preserve integer identity well enough for calculation
traces. More ordinary rule-training samples should not be expected to make
this exact.

State reads should instead be represented symbolically in the generated
program, for example:

```text
CALCULATION_READ_PLAYER_HP
CALC_FIELD_source
VALUE_player_hp
```

The runtime can resolve that reference exactly and materialize `left=79` in a
human-readable trace. If a learned numeric output is required, use a dedicated
bounded-integer classification head or a scalar regression head followed by
deterministic rounding and token serialization. Decimal digit tokens should
not be the storage format for values already present in the state vector.

Raw reports:

- `docs/continuous-to-number-token-experiment.json`
- `docs/continuous-to-number-token-zero-nuisance-experiment.json`

Reproduction script:

- `rl/experiment_continuous_to_number_tokens.py`
