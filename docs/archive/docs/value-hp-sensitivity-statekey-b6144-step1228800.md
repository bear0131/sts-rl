# Value HP sensitivity: fixed-statekey b6144, step 1228800

Checkpoint:
`artifacts/latest-nosl-az-s512-statekey-b6144-20260907/nosl_az_1228800_steps.zip`

SHA-256:
`2a5df960cfd92b9515eba296501026c61f6d306cda067a52f2c0fc76dd85fa1c`

The checkpoint was the newest complete file on rank 0 while the live run was
at approximately 1,388,544 environment steps. Its stored search configuration
is 512 simulations, unlimited depth, AZ batch size 6144, leaf batch size 16,
and no Dirichlet root noise.

## Method

Both probes use the same real seed-36 Cultist combat with Bash+ and Strike.
The enemy-HP probe changes only `monster_scalars[0,1]` (HP / 200) and
`monster_scalars[0,2]` (HP / max HP). The player-HP probe changes the current
HP entries in `player_scalars` and `run_hp`; the combat specialist consumes
`player_scalars`, while `run_hp` belongs to the overworld specialist. Every
other observation field remains byte-identical.

## Enemy HP

| Enemy HP | Value |
| ---: | ---: |
| 1 | 0.2560584843 |
| 5 | 0.2560486197 |
| 10 | 0.2560358047 |
| 20 | 0.2560086846 |
| 30 | 0.2559796274 |
| 40 | 0.2559486628 |
| 50 | 0.2559159100 |

- Slope: `-2.89939e-6` value / enemy HP.
- Low-HP minus full-HP value: `+0.0001425743`.
- The response is monotonic and has the correct sign: a lower enemy HP gives
  a higher predicted final player HP.

## Player HP

| Player HP | Value |
| ---: | ---: |
| 1 | 0.2538689971 |
| 10 | 0.2541025281 |
| 20 | 0.2543620765 |
| 40 | 0.2548809648 |
| 60 | 0.2553991377 |
| 80 | 0.2559159100 |

- Slope: `+2.59151e-5` value / player HP.
- 1-HP minus 80-HP value: `-0.0020469129`.
- The response is monotonic and has the correct sign.
- Because the label scale is final HP / 200, a direct one-for-one retained HP
  effect would have slope `0.005`. The observed local slope is about `0.52%`
  of that scale, so the value head remains weakly sensitive in absolute terms.

## Reference to the earlier 614400 checkpoint

The archived 614400 checkpoint came from a different run, so this is a model
comparison rather than a learning-curve measurement.

| Probe | Step 614400 slope | Current slope | Change |
| --- | ---: | ---: | --- |
| Enemy HP | `+2.31099e-6` | `-2.89939e-6` | sign corrected; 1.25x absolute magnitude |
| Player HP | `+2.05063e-6` | `+2.59151e-5` | correct sign; 12.64x magnitude |

The current checkpoint has learned the correct direction for both variables.
Player-HP sensitivity improved substantially, while enemy-HP sensitivity is
still extremely small and should not yet be treated as a reliable value
ordering signal between tactically close branches.
