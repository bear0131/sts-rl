# NOSL Gumbel Sequential Halving Experiment

Date: 2026-09-09

This experiment compares the native NOSL search with the latest local
checkpoint:

`artifacts/latest-nosl-lr1e4-s7680000/nosl_az_7680000_steps.zip`

The fixture is an Ironclad Cultist combat with a normal, non-unique draw pile.
The relevant initial hand contains `Bash`, `Clash`, and three `Strike`s.  The
model's direct policy strongly prefers `Clash`, while the intended order for
the test is to apply `Bash` first and then use `Clash` after the hand no longer
contains a Skill.

## Search configuration

- seed: `57`
- root simulations: `128` at depths `1, 2, 3, 4`
- root simulations: `512` at depth `0` (`depth=0` means unlimited)
- leaf batch size: `64`
- prior: model logits
- `c_puct=1.5`
- root Dirichlet noise: disabled
- every NOSL simulation independently redeterminizes the hidden state

The Gumbel implementation currently uses one Gumbel sample per legal root
edge, ranks roots by `log(prior) + Gumbel`, gives active roots equal simulation
quotas, and removes approximately half of the roots after each completed
round using raw empirical Q. Descendants continue to use PUCT. This is a
Gumbel top-k plus simplified sequential-halving allocator, not the complete
paper Gumbel MuZero completed-score/Q-normalization formulation.

## Main comparison

At `depth=0, simulations=512` for the controlled seed-57 hand:

| search | direct-policy argmax | Bash visits | Clash visits | Bash-child Clash visits | selected action |
|---|---:|---:|---:|---:|---|
| PUCT | Clash (about 100%) | 0 | 512 | 0 | Clash |
| Gumbel Sequential Halving | Clash (about 100%) | 172 | 171 | 161 | Bash |

The corresponding Gumbel root estimates were:

- Bash: `Q=0.38486`, prior `4.9e-6`, visits `172`
- Clash: `Q=0.38402`, prior `0.999995`, visits `171`

Thus Gumbel did not merely reproduce the prior. It forced the low-prior Bash
edge to receive a full initial allocation and expanded the important
continuation (`Bash -> Clash`) 161 times. The final Bash choice follows the
slightly higher empirical Q after the equalized root exploration.

At finite depths `1--4` with 128 simulations, Gumbel allocated Bash 7 visits
and Clash 43 visits (all six legal roots remained active in the default
configuration), but selected Clash. The cutoff prevents the search from
observing enough of the Bash -> Clash continuation and later draw/turn
effects. Sequential Halving can allocate budget to a candidate, but cannot
recover value information that is cut off before the relevant consequences.

## Repeated-world check

The primary stochastic check keeps the exact public hand/draw pile fixed and
changes only the NOSL nonce (20 repetitions below). This is preferable to
mixing seeds that can change which cards reach the opening hand; the older
per-seed JSON artifacts are retained for inspection but are not used as the
main aggregate claim.

## Budget threshold

The same public state was repeated with 20 independent NOSL nonces. The
opening hand and draw pile were unchanged; only hidden future RNG and the
Gumbel/search stream changed.

| simulations (depth 0) | Bash selections | mean Bash visits | mean Clash visits | mean Q(Bash)-Q(Clash) |
|---:|---:|---:|---:|---:|
| 32  | 0/20  | 1   | 12  | -0.01720 |
| 64  | 0/20  | 3   | 23  | -0.01577 |
| 128 | 0/20  | 7   | 43  | -0.01669 |
| 256 | 20/20 | 86  | 86  | +0.00131 |
| 512 | 20/20 | 172 | 171 | +0.00085 |

PUCT selected Bash `0/20` at every tested budget. Thus the observed switch at
256 simulations is a budget threshold in this checkpoint/fixture, not a single
fortunate Gumbel draw.

## Candidate ablations

- With model priors and `gumbel_candidates=2`, the top two sampled roots were
  Bash and Clash, receiving `256/256` visits at depth 0; Bash was selected.
  This is expected: Clash has prior about `0.999995`, Bash about `4.9e-6`, and
  each Strike is about `5.6e-9`, so Bash is roughly 865 times more likely than
  a Strike to be the second Gumbel-top-k item.
- At `gumbel_candidates=3`, the roots Bash/Clash/one Strike received
  `214/213/85` visits and Bash was selected. At `4`, the allocation was
  `192/192/64/64`; at `6` (or `0`, meaning all roots), it was the full-root
  allocation shown above. Candidate truncation is therefore a useful compute
  control, but should only be used when the candidate budget covers plausible
  actions.
- Uniform root priors retained the equal-allocation behavior and selected Bash
  in the controlled unlimited-depth fixture, but this removes useful model
  information.
- The strict no-prior ablation (`c_puct=0`) also explored Bash deeply, showing
  that the main effect is the root allocation policy rather than a stronger
  PUCT prior term.

## Artifacts

- [PUCT full comparison](../.tmp-order-puct-bash-clash-step7680000-full.json)
- [Gumbel full comparison](../.tmp-order-gumbel-bash-clash-step7680000.json)
- [Gumbel model-prior candidate-3 ablation](../.tmp-order-gumbel-model-c3-step7680000.json)
- [Gumbel uniform-prior ablation](../.tmp-order-gumbel-uniform-step7680000.json)
- [Gumbel no-prior ablation](../.tmp-order-gumbel-none-step7680000.json)
- [Clean-build full Gumbel comparison](../.tmp-order-gumbel-current-bash-clash-step7680000.json)
- [Clean-build candidate matrix, `c=0`](../.tmp-order-gumbel-current-c0-step7680000.json)
- [Clean-build candidate matrix, `c=2`](../.tmp-order-gumbel-current-c2-step7680000.json)
- [Clean-build candidate matrix, `c=3`](../.tmp-order-gumbel-current-c3-step7680000.json)
- [Clean-build candidate matrix, `c=4`](../.tmp-order-gumbel-current-c4-step7680000.json)
- [Clean-build candidate matrix, `c=6`](../.tmp-order-gumbel-current-c6-step7680000.json)
- [Clean-build uniform-prior ablation](../.tmp-order-gumbel-current-uniform-step7680000.json)
- [Clean-build no-prior ablation](../.tmp-order-gumbel-current-none-step7680000.json)
- [Paired budget repetitions, 20 NOSL nonces](../.tmp-gumbel-current-budget-repeats-step7680000.json)
- [Paired budget repetitions, `gumbel_candidates=2`](../.tmp-gumbel-current-budget-repeats-c2-step7680000.json)
- [Reproducible paired-matrix harness](../rl/eval_gumbel_bash_clash_matrix.py)

The clean native module was rebuilt from the current source at
`sim/sts_lightspeed/build-gumbel-current`; its SHA-256 is
`E7E7684B2A59439DC7843BB9A37B7E397A50E2582A73C0BDC9B3F0AE7399A51B`.

The Bash-first advantage in this fixture is not a claim that the value head is
always correct: Bash applies Vulnerable before the subsequent Clash, so the
immediate two-card damage is strictly higher than Clash-then-Bash. Paired NOSL
rollouts are still required for a full-game conclusion; the search result here
shows when the current value estimates have enough budget to detect that
advantage.

The relevant unit tests pass: `18 passed` across `test_nosl_tree_labels`,
`test_nosl_is_az`, and `test_nosl_sampler`.

## Strict Bash/Clash-only root rerun

To isolate the ordering question, a second fixture accepts a root only when
the energy is exactly 2 and the legal `PLAY_CARD` set is exactly `{Bash,
Clash}`.  Three/four-cost attacks occupy the other opening-hand slots and are
therefore not playable; the remaining deck contains 17 mixed ordinary cards,
so the hidden draw future is still non-empty.  `End Turn` remains a simulator
action but is excluded from the card-order count.

The state construction uses fixed seeds (`219, 357, 454, 575, 605`) and both
hand-slot layouts, rather than scanning seeds during the measurement.  Each
configuration has five independent NOSL future nonces (50 trials per search
budget).  The search first selects a root card, executes its complete native
card+target action, and then searches the resulting state for the second card.

| method | simulations | Bash -> Clash | Clash -> Bash | second End Turn | root other |
|---|---:|---:|---:|---:|---:|
| PUCT | 64 | 0/50 | 50/50 | 0 | 0 |
| PUCT | 128 | 0/50 | 50/50 | 0 | 0 |
| PUCT | 256 | 0/50 | 50/50 | 0 | 0 |
| PUCT | 512 | 0/50 | 50/50 | 0 | 0 |
| Gumbel, candidates=2 | 64 | 50/50 | 0/50 | 0 | 0 |
| Gumbel, candidates=2 | 128 | 50/50 | 0/50 | 0 | 0 |
| Gumbel, candidates=2 | 256 | 50/50 | 0/50 | 0 | 0 |
| Gumbel, candidates=2 | 512 | 49/50 | 1/50 | 0 | 0 |

The reproducible output is
`../.tmp-strict-bash-clash-sequence-normal-draw17-depth0-step7680000-5x.json`,
and the evaluator is `../rl/eval_strict_bash_clash_sequence.py`.  These counts
are an ordering/coverage diagnostic for this checkpoint and fixture, not a
full-combat win-rate estimate.
