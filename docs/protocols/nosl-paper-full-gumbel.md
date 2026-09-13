# NOSL strict paper Full Gumbel reproduction

The repository's `full_gumbel` mode contains project-specific extensions for
sparse NOSL trees. Use `paper_full_gumbel=true` (CLI:
`--nosl-is-az-paper-full-gumbel`) with `search_mode=full_gumbel` to run the
canonical Full Gumbel MuZero/AlphaZero path.

## Canonical parameters

The native session forces these values in strict mode:

```text
gumbel_scale=1
gumbel_value_scale=0.1
gumbel_max_visit_init=50
gumbel_candidates=16
prior_temperature=1
gumbel_rescale_q=true
gumbel_use_mixed_value=true
interior_uniformity=0
interior_min_visits=0
interior_coverage_fraction=0
interior_value_mode=0
dirichlet_alpha=0
dirichlet_epsilon=0
```

The root first keeps the top `K=16` legal actions by `G + prior_logits`
(or all legal actions when fewer than 16 exist), matching the default
`max_num_considered_actions=16` in `mctx.gumbel_muzero_policy`. It then uses
the official `mctx` considered-visit table. For a scheduled
visit count it selects the valid edge maximizing

```text
G(a) + prior_logits(a) + completed_Q(a)
```

where the other visit counts receive a `-infinity` schedule mask. At an
interior decision node it selects

```text
argmax_a softmax(prior_logits + completed_Q)[a]
              - N(a) / (1 + sum_b N(b))
```

At the end of search the action is selected from the most-visited root edges;
ties use the same Gumbel-plus-logit-plus-completed-Q score for deterministic
reproducibility. The training target is
`softmax(prior_logits + completed_Q)` and does not include the sampled Gumbel.

Completed Q follows mctx Appendix D:

```text
mixed = (raw_value + sum_a N(a) * prior(a) * Q(a)
                   / sum_{a:N(a)>0} prior(a)) / (1 + sum_a N(a))
completed_Q(a) = Q(a)       if N(a) > 0
                 mixed      otherwise
completed_Q <- (completed_Q - min) / max(max - min, epsilon)
completed_Q <- (50 + max_a N(a)) * completed_Q
```

## Differences from the current experiment mode

The running experiment uses `full_gumbel` plus non-paper options: a reduced
`gumbel_value_scale`, optional root Dirichlet noise, prior-independent
interior coverage/minimum visits, progressive mean-to-max decision values,
and structural `valueVisits` exclusions for coverage probes. Those options are
disabled by strict mode. The existing considered-visit root schedule and
deterministic interior selector are retained because they already match the
official `mctx` implementation.

NOSL-specific semantics remain unchanged: every simulation independently
redeterminizes hidden RNG, public-state keys separate different future
observations, random-world children are chance-averaged, and
`end_turn_as_leaf` may stop at the post-enemy-turn value boundary. These are
environment/information-set requirements rather than Full Gumbel choices.

## Reproduction command

For a fresh training run:

```text
uv run python -m rl.trainer --task wholerun --nosl-is-az --nosl-is-az-cpp \
  --nosl-is-az-search-mode full_gumbel \
  --nosl-is-az-paper-full-gumbel \
  --nosl-is-az-simulations 512 --nosl-is-az-depth 0 \
  --nosl-is-az-leaf-batch-size 1
```

`leaf_batch_size=1` reproduces the sequential reference order exactly. A
larger leaf batch is a parallel virtual-visit approximation: later selections
in that batch do not yet know the earlier leaves' neural values. Keep the
worker count, frozen outer policy, checkpoint, and learning settings identical
to the baseline when comparing results. For a controlled Bash/Clash evaluation, pass `paper_full_gumbel=True` through
`_run_search` or use the `--paper-full-gumbel` option of
`rl.eval_full_gumbel_bash_clash`.
