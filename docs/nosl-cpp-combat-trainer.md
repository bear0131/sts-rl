# Native NOSL Combat Trainer

The simulator now provides two native paths. `nosl_native_search` remains a
CPU-only model-free benchmark. `nosl_neural_search_begin` starts the complete
AlphaZero-style path used by whole-run training: C++ owns the tree, NOSL
redeterminization, simulator transitions, virtual visits, and backup; the
learner supplies neural policy/value predictions for public leaves.

## Search semantics

For every real combat decision the trainer creates a fresh tree. Each
simulation copies the live `BattleContext`, calls
`search::noslRedeterminize(copy, nonce)`, and then performs at most `depth`
actions with PUCT. The redeterminization replaces the hidden RNG streams and
canonicalizes/shuffles the remaining draw-pile multiset; it never changes the
live world-line. The tree therefore does not read the current world's future
RNG or hidden draw-pile order.

The legacy trainer's initial prior is uniform. The neural session instead
installs the current policy network's masked root prior and requests a policy
and value at every newly expanded non-terminal leaf. At the depth cutoff it
requests only the value. Terminal leaves use objective remaining HP.
The only objective is this HP ratio for both victory and loss states.

## Training prototype

After a combat ends, every root stores its visit distribution and is updated
with the same final HP target. The table key includes visible combat fields
(player, monsters, hand, potions, turn, and card-select task), but excludes RNG
state and draw-pile order. This is an online tabular baseline, not a claim of
neural-network quality; it is intended to validate data collection and native
execution before plugging in a policy/value network.

With `--threads N`, episodes are distributed across N independent trainers.
This is data-parallel benchmarking: each worker has its own table. Policy dumps
are written as `PATH.worker-N` when more than one worker is used.

## Build and run

```text
cmake -S sim/sts_lightspeed -B sim/sts_lightspeed/build -G Ninja
cmake --build sim/sts_lightspeed/build --target nosl-combat-train -j 8
sim/sts_lightspeed/build/nosl-combat-train.exe \
  --episodes 100 --simulations 128 --depth 3 --threads 1 --train 1 \
  --encounter JAW_WORM --dump-policy nosl-policy.tsv
```

The final line reports `decisions_per_s`, `simulations_per_s`, and
`transitions_per_s`. These have different meanings: a combat decision runs a
whole root search, a simulation is one tree traversal, and a transition is one
native `Action::execute` call.

For a pure efficiency run use `--train 0`; this avoids table updates while
keeping the same search and simulator work.

## Whole-run integration

The native search is available through the Python binding as
`BattleContext.nosl_native_search(...)`. The neural session is exposed as
`BattleContext.nosl_neural_search_begin(...)` and its
`NoslNeuralSearchSession.next_batch/backprop/result` methods. A batch contains
independent leaf clones; virtual visits prevent the leaves in one batch from
collapsing onto the same path.

`SlayTheSpireEnv` adapts its complete
bit-packed action to the whole-run thin action space and retains the native
action until the matching root step is submitted. Targeted cards therefore
execute atomically in the C++ simulator; the old Python `PLAY_CARD` then
`PICK_TARGET` search path is not used in this mode. Root visits are aggregated
by hand/potion/select slot and used as the policy target, while the real combat
terminal HP remains the value target.

Enable this path with `--nosl-is-az-cpp`. `--nosl-is-az-leaf-batch-size` controls
how many C++ leaves each worker returns per learner round (default 16); leaves
from all workers are concatenated into one GPU inference batch. Because its action timing differs from
the previous autoregressive Python search, a compatible outer checkpoint should
be passed through `--init-from` and the combat specialist should be reset:

```text
python -m rl.trainer --task wholerun \
  --init-from <outer-checkpoint> \
  --nosl-is-az --nosl-is-az-inner-only --nosl-is-az-cpp \
  --nosl-is-az-reset-combat
```

The first remote fresh-inner run used `simulations=128`, `depth=3`,
`n_envs=24`, `envs_per_worker=4`, and reached 150,528 steps. Its final
checkpoint is:

```text
/root/autodl-tmp/sts-rl/runs/nosl-cpp-inner-s128d3-fresh/20260905-033234/final.zip
```
