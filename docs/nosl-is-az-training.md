# NOSL-IS-AZ WholeRun Training

`--nosl-is-az` enables the first information-set AlphaZero combat mode in
`rl.trainer`. The run remains a WholeRun task: overworld transitions use the
existing hierarchical PPO objective, while combat actions are selected by a
worker-side NOSL PUCT search and combat PPO samples are excluded from the PPO
loss.

Each search simulation clones and redeterminizes the live combat state once at
the root. The branch is then advanced in place, so later observations do not
reuse the current world line's hidden RNG or draw-pile order. The learner sends
public policy logits to the worker. Non-terminal leaves are evaluated by the
current value network and terminal leaves use objective remaining HP. The
worker returns a legal action and a root visit distribution; the combat value
target is filled in only when the real fight ends, using that fight's final HP
for every root collected in the fight. There is no fixed-policy rollout stage.

Example smoke run:

```text
python -m rl.trainer --task wholerun --nosl-is-az --vec dummy \
  --n-envs 1 --n-steps 64 --batch-size 64 --n-epochs 1 \
  --nosl-is-az-simulations 4 --nosl-is-az-depth 2 \
  --nosl-is-az-rollout-steps 0 --steps 64 --device cpu
```

For remote CPU search with a GPU learner, use `--vec subproc` and increase
`--n-envs`. To let one subprocess own several independent environments/search
trees, set `--envs-per-worker` (for example `--envs-per-worker 4` with
`--n-envs 24` creates 6 subprocesses). Each tree remains independent;
only step and leaf-method IPC is grouped. Start with 4-8 simulations and depth
2-3, then scale after measuring `nosl_is_az/collection_seconds` and
`nosl_is_az/combat_roots`.

`rollout-steps` is retained only for checkpoint/CLI compatibility and is
ignored by the search. The distributed ring trainer reduces the combat
specialist gradients across all ranks; the outer policy remains frozen in the
inner-only experiment.

`card_language` must remain disabled. Ordered hidden draw-pile fields would
violate the information-set assumption even though the simulator branch itself
is redeterminized correctly.

For an isolated combat-from-scratch experiment with a fixed trained overworld
controller, add `--nosl-is-az-inner-only --nosl-is-az-reset-combat` together
with `--init-from <whole-run-checkpoint>`. The full checkpoint is loaded first,
the combat specialist is restored to its fresh random initialization, and then
the shared tokenizer plus overworld specialist are frozen. Overworld PPO is
skipped entirely, so only combat policy/value parameters receive optimizer
updates from NOSL search targets.

The native C++ search path is selected with `--nosl-is-az-cpp`. It is a complete
neural AlphaZero implementation: the C++ tree uses the current root policy
prior, requests neural policy/value predictions at expanded leaves, applies
virtual visits for batched simulations, and backs up the returned values. The
learner concatenates leaves from all workers into one GPU forward. Complete
bit-packed simulator actions are searched and targeted card actions are
submitted atomically, so its action timing is different from the staged Python
`PLAY_CARD`/`PICK_TARGET` path. `--nosl-is-az-leaf-batch-size` controls the
number of leaves per worker round (default 16). Treat it as a new inner-policy
experiment: initialize only the frozen outer policy from an existing
checkpoint and reset the combat specialist instead of resuming the old inner
weights.

## Four-machine training run (phase 1 completed)

The current experiment used four machines (one rank per machine), kept the
trained outer policy frozen, and trained the combat specialist from a fresh
initialization. It used the C++ AlphaZero search and reduced the inner-model
gradients through the SSH ring:

```text
python -m rl.trainer --task wholerun \
  --init-from /root/autodl-tmp/sts-rl/runs/nosl-cpp-inner-s128d3-fresh/20260905-033234/final.zip \
  --nosl-is-az --nosl-is-az-inner-only --nosl-is-az-cpp \
  --nosl-is-az-reset-combat --nosl-is-az-simulations 128 \
  --nosl-is-az-depth 3 --nosl-is-az-leaf-batch-size 16 \
  --nosl-is-az-batch-size 1536 --nosl-is-az-updates 1 \
  --n-envs 24 --envs-per-worker 4 --n-steps 128 \
  --batch-size 1536 --n-epochs 1 --steps 300000 \
  --device cuda --amp-mode bf16 --save-every 50000 \
  --wandb --wandb-project sts-rl \
  --wandb-name nosl-az-4way-s128d3-b16
```

Run id: `naz4w`  
W&B run: <https://wandb.ai/1283802076-tsinghua-university/sts-rl/runs/iz3vyusd>

The run completed 13 updates (`309720` global environment steps) and then
exited normally at the requested `300000`-step budget. Per update, the global
rollout contained `12288` transitions; collection took about `23` seconds and
the aggregate throughput was about `500` environment steps/s. The last logged
evaluation had mean floor `27.1`, Act 1 win rate `69.5%`, and Act 2 win rate
`14.4%` (Act 3 was not evaluated).

The final rank-0 checkpoint is stored remotely at:

```text
/root/autodl-tmp/sts-rl/checkpoints/nosl-az-4way-s128d3-b16-20260905/
  nosl_az_309720_steps.zip
```

It is archived locally at
`artifacts/paused-nosl-az-4way-s128d3-b16-20260905/nosl_az_309720_steps.zip`.
The SHA-256 is
`728228668b8ce9b8a781dfde86a2895bd6114fa37b0650b9ad18b0e246c1f716`.

The earlier single-machine run (`728j7mw9`, `s128d3-b16`) is retained only as
an archived baseline. It is paused and is not part of the current four-machine
training.

## Current four-machine continuation

The same W&B run is now continuing from the phase-1 checkpoint on all four
machines. This continuation keeps the C++ NOSL-IS-AZ and inner-only settings
unchanged and raises the absolute global target to `10000000` steps:

```text
checkpoint:
  /root/autodl-tmp/sts-rl/migration/nosl-az/maskable_ppo_309720_steps.zip
transport run-id: naz4w-cont10m
log directory:
  /root/autodl-tmp/sts-rl/logs/nosl-az-4way-s128d3-b16-cont10m-20260905/
checkpoint directory:
  /root/autodl-tmp/sts-rl/checkpoints/nosl-az-4way-s128d3-b16-cont10m-20260905/
absolute target: 10000000 global environment steps
W&B id: iz3vyusd
```

The four ranks passed startup and ring synchronization. The first observed
continuation updates reached global steps `322008` and `334296`; collection
remains about `21-23` seconds per update. Checkpoints are saved every 25
updates, with a final checkpoint written when the absolute target is reached.

Useful live metrics are `speed/env_steps_per_second_lifetime`,
`nosl_is_az/cpp_leaf_evaluations`, `nosl_is_az/collection_seconds`,
`nosl_is_az/policy_loss`, `nosl_is_az/value_loss`, `game/floor_mean`, and
the Act 1/Act 2 win rates. The run's checkpoint directory is
`/root/autodl-tmp/sts-rl/rl/runs/whole-run/<timestamp>/`.

## Restart after ring failure

The first continuation stopped at approximately `727512` steps because rank 3
hit a NumPy scalar-shape error while bootstrapping a TimeLimit terminal reward;
the other ranks then reported the expected ring peer-closed error. The terminal
value is now converted explicitly to a Python float before it is added to the
reward buffer. The most recent complete checkpoint was `nosl_az_616920_steps`
and was copied to all four machines with SHA-256
`e1758bac62ccafa4880e3228d58554d0db812e0a413d148ce24be94a1e5e95bf`.

The restarted depth-limited run uses transport id `naz4w-cont10m-r3`, logs at
`/root/autodl-tmp/sts-rl/logs/nosl-az-4way-s128d3-b16-cont10m-resume2-20260905/`,
and continues the same W&B run `iz3vyusd` from `616920` to the absolute
`10000000`-step target. It passed the first two updates at `629208` and
`641496`, and later reached `899544` with all four ranks alive. A detached
queue on rank 0 will start the independent unlimited-depth run only after the
The depth-limited run was then stopped at the user's request before reaching
the 10M target; its checkpoints and W&B history remain intact.

## Unlimited-depth run from a fresh combat specialist

The second experiment starts from global step `0`. It preserves the trained
outer/overworld policy from the `309720` checkpoint, reinitializes the combat
specialist with a fixed seed, clears the optimizer state, and sets
`nosl_is_az_depth=0`. The initialization archive is
`/root/autodl-tmp/sts-rl/migration/nosl-az/nosl_az_random_start_step0.zip`,
with SHA-256 `6ee9ae8b72460f5ce85e08f548b9c9588e97ce8746efc6e8d7519126fc8b8c5d`.

The four ranks use transport id `naz4w-unlimited-random0`, ring port `29764`,
and logs at
`/root/autodl-tmp/sts-rl/logs/nosl-az-4way-s128-unlimited-b16-20260905/`.
The W&B run is `qnbi62qu`:
<https://wandb.ai/1283802076-tsinghua-university/sts-rl/runs/qnbi62qu>.
All four ranks passed startup and completed the first four updates (currently
`61440` global steps). Unlimited node expansion currently takes about
`59-62` seconds per global update, or roughly `50` local environment steps/s
per rank; the first update evaluated about `266861` C++ leaves.

## Unlimited-depth run with 512 simulations

The 128-simulation unlimited-depth run above was stopped with its W&B history,
logs, and checkpoints retained. A replacement run starts again from global
step `0` and changes only the search budget from 128 to 512 simulations. It
uses the same trained and frozen outer policy, the same fixed-seed random
combat-specialist initialization, unlimited node expansion (`depth=0`), and a
leaf batch size of 16. The step-0 checkpoint is:

```text
/root/autodl-tmp/sts-rl/migration/nosl-az/nosl_az_random_start_s512_step0.zip
SHA-256: 2aeea09ca5abc5715d9f3d70805eb72f21c500449becb2e06a438ca16e1735cd
```

The four ranks use transport id `s512u0`, ring port `29764`, and logs at
`/root/autodl-tmp/sts-rl/logs/nosl-az-4way-s512-unlimited-b16-random0-20260905/`.
Checkpoints are written under
`/root/autodl-tmp/sts-rl/checkpoints/nosl-az-4way-s512-unlimited-b16-random0-20260905/`.
The W&B run is `m72jb2b9`:
<https://wandb.ai/1283802076-tsinghua-university/sts-rl/runs/m72jb2b9>.

The first two synchronized updates reached `24576` global environment steps.
Mean search collection time across ranks was `217.63` seconds for the first
update and `208.47` seconds for the second. First-update rank rollout times
ranged from `196.13` to `227.55` seconds, corresponding to about `13.5-15.7`
local environment steps/s per rank. The aggregate effective throughput is
about `54-55` global environment steps/s when measured against the slowest
rank. The
first update averaged `945562.75` C++ leaf evaluations per rank (about `3.78M`
across four ranks), `4032` leaf batches per rank, and `2250.75` combat roots
per rank. All four ranks completed both updates, synchronized gradients
through the SSH ring, and continued training toward `10000000` global steps;
the latest checked logs contained no training, ring, or OOM errors.

## Local dashboard checkpoint and search controls

The latest available s512 checkpoint was pulled from rank 0 for local browser
inspection:

```text
artifacts/latest-nosl-az-s512-20260905/nosl_az_307200_steps.zip
SHA-256: 87E3F2802A5598BED37A2FEA5FC560F090F65633A132645484321781556A7840
```

The dashboard loads this SB3 checkpoint with its custom policy metadata and
the rebuilt local `cp311` simulator binding. Start it with:

```text
python -m rl.dashboard --checkpoint \
  artifacts/latest-nosl-az-s512-20260905/nosl_az_307200_steps.zip
```

The watch toolbar now exposes `MCTS simulations`. Positive values run the
NOSL model-free MCTS selector at each combat decision (depth 3, random
terminal rollouts up to 128 simulator decisions); `0` uses the checkpoint's
direct policy. `Precompute route` is disabled by default. Leaving it disabled
shows the first state immediately and builds the map timeline only as the user
steps, avoiding the long full-route preprocessing pass.

## Restarted 512-simulation run with root noise disabled

The Dirichlet-noise run was stopped after approximately `221184` global steps.
The comparison run starts again from global step `0` using the same frozen
outer policy and fresh combat-specialist initialization, with unlimited native
AlphaZero expansion (`depth=0`) and `512` simulations. Root noise is disabled
explicitly (`alpha=0`, `epsilon=0`). The initialization checkpoint is

```text
/root/autodl-tmp/sts-rl/migration/nosl-az-dirichlet-init/nosl_az_dirichlet_step0.zip
SHA-256: 777a74f235fdbd372f0898b5eec92e390c4471c8c3b2bdaea82604a2c6ca080f
```

The four-rank run id is `s512u0n1`, using ring port `29766`. Logs are under
`/root/autodl-tmp/sts-rl/logs/nosl-az-4way-s512-unlimited-noiseoff-20260906/`
and checkpoints under
`/root/autodl-tmp/sts-rl/checkpoints/nosl-az-4way-s512-unlimited-noiseoff-20260906/`.
The W&B run is `bh2d8j6y`:
<https://wandb.ai/1283802076-tsinghua-university/sts-rl/runs/bh2d8j6y>.

The first synchronized update reached `12288` global environment steps. It
completed on all four ranks with about `240.6` seconds of collection time,
`1.06M` C++ leaf evaluations per rank, and no ring, simulator, or OOM errors.

## Startup audit and continued tests (2026-09-06)

The running noise-off job was checked without changing or stopping it. All
four ranks still run `s512u0n1` with `512` simulations, `depth=0`, root noise
`alpha=0, epsilon=0`, `n_envs=24`, `n_steps=128`, `batch_size=1536`, BF16,
and the SSH gradient ring. At the latest check it reached global step
`921600` (update `74`); rank 0 reported floor mean `25.47`, Act 1 win rate
`61.0%`, Act 2 win rate `15.75%`, collection `229.8s`, and `855269` C++ leaf
evaluations.

The important effective difference from `m72jb2b9`
(`nosl-cpp-az-4way-s512-unlimited-b16-random0`) is the combat AZ minibatch
size:

| setting | old `m72jb2b9` | current `bh2d8j6y` |
| --- | ---: | ---: |
| initialization | `nosl_az_random_start_s512_step0` | `nosl_az_dirichlet_step0` |
| `nosl_is_az_batch_size` in checkpoint | `1536` | `64` |
| checkpoint `n_steps` / `batch_size` | `128` / `1536` | `64` / `64` |
| checkpoint `amp_mode` | `bf16` | `off` |
| checkpoint root noise | disabled | `0.3` / `0.25` (overridden by the running command to `0` / `0`) |
| running command `n_steps` / `batch_size` | `128` / `1536` | `128` / `1536` |

The distributed command does not override `nosl_is_az_batch_size`, so the
loaded checkpoint value remains effective. Consequently the old run used
about two AZ minibatches per rank and update (`~1934` roots), while the current
run uses about thirty (`~1900` roots). This is visible in the logs: mean AZ
backward time was approximately `0.17s` for the old run and `1.17s` for the
current run. The initialization combat specialist is also different; tensor
inspection found the outer policy identical and all differing tensors confined
to `core.combat.*`.

The search implementation was separately checked on all four machines:

* `rl.test_nosl_sampler` passed all 3 tests when the active
  `build-handonly-v4` native module was selected. Running against the default
  stale build gives the expected interface error (`nosl_redeterminize` absent),
  but the training processes all load `build-handonly-v4`.
* A native model-free search with `128` simulations and `depth=0` produced
  identical results on every rank for three fixed seed/nonce pairs.
* A neural C++ session with a fixed zero prior and constant leaf value used
  exactly 8 leaf batches / 128 leaves and produced identical root visits and
  value on every rank.

The active native `.so` hashes form two rank groups (`rank0/1` and `rank2/3`),
but the deterministic multi-seed search checks found no behavioral difference.
For a future clean reproduction, explicitly pass
`--nosl-is-az-batch-size 1536` (or deliberately use `64`) and use the same
step-0 checkpoint; otherwise the two training curves are not an apples-to-
apples comparison.

## AlphaZero replay window

NOSL-IS-AZ now keeps a per-rank FIFO replay window of completed combat roots.
Each entry stores the combat-only observation, legal-action mask, MCTS visit
policy, and objective terminal `HP / 200` target. A fight may span collection
rollouts; its pending roots stay attached to that environment until the fight
ends and receives its objective HP label.

The default capacity is `131072` roots per rank. Categorical observations use
the smallest lossless integer dtype, while normalized scalar observations and
visit probabilities use float16 storage. With the current 181-action combat
contract this reserves about 550 MiB per rank once the replay is first used.
The arrays are allocated lazily and the replay payload is deliberately excluded
from SB3 model zip files, so parameter checkpoints stay small; restarting a
process starts a fresh replay window.

Training controls have the following meanings after this change:

| option | meaning | default |
| --- | --- | ---: |
| `--nosl-is-az-batch-size` | replay roots sampled without replacement per effective update | `64` |
| `--nosl-is-az-micro-batch-size` | maximum roots per GPU forward/backward chunk | `2048` |
| `--nosl-is-az-replay-capacity` | FIFO roots retained on each rank | `131072` |
| `--nosl-is-az-replay-min-size` | roots required before training starts | `1` |

Microbatch losses are weighted by their actual sample count, accumulated into
one effective-batch mean gradient, and then synchronized once through the SSH
ring. Thus an effective batch of `24576` does not require materializing all
24576 roots on the GPU simultaneously.

## Replay-disabled run at learning rate 1e-4 (2026-09-07)

The replay experiment was stopped after its artifacts were preserved. A fresh
four-rank run was then started from
`migration/nosl-az/nosl_az_random_start_s512_step0.zip`, keeping the outer
policy frozen and changing only the requested training controls:

```text
run id: s512noreplaylr1e4
W&B: https://wandb.ai/1283802076-tsinghua-university/sts-rl/runs/w6msc4ko
simulations: 512
depth: 0
nosl_is_az_batch_size: 24576
nosl_is_az_micro_batch_size: 2048
replay: disabled
learning rate: 0.0001
n_envs / n_steps / batch_size: 24 / 128 / 1536
amp / gradient transport: bf16 / ssh-ring
```

All four ranks passed the NOSL unit tests and native build check before launch.
The first four synchronized updates completed at global step `49152`.
The rank-aggregated diagnostics reported
`learning_rate=9.9999997e-05`, `replay_enabled=0`,
`replay_size=0`, `replay_allocated_mb=0`, and
`finite_fraction=1`; no ring or simulator errors were observed.
