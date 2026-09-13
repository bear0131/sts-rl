# RLlib PPO trainer

`rl.rllib_trainer` is the distributed alternative to the SB3 trainer. It uses
the same environment, shaped reward, split combat/overworld pointer cores, and
action masks. RLlib checkpoints are separate from SB3 `.zip` checkpoints.

Install on every Ray node:

```bash
pip install -r rl/requirements.txt
```

Single machine, one GPU and 16 environments. This follows the same resource
shape as balatrobot: one Ray CPU slot per env-runner actor, with several
vectorized environments inside each actor when the simulator is light enough.
It is not one CPU per environment.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m rl.rllib_trainer \
  --task wholerun \
  --steps 8000000 \
  --num-env-runners 8 \
  --envs-per-runner 2 \
  --num-cpus-per-env-runner 1 \
  --num-gpus 1 \
  --train-batch-size 8192 \
  --minibatch-size 512
```

Warm-start from an SB3 split/combat-specialist checkpoint and log to W&B:

```bash
uv run python -m rl.rllib_trainer \
  --task wholerun \
  --global-map-view \
  --map-dag \
  --steps 8000000 \
  --num-env-runners 8 \
  --envs-per-runner 2 \
  --num-cpus-per-env-runner 1 \
  --num-gpus 1 \
  --train-batch-size 7168 \
  --minibatch-size 512 \
  --init-from maskable_ppo_300000_steps.zip \
  --num-epochs 10 \
  --wandb \
  --wandb-project sts-rl \
  --wandb-name wholerun-rllib
```

Combat-gauntlet pretraining with card and combat language observations:

```bash
uv run python -m rl.rllib_trainer \
  --task combat \
  --steps 8000000 \
  --max-env-runners 16 \
  --num-env-runners 16 \
  --envs-per-runner 2 \
  --num-cpus-per-env-runner 1 \
  --num-gpus 1 \
  --batch-size 256 \
  --gradient-accumulation-steps 128 \
  --rollout-fragment-length 64 \
  --card-language \
  --combat-language \
  --wandb \
  --wandb-project sts-rl \
  --wandb-name combat-lang-card-combat-bs32768-mb256
```

`--init-from` copies current specialized weights from an SB3 `.zip` and starts
RLlib at step zero with a fresh optimizer. A combat checkpoint initializes the
shared tokenizer and complete combat actor-critic in a fresh split model; the
overworld actor-critic remains newly initialized. Old unified/flat policies are
not accepted.
The default `--rollout-fragment-length auto` chooses the largest exact divisor
no greater than `--auto-fragment-max 64` after the final runner layout is known.
On single-machine
training, prefer fewer Ray actors with several vectorized environments each
(`4x4` or `8x2`) before trying `16x1`; many one-env actors can waste time in Ray
scheduling and may wait forever if the head/learner process also reserves CPU
slots. The trainer now reserves no Ray CPU for the local main process by default,
does not create a hidden local env runner, and folds too many requested runners
back into fewer vectorized runners when the cluster CPU budget or the default
`--max-env-runners 8` actor cap would otherwise be exceeded. Pass
`--max-env-runners 0` to disable that cap on a real multi-node Ray cluster.
SB3-style `--n-envs`, `--n-steps`, `--batch-size`, and
`--gradient-accumulation-steps` are also accepted. On a 32 GiB GPU, use
`--batch-size 512 --gradient-accumulation-steps 64`: RLlib samples 32768
transitions and performs one optimizer update per 64 micro-batches. A 1024
micro-batch exhausts a 32 GiB RTX 4080 SUPER with the map-DAG model.
`--resume` accepts a native RLlib checkpoint. Passing an SB3 `.zip` to
`--resume` migrates its weights and preserves its step offset, but the SB3
optimizer state cannot be converted.

Single machine, two GPUs for RLlib's multi-GPU PPO optimizer:

```bash
python -m rl.rllib_trainer \
  --steps 8000000 \
  --num-env-runners 8 \
  --envs-per-runner 4 \
  --num-gpus 2
```

Resume from an RLlib checkpoint directory:

```bash
python -m rl.rllib_trainer \
  --resume rl/runs/rllib/<run>/final \
  --steps 16000000
```

For the local parameter dashboard, download only the policy weights file from
the newest checkpoint:

```text
rl/runs/rllib/<run>/step_checkpoints/policies/default_policy/policy_state.pkl
```

Upload that `.pkl` directly in `python -m rl.dashboard`. The dashboard does not
need `algorithm_state.pkl`; resuming training still requires the complete RLlib
checkpoint directory shown above.

The dashboard also exposes a reusable hybrid comparison interface on port 8765.
It compares pure `policy_state.pkl` against a hybrid that routes one phase to
the RLlib policy and everything else to the SB3 checkpoint:

```bash
python -m rl.dashboard --port 8765
```

```bash
curl -X POST http://127.0.0.1:8765/api/hybrid-eval/start \
  -H "Content-Type: application/json" \
  -d "{\"episodes\":100,\"seed\":10000,\"num_envs\":16,\"route_phase\":\"combat\"}"
```

By default the interface reads:

```text
C:\Users\Tony2\Downloads\policy_state.pkl
C:\Users\Tony2\Downloads\maskable_ppo_6400000_steps.zip
```

Use `GET /api/hybrid-eval/status` to poll progress and
`POST /api/hybrid-eval/stop` to stop the current run. Each run writes
`summary.json`, `episodes.csv`, `stdout.log`, and `stderr.log` under
`rl/runs/eval-hybrid/dashboard-*`.

Set `"route_phase":"map"` to test the route-selection hybrid the same way.

For a multi-node Ray cluster, start the head and workers with `ray start`, make
the repository and compiled simulator available on every node, then add
`--ray-address auto`. Cross-node training requires the nodes to be mutually
reachable; separate AutoDL instances often need explicit network configuration.
The trainer packages the repository for Ray workers while excluding `.venv`,
W&B history, run outputs, and checkpoint `.zip` files. The compiled simulator
extension remains included; this prevents Ray's 512 MiB working-directory limit
from being consumed by old training artifacts.

Each iteration prints sampling, learning, and weight-sync durations plus a
coarse bottleneck classification. RLlib also writes detailed `timers`, learner
statistics, CPU/RAM utilization, JSON, CSV, and TensorBoard logs under
`~/ray_results`.

The implementation currently pins RLlib 2.48 and uses its legacy ModelV2 stack
because that stack supports the existing custom pointer actor-critic and the
legacy multi-GPU PPO optimizer directly. RLlib emits deprecation warnings for
this stack; migrate to RLModule/Learner before upgrading beyond the pinned Ray
version.
