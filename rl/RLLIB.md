# RLlib PPO trainer

`rl.rllib_trainer` is the distributed alternative to the SB3 trainer. It uses
the same environment, shaped reward, observation encoder, map-DAG pointer core,
and action masks. RLlib checkpoints are separate from SB3 `.zip` checkpoints.

Install on every Ray node:

```bash
pip install -r rl/requirements.txt
```

Single machine, one GPU and 16 environments:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m rl.rllib_trainer \
  --task wholerun \
  --steps 8000000 \
  --num-env-runners 4 \
  --envs-per-runner 4 \
  --num-gpus 1 \
  --train-batch-size 8192 \
  --minibatch-size 512
```

Warm-start from an SB3 transformer-pointer checkpoint and log to W&B:

```bash
uv run python -m rl.rllib_trainer \
  --task wholerun \
  --global-map-view \
  --map-dag \
  --steps 8000000 \
  --num-env-runners 4 \
  --envs-per-runner 4 \
  --num-gpus 1 \
  --train-batch-size 7168 \
  --minibatch-size 512 \
  --init-from maskable_ppo_300000_steps.zip \
  --num-epochs 10 \
  --wandb \
  --wandb-project sts-rl \
  --wandb-name wholerun-rllib
```

`--init-from` copies the shared policy/value core from an SB3 `.zip` and starts
RLlib at step zero with a fresh optimizer. The checkpoint must use
`StSPointerPolicy`; flat SB3 policies cannot be mapped to RLlib's pointer head.
The default `--rollout-fragment-length auto` chooses the largest exact divisor
no greater than 256 after the final runner layout is known. On single-machine
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
