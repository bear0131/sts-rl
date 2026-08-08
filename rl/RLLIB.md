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
  --num-env-runners 16 \
  --envs-per-runner 1 \
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
The default `--rollout-fragment-length auto` makes RLlib choose 448 for the
16-runner/7168-batch example, so each sampling wave exactly fills one batch.
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

Each iteration prints sampling, learning, and weight-sync durations plus a
coarse bottleneck classification. RLlib also writes detailed `timers`, learner
statistics, CPU/RAM utilization, JSON, CSV, and TensorBoard logs under
`~/ray_results`.

The implementation currently pins RLlib 2.48 and uses its legacy ModelV2 stack
because that stack supports the existing custom pointer actor-critic and the
legacy multi-GPU PPO optimizer directly. RLlib emits deprecation warnings for
this stack; migrate to RLModule/Learner before upgrading beyond the pinned Ray
version.
