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
  --steps 8000000 \
  --num-env-runners 4 \
  --envs-per-runner 4 \
  --num-gpus 1 \
  --train-batch-size 8192 \
  --minibatch-size 512
```

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
