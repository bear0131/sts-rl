"""CPU/CUDA PPO benchmark with a subprocess wall-time budget and JSON evidence.

Run from the repo: python -m rl.benchmarks.train --help
Output intentionally stays outside source control under artifacts/local/.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--n-envs", type=int, default=2)
    p.add_argument("--vec-env", choices=("dummy", "subproc", "grouped"), default="dummy")
    p.add_argument("--envs-per-worker", type=int, default=2)
    p.add_argument("--n-steps", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--accumulation", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--updates", type=int, default=2)
    p.add_argument("--warmup-updates", type=int, default=1)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hierarchical", action="store_true")
    p.add_argument("--amp", choices=("off", "bf16"), default="off")
    p.add_argument("--search", choices=("none", "puct", "full", "strict"), default="none")
    p.add_argument("--simulations", type=int, default=8)
    p.add_argument("--replay-capacity", type=int, default=2048)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--sample-interval", type=float, default=0.5)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p


def make_environment():
    from gymnasium.wrappers import TimeLimit
    from env.core.run_env import SlayTheSpireEnv
    from rl.reward import ShapedRewardWrapper
    return TimeLimit(ShapedRewardWrapper(SlayTheSpireEnv(global_map_view=True)),
                     max_episode_steps=5000)


def run_worker(args):
    import numpy as np
    import psutil
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv
    from rl.cached_mask_vec_env import CachedMaskSubprocVecEnv, GroupedCachedMaskSubprocVecEnv
    from rl.accumulating_ppo import AccumulatingMaskablePPO
    from rl.split_policy import SplitStSPointerPolicy
    from rl.benchmarks.resources import ResourceMonitor
    import slaythespire as sts

    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.amp == "bf16" and args.device != "cuda":
        raise ValueError("this benchmark supports BF16 only on CUDA")
    def synchronize():
        if args.device == "cuda":
            torch.cuda.synchronize()

    class TimedPPO(AccumulatingMaskablePPO):
        collection_seconds = 0.0
        optimization_seconds = 0.0
        def collect_rollouts(self, *a, **kw):
            synchronize()
            start = time.perf_counter()
            result = super().collect_rollouts(*a, **kw)
            synchronize()
            self.collection_seconds += time.perf_counter() - start
            return result
        def train(self):
            synchronize()
            start = time.perf_counter()
            super().train()
            synchronize()
            self.optimization_seconds += time.perf_counter() - start

    start = time.perf_counter()
    env_fns = [make_environment] * args.n_envs
    if args.vec_env == "dummy":
        env = DummyVecEnv(env_fns)
    elif args.vec_env == "subproc":
        env = CachedMaskSubprocVecEnv(env_fns, start_method="spawn")
    else:
        env = GroupedCachedMaskSubprocVecEnv(
            env_fns, envs_per_worker=args.envs_per_worker, start_method="spawn")
    try:
        model = TimedPPO(
            SplitStSPointerPolicy, env, n_steps=args.n_steps, batch_size=args.batch_size,
            n_epochs=args.epochs, gradient_accumulation_steps=args.accumulation,
            hierarchical_combat_hp=args.hierarchical, amp_mode=args.amp,
            nosl_is_az=args.search != "none", nosl_is_az_cpp=args.search != "none",
            nosl_is_az_depth=0, nosl_is_az_simulations=args.simulations,
            nosl_is_az_search_mode="full_gumbel" if args.search in ("full", "strict") else "puct",
            nosl_is_az_paper_full_gumbel=args.search == "strict",
            nosl_is_az_replay_capacity=args.replay_capacity,
            nosl_is_az_micro_batch_size=args.batch_size,
            nosl_is_az_batch_size=args.batch_size,
            policy_kwargs=dict(d_model=args.d_model, n_layers=args.layers,
                               n_heads=args.heads, map_dag=True),
            seed=args.seed, device=args.device, learning_rate=3e-4,
        )
        synchronize()
        setup_seconds = time.perf_counter() - start
        rollout = args.n_envs * args.n_steps
        warmup_started = time.perf_counter()
        if args.warmup_updates:
            model.learn(rollout * args.warmup_updates)
        synchronize()
        warmup_seconds = time.perf_counter() - warmup_started
        model.collection_seconds = model.optimization_seconds = 0.0
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        before = model.num_timesteps
        with ResourceMonitor(interval=args.sample_interval, gpu=args.device == "cuda") as monitor:
            start = time.perf_counter()
            model.learn(rollout * args.updates, reset_num_timesteps=False)
            synchronize()
            seconds = time.perf_counter() - start
        transition_count = model.num_timesteps - before
        tensors = model.policy.state_dict()
        if not all(torch.isfinite(v).all().item() for v in tensors.values()):
            raise RuntimeError("training produced non-finite parameters")
        cuda = None
        if args.device == "cuda":
            cuda = {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        buffer_arrays = {k: v for k, v in vars(model.rollout_buffer).items()
                         if isinstance(v, np.ndarray)}
        observations = model.rollout_buffer.observations
        buffer_bytes = sum(x.nbytes for x in buffer_arrays.values())
        buffer_bytes += sum(x.nbytes for x in observations.values())
        replay = getattr(model, "_nosl_is_az_replay", None)
        if args.search != "none" and (replay is None or replay.total_added == 0):
            raise RuntimeError("search budget produced no completed-fight labels; increase steps")
        def git(*a):
            return subprocess.check_output(["git", *a], text=True).strip()
        result = {
            "status": "ok", "schema_version": 1, "config": vars(args).copy(),
            "identity": {"commit": git("rev-parse", "HEAD"),
                         "dirty": bool(git("status", "--porcelain")),
                         "native_module": str(Path(sts.__file__).resolve()),
                         "native_sha256": hashlib.sha256(Path(sts.__file__).read_bytes()).hexdigest(),
                         "python": platform.python_version(), "platform": platform.platform(),
                         "cpu_logical": psutil.cpu_count(), "cpu_physical": psutil.cpu_count(logical=False),
                         "ram_total_bytes": psutil.virtual_memory().total,
                         "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
                         "dependencies": {p: importlib.metadata.version(p) for p in
                                          ("torch", "numpy", "gymnasium", "stable-baselines3", "sb3-contrib", "psutil")}},
            "timing": {"setup_seconds": setup_seconds, "warmup_seconds": warmup_seconds,
                       "measured_seconds": seconds, "transitions": transition_count,
                       "transitions_per_second": transition_count / seconds,
                       "collection_seconds": model.collection_seconds,
                       "optimization_seconds": model.optimization_seconds},
            "memory": {"rollout_buffer_bytes": buffer_bytes,
                       "replay_bytes": 0 if replay is None else replay.allocated_bytes,
                       "parameter_bytes": sum(v.numel() * v.element_size() for v in model.policy.parameters()),
                       "cuda": cuda},
            "resources": monitor.summary, "samples": monitor.samples,
            "learner": {"replay_samples": 0 if replay is None else replay.size,
                        "replay_total_added": 0 if replay is None else replay.total_added},
        }
        result["config"]["output"] = str(args.output)
        return result
    finally:
        env.close()


def main():
    args = parser().parse_args()
    for field in ("threads", "n_envs", "n_steps", "batch_size", "accumulation",
                  "epochs", "updates", "d_model", "layers", "heads", "envs_per_worker",
                  "simulations", "replay_capacity"):
        if getattr(args, field) < 1:
            raise ValueError(field + " must be positive")
    if args.warmup_updates < 0 or args.timeout <= 0 or args.sample_interval <= 0:
        raise ValueError("invalid warmup/time budget")
    if args.n_envs * args.n_steps % args.batch_size:
        raise ValueError("use complete microbatches for controlled benchmark comparisons")
    if args.search != "none":
        args.hierarchical = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        try:
            result = run_worker(args)
        except Exception as exc:
            args.output.write_text(json.dumps({"status": "error", "error": repr(exc)}, indent=2))
            raise
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return
    if args.output.exists():
        raise FileExistsError("choose a new --output; benchmark evidence is not overwritten")
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = str(args.threads)
    start = time.perf_counter()
    process = subprocess.Popen([sys.executable, "-m", __name__.replace("__main__", "rl.benchmarks.train"),
                                *sys.argv[1:], "--worker"], env=env)
    try:
        code = process.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        import psutil
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        process.kill()
        process.wait(timeout=5)
        args.output.write_text(json.dumps({"status": "timeout", "timeout_seconds": args.timeout}))
        raise SystemExit(124)
    if code == 0:
        result = json.loads(args.output.read_text(encoding="utf-8"))
        result["timing"]["process_wall_seconds"] = time.perf_counter() - start
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(args.output), "timing": result["timing"],
                          "memory": result["memory"], "resources": result["resources"]}))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
