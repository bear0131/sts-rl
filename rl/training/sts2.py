"""Bounded STS2 training entry point; STS1 distributed CLI owns a different codec."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from env.sts2.game import EnvConfig
from rl.algorithms.sts2_ppo import TrainConfig, Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars="@")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2, help="absolute target, also when resuming")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--task", choices=("combat", "run"), default="combat")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--episode-steps", type=int, default=3000)
    for name, default in (("seed", 42), ("num-envs", 4), ("rollout-steps", 64), ("minibatch", 64), ("threads", 1)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.iterations < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("positive iterations and timeout required")
    # Each invocation gets its own output, including resumes: logs cannot mix
    # divergent checkpoint lineages or silently overwrite an earlier result.
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def check():
        if time.monotonic()-started > args.timeout:
            raise TimeoutError("STS2 training wall-clock budget exceeded")
    trainer = Trainer(TrainConfig(seed=args.seed, num_envs=args.num_envs,
        rollout_steps=args.rollout_steps, minibatch=args.minibatch, threads=args.threads, device=args.device),
        EnvConfig(task=args.task, ascension=args.ascension, max_steps=args.episode_steps))
    try:
        if args.resume:
            trainer.restore(args.resume)
        while trainer.iteration < args.iterations:
            row = trainer.iteration_step(check)
            check()
            save_started = time.perf_counter()
            trainer.save(args.output / f"iteration-{trainer.iteration}.pt")
            row["save_seconds"] = time.perf_counter()-save_started
            row["elapsed_seconds"] = time.monotonic()-started
            with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
            print(json.dumps({k: row[k] for k in ("iteration", "transitions", "loss", "parameter_delta_l2", "outcomes")}))
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
