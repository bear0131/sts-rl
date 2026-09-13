"""Run independent NOSL full-run evaluations in parallel CPU workers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--workers", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--master-nonce", type=lambda value: int(value, 0), default=0x4E4F534C4953415A)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _run_one(args, seed: int) -> tuple[int, Path, int]:
    output = args.output_dir / f"episode-{seed}.json"
    log = args.output_dir / f"episode-{seed}.log"
    command = [
        sys.executable,
        "-u",
        "-m",
        "rl.evaluation.nosl.eval_nosl_fullrun",
        str(args.checkpoint),
        "--episodes",
        "1",
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--simulations",
        str(args.simulations),
        "--depth",
        str(args.depth),
        "--rollout-steps",
        str(args.rollout_steps),
        "--c-puct",
        str(args.c_puct),
        "--master-nonce",
        hex(args.master_nonce),
        "--output-json",
        str(output),
    ]
    environment = os.environ.copy()
    # One process per CPU worker; avoid each PyTorch/OpenMP runtime spawning a
    # second pool and oversubscribing the 25 requested workers.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = "1"
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
    return seed, output, int(completed.returncode)


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes < 1 or args.workers < 1:
        raise ValueError("episodes and workers must be positive")
    args.workers = min(args.workers, args.episodes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows: list[dict] = []
    failures: list[dict] = []
    seeds = [args.seed + index for index in range(args.episodes)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run_one, args, seed) for seed in seeds]
        for future in as_completed(futures):
            seed, path, returncode = future.result()
            if returncode != 0 or not path.is_file():
                failures.append({"seed": seed, "returncode": returncode})
                print(f"[parallel-nosl] seed={seed} failed rc={returncode}", flush=True)
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(payload.get("rows", []))
            row = rows[-1]
            print(
                f"[parallel-nosl] {len(rows)}/{args.episodes}: seed={seed} "
                f"result={row.get('result')} floor={row.get('final_floor')} "
                f"steps={row.get('steps')}",
                flush=True,
            )

    rows.sort(key=lambda row: int(row.get("seed", 0)))
    floors = np.asarray([int(row["final_floor"]) for row in rows], dtype=np.float64)
    wins = sum(row.get("result") == "win" for row in rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "episodes_requested": args.episodes,
        "episodes_completed": len(rows),
        "workers": args.workers,
        "seed": args.seed,
        "simulations": args.simulations,
        "depth": args.depth,
        "rollout_steps": args.rollout_steps,
        "wins": int(wins),
        "win_rate": float(wins / len(rows)) if rows else 0.0,
        "final_floor_mean": float(floors.mean()) if len(floors) else 0.0,
        "final_floor_median": float(np.median(floors)) if len(floors) else 0.0,
        "elapsed_seconds": time.monotonic() - started,
        "failures": failures,
        "rows": rows,
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[parallel-nosl] completed={len(rows)}/{args.episodes} "
        f"workers={args.workers} sims={args.simulations} "
        f"win_rate={summary['win_rate']:.4f} "
        f"final_floor_mean={summary['final_floor_mean']:.2f} "
        f"elapsed={summary['elapsed_seconds']:.1f}s",
        flush=True,
    )
    print(f"[parallel-nosl] wrote {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
