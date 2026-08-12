"""Evaluate one or more MaskablePPO checkpoints on the combat task.

The environment, loadout distribution, reward wrapper, and episode limit match
``python -m rl.trainer --task combat``.  Every checkpoint is evaluated on the
same independently seeded episode set, so deterministic results are directly
comparable across checkpoints.

Examples:

    python -m rl.eval_combat rl/runs/combat/run/final.zip
    python -m rl.eval_combat a.zip b.zip --episodes 1000 --output-csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

# Register both custom policy families before SB3 unpickles a checkpoint.
from . import model as _model  # noqa: F401
from . import pointer_policy as _pointer_policy  # noqa: F401
from .tasks.combat import CombatTask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one or more MaskablePPO checkpoints on the same combat "
            "gauntlet episode seeds."
        )
    )
    parser.add_argument(
        "checkpoints",
        nargs="+",
        type=Path,
        help="MaskablePPO .zip checkpoint(s) to evaluate.",
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--floor-lo", type=int, default=1)
    parser.add_argument("--floor-hi", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using deterministic argmax actions.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N episodes; use 0 to disable.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
        help=(
            "Print an in-episode heartbeat after this many seconds without a "
            "completed episode; use 0 to disable."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optionally write one summary row per checkpoint.",
    )
    return parser


def _mean_and_se(values: list[int] | list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
    return mean, se


@dataclass
class EvalResult:
    checkpoint: str
    episodes: int
    deterministic: bool
    elapsed_seconds: float
    mean_steps: float
    truncated: int
    cleared_mean: float
    cleared_se: float
    act_counts: dict[int, int]
    act_means: dict[int, float]
    act_ses: dict[int, float]

    def csv_row(self) -> dict[str, object]:
        def optional_metric(value: float) -> float | str:
            return "" if math.isnan(value) else value

        row: dict[str, object] = {
            "checkpoint": self.checkpoint,
            "episodes": self.episodes,
            "deterministic": self.deterministic,
            "cleared_per_episode": self.cleared_mean,
            "cleared_per_episode_se": self.cleared_se,
            "mean_steps": self.mean_steps,
            "truncated": self.truncated,
            "elapsed_seconds": self.elapsed_seconds,
        }
        for act in (1, 2, 3):
            row[f"act{act}_episodes"] = self.act_counts[act]
            row[f"cleared_act{act}"] = optional_metric(self.act_means[act])
            row[f"cleared_act{act}_se"] = optional_metric(self.act_ses[act])
        return row


def evaluate_checkpoint(args: argparse.Namespace, checkpoint: Path) -> EvalResult:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    load_started = time.monotonic()
    print(f"[eval-combat] loading checkpoint: {checkpoint}", flush=True)
    model = MaskablePPO.load(checkpoint, device=args.device)
    card_language = _model.uses_card_language(model)
    combat_language = _model.uses_combat_language(model)
    env_args = argparse.Namespace(
        floor_lo=args.floor_lo,
        floor_hi=args.floor_hi,
        ascension=args.ascension,
        combat_hp_loss_coef=0.0,
        card_language=card_language,
        combat_language=combat_language,
    )
    env = CombatTask().make_env_fn(env_args)()
    model.set_random_seed(args.seed)
    print(
        f"[eval-combat] loaded {checkpoint.name} in "
        f"{time.monotonic() - load_started:.1f}s; "
        f"card_language={card_language} combat_language={combat_language}; "
        f"starting {args.episodes} episodes",
        flush=True,
    )

    cleared_by_act: dict[int, list[int]] = {1: [], 2: [], 3: []}
    lengths: list[int] = []
    truncations = 0
    started = time.monotonic()
    last_report = started

    try:
        for episode in range(args.episodes):
            # Independent episode seeds make the initial samples reproducible
            # and identical for every checkpoint in this invocation.
            obs, _info = env.reset(seed=args.seed + episode)
            terminated = truncated = False
            steps = 0

            while not (terminated or truncated):
                mask = get_action_masks(env)
                action, _ = model.predict(
                    obs,
                    action_masks=mask,
                    deterministic=not args.stochastic,
                )
                obs, _reward, terminated, truncated, _info = env.step(
                    int(np.asarray(action).item())
                )
                steps += 1
                now = time.monotonic()
                if (
                    args.heartbeat_seconds > 0
                    and now - last_report >= args.heartbeat_seconds
                ):
                    print(
                        f"[eval-combat] {checkpoint.name}: episode "
                        f"{episode + 1}/{args.episodes}, step {steps} in progress",
                        flush=True,
                    )
                    last_report = now

            backend = env.unwrapped.thick
            act = int(backend.gc.act)
            if act not in cleared_by_act:
                raise RuntimeError(f"combat factory produced unexpected act {act}")
            cleared_by_act[act].append(int(backend.combats_won))
            lengths.append(steps)
            truncations += int(truncated)

            done = episode + 1
            if args.progress_every > 0 and done % args.progress_every == 0:
                elapsed = time.monotonic() - started
                remaining = elapsed / done * (args.episodes - done)
                print(
                    f"[eval-combat] {checkpoint.name}: {done}/{args.episodes} "
                    f"episodes ({done / elapsed:.2f} ep/s, "
                    f"ETA {remaining / 60:.1f} min)",
                    flush=True,
                )
                last_report = time.monotonic()
    finally:
        env.close()

    act_means: dict[int, float] = {}
    act_ses: dict[int, float] = {}
    for act, values in cleared_by_act.items():
        act_means[act], act_ses[act] = _mean_and_se(values)

    all_clears = [value for values in cleared_by_act.values() for value in values]
    cleared_mean, cleared_se = _mean_and_se(all_clears)
    return EvalResult(
        checkpoint=str(checkpoint.resolve()),
        episodes=args.episodes,
        deterministic=not args.stochastic,
        elapsed_seconds=time.monotonic() - started,
        mean_steps=float(np.mean(lengths)),
        truncated=truncations,
        cleared_mean=cleared_mean,
        cleared_se=cleared_se,
        act_counts={act: len(values) for act, values in cleared_by_act.items()},
        act_means=act_means,
        act_ses=act_ses,
    )


def print_result(result: EvalResult, args: argparse.Namespace) -> None:
    print()
    print(f"[eval-combat] checkpoint={result.checkpoint}")
    print(
        f"[eval-combat] episodes={result.episodes} seed={args.seed} "
        f"ascension={args.ascension} floors={args.floor_lo}-{args.floor_hi} "
        f"deterministic={result.deterministic}"
    )
    for act in (1, 2, 3):
        if result.act_counts[act] == 0:
            print(f"combat/cleared_act{act}=n/a (episodes=0, SE=n/a)")
            continue
        print(
            f"combat/cleared_act{act}={result.act_means[act]:.4f} "
            f"(episodes={result.act_counts[act]}, SE={result.act_ses[act]:.4f})"
        )
    print(
        f"combat/cleared_per_episode={result.cleared_mean:.4f} "
        f"(SE={result.cleared_se:.4f})"
    )
    print(
        f"[eval-combat] mean_steps={result.mean_steps:.1f} "
        f"truncated={result.truncated}/{result.episodes} "
        f"elapsed={result.elapsed_seconds:.1f}s"
    )


def write_csv(path: Path, results: list[EvalResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [result.csv_row() for result in results]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval-combat] CSV written to {path.resolve()}")


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if not (1 <= args.floor_lo <= args.floor_hi <= 50):
        raise ValueError("expected 1 <= --floor-lo <= --floor-hi <= 50")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")
    if args.heartbeat_seconds < 0:
        raise ValueError("--heartbeat-seconds must be non-negative")

    results = []
    for checkpoint in args.checkpoints:
        result = evaluate_checkpoint(args, checkpoint)
        results.append(result)
        print_result(result, args)

    if args.output_csv is not None:
        write_csv(args.output_csv, results)


if __name__ == "__main__":
    main()
