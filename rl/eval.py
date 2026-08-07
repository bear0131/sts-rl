"""Evaluate a MaskablePPO checkpoint on complete simulated runs.

Every episode starts on floor 1 with the normal Ironclad starter state and then
plays the complete map, carrying HP, deck, relics, and potions between floors.
There is deliberately no step-count TimeLimit in this evaluator.

Examples:

    python -m rl.eval rl/runs/whole-run/final.zip
    python -m rl.eval checkpoint.zip --episodes 200 --output-csv runs.csv
    python -m rl.eval checkpoint.zip --episodes 50 --stochastic
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import numpy as np

from env.core.run_env import SlayTheSpireEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from env.core import action_space as A
from rl.eval_starter_policy import _dp_route_choices

# Register both custom policy families before SB3 unpickles a checkpoint.
from . import model as _model  # noqa: F401
from . import pointer_policy as _pointer_policy  # noqa: F401


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MaskablePPO from floor 1 on complete SlayTheSpireEnv runs "
            "and report the death-floor distribution."
        )
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to a MaskablePPO .zip checkpoint.",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--route",
        choices=("model", "min-elite-max-rest"),
        default="model",
        help="model lets the checkpoint choose map nodes; min-elite-max-rest "
        "overrides only map actions using full-map DP (minimize elites first, "
        "then maximize rest sites).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using deterministic argmax actions.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N completed runs; use 0 to disable.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
        help="Print current floor/step after this long without output; 0 disables.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optionally save one row per run for seed-level analysis.",
    )
    return parser


def _run_stats(env: SlayTheSpireEnv) -> dict:
    return env.unwrapped.thick.run_stats()


def is_win(env: SlayTheSpireEnv) -> bool:
    """A normal three-act clear advances to act 4 with outcome UNDECIDED."""
    stats = _run_stats(env)
    return stats["act"] > 3 or stats["outcome"] == "PLAYER_VICTORY"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval] CSV written to {path.resolve()}", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")
    if args.heartbeat_seconds < 0:
        raise ValueError("--heartbeat-seconds must be non-negative")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    load_started = time.monotonic()
    print(f"[eval] loading checkpoint: {args.checkpoint}", flush=True)
    model = MaskablePPO.load(args.checkpoint, device=args.device)
    env = SlayTheSpireEnv(
        ascension=args.ascension,
        global_map_view=_model.uses_global_map_observation(model),
        card_language=_model.uses_card_language(model),
        combat_language=_model.uses_combat_language(model),
    )
    model.set_random_seed(args.seed)
    print(
        f"[eval] loaded {args.checkpoint.name} in "
        f"{time.monotonic() - load_started:.1f}s; starting {args.episodes} "
        "full runs from floor 1 (no step limit)",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    death_floors: Counter[int] = Counter()
    dead_screen_floors: Counter[int] = Counter()
    returns: list[float] = []
    lengths: list[int] = []
    started = time.monotonic()
    last_report = started

    try:
        for episode_index in range(args.episodes):
            episode_seed = args.seed + episode_index
            obs, info = env.reset(seed=episode_seed)
            terminated = truncated = False
            episode_return = 0.0
            episode_length = 0
            route_choices = 0
            route_rng = np.random.default_rng(episode_seed ^ 0x51A7_7E12)

            while not (terminated or truncated):
                action_masks = get_action_masks(env)
                if args.route == "min-elite-max-rest" and info.get("phase") == "map":
                    legal = np.flatnonzero(action_masks).tolist()
                    map_choices = [
                        (offset, action_id)
                        for action_id in legal
                        for segment, offset in [A.segment_of(action_id)]
                        if segment == "MAP_NODE"
                    ]
                    if not map_choices:
                        raise RuntimeError("map phase has no legal MAP_NODE action")
                    backend = env.unwrapped.thick
                    optimal, _score = _dp_route_choices(
                        map_choices,
                        backend.rmap,
                        backend.gc.cur_map_node_y,
                    )
                    action = int(route_rng.choice(optimal))
                    route_choices += 1
                else:
                    action, _ = model.predict(
                        obs,
                        action_masks=action_masks,
                        deterministic=not args.stochastic,
                    )
                obs, reward, terminated, truncated, info = env.step(
                    int(np.asarray(action).item())
                )
                episode_return += float(reward)
                episode_length += 1

                now = time.monotonic()
                if (
                    args.heartbeat_seconds > 0
                    and now - last_report >= args.heartbeat_seconds
                ):
                    stats = _run_stats(env)
                    print(
                        f"[eval] run {episode_index + 1}/{args.episodes}: "
                        f"floor={stats['floor']} act={stats['act']} "
                        f"step={episode_length} in progress (no step limit)",
                        flush=True,
                    )
                    last_report = now

            stats = _run_stats(env)
            floor = int(stats["floor"])
            act = int(stats["act"])
            if is_win(env):
                result = "win"
            elif truncated and info.get("dead_screen"):
                result = "dead_screen"
                dead_screen_floors[floor] += 1
            elif truncated:
                # SlayTheSpireEnv currently only truncates unsupported screens;
                # retain this category so a future truncation cannot become a death.
                result = "other_truncation"
            else:
                result = "death"
                death_floors[floor] += 1

            returns.append(episode_return)
            lengths.append(episode_length)
            rows.append(
                {
                    "episode": episode_index + 1,
                    "seed": episode_seed,
                    "result": result,
                    "death_floor": floor if result == "death" else "",
                    "final_floor": floor,
                    "final_act": act,
                    "steps": episode_length,
                    "route_choices": route_choices,
                    "raw_return": episode_return,
                    "outcome": stats["outcome"],
                }
            )

            done = episode_index + 1
            if args.progress_every > 0 and done % args.progress_every == 0:
                elapsed = time.monotonic() - started
                remaining = elapsed / done * (args.episodes - done)
                print(
                    f"[eval] {done}/{args.episodes}: seed={episode_seed} "
                    f"result={result} floor={floor} steps={episode_length} "
                    f"ETA={remaining / 60:.1f} min",
                    flush=True,
                )
                last_report = time.monotonic()
    finally:
        env.close()

    result_counts = Counter(str(row["result"]) for row in rows)
    wins = result_counts["win"]
    deaths = result_counts["death"]
    elapsed = time.monotonic() - started
    final_floors = np.asarray([int(row["final_floor"]) for row in rows])

    print()
    print(
        f"[eval] episodes={args.episodes} ascension={args.ascension} "
        f"deterministic={not args.stochastic} route={args.route} "
        "no_step_limit=True"
    )
    print(f"[eval] wins={wins}/{args.episodes} win_rate={wins / args.episodes:.4f}")
    print(
        f"[eval] deaths={deaths} dead_screens={result_counts['dead_screen']} "
        f"other_truncations={result_counts['other_truncation']}"
    )
    print(
        f"[eval] final_floor_mean={final_floors.mean():.2f} "
        f"median={np.median(final_floors):.1f}"
    )
    print("[eval] death_floor_distribution:")
    if deaths == 0:
        print("  (no deaths)")
    else:
        for floor in sorted(death_floors):
            count = death_floors[floor]
            print(
                f"  floor={floor:02d} count={count:4d} "
                f"pct_of_deaths={count / deaths:7.2%} "
                f"pct_of_all_runs={count / args.episodes:7.2%}"
            )
    if dead_screen_floors:
        print("[eval] dead_screen_floor_distribution (excluded from deaths):")
        for floor in sorted(dead_screen_floors):
            print(f"  floor={floor:02d} count={dead_screen_floors[floor]:4d}")
    print(
        f"[eval] raw_return={np.mean(returns):.3f} +/- {np.std(returns):.3f}"
    )
    print(
        f"[eval] episode_length={np.mean(lengths):.1f} +/- "
        f"{np.std(lengths):.1f} elapsed={elapsed:.1f}s"
    )

    if args.output_csv is not None:
        _write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()
