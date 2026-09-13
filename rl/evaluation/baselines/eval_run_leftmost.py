"""Evaluate a simple full-run Ironclad baseline from floor 1.

Policy:
  * combat: play the leftmost currently legal card; otherwise end turn;
  * map: choose uniformly among currently reachable nodes;
  * campfire: rest whenever rest is legal;
  * all other required choices: take the leftmost legal action.

There is no step-count TimeLimit. Unsupported simulator screens are reported as
``dead_screen`` and are not counted as player deaths.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from env.core import action_space as A
from env.core import game_actions as ga
from env.core.run_env import SlayTheSpireEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Full-run baseline: leftmost playable card, random map route, "
            "and rest at campfires."
        )
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser


def choose_action(info: dict, route_rng: np.random.Generator) -> int:
    """Choose one legal thin action according to the baseline policy."""
    mask = np.asarray(info["action_mask"], dtype=bool)
    legal = np.flatnonzero(mask).tolist()
    if not legal:
        raise RuntimeError("environment returned an all-false action mask")

    by_segment: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for action_id in legal:
        segment, offset = A.segment_of(action_id)
        by_segment[segment].append((offset, action_id))

    # Resolve parameters opened by the most recently played card.
    if info.get("arg_step") == A.PICK_TARGET:
        return min(by_segment["PICK_TARGET"])[1]
    if info.get("arg_step") == A.PICK_HAND:
        if not info.get("multi_selected") and by_segment.get("PICK_CARD"):
            return min(by_segment["PICK_CARD"])[1]
        return by_segment["CONFIRM_SELECT"][0][1]
    if info.get("arg_step") == A.PICK_CARD:
        return min(by_segment["PICK_CARD"])[1]

    phase = info.get("phase")
    if phase == "combat":
        if by_segment.get("PLAY_CARD"):
            return min(by_segment["PLAY_CARD"])[1]
        if by_segment.get("END_TURN"):
            return by_segment["END_TURN"][0][1]

    if phase == "map" and by_segment.get("MAP_NODE"):
        choices = [action_id for _offset, action_id in by_segment["MAP_NODE"]]
        return int(route_rng.choice(choices))

    if phase == "rest":
        rest_id = A.seg_base("CAMPFIRE") + ga.CAMPFIRE_OPTIONS["rest"]
        if rest_id in legal:
            return rest_id
        if by_segment.get("PROCEED"):
            return by_segment["PROCEED"][0][1]

    # The request does not prescribe reward/event/shop decisions. Keeping this
    # fallback leftmost makes them deterministic and easy to reproduce.
    return min(legal)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[run-leftmost] CSV written to {path.resolve()}")


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")
    if args.heartbeat_seconds < 0:
        raise ValueError("--heartbeat-seconds must be non-negative")

    env = SlayTheSpireEnv(ascension=args.ascension)
    rows: list[dict[str, object]] = []
    death_floors: Counter[int] = Counter()
    dead_screen_floors: Counter[int] = Counter()
    started = time.monotonic()
    last_report = started

    try:
        for episode_index in range(args.episodes):
            episode_seed = args.seed + episode_index
            route_rng = np.random.default_rng(episode_seed ^ 0x5EED_5EED)
            _obs, info = env.reset(seed=episode_seed)
            terminated = truncated = False
            steps = 0
            route_choices = 0
            rests = 0
            rest_unavailable = 0

            while not (terminated or truncated):
                phase = info.get("phase")
                action = choose_action(info, route_rng)
                segment, offset = A.segment_of(action)
                if phase == "map" and segment == "MAP_NODE":
                    route_choices += 1
                if phase == "rest":
                    if segment == "CAMPFIRE" and offset == ga.CAMPFIRE_OPTIONS["rest"]:
                        rests += 1
                    else:
                        rest_unavailable += 1

                _obs, _reward, terminated, truncated, info = env.step(action)
                steps += 1

                now = time.monotonic()
                if (
                    args.heartbeat_seconds > 0
                    and now - last_report >= args.heartbeat_seconds
                ):
                    stats = env.unwrapped.thick.run_stats()
                    print(
                        f"[run-leftmost] episode {episode_index + 1}/"
                        f"{args.episodes}: floor={stats['floor']} step={steps} "
                        "in progress (no step limit)",
                        flush=True,
                    )
                    last_report = now

            stats = env.unwrapped.thick.run_stats()
            floor = int(stats["floor"])
            act = int(stats["act"])
            won = act > 3 or stats["outcome"] == "PLAYER_VICTORY"
            if won:
                result = "win"
            elif truncated and info.get("dead_screen"):
                result = "dead_screen"
                dead_screen_floors[floor] += 1
            elif truncated:
                result = "other_truncation"
            else:
                result = "death"
                death_floors[floor] += 1

            rows.append(
                {
                    "episode": episode_index + 1,
                    "seed": episode_seed,
                    "result": result,
                    "death_floor": floor if result == "death" else "",
                    "final_floor": floor,
                    "final_act": act,
                    "steps": steps,
                    "route_choices": route_choices,
                    "rests": rests,
                    "rest_unavailable": rest_unavailable,
                    "outcome": stats["outcome"],
                }
            )

            done = episode_index + 1
            if args.progress_every > 0 and done % args.progress_every == 0:
                elapsed = time.monotonic() - started
                remaining = elapsed / done * (args.episodes - done)
                print(
                    f"[run-leftmost] {done}/{args.episodes} "
                    f"({done / elapsed:.2f} ep/s, ETA {remaining / 60:.1f} min)",
                    flush=True,
                )
                last_report = time.monotonic()
    finally:
        env.close()

    result_counts = Counter(str(row["result"]) for row in rows)
    deaths = result_counts["death"]
    wins = result_counts["win"]
    final_floors = np.asarray([int(row["final_floor"]) for row in rows])
    lengths = np.asarray([int(row["steps"]) for row in rows])
    elapsed = time.monotonic() - started

    print()
    print(
        f"[run-leftmost] episodes={args.episodes} seed={args.seed} "
        f"ascension={args.ascension} no_step_limit=True"
    )
    print(
        "[run-leftmost] policy=leftmost_play + random_route + "
        "campfire_rest + leftmost_other"
    )
    print(f"[run-leftmost] wins={wins}/{args.episodes} win_rate={wins / args.episodes:.4f}")
    print(
        f"[run-leftmost] deaths={deaths} "
        f"dead_screens={result_counts['dead_screen']} "
        f"other_truncations={result_counts['other_truncation']}"
    )
    print(
        f"[run-leftmost] final_floor_mean={final_floors.mean():.2f} "
        f"median={np.median(final_floors):.1f}"
    )
    for cleared_act in (1, 2, 3):
        cleared = sum(int(row["final_act"]) > cleared_act for row in rows)
        print(
            f"[run-leftmost] cleared_act{cleared_act}={cleared / args.episodes:.4f} "
            f"({cleared}/{args.episodes})"
        )
    print(
        f"[run-leftmost] route_choices_mean="
        f"{np.mean([int(row['route_choices']) for row in rows]):.2f} "
        f"rests_mean={np.mean([int(row['rests']) for row in rows]):.2f} "
        f"rest_unavailable_total="
        f"{sum(int(row['rest_unavailable']) for row in rows)}"
    )
    print("[run-leftmost] death_floor_distribution:")
    if not death_floors:
        print("  (no deaths)")
    for floor in sorted(death_floors):
        count = death_floors[floor]
        print(
            f"  floor={floor:02d} count={count:4d} "
            f"pct_of_deaths={count / deaths:7.2%} "
            f"pct_of_all_runs={count / args.episodes:7.2%}"
        )
    if dead_screen_floors:
        print("[run-leftmost] dead_screen_floor_distribution (excluded from deaths):")
        for floor in sorted(dead_screen_floors):
            print(f"  floor={floor:02d} count={dead_screen_floors[floor]:4d}")
    print(
        f"[run-leftmost] episode_length={lengths.mean():.1f} +/- "
        f"{lengths.std():.1f} elapsed={elapsed:.1f}s"
    )

    if args.output_csv is not None:
        _write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()
