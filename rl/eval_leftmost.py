"""Evaluate a deterministic leftmost-play baseline on the combat gauntlet.

Policy at a normal combat decision:
  1. Play the leftmost currently legal hand card.
  2. If no hand card is playable, end the turn.

Autoregressive target/card choices use the leftmost legal option. On a MULTI
card-select screen, select the leftmost card once and then confirm. The env,
loadout distribution, reward wrapper, and episode limit are exactly those used
by ``python -m rl.trainer --task combat``.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict

import numpy as np

from env.core import action_space as A
from rl.tasks.combat import CombatTask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the leftmost-play heuristic on the combat task."
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--floor-lo", type=int, default=1)
    parser.add_argument("--floor-hi", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def choose_action(info: dict) -> int:
    """Choose one legal thin action according to the baseline policy."""
    mask = np.asarray(info["action_mask"], dtype=bool)
    legal = np.flatnonzero(mask).tolist()
    if not legal:
        raise RuntimeError("environment returned an all-false action mask")

    by_segment: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for action_id in legal:
        segment, offset = A.segment_of(action_id)
        by_segment[segment].append((offset, action_id))

    # A targeted play is already pending: pick the leftmost live monster.
    if info.get("arg_step") == A.PICK_TARGET:
        return min(by_segment["PICK_TARGET"])[1]

    # Avoid toggling the same card forever: pick the leftmost card once, confirm
    # on the next step. This only handles a card's internal selection prompt;
    # it does not change the root rule of always playing the leftmost legal card.
    if info.get("arg_step") == A.PICK_HAND:
        if not info.get("multi_selected") and by_segment.get("PICK_CARD"):
            return min(by_segment["PICK_CARD"])[1]
        return by_segment["CONFIRM_SELECT"][0][1]

    # Single-card prompts opened by the card just played (Armaments, True Grit,
    # Headbutt, etc.) choose the leftmost legal option.
    if info.get("arg_step") == A.PICK_CARD:
        return min(by_segment["PICK_CARD"])[1]

    # Root combat rule: leftmost PLAY_CARD slot, otherwise END_TURN.
    if by_segment.get("PLAY_CARD"):
        return min(by_segment["PLAY_CARD"])[1]
    if by_segment.get("END_TURN"):
        return by_segment["END_TURN"][0][1]

    # Combat gauntlet should not expose overworld actions. Keep this fallback
    # deterministic and loud in the final counters rather than sampling.
    return min(legal)


def _mean_and_se(values: list[int]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
    return mean, se


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if not (1 <= args.floor_lo <= args.floor_hi <= 50):
        raise ValueError("expected 1 <= --floor-lo <= --floor-hi <= 50")

    env_args = argparse.Namespace(
        floor_lo=args.floor_lo,
        floor_hi=args.floor_hi,
        ascension=args.ascension,
    )
    env = CombatTask().make_env_fn(env_args)()
    cleared_by_act: dict[int, list[int]] = {1: [], 2: [], 3: []}
    lengths: list[int] = []
    truncations = 0
    started = time.monotonic()

    try:
        for episode in range(args.episodes):
            # Seed the continuous gauntlet RNG once, like a seeded training env.
            reset_kwargs = {"seed": args.seed} if episode == 0 else {}
            _obs, info = env.reset(**reset_kwargs)
            terminated = truncated = False
            steps = 0

            while not (terminated or truncated):
                action = choose_action(info)
                _obs, _reward, terminated, truncated, info = env.step(action)
                steps += 1

            backend = env.unwrapped.thick
            act = int(backend.gc.act)
            cleared = int(backend.combats_won)
            cleared_by_act[act].append(cleared)
            lengths.append(steps)
            truncations += int(truncated)

            done = episode + 1
            if args.progress_every > 0 and done % args.progress_every == 0:
                elapsed = time.monotonic() - started
                print(
                    f"[leftmost] {done}/{args.episodes} episodes "
                    f"({done / elapsed:.2f} ep/s)"
                )
    finally:
        env.close()

    elapsed = time.monotonic() - started
    print(
        f"[leftmost] episodes={args.episodes} seed={args.seed} "
        f"ascension={args.ascension} floors={args.floor_lo}-{args.floor_hi}"
    )
    for act in (1, 2, 3):
        values = cleared_by_act[act]
        mean, se = _mean_and_se(values)
        print(
            f"combat/cleared_act{act}={mean:.4f} "
            f"(episodes={len(values)}, SE={se:.4f})"
        )
    all_clears = [v for values in cleared_by_act.values() for v in values]
    overall, overall_se = _mean_and_se(all_clears)
    print(
        f"combat/cleared_per_episode={overall:.4f} "
        f"(SE={overall_se:.4f})"
    )
    print(
        f"[leftmost] mean_steps={np.mean(lengths):.1f} "
        f"truncated={truncations}/{args.episodes} elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
