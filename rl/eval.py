"""Evaluate a MaskablePPO checkpoint on whole-run SlayTheSpireEnv.

Examples:

    python -m rl.eval rl/runs/whole-run/final.zip
    python -m rl.eval rl/runs/whole-run/final.zip --episodes 200
    python -m rl.eval rl/runs/whole-run/final.zip --episodes 50 --stochastic
"""

from __future__ import annotations

import argparse

import numpy as np

from env.core.run_env import SlayTheSpireEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

# Important: import custom model code before loading a checkpoint.
# This keeps custom extractor classes available to Python / cloudpickle.
from . import model as _model  # noqa: F401


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate MaskablePPO on whole-run SlayTheSpireEnv."
    )

    parser.add_argument(
        "checkpoint",
        help="Path to a MaskablePPO .zip checkpoint.",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using deterministic argmax actions.",
    )

    return parser


def is_win(env) -> bool:
    """Read the terminal win signal straight off the sim, mirroring SimGameEnv's
    own terminal logic (game_env.step): a full-run victory is
    outcome == PLAYER_VICTORY, and the v1 act-1 target is "cleared into act 2"
    (outcome stays UNDECIDED on a boss clear; act > 1 is the real signal). Never
    infer the win from the shaped return."""
    import slaythespire as sts

    gc = env.unwrapped.thick.gc
    if gc.outcome == sts.GameOutcome.PLAYER_VICTORY:
        return True
    return gc.outcome == sts.GameOutcome.UNDECIDED and gc.act > 1


def main() -> None:
    args = build_parser().parse_args()

    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")

    env = SlayTheSpireEnv(ascension=args.ascension)
    model = MaskablePPO.load(args.checkpoint, device=args.device)

    returns: list[float] = []
    lengths: list[int] = []
    wins = 0
    truncations = 0

    for episode_index in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode_index)

        terminated = False
        truncated = False
        episode_return = 0.0
        episode_length = 0

        while not (terminated or truncated):
            action_masks = get_action_masks(env)

            action, _ = model.predict(
                obs,
                action_masks=action_masks,
                deterministic=not args.stochastic,
            )

            action = int(np.asarray(action).item())

            obs, reward, terminated, truncated, info = env.step(action)

            episode_return += float(reward)
            episode_length += 1

        returns.append(episode_return)
        lengths.append(episode_length)
        wins += int(is_win(env))
        truncations += int(truncated)

    returns_array = np.asarray(returns, dtype=np.float64)
    lengths_array = np.asarray(lengths, dtype=np.float64)

    print(
        f"[eval] episodes={args.episodes} "
        f"ascension={args.ascension} "
        f"deterministic={not args.stochastic}"
    )
    print(
        f"[eval] win_rate={wins / args.episodes:.3f} "
        f"wins={wins}/{args.episodes}"
    )
    print(
        f"[eval] raw_return="
        f"{returns_array.mean():.3f} +/- {returns_array.std():.3f}"
    )
    print(
        f"[eval] episode_length="
        f"{lengths_array.mean():.1f} +/- {lengths_array.std():.1f}"
    )

    if truncations:
        print(f"[eval] truncated={truncations}/{args.episodes}")

    env.close()


if __name__ == "__main__":
    main()
