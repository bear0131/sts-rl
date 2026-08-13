"""Watch a MaskablePPO checkpoint play whole-run SlayTheSpireEnv, live.

A checkpoint -> sim rollout connector with a human-readable terminal view: each
step clears the screen and redraws the current game state (env.render()), the
model's probability over every legal action, the chosen action, and the reward.

Examples:

    python -m rl.watch rl/runs/whole-run/.../checkpoints/maskable_ppo_50000_steps.zip
    python -m rl.watch .../final.zip --delay 0.3
    python -m rl.watch .../final.zip --step          # press Enter to advance
    python -m rl.watch .../final.zip --episodes 3 --stochastic
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from env.core.run_env import SlayTheSpireEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

# Import custom model code before loading a checkpoint so the custom extractor
# classes are available to cloudpickle (same reason as eval.py).
from . import model as _model  # noqa: F401
from . import split_policy as _split_policy  # noqa: F401
from .eval import is_win


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Watch a MaskablePPO checkpoint play, live, in the terminal."
    )
    parser.add_argument("checkpoint", help="Path to a MaskablePPO .zip checkpoint.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of deterministic argmax.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.6,
        help="Seconds to pause between steps (ignored with --step).",
    )
    parser.add_argument(
        "--step",
        action="store_true",
        help="Pause for Enter after each step instead of auto-advancing.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=6,
        help="How many legal actions (by probability) to list per step.",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Scroll instead of clearing the screen each step.",
    )
    return parser


def _clear():
    # cls on Windows, ANSI elsewhere; both land the cursor at the top-left.
    if sys.platform == "win32":
        os.system("cls")
    else:
        print("\033[2J\033[H", end="")


def _action_dist(model, obs, action_masks):
    """(probs over N_ACTIONS, value estimate) for the current obs under the masked
    policy, or (None, None) if the policy API can't be coaxed into giving them —
    the viewer then just shows the chosen action without a distribution."""
    import torch

    try:
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_tensor, action_masks=action_masks)
            probs = dist.distribution.probs[0].cpu().numpy()
            value = model.policy.predict_values(obs_tensor)[0].item()
        return probs, value
    except Exception:
        return None, None


def _decision_panel(env, *, header, probs, value, topk, chosen):
    """Render the CURRENT decision point: state, the model's distribution over
    the (current) legal actions, and the chosen one. Must be called BEFORE
    env.step() — every env read here (render/action_masks/describe_action) and
    the probs all have to describe the SAME pre-step state, or they disagree."""
    lines = [header, "=" * 60, env.render(), "-" * 60]

    if value is not None:
        lines.append(f"V(s) = {value:+.3f}")

    if probs is not None:
        legal = np.flatnonzero(env.action_masks())
        ranked = sorted(legal, key=lambda i: probs[i], reverse=True)[:topk]
        lines.append("model:")
        for i in ranked:
            mark = ">" if i == chosen else " "
            lines.append(f" {mark} {probs[i] * 100:5.1f}%  {env.describe_action(i)}")
        if len(legal) > topk:
            lines.append(f"   ... (+{len(legal) - topk} more legal)")

    lines.append(f"--> {env.describe_action(chosen)}")
    return "\n".join(lines)


def watch_episode(model, env, *, seed, stochastic, delay, step_mode, topk, clear,
                  ep_idx, ep_total):
    obs, info = env.reset(seed=seed)
    terminated = truncated = False
    ep_return = 0.0
    n_steps = 0

    while not (terminated or truncated):
        masks = get_action_masks(env)
        probs, value = _action_dist(model, obs, masks)

        action, _ = model.predict(obs, action_masks=masks, deterministic=not stochastic)
        action = int(np.asarray(action).item())

        stats = info.get("run_stats", {})
        header = (
            f"episode {ep_idx}/{ep_total}   step {n_steps}   "
            f"floor {stats.get('floor', '?')}  act {stats.get('act', '?')}   "
            f"return {ep_return:+.2f}"
        )

        # Render the decision point BEFORE stepping — state, probs and the chosen
        # action must all describe this same pre-step state. reward is the only
        # thing known only after the step, so it's appended on its own line.
        panel = _decision_panel(env, header=header, probs=probs, value=value,
                                topk=topk, chosen=action)

        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += float(reward)
        n_steps += 1

        if clear:
            _clear()
        print(panel)
        print(f"    reward {reward:+.3f}")

        if step_mode:
            input("\n[Enter to step] ")
        elif delay > 0:
            time.sleep(delay)

    won = is_win(env)
    print(
        f"\n=== episode {ep_idx} done: "
        f"{'WIN' if won else 'LOSS'}  return {ep_return:+.2f}  steps {n_steps}  "
        f"floor {env.unwrapped.thick.gc.floor_num} ==="
    )
    return won, ep_return


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")

    model = MaskablePPO.load(args.checkpoint, device=args.device)
    env = SlayTheSpireEnv(
        ascension=args.ascension,
        global_map_view=_model.uses_global_map_observation(model),
        card_language=_model.uses_card_language(model),
        combat_language=_model.uses_combat_language(model),
        deck_size_observation=_model.uses_deck_size_observation(model),
        neow_observation=_model.uses_neow_observation(model),
        event_option_observation=_model.uses_event_option_observation(model),
    )

    wins = 0
    for ep in range(args.episodes):
        won, _ = watch_episode(
            model, env,
            seed=args.seed + ep,
            stochastic=args.stochastic,
            delay=args.delay,
            step_mode=args.step,
            topk=args.topk,
            clear=not args.no_clear,
            ep_idx=ep + 1,
            ep_total=args.episodes,
        )
        wins += int(won)
        if args.episodes > 1 and not args.step:
            time.sleep(max(args.delay, 1.0))

    if args.episodes > 1:
        print(f"\n[watch] {wins}/{args.episodes} wins")

    env.close()


if __name__ == "__main__":
    main()
