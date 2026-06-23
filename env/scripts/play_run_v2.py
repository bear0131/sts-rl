"""Interactive whole-run player on the THIN SlayTheSpireEnv (action v2).

Same idea as play_run.py but it drives the model-facing layer: a fixed
Discrete action space + per-step mask, with the autoregressive sub-step exposed.
A targeted card play takes two menu picks (the card, then its target), exactly as
the RL agent experiences it — so this doubles as a faithfulness check of the thin
action codec, the mask, and obs['arg_step'].

Run from repo root (needs the sim built):
    python -m env.scripts.play_run_v2
    python -m env.scripts.play_run_v2 --seed 42
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from env.core.run_env import SlayTheSpireEnv


def _menu(env):
    """(action_id, label) for every legal thin action id right now."""
    mask = env.action_masks()
    return [(i, env.describe_action(i)) for i in np.flatnonzero(mask)]


def _prompt(menu):
    for n, (_aid, label) in enumerate(menu):
        print(f"  {n:>2}: {label}")
    while True:
        raw = input("> action # (q to quit): ").strip().lower()
        if raw in ("q", "quit", "exit"):
            return None
        if raw.isdigit() and 0 <= int(raw) < len(menu):
            return menu[int(raw)][0]
        print(f"  ?? enter 0..{len(menu) - 1} or q")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Play a whole run on the thin SlayTheSpireEnv.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ascension", type=int, default=0)
    args = ap.parse_args(argv)

    env = SlayTheSpireEnv(ascension=args.ascension)
    env.reset(seed=args.seed)
    gc = env.thick.gc
    print(f"\nseed={gc.seed} ascension={args.ascension} — starting run\n")

    total_r, step, last = 0.0, 0, None
    while True:
        info = env._info()
        phase, arg = info["phase"], info["arg_step"]
        tag = f"{phase}/{arg}" if arg != "root" else phase
        if tag != last:
            print(f"\n--- now: {tag} (floor {env.thick.gc.floor_num}) ---")
            last = tag
        print(env.render())

        menu = _menu(env)
        if not menu:
            print("(no legal actions — run is over)")
            break
        choice = _prompt(menu)
        if choice is None:
            print("\n(quit)")
            return

        _obs, r, term, trunc, info = env.step(choice)
        total_r += r
        step += 1
        print(f"\n   -> reward {r:+.3f}  (cumulative {total_r:+.3f})")
        if term or trunc:
            break

    outcome = str(env.thick.gc.outcome).split(".")[-1]
    if outcome == "UNDECIDED" and env.thick.gc.act > 1:
        result = "CLEARED ACT 1"
    elif outcome == "PLAYER_VICTORY":
        result = "VICTORY"
    else:
        result = "DEFEAT"
    print(f"\n=== {result} after {step} actions, floor {env.thick.gc.floor_num}, "
          f"hp {env.thick.gc.cur_hp}, total reward {total_r:+.3f} ===")


if __name__ == "__main__":
    main()
