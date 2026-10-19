"""Interactive combat player — drive a SimCombatEnv by hand.

Requires the sim to be built (sim/sts_lightspeed/build). Run from the repo root:

    python -m env.scripts.play
    python -m env.scripts.play --pool elite --seed 42
    python -m env.scripts.play --encounter "Cultist" --seed 7

Each turn it prints the rendered state and a numbered menu of the LEGAL actions
(built from the env's action mask, so every choice is guaranteed playable). Type
the number to take that action, 'q' to quit. This is a human harness over the
exact same env/encoder/action-space the RL agent uses, so it doubles as a sanity
check that the action codec and state render are faithful.
"""

import argparse
import sys

import numpy as np

# Allow running as a script (python env/scripts/play.py) as well as a module.
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from env.core.combat_env import SimCombatEnv
from env.core import factories


def _build_env(args):
    if args.encounter:
        factory = factories.fixed_factory(
            seed=args.seed if args.seed is not None else 0,
            encounter=args.encounter,
            floor=args.floor,
            ascension=args.ascension,
        )
        return SimCombatEnv(gc_factory=factory, render_mode="ansi")
    return SimCombatEnv(pool=args.pool, ascension=args.ascension, render_mode="ansi")


def _legal_menu(env, info):
    """List (action_id, label) for every legal action in the current state."""
    mask = info["action_mask"]
    return [(aid, env.describe_action(aid)) for aid in np.flatnonzero(mask)]


def _prompt_choice(menu):
    """Show the numbered menu, return a chosen action_id or None to quit."""
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
    ap = argparse.ArgumentParser(description="Interactively play a SimCombatEnv combat.")
    ap.add_argument("--pool", default="normal", choices=["normal", "elite", "boss"],
                    help="random act-1 pool to draw from (ignored if --encounter set)")
    ap.add_argument("--encounter", default=None,
                    help="force a specific encounter name (e.g. 'Cultist'); see env/core/encounters.py")
    ap.add_argument("--seed", type=int, default=None, help="reset seed (reproducible)")
    ap.add_argument("--floor", type=int, default=1, help="floor for --encounter mode (perturbs combat RNG)")
    ap.add_argument("--ascension", type=int, default=0)
    args = ap.parse_args(argv)

    env = _build_env(args)
    obs, info = env.reset(seed=args.seed)
    print(f"\nEncounter: {env.encounter}\n")

    total_r = 0.0
    step = 0
    while True:
        print(env.render())
        menu = _legal_menu(env, info)
        choice = _prompt_choice(menu)
        if choice is None:
            print("\n(quit)")
            return
        obs, r, term, trunc, info = env.step(choice)
        total_r += r
        step += 1
        print(f"\n   -> reward {r:+.3f}  (cumulative {total_r:+.3f})\n")
        if term or trunc:
            print(env.render())
            outcome = env.bc.to_dict()["outcome"]
            result = {1: "VICTORY", 2: "DEFEAT", 3: "ESCAPE"}.get(outcome, "UNDECIDED")
            print(f"\n=== {result} after {step} actions, total reward {total_r:+.3f} ===")
            return


if __name__ == "__main__":
    main()
