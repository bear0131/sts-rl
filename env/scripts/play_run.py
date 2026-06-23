"""Interactive whole-run player — drive a SimGameEnv by hand, across screens.

Requires the sim to be built (sim/sts_lightspeed/build). Run from the repo root:

    python -m env.scripts.play_run
    python -m env.scripts.play_run --seed 42

Each decision point prints the rendered state (combat board OR overworld screen)
and a numbered menu of the LEGAL semantic actions (from the env itself, so every
choice is playable). Type the number to take it, 'q' to quit. Same env / action
vocabulary the RL agent will consume, so it doubles as a sanity check that the
whole-run state machine, semantic action set, and renderer are faithful.

Scope: act 1. Shops (real buys/removes), events, and campfires all enumerate
their true legal options via the sim's own action enumerator, so the menu is
always faithful. The run ends when you die or clear act 1.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from env.core.game_env import SimGameEnv


def _menu(env):
    """List (action, label) for every legal semantic action right now."""
    return [(a, env.describe_action(a)) for a in env.legal_semantic_actions()]


def _prompt_choice(menu):
    """Show the numbered menu, return a chosen action or None to quit."""
    for n, (_a, label) in enumerate(menu):
        print(f"  {n:>2}: {label}")
    while True:
        raw = input("> action # (q to quit): ").strip().lower()
        if raw in ("q", "quit", "exit"):
            return None
        if raw.isdigit() and 0 <= int(raw) < len(menu):
            return menu[int(raw)][0]
        print(f"  ?? enter 0..{len(menu) - 1} or q")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Interactively play a whole SimGameEnv run.")
    ap.add_argument("--seed", type=int, default=None, help="reset seed (reproducible)")
    ap.add_argument("--ascension", type=int, default=0)
    args = ap.parse_args(argv)

    env = SimGameEnv(ascension=args.ascension)
    canon, info = env.reset(seed=args.seed)
    print(f"\nseed={env.gc.seed} ascension={args.ascension} — starting run\n")

    total_r = 0.0
    step = 0
    last_phase = None
    while True:
        phase = env.phase()
        # A blank line + banner when the phase changes keeps the long scroll readable.
        if phase != last_phase:
            print(f"\n--- now: {phase} (floor {env.gc.floor_num}) ---")
            last_phase = phase
        print(env.render())

        menu = _menu(env)
        if not menu:
            print("(no legal actions — run is over)")
            break
        choice = _prompt_choice(menu)
        if choice is None:
            print("\n(quit)")
            return

        canon, r, term, info = env.step(choice)
        total_r += r
        step += 1
        print(f"\n   -> reward {r:+.3f}  (cumulative {total_r:+.3f})")
        if term:
            break

    # terminal summary
    outcome = str(env.gc.outcome).split(".")[-1]
    if outcome == "UNDECIDED" and env.gc.act > 1:
        result = "CLEARED ACT 1"
    elif outcome == "PLAYER_VICTORY":
        result = "VICTORY"
    else:
        result = "DEFEAT"
    print(f"\n=== {result} after {step} actions, floor {env.gc.floor_num}, "
          f"hp {env.gc.cur_hp}, total reward {total_r:+.3f} ===")


if __name__ == "__main__":
    main()
