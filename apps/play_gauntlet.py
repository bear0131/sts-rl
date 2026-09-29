"""Play the COMBAT GAUNTLET (manually, or watch a checkpoint), model action API.

Same idea as env/scripts/play_run_v2.py (thin Discrete action space + per-step
mask, the autoregressive sub-step exposed) but on the env rl/train_combat.py
pretrains on: SimCombatBackend driven by random_gauntlet_factory — ONE random
Ironclad loadout fighting an ENDLESS stream of random encounters, HP carried over
with no healing, the gauntlet ending only on DEATH. So this answers "how many
fights can you clear in this config", on the same menu the policy sees.
Score = combats_won.

By default YOU play (type the action #). Pass --watch <ckpt> to instead let a
MaskablePPO checkpoint drive the same gauntlet autoregressively.

Run from repo root (needs the sim built):
    python -m apps.play_gauntlet
    python -m apps.play_gauntlet --seed 42 --floor-lo 1 --floor-hi 50
    python -m apps.play_gauntlet --watch runs/combat/.../final.zip --delay 0.4
"""

import argparse
import time

from env.core import native_actions as C
import numpy as np

from env.core import factories
from env.core.run_env import SlayTheSpireEnv
from env.core.combat_backend import SimCombatBackend
from rl.policies import model as _model


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
    ap = argparse.ArgumentParser(description="Play the combat gauntlet (train_combat config).")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--floor-lo", type=int, default=1)
    ap.add_argument("--floor-hi", type=int, default=50)
    ap.add_argument("--watch", default=None,
                    help="MaskablePPO .zip to let the model play instead of you.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--stochastic", action="store_true",
                    help="--watch only: sample actions instead of deterministic argmax.")
    ap.add_argument("--delay", type=float, default=0.4,
                    help="--watch only: seconds to pause between steps.")
    args = ap.parse_args(argv)

    model = None
    if args.watch is not None:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import get_action_masks
        # Import the policy class before load so cloudpickle can find it.
        from rl.policies import pointer_policy as _pp  # noqa: F401
        from rl.policies import split_policy as _split_policy  # noqa: F401
        model = MaskablePPO.load(args.watch, device=args.device)

    def _builder(np_random):
        return factories.random_gauntlet_factory(
            np_random, ascension=args.ascension,
            floor_range=(args.floor_lo, args.floor_hi),
        )

    env = SlayTheSpireEnv(
        combat_action_contract=(C.policy_contract(model.observation_space) if model is not None else "legacy"),
        backend_factory=lambda: SimCombatBackend(_builder, ascension=args.ascension),
        card_language=(
            _model.uses_card_language(model) if model is not None else False
        ),
        combat_language=(
            _model.uses_combat_language(model) if model is not None else False
        ),
        deck_size_observation=(
            _model.uses_deck_size_observation(model)
            if model is not None else False
        ),
    )
    obs, _ = env.reset(seed=args.seed)
    # The episode's fixed level L (gauntlet has no map; gc.floor_num is L) sets the
    # loadout strength and which act's encounter pool the fights are drawn from.
    L = env.thick.gc.floor_num
    who = f"model={args.watch}" if model is not None else "manual"
    print(f"\nseed={args.seed} ascension={args.ascension} level L={L} [{who}] -- gauntlet start\n")

    total_r, step, last, prev_won = 0.0, 0, None, 0
    while True:
        info = env._info()
        phase, arg = info["phase"], info["arg_step"]
        tag = f"{phase}/{arg}" if arg != "root" else phase
        if tag != last:
            print(f"\n--- now: {tag} (fight #{env.thick.combats_won + 1}) ---")
            last = tag
        print(env.render())

        menu = _menu(env)
        if not menu:
            print("(no legal actions -- gauntlet is over)")
            break

        if model is None:
            choice = _prompt(menu)
            if choice is None:
                print("\n(quit)")
                break
        else:
            from sb3_contrib.common.maskable.utils import get_action_masks
            masks = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=masks,
                                      deterministic=not args.stochastic)
            choice = int(np.asarray(action).item())
            print(f"> model picks: {env.describe_action(choice)}")
            if args.delay > 0:
                time.sleep(args.delay)

        obs, r, term, trunc, info = env.step(choice)
        total_r += r
        step += 1
        print(f"\n   -> reward {r:+.3f}  (cumulative {total_r:+.3f})")

        if env.thick.combats_won > prev_won:
            prev_won = env.thick.combats_won
            hp, _ = env.thick.combat_state()
            print(f"   *** cleared fight #{prev_won}  (hp {hp}) ***")

        if term or trunc:
            break

    print(f"\n=== gauntlet over: cleared {env.thick.combats_won} fights "
          f"(level L={L}, {step} actions, total reward {total_r:+.3f}) ===")


if __name__ == "__main__":
    main()
