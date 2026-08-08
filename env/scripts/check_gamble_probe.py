"""Forced Gamble (combat MULTI card-select) probe.

The natural masked-random runs in check_select_binding reliably hit EXHAUST_MANY
(Purity etc.) but GAMBLE needs the Gambling Chip relic / Gambler's Brew potion,
which are rare in random act-1 play. This script FORCES a Gamble screen so the
thin layer's monotonic select+confirm machine is exercised on BOTH multi tasks.

Recipe: build an Ironclad GameContext, inject Gambling Chip via gc.set_relics
(keeping Burning Blood), create_battle a basic encounter. Gambling Chip fires
GambleAction at turn start -> a MULTI_CARD_SELECT screen over the opening hand.
We then drive the thin SlayTheSpireEnv's multi codec directly against that bc to
assert: selections light select_hand_mask, become masked, drop pick_remaining,
and confirm hands
the bitset to the sim (chooseGambleCards) so the chosen cards leave hand.

Run from repo root (needs the sim built):
    python -m env.scripts.check_gamble_probe
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from env.core import _BUILD  # noqa: F401
import slaythespire as sts

from env.core.run_env import SlayTheSpireEnv
from env.core.game_env import SimGameEnv
from env.core import action_space as A
from env.core import encode
from schemas.combat_state import normalize_sim


class _GambleBackend(SimGameEnv):
    """SimGameEnv whose first battle is forced to a Gambling-Chip combat, so the
    very first decision is a GAMBLE MULTI card-select over the opening hand."""

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        # inject Gambling Chip alongside Burning Blood, then drop straight into a
        # combat so the relic fires GambleAction at turn start.
        self.gc.set_relics(["Burning Blood", "Gambling Chip"])
        self.bc = sts.create_battle(self.gc, "Cultist")
        return self._canonical(), self._info()


def main():
    env = SlayTheSpireEnv(backend_factory=lambda: _GambleBackend())
    obs, info = env.reset(seed=7)
    space = env.observation_space

    thick = env.thick
    desc = thick.select_descriptor()
    assert desc is not None and desc.get("multi"), f"expected a MULTI screen, got {desc}"
    assert desc["context"] == "GAMBLE", desc["context"]
    n = int(desc["cards_in_hand"])
    cap = int(desc["pick_max"])
    print(f"GAMBLE screen: cards_in_hand={n} pick_max={cap} "
          f"can_pick_zero={desc['can_pick_zero']}")
    assert cap == n, f"Gamble cap should equal hand size: cap={cap} n={n}"

    # obs/mask sanity at landing
    assert space.contains(obs), "obs not in space on GAMBLE screen"
    assert obs["select_context"][0] != 0, "select_context PAD on GAMBLE"
    mask = env.action_masks()
    assert mask[A._CONFIRM], "CONFIRM not legal on GAMBLE"
    assert all(mask[A._PICK + h] for h in range(n)), "not all hand cards selectable"

    hand_before = [c["id"] for c in normalize_sim(thick.bc.to_dict())["hand"]]
    print(f"opening hand: {hand_before}")

    # Select cards 0 and 2 (if present), checking obs and monotonic masks each step.
    pick = [h for h in (0, 2) if h < n]
    for h in pick:
        obs, _r, _t, _tr, _i = env.step(A._PICK + h)
        assert obs["select_hand_mask"][h] == 1.0, f"selection {h} not in select_hand_mask"
        assert not env.action_masks()[A._PICK + h], f"selected card {h} remained legal"
    rem = float(obs["pick_remaining"][0]) * encode.PICK_SCALE
    assert abs(rem - (cap - len(pick))) < 0.5, (rem, cap, len(pick))
    print(f"after selecting {pick}: select_hand_mask="
          f"{[int(x) for x in obs['select_hand_mask'][:n]]} pick_remaining={rem}")

    discard_before = len(thick.bc.to_dict()["discard"])
    obs, _r, term, _tr, _i = env.step(A._CONFIRM)
    # screen dismissed; Gamble discards the chosen cards then draws that many.
    assert thick.bc is None or thick.bc.input_state != 2, "still on GAMBLE after confirm"
    assert space.contains(obs), "obs not in space after confirm"
    if thick.bc is not None and not thick.bc.is_battle_over:
        d = normalize_sim(thick.bc.to_dict())
        hand_after = [c["id"] for c in d["hand"]]
        discard_after = len(d["discard"])
        print(f"after confirm: hand={hand_after} discard {discard_before}->{discard_after}")
        assert discard_after >= discard_before + len(pick), (
            f"Gamble should discard {len(pick)} cards: "
            f"{discard_before}->{discard_after}")

    print(f"\nOBS_VERSION={encode.OBS_VERSION} ACTION_VERSION={A.ACTION_VERSION} "
          f"N_ACTIONS={A.N_ACTIONS}")
    print("OK: forced GAMBLE monotonic select+confirm drives the sim correctly.")


if __name__ == "__main__":
    main()
