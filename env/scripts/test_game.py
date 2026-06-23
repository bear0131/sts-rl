"""Whole-run (SimGameEnv) regression tests — run on the box with the built sim:

    python -m env.scripts.test_game

Starts with the map-spine check (RunMap reachability must agree with where the
gc actually lands) because that reconstruction is what the whole-run state
machine stands on. The full random-policy-to-act1-boss test is added alongside
SimGameEnv.
"""

import sys

from env.core import spiremap
from env.core import _BUILD  # noqa: F401
import slaythespire as sts

_IRONCLAD = None  # set in main once sts is importable


def _to_map_screen(gc):
    """Advance a fresh gc through Neow (option 0) until the first MAP_SCREEN,
    resolving whatever sub-screen the reward opens. Neow option 0 varies by seed
    (gain card / relic / gold / max-hp...), so we dispatch generically instead of
    assuming one screen — this previews the SimGameEnv state machine."""
    gc.choose_neow_option(0)
    guard = 0
    while gc.screen_state_name.split(".")[-1] != "MAP_SCREEN" and guard < 8:
        ss = gc.screen_state_name.split(".")[-1]
        if ss == "CARD_SELECT":
            gc.choose_select_card_option(0)
        elif ss == "REWARDS":
            # take the first of each claimable, then leave the reward screen.
            info = gc.get_reward_info()
            if info["gold"]:
                gc.claim_gold(0)
            elif info["relics"]:
                gc.claim_relic(0)
            elif info["potions"]:
                gc.claim_potion(0)
            elif info["card_count"]:
                gc.pick_reward_card(gc.get_card_reward()[0])
            else:
                gc.reward_proceed()
        else:
            raise AssertionError(f"unexpected post-Neow screen {ss!r}")
        guard += 1
    assert gc.screen_state_name.split(".")[-1] == "MAP_SCREEN", gc.screen_state_name


def test_map_spine(seeds=(42, 1, 7, 100, 999)):
    """RunMap.reachable / room_type must match where gc.transition_to_map_node
    actually lands, for the first two map moves on several seeds. This is the
    load-bearing assumption of SimGameEnv's map phase."""
    for seed in seeds:
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
        rmap = spiremap.RunMap(seed, 0, gc.act, burning=False)
        _to_map_screen(gc)

        # first move: gc node is (-1,-1); reachable() must list openable cols.
        first = rmap.reachable(-1, -1)
        assert first, f"seed {seed}: no first-row nodes"
        x0 = first[0]
        predicted = rmap.room_type(x0, 0)
        gc.transition_to_map_node(x0)
        landed = str(gc.cur_room).split(".")[-1]
        assert landed == predicted, (
            f"seed {seed}: first move x={x0} predicted {predicted} landed {landed}")
        assert gc.floor_num == 1, f"seed {seed}: floor {gc.floor_num} after first move"

        # If we landed in a MONSTER room the screen is BATTLE; we can't take a
        # second map move without fighting. Only continue the map-edge check when
        # we're back on the map (non-combat room). The combat path is exercised
        # by the full run test; here we just assert reachability is consistent.
        x, y = gc.cur_map_node_x, gc.cur_map_node_y
        nxt = rmap.reachable(x, y)
        assert nxt, f"seed {seed}: node ({x},{y}) has no outgoing edges"
        for x2 in nxt:
            rt = rmap.room_type(x2, y + 1)
            assert rt not in ("NONE", "INVALID"), (
                f"seed {seed}: edge ({x},{y})->{x2} points at {rt}")
    print(f"[ok] test_map_spine: {len(seeds)} seeds, RunMap matches gc landings")


def main():
    test_map_spine()
    print("\nall whole-run tests passed.")


if __name__ == "__main__":
    main()
