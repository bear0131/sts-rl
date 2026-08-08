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
from env.core.game_env import SimGameEnv
from env.core import game_actions as ga

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


def test_boss_transition_is_dag_choice(seed=0):
    rmap = spiremap.RunMap(seed, 0, 1, burning=False)
    x = next(
        col for col in range(spiremap.N_COLS)
        if rmap.room_type(col, spiremap.MAX_Y) not in ("NONE", "INVALID")
    )
    view = rmap.full_view(x, spiremap.MAX_Y)
    chosen = [node for node in view if node["choice"]]
    assert [(node["x"], node["y"]) for node in chosen] == [
        (x, spiremap.MAX_Y)
    ]
    print("[ok] test_boss_transition_is_dag_choice")


def _run_episode(seed, max_steps=2000, verbose=False):
    """Drive one whole run with a random legal policy. Returns a summary dict.
    Asserts the invariants that must hold at every decision point."""
    env = SimGameEnv(ascension=0)
    canon, info = env.reset(seed=seed)

    import random
    rnd = random.Random(seed)

    saw_card_reward = False     # did a card-reward screen ever appear?
    deck_before_pick = None     # deck size right before the first reward-card pick
    deck_after_pick = None      # ... and right after that same pick
    phases_seen = set()
    steps = 0
    while not env.is_over and steps < max_steps:
        legal = env.legal_semantic_actions()
        assert legal, f"seed {seed}: no legal actions at phase {env.phase()!r} " \
                      f"floor {env.gc.floor_num} (not terminal)"
        phases_seen.add(env.phase())

        # Track that picking a reward card actually grows the deck.
        if env.phase() == "combat_reward":
            info_r = env.gc.get_reward_info()
            if info_r["card_count"]:
                saw_card_reward = True

        action = rnd.choice(legal)
        # Snapshot deck size around the FIRST reward-card pick (same action).
        take_card = action[0] == ga.REWARD_CARD and deck_before_pick is None
        if take_card:
            deck_before_pick = len(env.gc.deck_ids())

        canon, reward, terminated, info = env.step(action)
        steps += 1

        if take_card:
            deck_after_pick = len(env.gc.deck_ids())

        if verbose and steps % 20 == 0:
            print(f"  seed {seed}: step {steps} floor {env.gc.floor_num} "
                  f"phase {env.phase()} hp {env.gc.cur_hp}")

    assert env.is_over or steps < max_steps, f"seed {seed}: hit step cap"
    return {
        "seed": seed,
        "steps": steps,
        "final_floor": env.gc.floor_num,
        "final_act": env.gc.act,
        "outcome": str(env.gc.outcome).split(".")[-1],
        "hp": env.gc.cur_hp,
        "saw_card_reward": saw_card_reward,
        "deck_before_pick": deck_before_pick,
        "deck_after_pick": deck_after_pick,
        "phases": phases_seen,
    }


def test_full_run(seeds=(7, 42, 1, 100, 999, 2024)):
    """Random legal policy must drive each run to a real terminal (player died,
    or cleared act 1 into act 2) without stalling, and the deck must grow when a
    reward card is taken (proves we can train cards beyond the starter deck)."""
    any_grew = False
    any_combat = False
    for seed in seeds:
        s = _run_episode(seed)
        # terminal: either died (outcome != UNDECIDED) or advanced past act 1
        assert s["outcome"] != "UNDECIDED" or s["final_act"] > 1, (
            f"seed {seed}: ended non-terminal {s}")
        assert "combat" in s["phases"], f"seed {seed}: never entered combat"
        any_combat = True
        if s["deck_before_pick"] is not None and s["deck_after_pick"] is not None:
            assert s["deck_after_pick"] > s["deck_before_pick"], (
                f"seed {seed}: deck did not grow on reward card "
                f"({s['deck_before_pick']} -> {s['deck_after_pick']})")
            any_grew = True
        print(f"  seed {seed:>4}: {s['steps']:>4} steps -> floor {s['final_floor']:>2} "
              f"act {s['final_act']} {s['outcome']} hp {s['hp']} "
              f"phases={sorted(s['phases'])}")
    assert any_combat, "no run ever fought"
    assert any_grew, "no run ever grew its deck via a reward card"
    print(f"[ok] test_full_run: {len(seeds)} runs to terminal, deck growth confirmed")


def main():
    test_map_spine()
    test_boss_transition_is_dag_choice()
    print()
    test_full_run()
    print("\nall whole-run tests passed.")


if __name__ == "__main__":
    main()
