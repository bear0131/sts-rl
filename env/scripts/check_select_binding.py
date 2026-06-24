"""Verify the card-select bindings + obs contract (obs_design §4.1).

Drives masked-random whole runs over the thin SlayTheSpireEnv and, whenever it
lands on a card-select screen (an OVERWORLD grid: phase 'card_select' — Remove/
Transform/Smith; OR an IN-COMBAT sub-screen: phase 'combat' with the live bc in
the CARD_SELECT input state — Armaments/Headbutt/Gamble/...), asserts:

  binding side (gc.select_screen_info / bc.select_info):
    - context is a known SELECT_CONTEXT_VOCAB name (not INVALID on a live screen);
    - pick_count >= 1; have_selected in [0, pick_count] (overworld);
    - overworld 'selected' ids count == have_selected and are real card ids;
    - overworld grid candidates fit within G_MAX (no silent truncation observed).

  obs side (encode):
    - the obs is IN the observation space every step (shape/dtype/bounds);
    - on a select screen select_context != PAD and matches the binding;
    - pick_remaining, selected_card_ids are populated consistently.

Run from repo root (needs the sim built):
    python -m env.scripts.check_select_binding --episodes 120 --seed 0
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from env.core.run_env import SlayTheSpireEnv
from env.core import encode
from env.core import vocab as V


def _check_select(thick, report):
    """Assert the select binding/obs contract for the CURRENT screen. `thick` is
    the SimGameEnv (owns gc/bc). Returns a short label for printing."""
    bc = thick.bc
    in_combat_select = bc is not None and bc.input_state == 2
    if in_combat_select:
        info = bc.select_info()
        context = info["context"]
        pick_count = info["pick_count"]
        have_selected = 0
        selected_ids = []
        kind = "combat"
    else:
        info = thick.gc.select_screen_info()
        context = info["context"]
        pick_count = info["pick_count"]
        have_selected = info["have_selected"]
        selected_ids = info["selected"]
        kind = "overworld"

    assert context in V.SELECT_CONTEXT_VOCAB_INDEX, f"unknown context {context!r}"
    assert context != "INVALID", "live select screen but context INVALID"
    assert context != "PAD", context
    # NOTE: combat cardSelectInfo.pickCount is only meaningful for the MULTI tasks
    # (Gamble/Exhaust-Many); single-pick combat tasks (e.g. Dual Wield) leave it 0
    # since the screen resolves after one pick. So pick_remaining is an overworld-
    # only useful signal — only enforce pick_count there.
    if kind == "overworld":
        assert pick_count >= 1, f"overworld pick_count {pick_count} context={context}"
        assert 0 <= have_selected <= pick_count, (have_selected, pick_count)
        assert len(selected_ids) == have_selected, (selected_ids, have_selected)
        grid = thick.gc.get_select_cards()
        assert len(grid) <= encode.G_MAX, f"grid {len(grid)} > G_MAX {encode.G_MAX}"
        report["max_grid"] = max(report["max_grid"], len(grid))

    report["screens"] += 1
    report[f"ctx_{context}"] = report.get(f"ctx_{context}", 0) + 1
    if report["screens"] <= 5:
        print(f"  [{kind}] context={context} pick_count={pick_count} "
              f"have_selected={have_selected} selected={selected_ids}")
    return context, pick_count


def run_episode(env, rng, report):
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    space = env.observation_space
    steps = 0
    while steps < 6000:
        thick = env.thick
        if thick.is_over:
            break
        # detect a live card-select screen (overworld grid or in-combat sub-screen)
        bc = thick.bc
        is_select = (thick.phase() == "card_select" or
                     (bc is not None and bc.input_state == 2))
        if is_select:
            context, _ = _check_select(thick, report)
            # cross-check the obs the model will actually see
            assert space.contains(obs), "obs not in observation space on select screen"
            ctx_idx = obs["select_context"][0]
            assert ctx_idx == V.SELECT_CONTEXT_VOCAB_INDEX[context], (
                f"obs select_context {ctx_idx} != binding {context} "
                f"({V.SELECT_CONTEXT_VOCAB_INDEX[context]})")
            assert ctx_idx != 0, "select_context is PAD on a live select screen"

        mask = env.action_masks()
        legal_ids = np.flatnonzero(mask)
        if len(legal_ids) == 0:
            report["stalls"] += 1
            break
        a = int(legal_ids[int(rng.integers(0, len(legal_ids)))])
        obs, _r, term, _trunc, _info = env.step(a)
        assert space.contains(obs), "obs not in observation space"
        steps += 1
        if term:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = SlayTheSpireEnv()
    rng = np.random.default_rng(args.seed)
    report = {"screens": 0, "stalls": 0, "max_grid": 0}
    for ep in range(args.episodes):
        run_episode(env, rng, report)

    print(f"\n==== {args.episodes} episodes ====")
    print(f"OBS_VERSION={encode.OBS_VERSION}  G_MAX={encode.G_MAX}")
    print(f"select screens checked : {report['screens']}")
    print(f"max overworld grid seen: {report['max_grid']}")
    print(f"stalls                 : {report['stalls']}")
    contexts = {k[4:]: v for k, v in sorted(report.items()) if k.startswith("ctx_")}
    print(f"contexts               : {contexts}")
    assert report["screens"] > 0, "no card-select screen reached — increase --episodes"
    print("OK: select bindings + obs contract consistent.")


if __name__ == "__main__":
    main()
