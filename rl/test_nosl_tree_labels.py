from __future__ import annotations

import numpy as np
import pytest

from bridge.schemas.combat_state import normalize_sim
from env.core import _BUILD  # noqa: F401  (installs the native module path)
from env.core import encode
from env.core import action_space as A
from env.core.run_env import SlayTheSpireEnv

sts = pytest.importorskip("slaythespire")


def test_unvisited_edge_q_uses_parent_value_minus_fpu_reduction():
    game = sts.GameContext(sts.CharacterClass.IRONCLAD, 8, 0)
    battle = sts.create_battle(game, "Jaw Worm")
    root_logits = np.zeros(200, dtype=np.float64)
    session = battle.nosl_neural_search_begin(
        root_logits,
        simulations=1,
        depth=1,
        c_puct=1.5,
        nonce=123,
        root_value=0.37,
    )

    requests = session.next_batch(1)
    assert len(requests) == 1
    session.backprop(
        requests[0]["request_id"],
        0.81,
        np.zeros(200, dtype=np.float64),
    )

    root = session.tree_snapshot([], 64)
    visited = [edge for edge in root["children"] if edge["visits"] == 1]
    unvisited = [edge for edge in root["children"] if edge["visits"] == 0]
    assert len(visited) == 1
    assert unvisited
    assert visited[0]["value"] == pytest.approx(0.81)
    assert all(edge["value"] == pytest.approx(0.36) for edge in unvisited)


def test_armaments_tree_uses_hypothetical_hand_labels():
    # Seed 8 draws Armaments and Anger together from this small controlled
    # deck.  The high Armaments root prior forces the relevant branch to be
    # visited; later levels must describe the upgraded Anger in that branch.
    deck = [
        ("ARMAMENTS", 0),
        ("ANGER", 0),
        ("STRIKE_RED", 0),
        ("STRIKE_RED", 0),
        ("STRIKE_RED", 0),
        ("DEFEND_RED", 0),
        ("DEFEND_RED", 0),
        ("DEFEND_RED", 0),
        ("BASH", 0),
        ("CLASH", 0),
    ]
    game = sts.GameContext(sts.CharacterClass.IRONCLAD, 8, 0)
    game.set_deck(deck)
    battle = sts.create_battle(game, "Jaw Worm")
    assert [card[0] for card in battle.to_dict()["hand"]][:2] == [
        "ANGER",
        "ARMAMENTS",
    ]

    root_logits = np.zeros(200, dtype=np.float64)
    root_logits[2] = 10.0  # thin action: play Armaments from hand slot 1
    session = battle.nosl_neural_search_begin(
        root_logits, simulations=64, depth=0, c_puct=1.5, nonce=123
    )
    leaf_logits = np.zeros(200, dtype=np.float64)
    while not session.done():
        for request in session.next_batch(16):
            session.backprop(request["request_id"], 0.8, leaf_logits)

    root = session.tree_snapshot([], 64)
    armaments = next(
        edge for edge in root["children"] if edge["action_bits"] == 1
    )
    assert armaments["visits"] == 64

    prompt = session.tree_snapshot([armaments["edge_index"]], 64)
    anger_pick = next(
        edge for edge in prompt["children"] if "Anger" in edge["label"]
    )
    assert "ARMAMENTS" in anger_pick["label"]

    upgraded = session.tree_snapshot(
        [armaments["edge_index"], anger_pick["edge_index"]], 64
    )
    assert any("Anger+" in edge["label"] for edge in upgraded["children"])


def test_decision_value_is_propagated_max_and_keeps_sample_mean():
    game = sts.GameContext(sts.CharacterClass.IRONCLAD, 8, 0)
    battle = sts.create_battle(game, "Jaw Worm")
    session = battle.nosl_neural_search_begin(
        np.zeros(200, dtype=np.float64),
        simulations=12,
        depth=1,
        c_puct=100.0,
        nonce=123,
        root_value=0.2,
        gumbel_interior_value_mode=1,
    )
    while not session.done():
        for request in session.next_batch(1):
            value = float(request["state"]["player"]["hp"]) / 200.0
            session.backprop(
                request["request_id"], value, np.zeros(200, dtype=np.float64)
            )

    root = session.tree_snapshot([], 64)
    visited = [edge for edge in root["children"] if edge["visits"] > 0]
    assert visited
    assert root["value"] == pytest.approx(max(edge["value"] for edge in visited))
    assert root["sample_value"] != pytest.approx(root["value"])


def test_action_edge_marks_multiple_nosl_world_lines_as_chance_average():
    game = sts.GameContext(sts.CharacterClass.IRONCLAD, 8, 0)
    battle = sts.create_battle(game, "Jaw Worm")
    root_logits = np.zeros(200, dtype=np.float64)
    root_logits[0] = 10.0  # End Turn, whose hidden continuation is random.
    session = battle.nosl_neural_search_begin(
        root_logits,
        simulations=32,
        depth=2,
        c_puct=100.0,
        nonce=123,
        root_value=0.2,
        gumbel_interior_value_mode=1,
    )
    while not session.done():
        for request in session.next_batch(8):
            value = float(request["state"]["player"]["hp"]) / 200.0
            session.backprop(
                request["request_id"], value, np.zeros(200, dtype=np.float64)
            )

    root = session.tree_snapshot([], 64)
    end_turn = next(edge for edge in root["children"] if edge["action_bits"] >> 29 == 4)
    assert end_turn["child_states"] > 1
    assert end_turn["random_split"] is True


def test_end_turn_value_boundary_is_fully_settled_and_non_random():
    game = sts.GameContext(sts.CharacterClass.IRONCLAD, 8, 0)
    battle = sts.create_battle(game, "Jaw Worm")
    before = battle.to_dict()

    boundary = battle.preview_end_turn_boundary(4 << 29)
    assert boundary is not None
    assert boundary["post_enemy_turn_boundary"] is True
    # The preview must execute queued enemy attacks before exposing the leaf.
    assert boundary["player"]["hp"] < before["player"]["hp"]
    # Expired action-state fields cannot leak the chosen play order into value.
    assert boundary["player"]["block"] == 0
    assert boundary["player"]["energy"] == 0
    counters = boundary["player"]["combat_counters"]
    assert counters["cards_played"] == 0
    assert counters["attacks_played"] == 0
    assert counters["skills_played"] == 0
    assert counters["cards_discarded"] == 0
    # The next intent is sampled beyond this boundary and must stay hidden.
    assert all(monster["move"] == "INVALID" for monster in boundary["monsters"])
    assert all(monster["base_damage"] == 0 for monster in boundary["monsters"])

    # Previewing is read-only, and the corresponding normal transition reaches
    # the same post-enemy HP before performing draw/recharge setup.
    assert battle.to_dict() == before
    live = battle.clone()
    live.execute(sts.Action(sts.ActionType.END_TURN))
    live_state = live.to_dict()
    assert live_state["player"]["hp"] == boundary["player"]["hp"]
    assert live_state["post_enemy_turn_boundary"] is False
    assert len(boundary["hand"]) == 0
    assert len(live_state["hand"]) > 0


def test_search_end_turn_leaf_uses_stable_boundary_state():
    game = sts.GameContext(sts.CharacterClass.IRONCLAD, 8, 0)
    battle = sts.create_battle(game, "Jaw Worm")
    root_hp = battle.to_dict()["player"]["hp"]
    logits = np.full(200, -100.0, dtype=np.float64)
    logits[0] = 100.0
    session = battle.nosl_neural_search_begin(
        logits,
        simulations=1,
        depth=0,
        c_puct=1.5,
        nonce=123,
        end_turn_as_leaf=True,
    )

    requests = session.next_batch(1)
    assert len(requests) == 1
    leaf = requests[0]
    assert leaf["needs_policy"] is False
    state = leaf["state"]
    assert state["post_enemy_turn_boundary"] is True
    assert state["player"]["hp"] < root_hp
    assert state["player"]["block"] == 0
    assert state["player"]["combat_counters"]["cards_played"] == 0


def test_boundary_observation_changes_only_enemy_hp_for_hp_only_input_change():
    game = sts.GameContext(sts.CharacterClass.IRONCLAD, 8, 0)
    low = sts.create_battle(game, "Jaw Worm")
    high = low.clone()
    low.apply_custom_state({"monsters": [{"hp": 30}]})
    high.apply_custom_state({"monsters": [{"hp": 40}]})

    low_raw = low.preview_end_turn_boundary(4 << 29)
    high_raw = high.preview_end_turn_boundary(4 << 29)
    low_obs = encode.encode_combat(normalize_sim(low_raw))
    high_obs = encode.encode_combat(normalize_sim(high_raw))

    changed = []
    for key in low_obs:
        difference = np.argwhere(np.asarray(low_obs[key]) != high_obs[key])
        if difference.size:
            changed.append((key, difference.tolist()))
    assert changed == [("monster_scalars", [[0, 1], [0, 2]])]


def test_native_end_turn_step_returns_value_boundary_observation():
    env = SlayTheSpireEnv()
    env.reset(seed=7)
    env.thick.bc = sts.create_battle(env.thick.gc, "Jaw Worm")
    env.thick._prev_hp = env.thick.gc.cur_hp
    env._invalidate_backend_cache()

    before_hp = env.thick.bc.to_dict()["player"]["hp"]
    end_turn_bits = 4 << 29
    env._nosl_cpp_pending = (A.END_TURN, end_turn_bits)
    _obs, _reward, _terminated, _truncated, info = env.step(A.END_TURN)

    boundary = info["nosl_value_boundary_observation"]
    assert boundary["post_enemy_turn_boundary"].tolist() == [1]
    assert boundary["player_scalars"][0] < min(before_hp, 200) / 200.0
