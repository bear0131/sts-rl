from __future__ import annotations

import numpy as np
import pytest

from env.core import _BUILD  # noqa: F401  (installs the native module path)

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
