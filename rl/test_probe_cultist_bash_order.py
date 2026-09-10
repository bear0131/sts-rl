from __future__ import annotations

import numpy as np

from env.core.ironclad_cards import ATTACK, IRONCLAD_CARDS
from rl.probe_cultist_bash_order import (
    ATTACK_CARD_IDS,
    BASH_ID,
    make_observation,
    summarize,
)


def test_all_non_bash_ironclad_attacks_are_included():
    expected = {
        card_id
        for card_id, info in IRONCLAD_CARDS.items()
        if info.kind == ATTACK and card_id != BASH_ID
    }
    assert set(ATTACK_CARD_IDS) == expected
    assert len(ATTACK_CARD_IDS) == 31


def test_probe_observation_has_only_two_deck_and_hand_cards_without_vulnerable():
    observation = make_observation(("Dropkick", BASH_ID), energy=6)
    reversed_hand = make_observation((BASH_ID, "Dropkick"), energy=6)

    assert np.count_nonzero(observation["hand_ids"]) == 2
    assert np.count_nonzero(observation["deck_ids"]) == 2
    np.testing.assert_array_equal(observation["deck_ids"], reversed_hand["deck_ids"])
    assert not np.array_equal(observation["hand_ids"], reversed_hand["hand_ids"])
    assert observation["hand_costs"][0] == 0.1
    assert observation["hand_costs"][1] == 0.2
    assert not observation["monster_buffs"].any()


def test_summary_averages_both_hand_layouts_and_conditions_on_playing():
    rows = [
        {
            "attack_id": "Strike_R",
            "energy": 6,
            "p_end_turn": 0.1,
            "p_attack": 0.3,
            "p_bash": 0.6,
            "deterministic": "bash",
        },
        {
            "attack_id": "Strike_R",
            "energy": 6,
            "p_end_turn": 0.3,
            "p_attack": 0.3,
            "p_bash": 0.4,
            "deterministic": "bash",
        },
    ]

    result = summarize(rows)[0]

    assert result["p_end_turn"] == 0.2
    assert result["p_attack"] == 0.3
    assert result["p_bash"] == 0.5
    assert result["p_bash_given_play"] == 0.625
    assert result["first_play"] == "bash"
