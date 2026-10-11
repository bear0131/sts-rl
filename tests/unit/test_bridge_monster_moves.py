import pytest

from bridge.schemas.combat_state import (
    _MONSTER_SIM2REAL,
    _REAL_MONSTER_MOVES,
    normalize_real,
    real_to_sim_monster_move,
)


def test_real_move_codec_covers_every_supported_real_monster():
    assert set(_MONSTER_SIM2REAL.values()) <= set(_REAL_MONSTER_MOVES)
    assert real_to_sim_monster_move("Cultist", 1) == "CULTIST_DARK_STRIKE"
    assert real_to_sim_monster_move("Cultist", 3) == "CULTIST_INCANTATION"
    assert real_to_sim_monster_move("Deca", 0) == "DECA_BEAM"
    assert real_to_sim_monster_move("Cultist", -1) == "INVALID"
    with pytest.raises(KeyError, match="unmapped real monster move"):
        real_to_sim_monster_move("Cultist", 2)


def test_real_normalizer_preserves_public_move_history():
    gs = {
        "gold": 99,
        "combat_state": {
            "turn": 3,
            "cards_discarded_this_turn": 0,
            "times_damaged": 0,
            "player": {
                "current_hp": 80, "max_hp": 80, "block": 0,
                "energy": 4, "powers": [],
            },
            "monsters": [{
                "id": "Cultist", "current_hp": 5, "max_hp": 49,
                "block": 0, "is_gone": False, "move_base_damage": 6,
                "move_adjusted_damage": 9, "move_hits": 1,
                "move_id": 1, "last_move_id": 1, "second_last_move_id": 3,
                "intent": "ATTACK",
                "powers": [
                    {"id": "Ritual", "amount": 3},
                    {"id": "Strength", "amount": 3},
                ],
            }],
            "hand": [], "draw_pile": [], "discard_pile": [], "exhaust_pile": [],
        },
    }

    monster = normalize_real(gs)["monsters"][0]
    assert monster["real_move_id"] == 1
    assert monster["real_last_move_id"] == 1
    assert monster["real_second_last_move_id"] == 3
    assert monster["powers"] == {"Ritual": 3, "Strength": 3}
