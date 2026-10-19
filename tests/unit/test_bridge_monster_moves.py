import pytest

from env.core import vocab as V
from bridge.schemas.combat_state import (
    _REAL_COMBAT_COUNTER_KEYS,
    _MONSTER_SIM2REAL,
    _REAL_MONSTER_MOVES,
    _sim_monster_id,
    normalize_real,
    real_to_sim_monster_move,
)
from bridge.schemas.entity_ids import SIMULATOR_PARITY_EXPORT_VERSION


def _real_combat_frame(monsters):
    return {
        "gold": 99,
        "relics": [],
        "combat_state": {
            "simulator_parity_export": SIMULATOR_PARITY_EXPORT_VERSION,
            "simulator_energy_per_turn": 4,
            "simulator_combat_counters": {
                key: 0 for key in _REAL_COMBAT_COUNTER_KEYS
            },
            "simulator_player_power_overrides": {},
            "turn": 3,
            "cards_discarded_this_turn": 0,
            "times_damaged": 0,
            "player": {
                "current_hp": 80, "max_hp": 80, "block": 0,
                "energy": 4, "powers": [],
            },
            "monsters": monsters,
            "hand": [], "draw_pile": [], "discard_pile": [],
            "exhaust_pile": [],
        },
    }


def test_real_move_codec_covers_every_supported_real_monster():
    assert set(_MONSTER_SIM2REAL.values()) <= set(_REAL_MONSTER_MOVES)
    assert real_to_sim_monster_move("Cultist", 1) == "CULTIST_DARK_STRIKE"
    assert real_to_sim_monster_move("Cultist", 3) == "CULTIST_INCANTATION"
    assert real_to_sim_monster_move("Deca", 0) == "DECA_BEAM"
    assert real_to_sim_monster_move("Cultist", -1) == "INVALID"
    with pytest.raises(KeyError, match="unmapped real monster move"):
        real_to_sim_monster_move("Cultist", 2)


def test_real_relic_sentinels_and_live_flags_are_explicit():
    from bridge.real_game_env import RealGameAdapter
    from bridge.schemas.entity_ids import native_relic_counter
    adapter = RealGameAdapter()
    adapter.gs = {"relics": [
        {"id": "Lizard Tail", "counter": -2}, {"id": "MawBank", "counter": -1}]}
    assert adapter._relic_counters() == [0, 1]
    for value in (-3, 0, 1, True, "-1"):
        with pytest.raises(ValueError, match="counter"):
            native_relic_counter("Lizard Tail", value)
    frame = _real_combat_frame([])
    counters = frame["combat_state"]["simulator_combat_counters"]
    counters.update(pen_nib=9, centennial_puzzle_active=True, lizard_tail_active=False)
    actual = normalize_real(frame)["player"]["combat_counters"]
    assert actual["pen_nib"] == -1
    assert actual["centennial_puzzle_active"] == 1
    assert actual["lizard_tail_active"] == 0
    counters["lizard_tail_active"] = -2
    with pytest.raises(ValueError, match="not boolean"):
        normalize_real(frame)


def test_real_normalizer_preserves_public_move_history():
    gs = _real_combat_frame([{
        "id": "Cultist", "current_hp": 5, "max_hp": 49,
        "block": 0, "is_gone": False, "move_base_damage": 6,
        "move_adjusted_damage": 9, "move_hits": 1,
        "move_id": 1, "last_move_id": 1, "second_last_move_id": 3,
        "intent": "ATTACK", "half_dead": False,
        "current_intent_export": SIMULATOR_PARITY_EXPORT_VERSION,
        "powers": [
            {"id": "Ritual", "amount": 3},
            {"id": "Strength", "amount": 3},
        ],
    }])

    monster = normalize_real(gs)["monsters"][0]
    assert monster["real_move_id"] == 1
    assert monster["real_last_move_id"] == 1
    assert monster["real_second_last_move_id"] == 3
    assert monster["move"] == "CULTIST_DARK_STRIKE"
    assert monster["move_id"] == V.MOVE_VOCAB_INDEX["CULTIST_DARK_STRIKE"]
    assert monster["powers"] == {"Ritual": 3, "Strength": 3}


def test_live_real_monster_move_cannot_be_fabricated_as_padding():
    monster = {
        "id": "Cultist", "current_hp": 5, "max_hp": 49,
        "block": 0, "is_gone": False, "move_base_damage": 6,
        "move_adjusted_damage": 6, "move_hits": 1, "powers": [],
        "intent": "ATTACK",
        "last_move_id": -1, "second_last_move_id": -1,
        "half_dead": False,
        "current_intent_export": SIMULATOR_PARITY_EXPORT_VERSION,
    }
    game_state = _real_combat_frame([monster])
    with pytest.raises(ValueError, match="missing.*move_id"):
        normalize_real(game_state)

    monster.update(
        current_hp=0, is_gone=True, move_id=-1,
        last_move_id=-1, second_last_move_id=-1,
    )
    normalized = normalize_real(game_state)["monsters"][0]
    assert normalized["move"] == "INVALID" and normalized["move_id"] == 0


def test_spire_growth_transport_id_is_not_model_padding():
    assert _MONSTER_SIM2REAL["SPIRE_GROWTH"] == "Serpent"
    assert _sim_monster_id("SPIRE_GROWTH") == "SpireGrowth"
    assert V.MONSTER_VOCAB_INDEX["SpireGrowth"] > 0
