"""Regression coverage for real-game combat root state injection."""

from copy import deepcopy

import numpy as np
import pytest

from env.core import native_actions as C
from env.core import game_actions as ga
from bridge.real_game_env import RealGameAdapter
from bridge.search_bridge import RealCombatSearch, _card_rows, _find_shadow_battle
from bridge.schemas.combat_state import (
    _REAL_MONSTER_MOVES,
    _monster_real,
    _monster_powers_real,
    _player_powers_real,
    diff,
    normalize_sim,
)
import slaythespire as sts


def _powers(rows):
    return {str(name): int(amount) for name, amount in rows}


def test_custom_state_replaces_public_powers_and_preserves_signed_stats(combat_factory):
    env = combat_factory(
        ["FLEX", "BASH", "STRIKE_RED"],
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    battle = env.thick.bc
    battle.apply_custom_state({
        "player": {"powers": [("STRENGTH", 5), ("VULNERABLE", 2)]},
        "monsters": [{"powers": [("STRENGTH", 4), ("WEAK", 2)]}],
    })
    battle.apply_custom_state({
        "player": {"powers": [("STRENGTH", -3), ("DEXTERITY", -2)]},
        "monsters": [{"powers": [("STRENGTH", -4)]}],
    })

    raw = battle.to_dict()
    player = _powers(raw["player"]["powers"])
    monster = _powers(raw["monsters"][0]["powers"])
    assert player["Strength"] == -3
    assert player["Dexterity"] == -2
    assert "Vulnerable" not in player
    assert monster["Strength"] == -4
    assert "Weak" not in monster


def test_custom_state_syncs_real_monster_move_history_and_intrinsic_power(combat_factory):
    env = combat_factory(
        ["BASH", "STRIKE_RED"],
        encounter="Cultist",
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    battle = env.thick.bc
    battle.apply_custom_state({
        "monsters": [{
            "hp": 5,
            "max_hp": 49,
            "move_history": ["CULTIST_DARK_STRIKE", "CULTIST_INCANTATION"],
            "powers": [("RITUAL", 3), ("STRENGTH", 3)],
        }],
    })

    monster = battle.to_dict()["monsters"][0]
    assert monster["move"] == "CULTIST_DARK_STRIKE"
    assert monster["last_move"] == "CULTIST_INCANTATION"
    assert (monster["base_damage"], monster["hits"]) == (6, 1)
    assert _powers(monster["powers"]) == {"Ritual": 3, "Strength": 3}


def test_custom_state_syncs_real_louse_hidden_bite_damage(combat_factory):
    env = combat_factory(
        ["BASH", "STRIKE_RED"],
        encounter="Two Louse",
        seed=1,
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    battle = env.thick.bc
    battle.apply_custom_state({
        "monsters": [
            {
                "move_history": ["RED_LOUSE_BITE", "INVALID"],
                "base_damage": 5,
                "hits": 1,
            },
            {
                "move_history": ["GREEN_LOUSE_BITE", "INVALID"],
                "base_damage": 6,
                "hits": 1,
            },
        ],
    })

    monsters = battle.to_dict()["monsters"]
    assert [(m["base_damage"], m["hits"]) for m in monsters] == [(5, 1), (6, 1)]


def test_real_monster_move_codec_uses_only_native_move_names(combat_factory):
    env = combat_factory(["STRIKE_RED"])
    battle = env.thick.bc
    for move in sorted({
        move
        for per_monster in _REAL_MONSTER_MOVES.values()
        for move in per_monster.values()
    }):
        battle.apply_custom_state({
            "monsters": [{"move_history": [move, "INVALID"]}],
        })
        assert battle.to_dict()["monsters"][0]["move"] == move


def test_custom_state_syncs_runtime_cost_counters_and_hp_order(combat_factory):
    env = combat_factory(
        ["BASH", "STRIKE_RED"],
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    battle = env.thick.bc
    battle.apply_custom_state({
        "player": {
            "hp": 91,
            "max_hp": 99,
            "energy": 1,
            "counters": {"cards_discarded": 7, "times_damaged": 11},
        },
        "hand": [("BASH", 0, 0), ("STRIKE_RED", 0, 2)],
    })

    raw = battle.to_dict()
    assert (raw["player"]["hp"], raw["player"]["max_hp"]) == (91, 99)
    assert raw["player"]["combat_counters"]["cards_discarded"] == 7
    assert raw["player"]["combat_counters"]["times_damaged"] == 11
    assert raw["hand"] == [("BASH", 0, 0), ("STRIKE_RED", 0, 2)]
    legal = {action_id for action_id, _bits in battle.autoregressive_policy_actions()}
    assert C.AUTOREGRESSIVE_PLAY in legal
    assert C.AUTOREGRESSIVE_PLAY + 1 not in legal


def test_custom_state_recomputes_card_pile_caches_before_damage(combat_factory):
    env = combat_factory(
        ["BLOOD_FOR_BLOOD"] * 6,
        encounter="Jaw Worm",
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    battle = env.thick.bc
    battle.apply_custom_state({
        "player": {"hp": 80, "max_hp": 80, "energy": 0},
        "monsters": [{
            "move_history": ["JAW_WORM_CHOMP", "INVALID"],
            "base_damage": 11,
            "hits": 1,
        }],
        "hand": ["DEFEND_RED"],
        "draw": ["STRIKE_RED"],
        "discard": [],
        "exhaust": [],
    })

    result = battle.preview_end_turn_result(4 << 29)

    assert result["player"]["hp"] == 69


def test_real_search_aligns_state_diff_to_real_monster_slot_order(combat_factory):
    env = combat_factory(
        ["STRIKE_RED"] * 5,
        encounter="Two Louse",
        seed=1,
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    sim = normalize_sim(env.thick.bc.to_dict())
    assert [m["id"] for m in sim["monsters"]] == [
        "FuzzyLouseNormal",
        "FuzzyLouseDefensive",
    ]
    real = deepcopy(sim)
    real["monsters"].reverse()

    search = RealCombatSearch.__new__(RealCombatSearch)
    search.env = env
    search.sim_to_real_targets = {}
    search.sim_to_real_targets = search._target_mapping(real)

    assert search.sim_to_real_targets == {0: 1, 1: 0}
    assert diff(real, search._sim_state_in_real_target_order()) == []


def test_confirmed_target_mapping_survives_shadow_kill(combat_factory):
    env = combat_factory(["STRIKE_RED"] * 5, encounter="Three Darklings",
                         combat_action_contract=C.AUTOREGRESSIVE_CONTRACT)
    real = normalize_sim(env.thick.bc.to_dict())
    env.thick.bc.apply_custom_state({"monsters": [{"gone": True}, {}, {}]})
    search = RealCombatSearch.__new__(RealCombatSearch)
    search.env = env
    search.sim_to_real_targets = {0: 0, 1: 1, 2: 2}

    # All three share an ID: matching only live shadow slots would either
    # lose the first entity or shift one of its siblings into its place.
    assert search._target_mapping(real) == {0: 0, 1: 1, 2: 2}
    env.thick.bc.apply_custom_state({"monsters": [
        {"hp": m["hp"], "max_hp": m["max_hp"]} for m in real["monsters"]
    ]})
    assert [m["hp"] for m in normalize_sim(env.thick.bc.to_dict())["monsters"]] == [
        m["hp"] for m in real["monsters"]]


def test_shadow_battle_nonce_recovers_randomized_encounter_roster():
    def make_gc(seed_nonce):
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 159066251 + seed_nonce, 0)
        gc.floor_num = 11
        gc.set_deck([("STRIKE_RED", 0)] * 5)
        return gc

    gc, battle, encounter, nonce = _find_shadow_battle(
        make_gc,
        ("AcidSlime_M", "FungiBeast"),
        ("Exordium Wildlife",),
    )

    assert gc is not None
    assert encounter == "Exordium Wildlife"
    assert nonce == 7
    assert sorted(m["id"] for m in normalize_sim(battle.to_dict())["monsters"]) == [
        "AcidSlime_M",
        "FungiBeast",
    ]


def test_game_context_potion_state_can_match_real_slots(combat_factory):
    env = combat_factory(["STRIKE_RED"])
    env.thick.gc.set_potion_state(["AttackPotion", "EMPTY_POTION_SLOT"])
    assert list(env.thick.gc.overworld_potions)[:2] == [
        "Attack Potion",
        "EMPTY_POTION_SLOT",
    ]
    env.thick.bc.apply_custom_state({
        "potions": ["AttackPotion", "EMPTY_POTION_SLOT"],
    })
    assert env.thick.bc.to_dict()["potions"][:2] == [
        "Attack Potion",
        "EMPTY_POTION_SLOT",
    ]


def test_java_unplayable_cost_preserves_native_curse_encoding(combat_factory):
    env = combat_factory(
        ["PAIN", "STRIKE_RED"],
        encounter="Jaw Worm",
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    rows = _card_rows([{"id": "Pain", "up": 0, "cost": -2}], costs=True)
    assert rows == [("PAIN", 0, None)]

    env.thick.bc.apply_custom_state({"hand": rows})
    raw = env.thick.bc.to_dict()
    assert raw["hand"] == [("PAIN", 0, -3)]
    assert normalize_sim(raw)["hand"] == [{"id": "Pain", "up": 0, "cost": -2}]


def test_java_confusion_power_maps_to_native_confused_presence(combat_factory):
    assert _player_powers_real([
        {"id": "Confusion", "amount": -1},
    ]) == {"Confused": 1}

    env = combat_factory(
        ["STRIKE_RED"],
        encounter="Snecko",
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    env.thick.bc.apply_custom_state({
        "player": {"powers": [("CONFUSED", 1)]},
    })
    assert normalize_sim(env.thick.bc.to_dict())["player"]["powers"]["Confused"] == 1


def test_java_no_draw_power_maps_to_native_presence(combat_factory):
    assert _player_powers_real([
        {"id": "No Draw", "amount": -1},
    ]) == {"No Draw": 1}

    env = combat_factory(
        ["BATTLE_TRANCE", "STRIKE_RED"],
        encounter="Lagavulin",
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    env.thick.bc.apply_custom_state({
        "player": {"powers": [("NO_DRAW", 1)]},
    })
    assert normalize_sim(env.thick.bc.to_dict())["player"]["powers"]["No Draw"] == 1


def test_java_duplication_power_maps_to_native_duplication(combat_factory):
    assert _player_powers_real([
        {"id": "DuplicationPower", "amount": 1},
    ]) == {"Duplication": 1}

    env = combat_factory(
        ["STRIKE_RED"],
        encounter="The Maw",
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    env.thick.bc.apply_custom_state({
        "player": {"powers": [("DUPLICATION", 1)]},
    })
    assert normalize_sim(env.thick.bc.to_dict())["player"]["powers"]["Duplication"] == 1


def test_java_flex_power_maps_to_native_lose_strength_amount(combat_factory):
    assert _player_powers_real([
        {"id": "Flex", "amount": 3},
        {"id": "Strength", "amount": 3},
    ]) == {"Lose Strength": 3, "Strength": 3}

    env = combat_factory(
        ["STRIKE_RED"],
        encounter="Lagavulin",
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    env.thick.bc.apply_custom_state({
        "player": {"powers": [("LOSE_STRENGTH", 3), ("STRENGTH", 3)]},
    })
    assert normalize_sim(env.thick.bc.to_dict())["player"]["powers"] == {
        "Lose Strength": 3,
        "Strength": 3,
    }


def test_java_life_link_power_maps_to_native_regrow_presence(combat_factory):
    assert _monster_powers_real([
        {"id": "Life Link", "amount": -1},
    ]) == {"Regrow": 1}

    env = combat_factory(
        ["STRIKE_RED"],
        encounter="Three Darklings",
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    powers = normalize_sim(env.thick.bc.to_dict())["monsters"][0]["powers"]
    assert powers["Regrow"] == 1


def test_darkling_chomp_matches_real_two_hit_move(combat_factory):
    env = combat_factory(
        ["STRIKE_RED"],
        encounter="Three Darklings",
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    env.thick.bc.apply_custom_state({
        "monsters": [{
            "move_history": ["DARKLING_CHOMP", "DARKLING_HARDEN"],
            "base_damage": 8,
            "hits": 2,
            "powers": [("REGROW", 1)],
        }, {}, {}],
    })
    monster = normalize_sim(env.thick.bc.to_dict())["monsters"][0]
    assert (monster["base_damage"], monster["hits"]) == (8, 2)


def test_java_generic_strength_up_power_maps_to_native_name(combat_factory):
    assert _monster_powers_real([
        {"id": "Generic Strength Up Power", "amount": 3},
    ]) == {"Generic Strength Up": 3}

    env = combat_factory(
        ["STRIKE_RED"],
        encounter="Orb Walker",
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    powers = normalize_sim(env.thick.bc.to_dict())["monsters"][0]["powers"]
    assert powers["Generic Strength Up"] == 3


def test_time_warp_zero_remains_an_active_native_counter(combat_factory):
    assert _monster_powers_real([
        {"id": "Time Warp", "amount": 0},
    ]) == {"Time Warp": 0}

    env = combat_factory(
        ["STRIKE_RED", "DEFEND_RED"],
        encounter="Time Eater",
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    battle = env.thick.bc
    battle.apply_custom_state({
        "player": {"energy": 3},
        "hand": ["STRIKE_RED", "DEFEND_RED"],
        "draw": [],
        "discard": [],
        "exhaust": [],
        "monsters": [{"powers": [("TIME_WARP", 0)]}],
    })

    before = normalize_sim(battle.to_dict())["monsters"][0]["powers"]
    assert "Time Warp" in before
    assert before["Time Warp"] == 0

    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    after = normalize_sim(battle.to_dict())["monsters"][0]["powers"]
    assert after["Time Warp"] == 1


def test_slow_zero_remains_an_active_native_counter(combat_factory):
    assert _monster_powers_real([
        {"id": "Slow", "amount": 0},
    ]) == {"Slow": 0}

    env = combat_factory(
        ["STRIKE_RED", "DEFEND_RED"],
        encounter="Giant Head",
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    battle = env.thick.bc
    battle.apply_custom_state({
        "player": {"energy": 3},
        "hand": ["STRIKE_RED", "DEFEND_RED"],
        "draw": [],
        "discard": [],
        "exhaust": [],
        "monsters": [{"powers": [("SLOW", 0)]}],
    })

    before = normalize_sim(battle.to_dict())["monsters"][0]["powers"]
    assert "Slow" in before
    assert before["Slow"] == 0

    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    after = normalize_sim(battle.to_dict())["monsters"][0]["powers"]
    assert after["Slow"] == 1
    battle.apply_custom_state({"monsters": [{"powers": []}]})
    assert "Slow" not in normalize_sim(battle.to_dict())["monsters"][0]["powers"]


def test_java_explosive_display_counter_uses_native_move_history(combat_factory):
    assert _monster_powers_real([
        {"id": "Explosive", "amount": 3},
    ]) == {}

    env = combat_factory(
        ["STRIKE_RED"],
        encounter="Three Shapes",
        seed=4,
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    exploder = next(
        monster
        for monster in normalize_sim(env.thick.bc.to_dict())["monsters"]
        if monster["id"] == "Exploder"
    )
    assert exploder["powers"] == {}
    assert exploder["move"] == "EXPLODER_SLAM"


def test_java_unawakened_power_syncs_native_boss_phase(combat_factory):
    assert _monster_powers_real([
        {"id": "Regenerate", "amount": 10},
        {"id": "Curiosity", "amount": 1},
        {"id": "Unawakened", "amount": -1},
    ]) == {"Regen": 10, "Curiosity": 1}

    env = combat_factory(
        ["STRIKE_RED"],
        encounter="Awakened One",
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    raw = env.thick.bc.to_dict()
    assert raw["monsters"][2]["id"] == "AWAKENED_ONE"
    assert raw["monsters"][2]["unawakened"] is True
    assert normalize_sim(raw)["monsters"][2]["unawakened"] is True

    env.thick.bc.apply_custom_state({
        "monsters": [{}, {}, {"unawakened": False}],
    })
    assert env.thick.bc.to_dict()["monsters"][2]["unawakened"] is False


def test_java_compulsive_power_syncs_writhing_mass_reactive_presence(combat_factory):
    real = _monster_real({
        "id": "WrithingMass",
        "current_hp": 160,
        "max_hp": 160,
        "block": 0,
        "is_gone": False,
        "move_base_damage": 15,
        "move_adjusted_damage": 15,
        "move_hits": 1,
        "move_id": 2,
        "powers": [
            {"id": "Malleable", "amount": 3},
            {"id": "Compulsive", "amount": -1},
        ],
    })
    assert real["reactive"] is True
    assert real["powers"] == {"Malleable": 3}

    env = combat_factory(
        ["STRIKE_RED"],
        encounter="Writhing Mass",
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    raw = env.thick.bc.to_dict()
    assert raw["monsters"][0]["reactive"] is True
    assert normalize_sim(raw)["monsters"][0]["reactive"] is True

    env.thick.bc.apply_custom_state({
        "monsters": [{"reactive": False}],
    })
    assert env.thick.bc.to_dict()["monsters"][0]["reactive"] is False
    env.thick.bc.apply_custom_state({
        "monsters": [{"reactive": True}],
    })
    restored = env.thick.bc.to_dict()["monsters"][0]
    assert restored["reactive"] is True
    assert "Reactive" not in _powers(restored["powers"])


def test_writhing_mass_wither_reports_real_intent_damage(combat_factory):
    env = combat_factory(
        ["STRIKE_RED"],
        encounter="Writhing Mass",
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    env.thick.bc.apply_custom_state({
        "monsters": [{
            "move_history": ["WRITHING_MASS_WITHER", "WRITHING_MASS_FLAIL"],
            "base_damage": 10,
            "hits": 1,
            "powers": [("MALLEABLE", 4)],
            "reactive": True,
        }],
    })
    monster = normalize_sim(env.thick.bc.to_dict())["monsters"][0]
    assert (monster["base_damage"], monster["hits"]) == (10, 1)
    assert monster["reactive"] is True


def test_gremlin_leader_dynamic_minion_species_reuses_native_slots(combat_factory):
    env = combat_factory(
        ["STRIKE_RED"],
        encounter="Gremlin Leader",
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    native = normalize_sim(env.thick.bc.to_dict())
    live = [deepcopy(monster) for monster in native["monsters"] if not monster["gone"]]
    leader = next(monster for monster in live if monster["id"] == "GremlinLeader")
    minions = [monster for monster in live if monster["id"] != "GremlinLeader"]
    summoned = deepcopy(minions[0])
    summoned.update({"id": "GremlinFat", "hp": 16, "max_hp": 16})
    real = deepcopy(native)
    real["monsters"] = minions + [summoned, leader]

    search = RealCombatSearch.__new__(RealCombatSearch)
    search.env = env
    search.sim_to_real_targets = {
        index: index - 1
        for index, monster in enumerate(native["monsters"])
        if index > 0 and not monster["gone"]
    }
    mapping = search._target_mapping(real)
    assert set(mapping.values()) == set(range(4))
    summoned_native_slot = next(
        index for index, real_index in mapping.items() if real_index == 2
    )

    monsters = [{"gone": True} for _ in env.thick.bc.to_dict()["monsters"]]
    monsters[summoned_native_slot] = {
        "id": "FAT_GREMLIN",
        "hp": 16,
        "max_hp": 16,
        "block": 0,
        "powers": [("MINION", 1)],
    }
    env.thick.bc.apply_custom_state({"monsters": monsters})
    rebuilt = env.thick.bc.to_dict()["monsters"][summoned_native_slot]
    assert rebuilt["id"] == "FAT_GREMLIN"
    assert rebuilt["hp"] == 16
    assert _powers(rebuilt["powers"])["Minion"] == 1


def test_java_regenerate_power_maps_to_native_regen_amount(combat_factory):
    assert _monster_powers_real([
        {"id": "Regenerate", "amount": 3},
    ]) == {"Regen": 3}

    env = combat_factory(
        ["STRIKE_RED"],
        encounter="Lagavulin",
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    env.thick.bc.apply_custom_state({
        "monsters": [{"powers": [("REGEN", 3)]}],
    })
    assert normalize_sim(env.thick.bc.to_dict())["monsters"][0]["powers"]["Regen"] == 3


def test_java_player_regeneration_power_maps_to_native_regen_amount(combat_factory):
    assert _player_powers_real([
        {"id": "Regeneration", "amount": 5},
    ]) == {"Regen": 5}

    env = combat_factory(
        ["STRIKE_RED"],
        encounter="Lagavulin",
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    env.thick.bc.apply_custom_state({
        "player": {"powers": [("REGEN", 5)]},
    })
    assert normalize_sim(env.thick.bc.to_dict())["player"]["powers"]["Regen"] == 5


@pytest.mark.parametrize("potion_id", ["SmokeBomb", "Explosive Potion"])
def test_real_dummy_target_potion_maps_to_native_untargeted_action(
    combat_factory,
    potion_id,
):
    env = combat_factory(
        ["STRIKE_RED"] * 5,
        encounter="Jaw Worm",
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    env.thick.gc.set_potion_state([potion_id, "EMPTY_POTION_SLOT"])
    env.thick.bc.apply_custom_state({
        "potions": [potion_id, "EMPTY_POTION_SLOT"],
    })
    env._invalidate_backend_cache()

    from bridge.real_game_env import RealGameAdapter
    adapter = RealGameAdapter(combat_action_contract=C.AUTOREGRESSIVE_CONTRACT)
    raw = env.thick.bc.to_dict()
    monster = raw["monsters"][0]
    adapter.update({
        "in_game": True,
        "ready_for_command": True,
        "available_commands": ["play", "end", "potion", "state"],
        "game_state": {
            "room_phase": "COMBAT",
            "screen_type": "NONE",
            "floor": 1,
            "potions": [{
                "id": potion_id, "requires_target": True, "can_use": True,
            }],
            "combat_state": {
                "turn": 1,
                "hand": [],
                "monsters": [{
                    "id": "JawWorm",
                    "current_hp": monster["hp"],
                    "max_hp": monster["max_hp"],
                    "is_gone": False,
                }],
            },
        },
    })

    search = RealCombatSearch.__new__(RealCombatSearch)
    search.env = env
    search.adapter = adapter
    search.sim_to_real_targets = {0: 0}
    real_mask = adapter.action_mask()
    assert real_mask[C.AUTOREGRESSIVE_POTION]
    assert not real_mask[C.AUTOREGRESSIVE_POTION + 5]
    remapped = search._remap_real_mask(real_mask)
    assert remapped[C.AUTOREGRESSIVE_POTION + 5]
    assert not remapped[C.AUTOREGRESSIVE_POTION]
    assert search._map_action_to_real(C.AUTOREGRESSIVE_POTION + 5) == C.AUTOREGRESSIVE_POTION
    assert adapter.command_for_joint_action(C.AUTOREGRESSIVE_POTION) == "POTION use 0 0"


def test_generated_combat_choices_can_match_real_screen(combat_factory):
    env = combat_factory(["STRIKE_RED"] * 5, relics=("Nilry's Codex",))
    battle = env.thick.bc
    battle.execute(sts.Action(sts.ActionType.END_TURN))
    assert battle.select_info()["context"] == "CODEX"

    battle.set_generated_selection_cards(["ANGER", "BASH", "CLEAVE"])

    candidates = [
        tuple(row["card"][:2])
        for row in battle.select_info()["candidates"]
        if not row["skip"]
    ]
    assert candidates == [("ANGER", 0), ("BASH", 0), ("CLEAVE", 0)]


def test_real_search_reconciles_generated_candidates_before_decision(combat_factory):
    env = combat_factory(["STRIKE_RED"] * 5, relics=("Nilry's Codex",))
    env.thick.bc.execute(sts.Action(sts.ActionType.END_TURN))
    env._invalidate_backend_cache()
    real_desc = env.thick.select_descriptor()
    for candidate, card_id in zip(
        [c for c in real_desc["candidates"] if c["card"] is not None],
        ["HandOfGreed", "PanicButton", "Ghostly"],
    ):
        candidate["card"] = {"id": card_id, "up": 0, "cost": 0}

    search = RealCombatSearch.__new__(RealCombatSearch)
    search.env = env
    search.run_alignment_checks = 0
    # CommunicationMod describes a single generated-card choice with a
    # one-card maximum and three visible cards.  Native select_info reserves
    # pick_max/cards_in_hand for MULTI hand selections, so those two fields
    # deliberately differ while the actual single-choice contract agrees.
    real_desc["pick_max"] = 1
    real_desc["cards_in_hand"] = 3
    search._reconcile_selection(real_desc)

    actual = [
        c["card"][0]
        for c in env.thick.bc.select_info()["candidates"]
        if not c["skip"]
    ]
    assert actual == ["HAND_OF_GREED", "PANIC_BUTTON", "APPARITION"]
    assert search.run_alignment_checks == 1


def test_real_search_recognizes_live_nilry_codex_action_and_skip(combat_factory):
    adapter = RealGameAdapter()
    adapter.update({
        "in_game": True,
        "ready_for_command": True,
        "available_commands": ["choose", "skip", "state"],
        "game_state": {
            "screen_type": "CARD_REWARD",
            "screen_state": {"cards": [
                {"id": "Body Slam", "upgrades": 0},
                {"id": "Rage", "upgrades": 0},
                {"id": "Flex", "upgrades": 0},
            ]},
            "choice_list": ["Body Slam", "Rage", "Flex"],
            "current_action": "CodexAction",
            "room_phase": "COMBAT",
            "combat_state": {"turn": 1},
        },
    })
    real_desc = adapter.select_descriptor()
    assert real_desc is not None
    assert real_desc["context"] == "CODEX"
    assert [candidate["card"] for candidate in real_desc["candidates"]] == [
        {"id": "Body Slam", "up": 0},
        {"id": "Rage", "up": 0},
        {"id": "Flex", "up": 0},
        None,
    ]
    assert adapter.command_for_semantic(ga.select(3)) == "SKIP"

    env = combat_factory(["STRIKE_RED"] * 5, relics=("Nilry's Codex",))
    env.thick.bc.execute(sts.Action(sts.ActionType.END_TURN))
    env._invalidate_backend_cache()
    search = RealCombatSearch.__new__(RealCombatSearch)
    search.env = env
    search.run_alignment_checks = 0
    search._reconcile_selection(real_desc)

    assert search.run_alignment_checks == 1
    assert [
        candidate["card"][0]
        for candidate in env.thick.bc.select_info()["candidates"]
        if not candidate["skip"]
    ] == ["BODY_SLAM", "RAGE", "FLEX"]


def test_real_search_still_rejects_single_select_progress_mismatch(combat_factory):
    env = combat_factory(["STRIKE_RED"] * 5, relics=("Nilry's Codex",))
    env.thick.bc.execute(sts.Action(sts.ActionType.END_TURN))
    env._invalidate_backend_cache()
    real_desc = env.thick.select_descriptor()
    real_desc["pick_remaining"] = 2

    search = RealCombatSearch.__new__(RealCombatSearch)
    search.env = env
    search.run_alignment_checks = 0
    with pytest.raises(RuntimeError, match="pick_remaining mismatch"):
        search._reconcile_selection(real_desc)


def test_real_search_maps_compacted_armaments_selection_handles(combat_factory):
    env = combat_factory(
        ["ARMAMENTS", "STRIKE_RED", "REAPER", "RECKLESS_CHARGE",
         "HEMOKINESIS", "STRIKE_RED"],
        encounter="The Champ",
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    env.thick.bc.apply_custom_state({
        "hand": ["STRIKE_RED", "REAPER", "RECKLESS_CHARGE", "ARMAMENTS",
                 "HEMOKINESIS", "STRIKE_RED"],
        "player": {"energy": 3},
    })
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 3))
    env._invalidate_backend_cache()

    compact_desc = env.thick.select_descriptor()
    assert compact_desc["context"] == "ARMAMENTS"
    assert np.flatnonzero(env.action_masks()).tolist() == list(
        range(C.AUTOREGRESSIVE_PICK, C.AUTOREGRESSIVE_PICK + 5)
    )

    real_desc = deepcopy(compact_desc)
    sim_desc = deepcopy(compact_desc)
    sim_handles = [0, 1, 2, 4, 5]
    for candidate, native_handle in zip(sim_desc["candidates"], sim_handles):
        candidate["native_index"] = native_handle

    class _SelectionThick:
        @staticmethod
        def select_descriptor():
            return sim_desc

    class _SelectionEnv:
        thick = _SelectionThick()

        @staticmethod
        def _invalidate_backend_cache():
            return None

    search = RealCombatSearch.__new__(RealCombatSearch)
    search.env = _SelectionEnv()
    search.run_alignment_checks = 0
    search.sim_to_real_selection_handles = {}
    search._reconcile_selection(real_desc)

    assert search.sim_to_real_selection_handles == {0: 0, 1: 1, 2: 2, 4: 3, 5: 4}
    pick_three_bits = C.autoregressive_native_bits(
        C.AUTOREGRESSIVE_PICK + 3, candidate_indices=sim_handles
    )
    assert pick_three_bits == 2 << 29 | 4
    assert C.autoregressive_action_id(
        pick_three_bits, candidate_indices=sim_handles
    ) == C.AUTOREGRESSIVE_PICK + 3
    assert search.run_alignment_checks == 1


def test_real_search_reconciles_warcry_single_pick_remaining(combat_factory):
    env = combat_factory(
        ["WARCRY", "STRIKE_RED", "DEFEND_RED"],
        encounter="The Maw",
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    env.thick.bc.apply_custom_state({
        "hand": ["WARCRY", "STRIKE_RED"],
        "draw": ["DEFEND_RED"],
        "player": {"energy": 3},
    })
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0))
    env._invalidate_backend_cache()

    sim_desc = env.thick.select_descriptor()
    assert sim_desc["context"] == "WARCRY"
    assert sim_desc["multi"] is False
    assert sim_desc["pick_remaining"] == 1

    # CommunicationMod reports max_cards=1 and the compacted visible hand.
    # pick_max/cards_in_hand are transport details for SINGLE selections; the
    # semantic one-pick progress and candidates must align.
    real_desc = deepcopy(sim_desc)
    real_desc["pick_remaining"] = 1
    real_desc["pick_max"] = 1
    real_desc["cards_in_hand"] = len(real_desc["candidates"])

    search = RealCombatSearch.__new__(RealCombatSearch)
    search.env = env
    search.run_alignment_checks = 0
    search.sim_to_real_selection_handles = {}
    search._reconcile_selection(real_desc)
    assert search.run_alignment_checks == 1


def test_real_search_normalizes_elixir_unbounded_sentinel(combat_factory):
    from bridge.real_game_env import RealGameAdapter

    env = combat_factory(
        ["STRIKE_RED"] * 5,
        encounter="Jaw Worm",
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
    )
    env.thick.bc.apply_custom_state({
        "hand": ["STRIKE_RED"] * 5,
        "potions": ["Elixir Potion"],
    })
    env.thick.bc.execute(sts.Action(sts.ActionType.POTION, 0))
    env._invalidate_backend_cache()
    sim_desc = env.thick.select_descriptor()
    assert sim_desc["context"] == "EXHAUST_MANY"
    assert sim_desc["pick_remaining"] == 10
    assert sim_desc["pick_max"] == 10
    assert sim_desc["cards_in_hand"] == 5

    adapter = RealGameAdapter(combat_action_contract=C.AUTOREGRESSIVE_CONTRACT)
    adapter.update({
        "in_game": True,
        "ready_for_command": True,
        "available_commands": ["choose", "confirm", "state"],
        "game_state": {
            "room_phase": "COMBAT",
            "screen_type": "HAND_SELECT",
            "action_phase": "EXECUTING_ACTIONS",
            "current_action": "ExhaustAction",
            "screen_state": {"max_cards": 99, "can_pick_zero": True},
            "combat_state": {
                "hand": [{"id": "Strike_R", "upgrades": 0}] * 5,
                "monsters": [],
            },
        },
    })
    real_desc = adapter.select_descriptor()
    assert real_desc["pick_remaining"] == 5
    assert real_desc["pick_max"] == 5
    assert real_desc["can_pick_any"] is True

    search = RealCombatSearch.__new__(RealCombatSearch)
    search.env = env
    search.run_alignment_checks = 0
    search._reconcile_selection(real_desc)
    assert search.run_alignment_checks == 1


@pytest.mark.parametrize("room,relics,berserk,expected", [
    ("MonsterRoom", [], 1, 4),
    ("MonsterRoom", ["Cursed Key", "SlaversCollar"], 2, 6),
    ("MonsterRoomElite", ["Cursed Key", "SlaversCollar"], 2, 7),
    ("MonsterRoomBoss", ["SlaversCollar"], 0, 4),
])
def test_real_berserk_syncs_next_turn_energy(combat_factory, room, relics, berserk, expected):
    from bridge.schemas.combat_state import _real_energy_per_turn

    real = {"room_type": room, "relics": [{"id": r} for r in relics],
            "combat_state": {"player": {"energy": 0,
                "powers": [{"id": "Berserk", "amount": berserk}]}}}
    energy = _real_energy_per_turn(real)
    assert energy == expected
    env = combat_factory(["DEFEND_RED"] * 10, encounter="Jaw Worm",
                         combat_action_contract=C.AUTOREGRESSIVE_CONTRACT)
    bc = env.thick.bc
    # Repeated state synchronization must not add Berserk a second time.
    for _ in range(2):
        bc.apply_custom_state({"player": {"energy": 0, "energy_per_turn": energy}})
    assert bc.to_dict()["player"]["energy_per_turn"] == expected
    bc.execute(sts.Action(sts.ActionType.END_TURN))
    assert bc.to_dict()["player"]["energy"] == expected


def test_false_reactive_does_not_erase_guardian_sharp_hide(combat_factory):
    env = combat_factory(["STRIKE_RED"] * 5, encounter="The Guardian",
                         combat_action_contract=C.AUTOREGRESSIVE_CONTRACT)
    bc = env.thick.bc
    bc.apply_custom_state({"monsters": [{"powers": [("SHARP_HIDE", 3)],
                                         "reactive": False}]})
    assert _powers(bc.to_dict()["monsters"][0]["powers"])["Sharp Hide"] == 3
    hp = bc.to_dict()["player"]["hp"]
    bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert bc.to_dict()["player"]["hp"] == hp - 3


def test_real_speed_potion_dexterity_expires_at_turn_end(combat_factory):
    assert _player_powers_real([
        {"id": "DexLoss", "amount": 5}, {"id": "Dexterity", "amount": 5},
    ]) == {"Lose Dexterity": 5, "Dexterity": 5}
    env = combat_factory(["DEFEND_RED"] * 10, encounter="Jaw Worm",
                         combat_action_contract=C.AUTOREGRESSIVE_CONTRACT)
    bc = env.thick.bc
    bc.apply_custom_state({"player": {"powers": [("DEXTERITY", 5), ("LOSE_DEXTERITY", 5)]}})
    assert _powers(bc.to_dict()["player"]["powers"])["Lose Dexterity"] == 5
    bc.execute(sts.Action(sts.ActionType.END_TURN))
    powers = _powers(bc.to_dict()["player"]["powers"])
    assert powers.get("Dexterity", 0) == 0
    assert powers.get("Lose Dexterity", 0) == 0
    bc.apply_custom_state({"player": {"powers": [("LOSE_DEXTERITY", 5)]}})
    bc.apply_custom_state({"player": {"powers": []}})
    assert "Lose Dexterity" not in _powers(bc.to_dict()["player"]["powers"])


@pytest.mark.parametrize("native,java", [
    ("APPARITION", "Ghostly"), ("BLOOD_FOR_BLOOD", "Blood for Blood"),
    ("FLASH_OF_STEEL", "Flash of Steel"), ("HAND_OF_GREED", "HandOfGreed"),
    ("JAX", "J.A.X."), ("MASTER_OF_STRATEGY", "Master of Strategy"),
    ("PANIC_BUTTON", "PanicButton"), ("RITUAL_DAGGER", "RitualDagger"),
])
def test_irregular_real_card_ids_roundtrip_native_piles(combat_factory, native, java):
    from bridge.schemas.combat_state import _card, _sim_card_id, real_to_sim_card_enum
    assert real_to_sim_card_enum(java) == native
    assert _sim_card_id(native) == java
    env = combat_factory(["STRIKE_RED"] * 5)
    env.thick.bc.apply_custom_state({"hand": _card_rows([{"id": java, "up": 0}])})
    assert normalize_sim(env.thick.bc.to_dict())["hand"][0]["id"] == _card(java, 0)["id"]


def test_exploder_unknown_intent_preserves_native_explosion(combat_factory):
    from types import SimpleNamespace

    env = combat_factory(["DEFEND_RED"] * 10, encounter="Three Shapes", seed=4,
                         combat_action_contract=C.AUTOREGRESSIVE_CONTRACT)
    bc = env.thick.bc
    raw = bc.to_dict()
    idx = next(i for i, m in enumerate(raw["monsters"]) if m["id"] == "EXPLODER")
    real = normalize_sim(raw)
    for m in real["monsters"]:
        m["real_move_id"] = 1
        m["real_last_move_id"] = 1
    m = real["monsters"][idx]
    m.update(real_move_id=2, base_damage=0, hits=0)
    search = RealCombatSearch.__new__(RealCombatSearch)
    search.env = env
    search.adapter = SimpleNamespace(_potions=lambda: [])
    search._target_mapping = lambda state: {idx: idx}
    spec = search._state_spec(real)
    assert "base_damage" not in spec["monsters"][idx]
    bc.apply_custom_state(spec)
    visible = normalize_sim(bc.to_dict())["monsters"][idx]
    assert (visible["base_damage"], visible["hits"]) == (0, 0)
    hp = bc.to_dict()["player"]["hp"]
    bc.execute(sts.Action(sts.ActionType.END_TURN))
    assert bc.to_dict()["player"]["hp"] == hp - 30
    assert bc.to_dict()["monsters"][idx]["hp"] == 0


def test_confirmed_real_death_clears_predicted_live_slot(combat_factory):
    env = combat_factory(["STRIKE_RED"], encounter="Three Shapes", seed=4,
                         combat_action_contract=C.AUTOREGRESSIVE_CONTRACT)
    real = normalize_sim(env.thick.bc.to_dict())
    idx = next(i for i, m in enumerate(real["monsters"]) if m["id"] == "Exploder")
    search = RealCombatSearch.__new__(RealCombatSearch)
    search.env = env
    search.sim_to_real_targets = {i: i for i in range(len(real["monsters"]))}
    real["monsters"][idx].update(hp=0, gone=True)
    mapping = search._target_mapping(real)
    assert mapping[idx] == idx
    search.sim_to_real_targets = mapping
    assert search._target_mapping(real) == mapping
    assert set(mapping.values()) == set(range(len(real["monsters"])))
    # Unknown entity mismatches remain errors; only confirmed deaths are trusted.
    real["monsters"][idx]["id"] = "Repulsor"
    with pytest.raises(RuntimeError, match="cannot map native monster"):
        search._target_mapping(real)


def test_collector_resummons_replace_confirmed_dead_transport_slots(combat_factory):
    env = combat_factory(["DEFEND_RED"] * 10, encounter="Collector",
                         combat_action_contract=C.AUTOREGRESSIVE_CONTRACT)
    bc = env.thick.bc
    bc.execute(sts.Action(sts.ActionType.END_TURN))
    native = normalize_sim(bc.to_dict())
    heads = [deepcopy(m) for m in native["monsters"] if m["id"] == "TorchHead"]
    leader = next(deepcopy(m) for m in native["monsters"] if m["id"] == "TheCollector")
    assert len(heads) == 2
    dead = deepcopy(heads)
    for m in dead: m.update(hp=0, gone=True)
    real = deepcopy(native)
    real["monsters"] = [heads[0], dead[0], heads[1], dead[1], leader]
    search = RealCombatSearch.__new__(RealCombatSearch)
    search.env = env
    head_indices = [i for i,m in enumerate(native["monsters"]) if m["id"] == "TorchHead"]
    leader_index = next(i for i,m in enumerate(native["monsters"]) if m["id"] == "TheCollector")
    search.sim_to_real_targets = {head_indices[0]: 1, head_indices[1]: 3, leader_index: 4}
    mapping = search._target_mapping(real)
    assert set(mapping.values()) == {0, 2, 4}
    assert mapping[leader_index] == 4
    search.sim_to_real_targets = mapping
    assert search._target_mapping(real) == mapping


def test_real_plated_armor_replaces_and_clears_predicted_amount(combat_factory):
    from bridge.search_bridge import _power_rows, _SUPPORTED_PLAYER_POWERS
    env = combat_factory(["STRIKE_RED", "DEFEND_RED"])
    battle = env.thick.bc
    for amount in (4, 3, 1):
        rows = _power_rows({"Plated Armor": amount}, _SUPPORTED_PLAYER_POWERS)
        assert rows == [("PLATED_ARMOR", amount)]
        battle.apply_custom_state({"player": {"powers": rows}})
        assert normalize_sim(battle.to_dict())["player"]["powers"]["Plated Armor"] == amount
    battle.apply_custom_state({"player": {"powers": []}})
    assert "Plated Armor" not in normalize_sim(battle.to_dict())["player"]["powers"]


def test_real_buffer_consumption_clears_shadow_power(combat_factory):
    from bridge.search_bridge import _power_rows, _SUPPORTED_PLAYER_POWERS
    env = combat_factory(["STRIKE_RED", "DEFEND_RED"])
    battle = env.thick.bc
    for amount in (2, 1):
        rows = _power_rows({"Buffer": amount}, _SUPPORTED_PLAYER_POWERS)
        assert rows == [("BUFFER", amount)]
        battle.apply_custom_state({"player": {"powers": rows}})
        assert normalize_sim(battle.to_dict())["player"]["powers"]["Buffer"] == amount
    battle.apply_custom_state({"player": {"powers": [("STRENGTH", 3), ("DEMON_FORM", 2)]}})
    assert normalize_sim(battle.to_dict())["player"]["powers"] == {"Strength": 3, "Demon Form": 2}


def test_all_supported_player_powers_clear_on_authoritative_replacement(combat_factory):
    from bridge.search_bridge import _SUPPORTED_PLAYER_POWERS
    env = combat_factory(["STRIKE_RED", "DEFEND_RED"])
    battle = env.thick.bc
    for power in sorted(_SUPPORTED_PLAYER_POWERS):
        battle.apply_custom_state({"player": {"powers": [(power, 1)]}})
        battle.apply_custom_state({"player": {"powers": []}})
        assert normalize_sim(battle.to_dict())["player"]["powers"] == {}, power


@pytest.mark.parametrize('amount', [2, 1])
def test_real_no_block_preserves_duration_and_blocks_defend(combat_factory, amount):
    from bridge.search_bridge import _power_rows, _SUPPORTED_PLAYER_POWERS
    battle = combat_factory(["DEFEND_RED"] * 10).thick.bc
    real = _player_powers_real([{'id': 'NoBlockPower', 'amount': amount},
                               {'id': 'Thorns', 'amount': 3}])
    assert real == {'No Block': amount, 'Thorns': 3}
    rows = _power_rows(real, _SUPPORTED_PLAYER_POWERS)
    assert ('NO_BLOCK', amount) in rows
    battle.apply_custom_state({'player': {'powers': rows, 'block': 0, 'energy': 3},
                               'hand': ['DEFEND_RED', 'DEFEND_RED']})
    assert normalize_sim(battle.to_dict())['player']['powers'] == real
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert battle.to_dict()['player']['block'] == 0
    battle.apply_custom_state({'player': {'powers': [('THORNS', 3)], 'energy': 3}})
    assert 'No Block' not in normalize_sim(battle.to_dict())['player']['powers']
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert battle.to_dict()['player']['block'] == 5


def test_no_block_duration_expires_naturally(combat_factory):
    battle = combat_factory(['DEFEND_RED'] * 10, encounter='Cultist').thick.bc
    battle.apply_custom_state({'player': {'powers': [('NO_BLOCK', 2)]}})
    battle.execute(sts.Action(sts.ActionType.END_TURN))
    assert normalize_sim(battle.to_dict())['player']['powers']['No Block'] == 1
    battle.execute(sts.Action(sts.ActionType.END_TURN))
    assert 'No Block' not in normalize_sim(battle.to_dict())['player']['powers']


def test_secret_technique_real_grid_sorts_native_candidates_and_keeps_ui_handles(combat_factory):
    env = combat_factory(['SECRET_TECHNIQUE', 'ARMAMENTS', 'DEFEND_RED', 'SPOT_WEAKNESS', 'SHRUG_IT_OFF'])
    env.thick.bc.apply_custom_state({'hand':['SECRET_TECHNIQUE'],
        'draw':['ARMAMENTS','DEFEND_RED','SPOT_WEAKNESS','SHRUG_IT_OFF',('DEFEND_RED',1),('DEFEND_RED',1)],
        'player':{'energy':3}})
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD,0))
    env._invalidate_backend_cache()
    cards = [{'id':c,'upgrades':u} for c,u in [('Armaments',0),('Defend_R',0),
             ('Spot Weakness',0),('Shrug It Off',0),('Defend_R',1),('Defend_R',1)]]
    adapter=RealGameAdapter(combat_action_contract=C.AUTOREGRESSIVE_CONTRACT)
    adapter.update({'in_game':True,'ready_for_command':True,'available_commands':['choose','state'],
        'game_state':{'current_action':'SkillFromDeckToHandAction','screen_type':'GRID',
            'room_phase':'COMBAT','combat_state':{'turn':1},
            'choice_list':[c['id'] for c in cards],
            'screen_state':{'cards':cards,'num_cards':1,'selected_cards':[]}}})
    desc=adapter.select_descriptor()
    assert desc['context']=='SECRET_TECHNIQUE'
    assert desc['zone']=='draw'
    search=RealCombatSearch.__new__(RealCombatSearch)
    search.env=env
    search.run_alignment_checks=0
    search._reconcile_selection(desc)
    assert [c['native_index'] for c in desc['candidates']]==[0,1,4,5,3,2]
    assert adapter.command_for_semantic(ga.select(2))=='CHOOSE 4'
    assert adapter.command_for_semantic(ga.select(5))=='CHOOSE 2'
    native_handle=env.thick.select_descriptor()['candidates'][2]['native_index']
    env.thick.bc.execute(sts.Action(2 << 29 | native_handle))
    assert normalize_sim(env.thick.bc.to_dict())['hand'][0]['id']=='Defend_R'
    assert normalize_sim(env.thick.bc.to_dict())['hand'][0]['up']==1
