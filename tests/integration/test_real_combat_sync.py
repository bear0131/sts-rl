"""Regression coverage for real-game combat root state injection."""

from copy import deepcopy

import pytest

from env.core import native_actions as C
from bridge.search_bridge import RealCombatSearch, _card_rows, _find_shadow_battle
from bridge.schemas.combat_state import (
    _REAL_MONSTER_MOVES,
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
        ["Anger", "Bash", "Cleave"],
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
    assert actual == ["ANGER", "BASH", "CLEAVE"]
    assert search.run_alignment_checks == 1


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
