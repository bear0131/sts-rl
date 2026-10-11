"""Regression coverage for real-game combat root state injection."""

from env.core import native_actions as C
from bridge.search_bridge import RealCombatSearch
from bridge.schemas.combat_state import _REAL_MONSTER_MOVES
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


def test_game_context_potion_state_can_match_real_slots(combat_factory):
    env = combat_factory(["STRIKE_RED"])
    env.thick.gc.set_potion_state(["Fire Potion", "EMPTY_POTION_SLOT"])
    assert list(env.thick.gc.overworld_potions)[:2] == [
        "Fire Potion",
        "EMPTY_POTION_SLOT",
    ]


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
    search._reconcile_selection(real_desc)

    actual = [
        c["card"][0]
        for c in env.thick.bc.select_info()["candidates"]
        if not c["skip"]
    ]
    assert actual == ["ANGER", "BASH", "CLEAVE"]
    assert search.run_alignment_checks == 1
