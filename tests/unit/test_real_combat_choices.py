from copy import deepcopy

import numpy as np
import pytest

from bridge.real_game_env import RealGameAdapter
from bridge.schemas.combat_state import _REAL_COMBAT_COUNTER_KEYS
from bridge.schemas.entity_ids import SIMULATOR_PARITY_EXPORT_VERSION
from env.core import action_space as A, game_actions as ga, encode


def _grid_export(
    state, *, context, source="com.megacrit.cardcrawl.rooms.EventRoom",
    for_upgrade=False, for_purge=False, for_transform=False,
):
    state.update({
        "selection_context": context,
        "selection_source": source,
        "simulator_parity_export": SIMULATOR_PARITY_EXPORT_VERSION,
        "selected_cards": state.get("selected_cards", []),
        "num_cards": state.get("num_cards", 1),
        "any_number": state.get("any_number", False),
        "for_upgrade": for_upgrade,
        "for_purge": for_purge,
        "for_transform": for_transform,
    })
    return state


def _complete_run_state(state):
    defaults = {
        "seed": 42,
        "simulator_parity_export": SIMULATOR_PARITY_EXPORT_VERSION,
        "floor": 1,
        "current_hp": 80,
        "max_hp": 80,
        "gold": 99,
        "class": "IRONCLAD",
        "ascension_level": 0,
        "act": 1,
        "room_type": "MonsterRoom",
        "act_boss": "Slime Boss",
        "has_ruby_key": False,
        "has_emerald_key": False,
        "has_sapphire_key": False,
        "deck": [],
        "relics": [{"id": "Burning Blood", "counter": -1}],
        "potions": [{"id": "Potion Slot"}] * 3,
    }
    return defaults | state


def test_real_grid_candidates_and_command_preserve_indices():
    adapter = RealGameAdapter()
    cards = [{"id": "Strike_R", "upgrades": 0}] * 12 + [{"id": "Demon Form", "upgrades": 1}]
    adapter.update({"in_game": True, "available_commands": ["choose"], "game_state": {
        "room_phase": "COMBAT", "screen_type": "GRID",
        "combat_state": {"hand": []},
        "screen_state": _grid_export(
            {"cards": cards}, context="HEADBUTT",
            source="com.megacrit.cardcrawl.actions.utility.DiscardPileToTopOfDeckAction",
        ),
    }})
    legal = adapter.legal_semantic_actions()
    assert legal == [ga.select(i) for i in range(13)]
    semantic, pending = A.resolve(A._PICK + 12, legal, None, "combat")
    assert pending is None and adapter.command_for_semantic(semantic) == "CHOOSE 12"
    desc = adapter.select_descriptor()
    obs = encode.encode_combat({"options": {"select": desc}})
    assert obs["select_card_ids"][12] == encode.CARD_INDEX["Demon Form"]
    assert obs["select_card_upgrades"][12] == 1 / encode.UPGRADE_SCALE
    assert obs["select_native_indices"][12] == 13


def test_real_grid_uses_selected_cards_as_authoritative_progress():
    adapter = RealGameAdapter()
    cards = [
        {"id": "Strike_R", "upgrades": 0, "uuid": f"strike-{i}"}
        for i in range(5)
    ] + [
        {"id": "Defend_R", "upgrades": 0, "uuid": f"defend-{i}"}
        for i in range(4)
    ] + [
        {"id": "Bash", "upgrades": 0, "uuid": "bash-0"}
    ]
    frame = {
        "in_game": True,
        "ready_for_command": True,
        "available_commands": ["choose", "state"],
        "game_state": _complete_run_state({
            "floor": 0,
            "act_boss": "Slime Boss",
            "room_phase": "INCOMPLETE",
            "screen_type": "GRID",
            "screen_state": _grid_export({
                "cards": cards,
                "selected_cards": [cards[5]],
                "num_cards": 3,
                "any_number": False,
            }, context="TRANSFORM", for_transform=True),
            "relics": [{"id": "Burning Blood", "counter": -1}],
            "deck": cards,
        }),
    }
    adapter.update(frame)

    desc = adapter.select_descriptor()
    assert desc["pick_remaining"] == 2
    assert desc["selected"] == [("Defend_R", 0)]
    assert [candidate["native_index"] for candidate in desc["candidates"]] == [
        0, 1, 2, 3, 4, 6, 7, 8, 9
    ]
    assert adapter.legal_semantic_actions() == [ga.select(i) for i in range(9)]
    assert adapter.command_for_semantic(ga.select(5)) == "CHOOSE 6"

    obs = adapter.observation()
    assert np.flatnonzero(obs["select_native_indices"]).tolist() == list(range(9))
    assert obs["select_native_indices"][:9].tolist() == [1, 2, 3, 4, 5, 7, 8, 9, 10]
    assert obs["selected_card_ids"][0] == encode.CARD_INDEX["Defend_R"]

    frame["available_commands"] = ["choose", "confirm", "state"]
    frame["game_state"]["screen_state"]["selected_cards"] = cards[5:8]
    adapter.update(frame)
    assert adapter.auto_command() == ("CONFIRM", "complete_grid_selection")


def test_real_astrolabe_waits_for_transformed_cards_before_leaving_neow():
    adapter = RealGameAdapter()
    starter = [
        {"id": "Strike_R", "upgrades": 0, "uuid": f"strike-{i}"}
        for i in range(5)
    ] + [
        {"id": "Defend_R", "upgrades": 0, "uuid": f"defend-{i}"}
        for i in range(4)
    ] + [{"id": "Bash", "upgrades": 0, "uuid": "bash-0"}]
    base = {
        "in_game": True,
        "ready_for_command": True,
        "available_commands": ["choose", "wait", "state"],
        "game_state": {
            "seed": 42,
            "floor": 0,
            "room_phase": "EVENT",
            "screen_type": "EVENT",
            "choice_list": ["leave"],
            "screen_state": {"event_id": "Neow"},
            "deck": starter,
            "relics": [{"id": "Burning Blood"}],
        },
    }
    adapter.update(base)

    grid = deepcopy(base)
    grid["game_state"].update({
        "screen_type": "GRID",
        "choice_list": [],
        "screen_state": _grid_export({
            "cards": starter,
            "selected_cards": [],
            "num_cards": 3,
        }, context="TRANSFORM_UPGRADE"),
        "relics": [{"id": "Astrolabe"}],
    })
    adapter.update(grid)

    missing_transforms = deepcopy(base)
    missing_transforms["game_state"].update({
        "deck": starter[:5] + starter[8:],
        "relics": [{"id": "Astrolabe"}],
    })
    adapter.update(missing_transforms)
    assert adapter.auto_command() == ("WAIT 1", "settle_overworld_grid")

    adapter.update(missing_transforms)
    assert adapter.auto_command() == ("WAIT 1", "await_astrolabe_transform")

    settled = deepcopy(missing_transforms)
    settled["game_state"]["deck"] += [
        {"id": "Uppercut", "upgrades": 1},
        {"id": "Twin Strike", "upgrades": 1},
        {"id": "Inflame", "upgrades": 1},
    ]
    adapter.update(settled)
    assert adapter.auto_command() == ("CHOOSE 0", "neow_intro")


def test_real_pending_sources_are_always_encoded(monkeypatch):
    adapter = RealGameAdapter()
    monkeypatch.setattr(adapter, "_base_canonical", lambda: {"phase": "combat"})
    first = adapter.observation(("play", 2))
    second = adapter.observation(("potion", 1))
    assert first["pending_card_slot"].tolist() == [3]
    assert first["pending_potion_slot"].tolist() == [0]
    assert second["pending_potion_slot"].tolist() == [2]
    assert not np.array_equal(first["pending_card_slot"], second["pending_card_slot"])


def joint_frame(hand, *, selected=0, multi=False):
    hand = [dict(upgrades=0, cost=1, **card) for card in hand]
    selection_state = {
        "selection_context": "EXHAUST_MANY",
        "selection_source": "com.megacrit.cardcrawl.actions.common.ExhaustAction",
        "simulator_parity_export": SIMULATOR_PARITY_EXPORT_VERSION,
        "selected": [
            {"id": "Strike_R", "upgrades": 0}
            for _ in range(selected)
        ],
        "max_cards": 3,
        "can_pick_zero": True,
    }
    counters = {key: 0 for key in _REAL_COMBAT_COUNTER_KEYS}
    return {"in_game":True, "ready_for_command": True,
        "available_commands":["choose","confirm","state"] if multi else ["play","end","potion"],
        "game_state":{"room_phase":"COMBAT", "screen_type":"HAND_SELECT" if multi else "NONE",
            "seed":42, "floor":1, "act_boss":"Slime Boss", "boss": "SLIME_BOSS",
            "simulator_parity_export": SIMULATOR_PARITY_EXPORT_VERSION,
            "class": "IRONCLAD", "ascension_level": 0, "act": 1,
            "room_type": "MonsterRoom", "current_hp": 80, "max_hp": 80,
            "gold": 0, "deck": [],
            "has_ruby_key": False, "has_emerald_key": False,
            "has_sapphire_key": False,
            "relics": [], "potions": [{"id": "Potion Slot"}] * 3,
            "screen_state":selection_state if multi else {},
            "combat_state":{
                "simulator_parity_export": SIMULATOR_PARITY_EXPORT_VERSION,
                "simulator_energy_per_turn": 3,
                "simulator_combat_counters": counters,
                "simulator_player_power_overrides": {},
                "turn":1,"hand":hand,"draw_pile":[],
                "discard_pile":[],"exhaust_pile":[],
                "player":{"current_hp":80,"max_hp":80,"block":0,"energy":3,"powers":[]},
                "monsters":[
                    {"id":"Cultist","intent":"NONE","current_hp":0,"max_hp":48,"block":0,
                     "is_gone":True,"powers":[], "move_id": -1,
                     "last_move_id": -1, "second_last_move_id": -1,
                     "move_base_damage": -1, "move_adjusted_damage": -1,
                     "move_hits": 0, "half_dead": False,
                     "current_intent_export": SIMULATOR_PARITY_EXPORT_VERSION},
                    {"id":"JawWorm","intent":"ATTACK","current_hp":10,"max_hp":42,"block":0,
                     "is_gone":False,"move_id":1,"last_move_id":-1,
                     "second_last_move_id":-1,
                     "move_base_damage":11,"move_adjusted_damage":11,
                     "move_hits":1,"powers":[], "half_dead": False,
                     "current_intent_export": SIMULATOR_PARITY_EXPORT_VERSION}]}}}


def test_joint_bridge_keeps_target_slot_and_executes_complete_play():
    from env.core import native_actions as C
    adapter = RealGameAdapter(combat_action_contract=C.JOINT_CONTRACT)
    adapter.update(joint_frame([{"id":"Strike_R","has_target":True,"is_playable":True}]))
    mask = adapter.action_mask()
    assert mask[C.JOINT_PLAY + 1] and not mask[C.JOINT_PLAY]
    assert adapter.command_for_joint_action(C.JOINT_PLAY+1) == "PLAY 1 1"
    assert adapter.describe_action_id(C.JOINT_PLAY+1)


def test_joint_bridge_compacts_overflowed_gremlin_history_and_restores_java_target():
    from bridge.search_bridge import RealCombatSearch
    from env.core import native_actions as C

    def monster(monster_id, hp, gone=False):
        return {
            "id": monster_id,
            "current_hp": hp,
            "max_hp": max(hp, 1),
            "block": 0,
            "is_gone": gone,
            "powers": [],
        }

    # CommunicationMod retains four dead summons, so the live leader has Java
    # index 6 even though the policy/native search contract has five targets.
    monsters = [
        monster("GremlinFat", 0, True),
        monster("GremlinFat", 14),
        monster("GremlinWarrior", 0, True),
        monster("GremlinWizard", 25),
        monster("GremlinFat", 0, True),
        monster("GremlinFat", 0, True),
        monster("GremlinLeader", 7),
    ]
    frame = joint_frame([
        {"id": "Strike_R", "has_target": True, "is_playable": True},
    ])
    frame["game_state"]["combat_state"]["monsters"] = monsters
    adapter = RealGameAdapter(combat_action_contract=C.AUTOREGRESSIVE_CONTRACT)
    adapter.update(frame)

    assert adapter.combat_target_layout() == (1, 3, 6)
    assert np.flatnonzero(adapter.action_mask()).tolist() == [0, 1, 2, 3]
    assert adapter.command_for_joint_action(C.AUTOREGRESSIVE_PLAY + 2) == "PLAY 1 6"

    search = RealCombatSearch.__new__(RealCombatSearch)
    search.adapter = adapter
    search.sim_to_real_targets = {0: 1, 1: 3, 3: 6}
    remapped = search._remap_real_mask(adapter.action_mask())
    assert np.flatnonzero(remapped).tolist() == [0, 1, 2, 4]
    assert search._map_action_to_real(C.AUTOREGRESSIVE_PLAY + 3) == (
        C.AUTOREGRESSIVE_PLAY + 2
    )


def test_joint_bridge_expands_subset_without_extra_policy_and_checks_ack():
    from env.core import native_actions as C
    hand = [{"id":cid} for cid in ("Strike_R", "Bash", "Defend_R")]
    adapter = RealGameAdapter(combat_action_contract=C.JOINT_CONTRACT)
    adapter.update(joint_frame(hand, multi=True))
    assert adapter.action_mask()[C.JOINT_MULTI+5]
    assert adapter.command_for_joint_action(C.JOINT_MULTI+5) == "CHOOSE 2"
    assert adapter.auto_command()[0] == "STATE"  # Same frame is not acknowledgement.
    adapter.update(joint_frame(hand[:2], selected=1, multi=True))
    assert adapter.auto_command()[0] == "CHOOSE 0"
    adapter.update(joint_frame(hand[1:2], selected=2, multi=True))
    assert adapter.auto_command()[0] == "CONFIRM"
    assert adapter._joint_multi_plan is None
    adapter.update(joint_frame(hand, multi=True))
    assert adapter.command_for_joint_action(C.JOINT_MULTI) == "CONFIRM"


def test_actionable_hand_select_overrides_executing_action_phase():
    from env.core import native_actions as C

    hand = [{"id": cid} for cid in ("Thunderclap", "Carnage", "Strike_R", "Pain", "Defend_R")]
    frame = joint_frame(hand, multi=True)
    frame["game_state"]["action_phase"] = "EXECUTING_ACTIONS"
    frame["game_state"]["current_action"] = "ExhaustAction"
    frame["game_state"]["screen_state"].update({
        "max_cards": 99,
        "can_pick_zero": True,
    })
    adapter = RealGameAdapter(combat_action_contract=C.JOINT_CONTRACT)
    adapter.update(frame)

    assert adapter.auto_command() is None
    assert adapter.select_descriptor()["multi"] is True
    assert adapter.select_descriptor()["can_pick_any"] is False
    assert adapter.select_descriptor()["can_pick_zero"] is True
    assert adapter.select_descriptor()["pick_max"] == len(hand)
    assert adapter.select_descriptor()["pick_remaining"] == len(hand)
    assert adapter.action_mask()[C.JOINT_MULTI]
    assert adapter.action_mask()[C.JOINT_MULTI + 1]


def test_combat_map_overlay_returns_before_native_search():
    from env.core import native_actions as C

    adapter = RealGameAdapter(combat_action_contract=C.JOINT_CONTRACT)
    adapter.update({
        "in_game": True,
        "ready_for_command": True,
        "available_commands": ["choose", "return", "wait", "state"],
        "game_state": {
            "room_phase": "COMBAT",
            "screen_type": "MAP",
            "screen_name": "MAP",
            "is_screen_up": True,
            "action_phase": "WAITING_ON_USER",
            "choice_list": ["x=0"],
            "screen_state": {"next_nodes": [{"symbol": "?", "x": 0, "y": 10}]},
            "combat_state": {
                "turn": 2,
                "hand": [{"id": "Strike_R", "is_playable": True}],
                "player": {"energy": 1},
                "monsters": [{
                    "id": "JawWorm",
                    "current_hp": 32,
                    "is_gone": False,
                }],
            },
        },
    })

    assert adapter.phase() == "combat"
    assert adapter.auto_command() == ("RETURN", "close_combat_map_overlay")


def test_combat_map_overlay_refreshes_until_return_is_available():
    adapter = RealGameAdapter()
    adapter.update({
        "in_game": True,
        "ready_for_command": True,
        "available_commands": ["wait", "state"],
        "game_state": {
            "room_phase": "COMBAT",
            "screen_type": "MAP",
            "combat_state": {"turn": 2, "monsters": []},
        },
    })

    assert adapter.auto_command() == (
        "STATE", "await_combat_map_overlay_close")


def test_single_hand_select_confirms_selected_alias_without_second_policy_call():
    from env.core import native_actions as C

    hand = [
        {"id": "Bash"}, {"id": "Strike_R"},
        {"id": "Strike_R"}, {"id": "Strike_R"},
    ]
    frame = joint_frame(hand, multi=False)
    frame["available_commands"] = ["choose", "state"]
    frame["game_state"].update({
        "screen_type": "HAND_SELECT",
        "action_phase": "EXECUTING_ACTIONS",
        "current_action": "ExhaustAction",
    })
    frame["game_state"]["screen_state"] = {
        "selection_context": "EXHAUST_ONE",
        "selection_source": "com.megacrit.cardcrawl.actions.common.ExhaustAction",
        "simulator_parity_export": SIMULATOR_PARITY_EXPORT_VERSION,
        "hand": [dict(upgrades=0, **card) for card in hand],
        "selected": [],
        "max_cards": 1,
        "can_pick_zero": False,
    }
    adapter = RealGameAdapter(combat_action_contract=C.JOINT_CONTRACT)
    adapter.update(frame)
    assert adapter.auto_command() is None
    assert adapter.command_for_joint_action(C.JOINT_PICK) == "CHOOSE 0"

    acknowledged = deepcopy(frame)
    acknowledged["available_commands"] = ["confirm", "state"]
    acknowledged["game_state"]["screen_state"].update({
        "hand": [dict(upgrades=0, **card) for card in hand[1:]],
        "selected": [dict(upgrades=0, **hand[0])],
    })
    adapter.update(acknowledged)
    assert adapter.select_descriptor()["pick_remaining"] == 0
    assert adapter.auto_command() == ("CONFIRM", "complete_hand_selection")


def test_actionable_selection_without_context_fails_closed():
    frame = joint_frame([{"id": "Strike_R"}], multi=True)
    frame["game_state"]["screen_state"].pop("selection_context")
    adapter = RealGameAdapter()
    with pytest.raises(ValueError, match="missing a recognized exported context"):
        adapter.update(frame)


def test_hand_contents_cannot_guess_selection_context():
    frame = joint_frame([{"id": "Calculated Gamble"}], multi=True)
    frame["game_state"]["screen_state"].pop("selection_context")
    adapter = RealGameAdapter()
    with pytest.raises(ValueError, match="missing a recognized exported context"):
        adapter.update(frame)


def test_current_action_cannot_replace_exported_selection_context():
    frame = joint_frame([{"id": "Strike_R"}], multi=True)
    frame["game_state"]["screen_state"].pop("selection_context")
    frame["game_state"]["current_action"] = "GamblingChipAction"
    adapter = RealGameAdapter()
    with pytest.raises(ValueError, match="missing a recognized exported context"):
        adapter.update(frame)


def test_joint_bridge_handles_stable_hand_and_discards_plan_on_screen_change():
    from env.core import native_actions as C
    hand = [{"id":cid} for cid in ("Strike_R", "Bash", "Defend_R")]
    adapter = RealGameAdapter(combat_action_contract=C.JOINT_CONTRACT)
    adapter.update(joint_frame(hand, multi=True))
    assert adapter.command_for_joint_action(C.JOINT_MULTI+5) == "CHOOSE 2"
    adapter.update(joint_frame(hand, selected=1, multi=True))
    assert adapter.auto_command()[0] == "CHOOSE 0"
    adapter.update(joint_frame(hand, selected=2, multi=True))
    assert adapter.auto_command()[0] == "CONFIRM"
    adapter.update(joint_frame(hand, multi=True))
    adapter.command_for_joint_action(C.JOINT_MULTI+5)
    adapter.update(joint_frame(hand))
    assert adapter.auto_command() is None
    assert adapter._joint_multi_plan is None


def test_autoregressive_bridge_keeps_prefix_internal_until_stop():
    from env.core import native_actions as C
    hand = [{"id": cid} for cid in ("Strike_R", "Bash", "Defend_R")]
    adapter = RealGameAdapter(combat_action_contract=C.AUTOREGRESSIVE_CONTRACT)
    frame = joint_frame(hand, multi=True)
    adapter.update(frame)
    assert adapter.action_mask().shape == (198,)
    assert adapter.command_for_joint_action(C.AUTOREGRESSIVE_MULTI) is None
    assert adapter.command_for_joint_action(C.AUTOREGRESSIVE_MULTI + 2) is None
    assert adapter.frame == frame
    assert adapter.obs_canonical()['options']['select']['hand_selected'] == [0, 2]
    assert np.flatnonzero(adapter.action_mask()).tolist() == [C.AUTOREGRESSIVE_STOP]
    assert adapter.command_for_joint_action(C.AUTOREGRESSIVE_STOP) == 'CHOOSE 2'
    assert adapter.auto_command()[0] == 'STATE'
    adapter.update(joint_frame(hand[:2], selected=1, multi=True))
    assert adapter.auto_command()[0] == 'CHOOSE 0'
    adapter.update(joint_frame(hand[1:2], selected=2, multi=True))
    assert adapter.auto_command()[0] == 'CONFIRM'


def test_combat_reward_opens_card_candidates_before_policy_decides():
    adapter = RealGameAdapter()
    adapter.update({
        "in_game": True,
        "ready_for_command": True,
        "available_commands": ["choose", "proceed", "state"],
        "game_state": _complete_run_state({
            "screen_type": "COMBAT_REWARD",
            "act_boss": "Slime Boss",
            "choice_list": ["card"],
            "screen_state": {"rewards": [{"reward_type": "CARD"}]},
        }),
    })

    assert adapter.auto_command() == ("CHOOSE 0", "open_card_reward")

    adapter.update({
        "in_game": True,
        "ready_for_command": True,
        "available_commands": ["choose", "skip", "state"],
        "game_state": _complete_run_state({
            "screen_type": "CARD_REWARD",
            "act_boss": "Slime Boss",
            "choice_list": ["Anger", "Shrug It Off", "Cleave"],
            "screen_state": {"cards": [
                {"id": "Anger", "upgrades": 0},
                {"id": "Shrug It Off", "upgrades": 0},
                {"id": "Cleave", "upgrades": 0},
            ]},
        }),
    })

    assert adapter.auto_command() is None
    assert ga.reward_card(0) in adapter.legal_semantic_actions()
    assert ga.reward_card(2) in adapter.legal_semantic_actions()
    assert ga.skip_cards() in adapter.legal_semantic_actions()


def _reward_outer_frame(*choices):
    rewards = {
        "gold": {"reward_type": "GOLD", "gold": 27},
        "relic": {
            "reward_type": "RELIC",
            "relic": {"id": "Matryoshka", "counter": 2},
        },
        "potion": {
            "reward_type": "POTION",
            "potion": {"id": "Regen Potion", "can_discard": True},
        },
        "card": {"reward_type": "CARD"},
        "emerald_key": {"reward_type": "EMERALD_KEY"},
        "sapphire_key": {"reward_type": "SAPPHIRE_KEY"},
    }
    return {
        "in_game": True,
        "ready_for_command": True,
        "available_commands": ["choose", "potion", "proceed", "state"],
        "game_state": _complete_run_state({
            "screen_type": "COMBAT_REWARD",
            "act_boss": "Slime Boss",
            "choice_list": list(choices),
            "screen_state": {"rewards": [rewards[choice] for choice in choices]},
        }),
    }


def _reward_card_frame():
    cards = [
        {"id": "Anger", "upgrades": 0},
        {"id": "Searing Blow", "upgrades": 0},
        {"id": "Whirlwind", "upgrades": 0},
    ]
    return {
        "in_game": True,
        "ready_for_command": True,
        "available_commands": ["choose", "skip", "state"],
        "game_state": _complete_run_state({
            "screen_type": "CARD_REWARD",
            "act_boss": "Slime Boss",
            "choice_list": [card["id"] for card in cards],
            "screen_state": {"cards": cards, "skip_available": True},
        }),
    }


def test_combat_reward_preview_matches_training_candidates_and_outer_rewards():
    adapter = RealGameAdapter()
    adapter.update(_reward_outer_frame("gold", "relic", "potion", "card"))
    assert adapter.auto_command() == ("CHOOSE 3", "open_card_reward")

    adapter.update(_reward_card_frame())

    assert adapter.obs_canonical()["options"] == {
        "gold": [27],
        "relics": ["Matryoshka"],
        "potions": ["Regen Potion"],
        "keys": [],
        "card_count": 1,
        "cards": [("Anger", 0), ("Searing Blow", 0), ("Whirlwind", 0)],
    }
    assert adapter.legal_semantic_actions() == [
        ga.claim(ga.CLAIM_GOLD, 0),
        ga.claim(ga.CLAIM_RELIC, 0),
        ga.claim(ga.CLAIM_POTION, 0),
        ga.reward_card(0),
        ga.reward_card(1),
        ga.reward_card(2),
        ga.skip_cards(),
        ga.proceed(),
    ]
    observation = adapter.observation()
    assert np.isclose(observation["reward_gold"][0], 27 / encode.GOLD_SCALE)
    assert observation["offer_card_ids"][0] == encode.CARD_INDEX["Anger"]
    assert adapter.command_for_semantic(ga.reward_card(2)) == "CHOOSE 2"
    assert adapter._deferred_combat_reward_action is None


@pytest.mark.parametrize("key_name,slot", [
    ("emerald_key", 0),
    ("sapphire_key", 1),
])
def test_real_key_reward_is_an_explicit_observation_and_action(key_name, slot):
    adapter = RealGameAdapter()
    adapter.update(_reward_outer_frame(key_name))

    canonical = adapter.obs_canonical()
    assert canonical["global"]["keys"] == {
        "ruby": False, "emerald": False, "sapphire": False,
    }
    assert canonical["options"]["keys"] == [key_name.upper()]
    assert adapter.legal_semantic_actions() == [
        ga.claim(ga.CLAIM_KEY, 0), ga.proceed(),
    ]
    assert np.flatnonzero(adapter.action_mask()).tolist() == [A._PROCEED, A._RKEY]
    assert adapter.command_for_semantic(ga.claim(ga.CLAIM_KEY, 0)) == "CHOOSE 0"
    expected = [0.0, 0.0]
    expected[slot] = 1.0
    assert adapter.observation()["reward_keys"].tolist() == expected


def test_real_reward_labels_and_records_must_name_the_same_key():
    adapter = RealGameAdapter()
    frame = _reward_outer_frame("emerald_key")
    frame["game_state"]["screen_state"]["rewards"][0]["reward_type"] = "SAPPHIRE_KEY"
    with pytest.raises(ValueError, match="keywords disagree"):
        adapter.update(frame)

    frame = _reward_outer_frame("emerald_key")
    frame["game_state"]["choice_list"] = ["ruby_key"]
    with pytest.raises(ValueError, match="unknown combat reward choice keyword"):
        adapter.update(frame)


def test_combat_reward_preview_defers_outer_actions_without_second_policy_call():
    for action, expected in [
        (ga.claim(ga.CLAIM_GOLD, 0), "CHOOSE 0"),
        (ga.proceed(), "PROCEED"),
    ]:
        adapter = RealGameAdapter()
        outer = _reward_outer_frame("gold", "card")
        adapter.update(outer)
        assert adapter.auto_command() == ("CHOOSE 1", "open_card_reward")
        adapter.update(_reward_card_frame())

        assert adapter.command_for_semantic(action) == "SKIP"
        assert adapter._skipped_combat_card_reward is False
        adapter.update(deepcopy(outer))
        assert adapter.auto_command() == (
            expected, "complete_deferred_reward_action")
        assert adapter._deferred_combat_reward_action is None


def test_combat_reward_semantic_skip_masks_cancelled_real_placeholder():
    adapter = RealGameAdapter()
    outer = _reward_outer_frame("gold", "card")
    adapter.update(outer)
    assert adapter.auto_command() == ("CHOOSE 1", "open_card_reward")
    adapter.update(_reward_card_frame())

    assert adapter.command_for_semantic(ga.skip_cards()) == "SKIP"
    adapter.update(deepcopy(outer))

    assert adapter.auto_command() is None
    assert adapter.obs_canonical()["options"] == {
        "gold": [27],
        "relics": [],
        "potions": [],
        "keys": [],
        "card_count": 0,
        "cards": [],
    }
    assert adapter.legal_semantic_actions() == [
        ga.claim(ga.CLAIM_GOLD, 0), ga.proceed()]
