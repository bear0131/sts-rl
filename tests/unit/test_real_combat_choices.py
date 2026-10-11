from copy import deepcopy

import numpy as np

from bridge.real_game_env import RealGameAdapter
from env.core import action_space as A, game_actions as ga, encode


def test_real_grid_candidates_and_command_preserve_indices():
    adapter = RealGameAdapter()
    cards = [{"id": "Strike_R", "upgrades": 0}] * 12 + [{"id": "Demon Form", "upgrades": 1}]
    adapter.update({"in_game": True, "available_commands": ["choose"], "game_state": {
        "room_phase": "COMBAT", "screen_type": "GRID",
        "combat_state": {"hand": []},
        "screen_state": {"cards": cards, "context": "HEADBUTT"},
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
        "game_state": {
            "floor": 0,
            "room_phase": "INCOMPLETE",
            "screen_type": "GRID",
            "screen_state": {
                "cards": cards,
                "selected_cards": [cards[5]],
                "num_cards": 3,
                "any_number": False,
            },
        },
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
        "screen_state": {
            "cards": starter,
            "selected_cards": [],
            "num_cards": 3,
        },
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
    hand = [dict(upgrades=0, **card) for card in hand]
    return {"in_game":True, "available_commands":["choose","confirm","state"] if multi else ["play","end","potion"],
        "game_state":{"room_phase":"COMBAT", "screen_type":"HAND_SELECT" if multi else "NONE",
            "floor":1, "screen_state":{"context":"EXHAUST_MANY", "multi":multi, "pick_max":3,
                                      "have_selected":selected},
            "combat_state":{"turn":1,"hand":hand,"monsters":[
                {"current_hp":0,"is_gone":True}, {"current_hp":10,"is_gone":False}]}}}


def test_joint_bridge_keeps_target_slot_and_executes_complete_play():
    from env.core import native_actions as C
    adapter = RealGameAdapter(combat_action_contract=C.JOINT_CONTRACT)
    adapter.update(joint_frame([{"id":"Strike_R","has_target":True,"is_playable":True}]))
    mask = adapter.action_mask()
    assert mask[C.JOINT_PLAY + 1] and not mask[C.JOINT_PLAY]
    assert adapter.command_for_joint_action(C.JOINT_PLAY+1) == "PLAY 1 1"
    assert adapter.describe_action_id(C.JOINT_PLAY+1)


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
    frame["game_state"]["screen_state"].pop("context")
    frame["game_state"]["screen_state"].pop("pick_max")
    frame["game_state"]["screen_state"].update({
        "max_cards": 99,
        "can_pick_zero": True,
    })
    adapter = RealGameAdapter(combat_action_contract=C.JOINT_CONTRACT)
    adapter.update(frame)

    assert adapter.auto_command() is None
    assert adapter.select_descriptor()["multi"] is True
    assert adapter.select_descriptor()["can_pick_any"] is True
    assert adapter.select_descriptor()["pick_max"] == len(hand)
    assert adapter.select_descriptor()["pick_remaining"] == len(hand)
    assert adapter.action_mask()[C.JOINT_MULTI]
    assert adapter.action_mask()[C.JOINT_MULTI + 1]


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
        "game_state": {
            "screen_type": "COMBAT_REWARD",
            "choice_list": ["card"],
            "screen_state": {"rewards": [{"reward_type": "CARD"}]},
        },
    })

    assert adapter.auto_command() == ("CHOOSE 0", "open_card_reward")

    adapter.update({
        "in_game": True,
        "ready_for_command": True,
        "available_commands": ["choose", "skip", "state"],
        "game_state": {
            "screen_type": "CARD_REWARD",
            "choice_list": ["Anger", "Shrug It Off", "Cleave"],
            "screen_state": {"cards": [
                {"id": "Anger", "upgrades": 0},
                {"id": "Shrug It Off", "upgrades": 0},
                {"id": "Cleave", "upgrades": 0},
            ]},
        },
    })

    assert adapter.auto_command() is None
    assert ga.reward_card(0) in adapter.legal_semantic_actions()
    assert ga.reward_card(2) in adapter.legal_semantic_actions()
    assert ga.skip_cards() in adapter.legal_semantic_actions()


def test_combat_reward_keeps_policy_control_while_claimable_reward_remains():
    adapter = RealGameAdapter()
    adapter.update({
        "in_game": True,
        "ready_for_command": True,
        "available_commands": ["choose", "proceed", "state"],
        "game_state": {
            "screen_type": "COMBAT_REWARD",
            "choice_list": ["gold", "card"],
            "screen_state": {"rewards": [
                {"reward_type": "GOLD"}, {"reward_type": "CARD"},
            ]},
        },
    })

    assert adapter.auto_command() is None
