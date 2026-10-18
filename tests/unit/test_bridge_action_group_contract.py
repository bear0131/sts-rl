from types import SimpleNamespace
from zipfile import ZipFile
from unittest.mock import MagicMock

import pytest


@pytest.mark.parametrize('override, expected', [(None, 1.0), (0.0, 0.0)])
def test_real_search_gumbel_override_reaches_strategy(monkeypatch, override, expected):
    from bridge import search_bridge as bridge
    model = SimpleNamespace(observation_space=object(), device='cpu',
        nosl_is_az_gumbel_scale=1.0, nosl_is_az_root_search_mode=2,
        nosl_is_az_paper_full_gumbel=True)
    monkeypatch.setattr(bridge.C, 'policy_contract', lambda _: next(iter(bridge.C.NEURAL_CONTRACTS)))
    for name in ('uses_global_map_observation', 'uses_card_language', 'uses_combat_language',
                 'uses_deck_size_observation', 'uses_neow_observation', 'uses_event_option_observation'):
        monkeypatch.setattr(bridge.model_utils, name, lambda _: False)
    strategy = MagicMock()
    monkeypatch.setattr(bridge, 'NativeStrategy', strategy)
    monkeypatch.setattr(bridge, 'SlayTheSpireEnv', MagicMock())
    search = bridge.RealCombatSearch(model, SimpleNamespace(ascension=0), gumbel_scale=override)
    assert search.config.gumbel_scale == expected
    assert strategy.call_args.args[1].gumbel_scale == expected
    search.close()


def test_bridge_gumbel_cli_defaults_to_checkpoint():
    from bridge.client.agent_orchestrator import build_parser
    assert build_parser().parse_args(['model.zip']).gumbel_scale is None
    assert build_parser().parse_args(['model.zip', '--gumbel-scale', '0']).gumbel_scale == 0.0


def test_bridge_selects_accumulating_algorithm_for_upper_optimizer(tmp_path):
    from bridge.client import agent_orchestrator as agent
    path = tmp_path/'joint.zip'
    with ZipFile(path, 'w') as archive:
        archive.writestr('policy.upper_optimizer.pth', b'optimizer')

    assert agent._checkpoint_model_class(path) is agent.AccumulatingMaskablePPO


def test_bridge_rejects_unavailable_instance_group_contract(tmp_path, monkeypatch):
    from bridge.client import agent_orchestrator as agent
    path = tmp_path/'model.zip'
    path.touch()
    monkeypatch.setattr(agent.MaskablePPO, 'load', lambda *_a, **_k:
        SimpleNamespace(nosl_action_group_contract='equivalent-plays-v1'))
    with pytest.raises(ValueError, match='per-instance fields'):
        agent._load_model(str(path), 'cpu')


def test_bridge_allows_explicit_direct_policy_diagnostic(tmp_path, monkeypatch):
    from bridge.client import agent_orchestrator as agent
    path = tmp_path/'model.zip'
    path.touch()
    expected = SimpleNamespace(nosl_action_group_contract='equivalent-plays-v1')
    monkeypatch.setattr(agent.MaskablePPO, 'load', lambda *_a, **_k: expected)

    actual = agent._load_model(
        str(path), 'cpu', allow_direct_combat_policy=True)

    assert actual is expected


def test_bridge_allows_grouped_checkpoint_for_simulator_search(tmp_path, monkeypatch):
    from bridge.client import agent_orchestrator as agent
    path = tmp_path/'model.zip'
    path.touch()
    expected = SimpleNamespace(nosl_action_group_contract='equivalent-plays-v1')
    monkeypatch.setattr(agent.MaskablePPO, 'load', lambda *_a, **_k: expected)

    actual = agent._load_model(
        str(path), 'cpu', simulator_combat_search=True)

    assert actual is expected


def test_bridge_relocates_frozen_card_vector_checkpoint(tmp_path, monkeypatch):
    from bridge.client import agent_orchestrator as agent
    path = tmp_path/'model.zip'
    card_vectors = tmp_path/'card-vectors.pt'
    path.touch()
    card_vectors.touch()
    saved = {
        'policy_kwargs': {
            'entity_config': {
                'width': 256,
                'card_vector_checkpoint': '/remote/card-vectors.pt',
            },
            'upper_config': {'width': 256},
        }
    }
    monkeypatch.setattr(
        agent, 'load_from_zip_file', lambda *_a, **_k: (saved, {}, {}))
    observed = {}

    def fake_load(*_args, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(nosl_action_group_contract='individual-actions')

    monkeypatch.setattr(agent.MaskablePPO, 'load', fake_load)

    agent._load_model(
        str(path), 'cpu', card_vector_checkpoint=str(card_vectors))

    relocated = observed['custom_objects']['policy_kwargs']
    assert relocated['entity_config']['width'] == 256
    assert relocated['entity_config']['card_vector_checkpoint'] == str(card_vectors.resolve())
    assert relocated['upper_config'] == {'width': 256}
    assert saved['policy_kwargs']['entity_config']['card_vector_checkpoint'] == '/remote/card-vectors.pt'


def test_mod_error_refreshes_state_instead_of_restarting_run():
    from bridge.client import agent_orchestrator as agent
    from bridge.real_game_env import RealGameAdapter

    driver = agent.RealPolicyAgent(
        object(),
        RealGameAdapter(),
        stochastic=False,
        topk=0,
        max_inner_decisions=1,
        auto_enter_shop=True,
        auto_start=True,
        start_seed="108",
        ascension=0,
        character="IRONCLAD",
    )

    decision = driver.decide({
        "ready_for_command": False,
        "error": "Selected card requires an enemy target.",
        "available_commands": ["state"],
    })

    assert decision.command == "STATE"
    assert decision.reason == "mod_error_state_refresh"


def test_reward_preview_transport_does_not_call_policy_between_commands():
    from bridge.client import agent_orchestrator as agent
    from bridge.real_game_env import RealGameAdapter
    from env.core import action_space as A, game_actions as ga

    driver = agent.RealPolicyAgent(
        object(),
        RealGameAdapter(),
        stochastic=False,
        topk=0,
        max_inner_decisions=1,
        auto_enter_shop=True,
        auto_start=False,
        start_seed=None,
        ascension=0,
        character="IRONCLAD",
    )
    policy_calls = []
    driver._predict = lambda *_args, **_kwargs: (
        policy_calls.append("predict") or A._RGOLD, ())
    outer = {
        "in_game": True,
        "ready_for_command": True,
        "available_commands": ["choose", "proceed", "state"],
        "game_state": {
            "screen_type": "COMBAT_REWARD",
            "choice_list": ["gold", "card"],
            "screen_state": {"rewards": [
                {"reward_type": "GOLD", "gold": 27},
                {"reward_type": "CARD"},
            ]},
            "potions": [{"id": "Potion Slot"}] * 3,
        },
    }
    card_page = {
        "in_game": True,
        "ready_for_command": True,
        "available_commands": ["choose", "skip", "state"],
        "game_state": {
            "screen_type": "CARD_REWARD",
            "choice_list": ["Anger", "Searing Blow", "Whirlwind"],
            "screen_state": {"cards": [
                {"id": "Anger", "upgrades": 0},
                {"id": "Searing Blow", "upgrades": 0},
                {"id": "Whirlwind", "upgrades": 0},
            ]},
            "potions": [{"id": "Potion Slot"}] * 3,
        },
    }

    open_page = driver.decide(outer)
    choose_outer = driver.decide(card_page)
    complete_outer = driver.decide(outer)

    assert (open_page.command, open_page.reason) == (
        "CHOOSE 1", "open_card_reward")
    assert choose_outer.command == "SKIP"
    assert choose_outer.semantic == ga.claim(ga.CLAIM_GOLD, 0)
    assert (complete_outer.command, complete_outer.reason) == (
        "CHOOSE 0", "complete_deferred_reward_action")
    assert policy_calls == ["predict"]


@pytest.mark.parametrize('screen,room,command,delayed', [
    ('EVENT', 'EVENT', 'CHOOSE 0', True),
    ('EVENT', 'EVENT', 'PROCEED', True),
    ('CARD_REWARD', 'COMPLETE', 'CHOOSE 1', True),
    ('CARD_REWARD', 'COMPLETE', 'SKIP', True),
    ('COMBAT_REWARD', 'COMPLETE', 'CHOOSE card', True),
    ('GRID', 'EVENT', 'CHOOSE 0', True),
    ('CARD_REWARD', 'COMBAT', 'CHOOSE 0', False),
    ('NONE', 'COMBAT', 'PLAY 1 0', False),
    ('EVENT', 'EVENT', 'STATE', False),
    ('SHOP_SCREEN', 'INCOMPLETE', 'CHOOSE 0', True),
    ('SHOP_SCREEN', 'INCOMPLETE', 'PROCEED', True),
    ('SHOP_ROOM', 'INCOMPLETE', 'CHOOSE shop', True),
    ('SHOP_SCREEN', 'INCOMPLETE', 'STATE', False),
])
def test_real_event_reward_command_pacing(monkeypatch, screen, room, command, delayed):
    from bridge.client import agent_orchestrator as agent
    calls = []
    monkeypatch.setattr(agent.time, 'sleep', lambda seconds: calls.append(('wait', seconds)))
    monkeypatch.setattr(agent, 'send', lambda value: calls.append(('send', value)))
    agent.send_decision({'game_state': {'screen_type': screen, 'room_phase': room}},
                        SimpleNamespace(command=command))
    assert calls == ([('wait', 0.5)] if delayed else []) + [('send', command)]


def test_duplicate_frame_after_play_waits_for_actual_hand_transition():
    from copy import deepcopy
    from bridge.client.agent_orchestrator import stale_combat_play_frame

    gs = {"floor": 35, "combat_state": {"hand": [{"uuid": "inflame"}], "turn": 2}}
    assert stale_combat_play_frame({"game_state": deepcopy(gs)}, gs)
    # Transport readiness changes do not acknowledge the card action.
    assert stale_combat_play_frame({"game_state": deepcopy(gs), "ready_for_command": True}, gs)
    after = deepcopy(gs)
    after["combat_state"]["hand"] = []
    assert not stale_combat_play_frame({"game_state": after}, gs)
    assert not stale_combat_play_frame({"error": "rejected"}, gs)
    assert not stale_combat_play_frame({"game_state": gs}, None)


def test_search_send_requires_fresh_unchanged_state_and_command_permission():
    from copy import deepcopy
    from bridge.client.agent_orchestrator import searched_command_still_current
    original = {"ready_for_command": True, "available_commands": ["play", "end"],
                "game_state": {"screen_type": "NONE", "is_screen_up": False}}
    assert searched_command_still_current(deepcopy(original), original, "PLAY 2 0")
    changed = deepcopy(original)
    changed["game_state"]["is_screen_up"] = True
    assert not searched_command_still_current(changed, original, "PLAY 2 0")
    changed = deepcopy(original)
    changed["available_commands"] = ["state"]
    assert not searched_command_still_current(changed, original, "PLAY 2 0")
    changed = deepcopy(original)
    changed["ready_for_command"] = False
    assert not searched_command_still_current(changed, original, "PLAY 2 0")


def test_cancel_unsent_search_restores_native_and_selection_state():
    from types import SimpleNamespace
    from bridge.search_bridge import RealCombatSearch
    bridge = RealCombatSearch.__new__(RealCombatSearch)
    saved = object()
    bridge._awaiting_command = True
    bridge._rollback_bc = saved
    bridge._rollback_thin = {"pending": {"source": "deck"}, "multi": None,
                             "real_multi_selected": {2}, "real_joint_multi_plan": [2]}
    bridge.env = SimpleNamespace(thick=SimpleNamespace(bc=object()),
                                 _invalidate_backend_cache=lambda: None)
    bridge.adapter = SimpleNamespace(multi_selected={3}, _joint_multi_plan=[3])
    bridge.cancel_unsent_command()
    assert bridge.env.thick.bc is saved
    assert bridge.env._pending == {"source": "deck"}
    assert bridge.adapter.multi_selected == {2}
    assert bridge.adapter._joint_multi_plan == [2]
    assert not bridge._awaiting_command and bridge._rollback_bc is None
    bridge.cancel_unsent_command()  # idempotent, no command has been sent


@pytest.mark.parametrize("command,error,recover", [
    ("END", "Invalid command: end. Possible commands: [key, click, wait, state]", True),
    ("END", "Invalid command: end. Possible commands: [choose, return, key, click, wait, state]", True),
    ("PLAY 1", "Invalid command: end. Possible commands: [key, click, wait, state]", False),
    ("PLAY 1", "Invalid command: play. Possible commands: [key, click, wait, state]", True),
    ("END", "Invalid argument: end", False),
])
def test_predispatch_end_rejection_is_audited_and_rolled_back(command, error, recover, caplog):
    from types import SimpleNamespace
    from bridge.search_bridge import RealCombatSearch
    bridge = RealCombatSearch.__new__(RealCombatSearch)
    saved = object()
    bridge._awaiting_command = True
    bridge._awaiting_command_text = command
    bridge._rollback_bc = saved
    bridge._rollback_thin = {}
    bridge.run_rejected_commands = 0
    bridge._consecutive_rejected_commands = 0
    bridge.env = SimpleNamespace(thick=SimpleNamespace(bc=object()),
                                 _invalidate_backend_cache=lambda: None)
    bridge.adapter = SimpleNamespace(multi_selected=set(), _joint_multi_plan=None)
    bridge.observe_frame({"error": error})
    assert bridge.env.thick.bc is saved
    assert not bridge._awaiting_command
    assert bridge.run_rejected_commands == int(recover)
    assert ("ERROR" not in caplog.text) == recover
    if recover:
        assert "transport_rejected_commands=1" in caplog.text
        bridge._consecutive_rejected_commands = 3
        bridge._awaiting_command = True
        bridge._rollback_bc = saved
        with pytest.raises(RuntimeError, match="exceeded 3"):
            bridge.observe_frame({"error": error})
        assert not bridge._awaiting_command


def test_end_requires_turn_ack_not_changed_commands_or_player_state():
    from copy import deepcopy
    from bridge.client.agent_orchestrator import stale_combat_end_frame
    gs = {"seed": 200200, "floor": 18, "room_phase": "COMBAT",
          "combat_state": {"turn": 1, "hand": [], "player": {"energy": 1}}}
    after = deepcopy(gs)
    after["combat_state"]["player"]["energy"] = 5
    assert stale_combat_end_frame({"game_state": after, "available_commands": ["end"]}, gs)
    after["combat_state"]["turn"] = 2
    assert not stale_combat_end_frame({"game_state": after}, gs)
    assert not stale_combat_end_frame({"error": "rejected"}, gs)
    assert not stale_combat_end_frame({"game_state": {"room_phase": "COMPLETE"}}, gs)
    terminal = dict(gs, screen_type="GAME_OVER", screen_state={"victory": False})
    assert not stale_combat_end_frame({"game_state": terminal}, gs)


def test_opening_empty_frame_and_inflight_draw_are_not_policy_ready():
    from bridge.real_game_env import RealGameAdapter
    adapter = RealGameAdapter()
    frame = {"in_game": True, "ready_for_command": True,
             "available_commands": ["end", "state"], "game_state": {
                 "screen_type": "NONE", "room_phase": "COMBAT",
                 "action_phase": "WAITING_ON_USER", "combat_state": {
                     "turn": 1, "hand": [], "draw_pile": [{"id": "Strike_R"}],
                     "player": {"energy": 0}}}}
    adapter.update(frame)
    assert adapter.auto_command() == ("STATE", "combat_not_waiting")
    frame["game_state"]["combat_state"]["player"]["energy"] = 5
    frame["game_state"]["current_action"] = "DrawCardAction"
    adapter.update(frame)
    assert adapter.auto_command() == ("STATE", "combat_not_waiting")
    frame["game_state"].pop("current_action")
    frame["game_state"]["combat_state"]["turn"] = 2
    adapter.update(frame)
    assert adapter._combat_ready_for_policy()  # a legitimate empty hand remains legal


def test_reward_open_waits_for_card_screen_not_stale_or_overlay_frames():
    from copy import deepcopy
    from bridge.client.agent_orchestrator import RewardCommandAck
    from bridge.real_game_env import RealDecision
    gate = RewardCommandAck()
    frame = {'game_state': {'seed': 1, 'floor': 25, 'screen_type': 'COMBAT_REWARD',
                            'deck': [], 'choice_list': ['gold', 'potion', 'card']}}
    gate.arm(frame, RealDecision('CHOOSE 2', 'open_card_reward'), 0)
    assert gate.waiting(frame, 0.5)
    assert gate.waiting(frame, 1.0)
    overlay = deepcopy(frame)
    overlay['game_state']['is_screen_up'] = True
    assert gate.waiting(overlay, 1.5)
    with pytest.raises(RuntimeError, match='not resending'):
        gate.waiting(frame, 11)
    frame['game_state']['screen_type'] = 'CARD_REWARD'
    assert not gate.waiting(frame, 12)


def test_reward_pick_requires_intended_card_to_enter_deck():
    from bridge.client.agent_orchestrator import RewardCommandAck
    from bridge.real_game_env import RealDecision
    gate = RewardCommandAck()
    frame = {'game_state': {'seed': 1, 'floor': 25, 'screen_type': 'CARD_REWARD',
                            'deck': [], 'screen_state': {'cards': [{'id': 'Searing Blow'}]}}}
    gate.arm(frame, RealDecision('CHOOSE 0', 'policy', semantic=('reward_card', 0)), 0)
    frame['game_state']['screen_type'] = 'COMBAT_REWARD'
    frame['game_state']['deck'] = [{'id': 'Feel No Pain'}]
    assert gate.waiting(frame, 1)
    with pytest.raises(RuntimeError, match='not resending'):
        gate.waiting(frame, 11)
    frame['game_state']['deck'] = [{'id': 'Searing Blow', 'upgrades': 2}]
    assert not gate.waiting(frame, 12)


def test_reward_rejection_is_not_silently_retried():
    from bridge.client.agent_orchestrator import RewardCommandAck
    from bridge.real_game_env import RealDecision
    gate = RewardCommandAck()
    gate.arm({'game_state': {'screen_type': 'COMBAT_REWARD'}},
             RealDecision('PROCEED', 'policy'), 0)
    with pytest.raises(RuntimeError, match='reward command rejected'):
        gate.waiting({'error': 'Invalid command'}, 1)


@pytest.mark.parametrize('command', ['CHOOSE 0', 'PROCEED', 'SKIP'])
def test_reward_mutation_waits_for_state_change(command):
    from copy import deepcopy
    from bridge.client.agent_orchestrator import RewardCommandAck
    from bridge.real_game_env import RealDecision
    gate = RewardCommandAck()
    frame = {'game_state': {'screen_type': 'COMBAT_REWARD', 'gold': 10}}
    gate.arm(frame, RealDecision(command, 'policy'), 0)
    assert gate.waiting({}, .1)
    assert gate.waiting(deepcopy(frame), .5)
    frame['game_state']['gold'] = 25
    assert not gate.waiting(frame, 1)


def test_combat_stability_requires_fresh_unchanged_frame_after_animation():
    from copy import deepcopy
    from bridge.client.agent_orchestrator import CombatFrameStability
    gate = CombatFrameStability()
    frame = {'ready_for_command': True, 'game_state': {
        'room_phase': 'COMBAT', 'screen_type': 'NONE', 'action_phase': 'WAITING_ON_USER',
        'combat_state': {'turn': 1, 'hand': [], 'player': {'energy': 0}}}}
    assert gate.waiting(frame, 0)
    assert gate.waiting(frame, 0.4)
    frame['game_state']['combat_state']['player']['energy'] = 5
    assert gate.waiting(frame, 0.45)
    frame['game_state']['combat_state']['hand'] = [{'id': 'Strike_R'}]
    assert gate.waiting(frame, 0.6)
    assert gate.waiting(frame, 1.0)
    assert not gate.waiting(deepcopy(frame), 1.11)
    frame['game_state']['current_action'] = 'DrawCardAction'
    assert gate.waiting(frame, 2)
    frame['game_state'].pop('current_action')
    assert gate.waiting(frame, 3)
    assert not gate.waiting(frame, 3.51)


def test_combat_stability_does_not_deadlock_executing_selection():
    from bridge.client.agent_orchestrator import CombatFrameStability
    gate = CombatFrameStability()
    frame = {'game_state': {'room_phase': 'COMBAT', 'screen_type': 'HAND_SELECT',
                           'action_phase': 'EXECUTING_ACTIONS', 'combat_state': {'turn': 1}}}
    assert not gate.waiting(frame, 0)


def test_deck_overlay_waits_until_closed_then_requires_fresh_stability():
    from bridge.client.agent_orchestrator import CombatFrameStability
    gate = CombatFrameStability()
    frame = {'ready_for_command': True, 'game_state': {
        'room_phase': 'COMBAT', 'screen_type': 'NONE', 'is_screen_up': True,
        'combat_state': {'turn': 3, 'hand': [], 'player': {'energy': 0}}}}
    assert gate.waiting(frame, 0)
    assert gate.waiting(frame, 60)
    frame['game_state']['is_screen_up'] = False
    assert gate.waiting(frame, 61)
    assert gate.waiting(frame, 61.4)
    assert not gate.waiting(frame, 61.51)


def test_opening_guard_allows_spent_first_turn_and_resets_next_battle():
    from copy import deepcopy
    from bridge.real_game_env import RealGameAdapter
    adapter = RealGameAdapter()
    frame = {'in_game': True, 'ready_for_command': True,
             'available_commands': ['end', 'state'], 'game_state': {
                 'seed': 200200, 'floor': 14, 'screen_type': 'NONE',
                 'room_phase': 'COMBAT', 'action_phase': 'WAITING_ON_USER',
                 'combat_state': {'turn': 1, 'hand': [],
                                  'draw_pile': [{'id': 'Strike_R'}],
                                  'player': {'energy': 0}}}}
    adapter.update(deepcopy(frame))
    assert not adapter._combat_ready_for_policy()
    ready = deepcopy(frame)
    ready['game_state']['combat_state']['hand'] = [{'id': 'Strike_R'}]
    ready['game_state']['combat_state']['player']['energy'] = 4
    adapter.update(ready)
    assert adapter._combat_ready_for_policy()
    adapter.update({'error': 'transient rejection'})
    adapter.update(deepcopy(frame))  # all cards/energy spent on the same turn
    assert adapter._combat_ready_for_policy()
    frame['game_state']['floor'] = 15
    adapter.update(frame)
    assert not adapter._combat_ready_for_policy()


@pytest.mark.parametrize('command', ['POTION use 0', 'CHOOSE 4', 'CONFIRM', 'RETURN', 'PLAY 1'])
def test_combat_mutations_wait_for_real_ack(command):
    from copy import deepcopy
    from bridge.client.agent_orchestrator import combat_command_needs_ack, stale_combat_play_frame
    gs = {'combat_state': {'turn': 1, 'hand': [{'id': 'Armaments'}]},
          'potions': [{'id': 'GamblersBrew'}], 'screen_type': 'NONE'}
    frame = {'game_state': gs}
    assert combat_command_needs_ack(frame, command)
    assert not combat_command_needs_ack(frame, 'STATE')
    assert not combat_command_needs_ack({'game_state': {}}, command)
    for _ in range(4):
        assert stale_combat_play_frame({'game_state': deepcopy(gs), 'ready_for_command': True}, gs)
    after = deepcopy(gs)
    after['potions'] = [{'id': 'Potion Slot'}]
    after['screen_type'] = 'HAND_SELECT'
    assert not stale_combat_play_frame({'game_state': after}, gs)
