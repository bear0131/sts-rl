from types import SimpleNamespace
from zipfile import ZipFile

import pytest


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
])
def test_real_event_reward_command_pacing(monkeypatch, screen, room, command, delayed):
    from bridge.client import agent_orchestrator as agent
    calls = []
    monkeypatch.setattr(agent.time, 'sleep', lambda seconds: calls.append(('wait', seconds)))
    monkeypatch.setattr(agent, 'send', lambda value: calls.append(('send', value)))
    agent.send_decision({'game_state': {'screen_type': screen, 'room_phase': room}},
                        SimpleNamespace(command=command))
    assert calls == ([('wait', 0.5)] if delayed else []) + [('send', command)]
