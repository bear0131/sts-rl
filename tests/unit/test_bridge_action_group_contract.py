from types import SimpleNamespace

import pytest


def test_bridge_rejects_unavailable_instance_group_contract(tmp_path, monkeypatch):
    from bridge.client import agent_orchestrator as agent
    path = tmp_path/'model.zip'
    path.touch()
    monkeypatch.setattr(agent.MaskablePPO, 'load', lambda *_a, **_k:
        SimpleNamespace(nosl_action_group_contract='equivalent-plays-v1'))
    with pytest.raises(ValueError, match='per-instance fields'):
        agent._load_model(str(path), 'cpu')
