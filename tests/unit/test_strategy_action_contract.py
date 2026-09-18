"""Environment requirements derive from dispatch, never strategy names."""
from types import SimpleNamespace
import json
import sys

import pytest

from rl.evaluation.strategies import Decision, DecisionRequest, PhaseStrategy, strategy_manifest


class Custom:
    name = 'same-name-for-both-contracts'
    def __init__(self, contract='legacy'):
        self.combat_action_contract = contract
    def decide(self, requests, *, check):
        return [Decision(7, self.combat_action_contract) for _ in requests]


def test_nested_noncombat_routes_preserve_only_actual_combat_requirement():
    native, legacy = Custom('native-v1'), Custom()
    nested = PhaseStrategy(PhaseStrategy(legacy, {'combat': native}), {'map': legacy})
    assert nested.combat_action_contract == 'native-v1'
    assert strategy_manifest(nested)['combat_action_contract'] == 'native-v1'
    # Native default is unreachable during combat without a rejecting predicate.
    assert PhaseStrategy(native, {'combat': legacy}).combat_action_contract == 'legacy'
    assert PhaseStrategy(legacy, {'map': native}).combat_action_contract == 'legacy'
    assert PhaseStrategy(native, {}).combat_action_contract == 'native-v1'


@pytest.mark.parametrize('default,route', [('native-v1', 'legacy'), ('legacy', 'native-v1')])
def test_predicate_requires_union_even_when_one_path_returns_thin(default, route):
    strategy = PhaseStrategy(Custom(default), {'combat': Custom(route)},
        predicates={'combat': lambda request: request.step > 0})
    assert strategy.combat_action_contract == 'native-v1'
    requests = [DecisionRequest(None, {}, {'phase': 'combat'}, 1, step) for step in (0, 1)]
    decisions = strategy.decide(requests)
    assert [d.source for d in decisions] == [default, route]
    assert [d.action for d in decisions] == [7, 7]


def test_missing_declaration_is_legacy_unknown_contract_is_rejected():
    assert PhaseStrategy(SimpleNamespace(name='custom'), {}).combat_action_contract == 'legacy'
    with pytest.raises(ValueError, match='unsupported combat action'):
        _ = PhaseStrategy(Custom('future-unsupported'), {}).combat_action_contract


def test_cli_preserves_candidate_construction_error_in_final_payload(tmp_path, monkeypatch):
    from rl.evaluation import eval as cli, strategies as strategies
    model = SimpleNamespace(set_random_seed=lambda seed: None)
    monkeypatch.setattr(cli, 'load_checkpoint', lambda *a, **kw: model)
    calls = []
    def factory(model, args):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError('candidate setup failed')
        return strategies.turn_planner_strategy(model, args)
    monkeypatch.setitem(strategies.STRATEGIES, 'turn-planner', factory)
    output = tmp_path / 'failed.json'
    monkeypatch.setattr(sys, 'argv', ['eval', 'unused.zip', '--solver', 'turn-planner',
        '--compare-combats', '1', '--output', str(output)])
    with pytest.raises(RuntimeError, match='candidate setup failed'):
        cli.main()
    payload = json.loads(output.read_text())
    assert payload['state'] == 'error'
    assert payload['error']['message'] == 'candidate setup failed'
    assert payload['combat_audit'] is None
    assert payload['rows'] == payload['combat_rows'] == []
