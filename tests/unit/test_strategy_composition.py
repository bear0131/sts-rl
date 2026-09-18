"""Independent upper/combat configuration must preserve phase ownership."""
from argparse import Namespace
from types import SimpleNamespace
import sys

import pytest

from rl.evaluation import strategies as S


class Sentinel:
    def __init__(self, name):
        self.name = name
        self.phases = []

    def decide(self, requests, *, check):
        self.phases.extend(r.info['phase'] for r in requests)
        return [S.Decision(7, self.name) for _ in requests]


def requests():
    return [S.DecisionRequest(None, {}, {'phase': phase}, 1, 0)
            for phase in ('map', 'combat', 'shop', 'combat')]


def test_separate_combat_model_never_takes_upper_requests(monkeypatch):
    upper, lower = Sentinel('upper'), Sentinel('lower')
    monkeypatch.setattr(S, 'policy_strategy', lambda model, args: model)
    monkeypatch.setitem(S.STRATEGIES, 'fixture', lambda model, args: model)
    args = Namespace(solver='fixture', upper_strategy='model', route='model')
    result = S.compose_strategy(upper, args, lower).decide(requests())
    assert [r.source for r in result] == ['upper', 'lower', 'upper', 'lower']
    assert upper.phases == ['map', 'shop']
    assert lower.phases == ['combat', 'combat']


def test_upper_rules_composition_has_explicit_fallback_and_no_combat_calls(monkeypatch):
    upper, lower = Sentinel('upper-model'), Sentinel('lower')
    class UpperRules(Sentinel):
        version = 'test-v1'
        def __init__(self, *, fallback, trace):
            super().__init__('upper-rules')
            self.fallback, self.trace = fallback, trace
    monkeypatch.setattr(S, 'policy_strategy', lambda model, args: model)
    monkeypatch.setitem(S.STRATEGIES, 'fixture', lambda model, args: model)
    monkeypatch.setitem(sys.modules, 'rl.evaluation.baselines.overworld_rules',
                        SimpleNamespace(OverworldRuleStrategy=UpperRules))
    args = Namespace(solver='fixture', upper_strategy='rules', route='model')
    strategy = S.compose_strategy(upper, args, lower)
    result = strategy.decide(requests())
    assert [r.source for r in result] == ['upper-rules', 'lower', 'upper-rules', 'lower']
    assert strategy.default.phases == ['map', 'shop']
    assert not upper.phases
    assert S.strategy_manifest(strategy)['default'] == {
        'name': 'upper-rules', 'version': 'test-v1', 'fallback': {'name': 'upper-model'}}


@pytest.mark.parametrize('selection', [None, 'model'])
def test_default_upper_selection_preserves_existing_solver(monkeypatch, selection):
    solver = Sentinel('arbitrary-solver')
    monkeypatch.setitem(S.STRATEGIES, 'fixture', lambda model, args: solver)
    args = Namespace(solver='fixture', route='model')
    if selection is not None:
        args.upper_strategy = selection
    strategy = S.compose_strategy(object(), args)
    assert strategy is solver
    assert [d.source for d in strategy.decide(requests(), check=lambda: None)] == [solver.name] * 4


def test_custom_upper_factory_receives_upper_model_trace_and_never_combat(monkeypatch):
    upper, lower, selected = Sentinel('upper'), Sentinel('lower'), Sentinel('custom')
    trace = object()
    monkeypatch.setattr(S, 'policy_strategy', lambda model, args: model)
    monkeypatch.setitem(S.STRATEGIES, 'fixture', lambda model, args: model)
    def factory(model, args, *, trace):
        assert model is upper
        selected.trace = trace
        return selected
    monkeypatch.setitem(S.UPPER_STRATEGIES, 'custom', factory)
    args = Namespace(solver='fixture', upper_strategy='custom', route='model')
    strategy = S.compose_strategy(upper, args, lower, trace=trace)
    assert [d.source for d in strategy.decide(requests())] == ['custom', 'lower', 'custom', 'lower']
    assert selected.phases == ['map', 'shop']
    assert selected.trace is trace and upper.phases == []


def test_demand_factory_manifest_preserves_exact_supplied_model_fallback():
    model = object()
    args = Namespace(stochastic=False)
    strategy = S.UPPER_STRATEGIES['demand'](model, args)
    assert strategy.fallback.model is model
    assert strategy._base.fallback is strategy.fallback
    assert S.strategy_manifest(strategy) == {
        'name': 'overworld-demand', 'version': 'ironclad-overworld-demand-v1',
        'fallback': {'name': 'policy', 'deterministic': True}}


def test_cli_registers_upper_factories_without_changing_defaults():
    from rl.evaluation.eval import build_parser
    parser = build_parser()
    assert parser.parse_args(['unused.zip']).upper_strategy == 'model'
    for name in ('model', 'rules', 'demand'):
        assert parser.parse_args(['unused.zip', '--upper-strategy', name]).upper_strategy == name


def test_unknown_upper_selection_fails_explicitly(monkeypatch):
    monkeypatch.setitem(S.STRATEGIES, 'fixture', lambda model, args: Sentinel('x'))
    with pytest.raises(ValueError, match='unknown upper strategy'):
        S.compose_strategy(None, Namespace(solver='fixture', upper_strategy='missing'))
