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


def test_combat_only_model_automatically_uses_rules_without_neural_fallback(monkeypatch):
    lower=Sentinel('combat-only')
    lower.policy=SimpleNamespace(core=SimpleNamespace(combat_only=True))
    class UpperRules(Sentinel):
        def __init__(self,*,fallback,trace):
            super().__init__('upper-rules')
            self.fallback=fallback
    monkeypatch.setitem(S.STRATEGIES,'fixture',lambda model,args:model)
    monkeypatch.setitem(sys.modules,'rl.evaluation.baselines.overworld_rules',
                        SimpleNamespace(OverworldRuleStrategy=UpperRules))
    strategy=S.compose_strategy(lower,Namespace(solver='fixture',upper_strategy='model',route='model'))
    assert strategy.default.fallback is None
    assert [r.source for r in strategy.decide(requests())]==['upper-rules','combat-only','upper-rules','combat-only']
    assert lower.phases==['combat','combat']


@pytest.mark.parametrize('options,message', [
    (['--simulations', '1'], 'collection search budget'),
    (['--depth', '1'], 'collection search budget'),
    (['--trace', 'trace.jsonl'], 'combat-audit wrapper'),
    (['--compare-combats', '1'], 'combat-audit wrapper'),
    (['--native-search-threads', '-1'], 'non-negative'),
    (['--task', 'combat'], 'whole-run checkpoint'),
])
def test_small_playout_cli_rejects_unsupported_combinations(options, message):
    from rl.evaluation.eval import build_parser
    from env.core.native_actions import AUTOREGRESSIVE_WIDTH
    from env.search.small_playout import SEARCH_CONTRACT
    model = SimpleNamespace(
        observation_space=SimpleNamespace(spaces={'combat_legal': SimpleNamespace(shape=(AUTOREGRESSIVE_WIDTH,))}),
        policy=SimpleNamespace(small_playout=object()), small_playout_search_contract=SEARCH_CONTRACT)
    args = build_parser().parse_args(['model.zip', '--solver', 'small-playout', *options])
    with pytest.raises(ValueError, match=message):
        S.small_playout_strategy(model, args)
