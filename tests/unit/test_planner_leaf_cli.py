"""Reject unsupported leaf experiments before any checkpoint or game work."""
import pytest

from rl.evaluation import eval as cli
from rl.evaluation.strategies import compose_strategy, strategy_manifest


@pytest.mark.parametrize('extra,match', [
    (['--planner-leaf-value', 'checkpoint'], 'requires --planner-leaf-boundary'),
    (['--planner-leaf-boundary', 'next-decision', '--planner-turns', '2'], 'require --task wholerun'),
    (['--planner-leaf-boundary', 'next-decision', '--task', 'combat'], 'require --task wholerun'),
    (['--planner-leaf-boundary', 'next-decision', '--solver', 'policy'], 'require --task wholerun'),
    (['--planner-leaf-boundary', 'next-decision', '--planner-turns', '2',
      '--planner-turn-scope', 'boss-elite-v1'], 'require --task wholerun'),
    (['--planner-leaf-boundary', 'next-decision', '--combat-search-depth', '1'],
     'combat-search cannot be combined'),
])
def test_leaf_cli_rejects_unsupported_combinations_before_load(monkeypatch, extra, match):
    monkeypatch.setattr('sys.argv', ['eval', 'unused.zip', '--solver', 'turn-planner', *extra])
    monkeypatch.setattr(cli, 'load_checkpoint', lambda *a, **k: pytest.fail('checkpoint loaded'))
    with pytest.raises(ValueError, match=match):
        cli.main()


def test_next_decision_heuristic_has_explicit_manifest_and_same_budget():
    args = cli.build_parser().parse_args(['unused.zip', '--solver', 'turn-planner',
        '--planner-leaf-boundary', 'next-decision', '--planner-nodes', '8192'])
    strategy = compose_strategy(object(), args)
    manifest = strategy_manifest(strategy)['phases']['combat']
    assert manifest['config']['leaf_boundary'] == 'next-decision'
    assert manifest['config']['lookahead_turns'] == 1
    assert manifest['config']['node_budget'] == 8192
    assert manifest['config']['worlds'] == 2
    assert manifest['horizon'] == 'next-legal-player-decision'
    assert manifest['leaf_value'] == 'heuristic'
    assert strategy.combat_action_contract == 'native-v1'


def test_checkpoint_option_requires_actual_compatible_model():
    args = cli.build_parser().parse_args(['unused.zip', '--solver', 'turn-planner',
        '--planner-leaf-boundary', 'next-decision', '--planner-leaf-value', 'checkpoint'])
    with pytest.raises(ValueError, match='requires Entity combat-only'):
        compose_strategy(object(), args)
