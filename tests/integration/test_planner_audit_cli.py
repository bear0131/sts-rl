"""Shared CLI/runner integration on one controlled battle, no model training."""
import json
import sys

import pytest
import torch

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.evaluation import eval as cli
from rl.evaluation.runner import Scenario
from rl.policies.split_policy import SplitStSPointerPolicy


def combat_leaf(strategy):
    while hasattr(strategy, 'routes'):
        strategy = strategy.routes.get('combat', strategy.default)
    return strategy


@pytest.mark.parametrize('upper', ['model', 'rules', 'demand', 'energy'])
def test_cli_planner_audit_runs_all_four_candidates_without_sharing_reference(
        tmp_path, monkeypatch, combat_factory, upper):
    root = combat_factory(['STRIKE_RED'] * 5)
    root.thick.bc.set_player_strength(999)
    root._invalidate_backend_cache()
    before_native = json.dumps(root.thick.bc.to_dict(), sort_keys=True)
    # An initialized tiny policy exercises real inference/encoding/native search;
    # no learn()/optimizer update or supplied checkpoint modification occurs.
    model = AccumulatingMaskablePPO(SplitStSPointerPolicy, root, device='cpu', seed=7,
        n_steps=2, batch_size=2, policy_kwargs=dict(d_model=32, n_layers=2, n_heads=4))
    model.policy.set_training_mode(False)
    weights = {key: value.clone() for key, value in model.policy.state_dict().items()}
    monkeypatch.setattr(cli, 'load_checkpoint', lambda *a, **kw: model)
    made_contracts = []

    def fixture_scenario(model, args, lower, contract):
        made_contracts.append(contract)
        def make():
            branch = root.fork_combat()
            branch.combat_action_contract = contract
            branch._invalidate_backend_cache()
            return branch
        return Scenario('bounded-cli-combat-fixture', make,
            outcome=lambda env, info, terminated, truncated: dict(
                result='win' if env.thick.bc.outcome == 1 else 'death', floor=1, act=1),
            reset=lambda env, seed: (env._obs(), env._info()))
    monkeypatch.setattr(cli, 'make_scenario', fixture_scenario)
    runner = cli.evaluate
    inspected = []

    def inspect_then_run(audit, *args, **kwargs):
        reference = combat_leaf(audit.reference)
        candidate = combat_leaf(audit.candidates['turn-planner'])
        assert reference is not candidate
        assert reference.planner is not candidate.planner
        assert reference.planner.sampler is not candidate.planner.sampler
        assert reference.config == candidate.config
        assert set(audit.candidates) == {'rules', 'policy', 'native', 'turn-planner'}
        inspected.append(True)
        yield from runner(audit, *args, **kwargs)
    monkeypatch.setattr(cli, 'evaluate', inspect_then_run)
    output, trace = tmp_path / 'result.json', tmp_path / 'trace.jsonl'
    monkeypatch.setattr(sys, 'argv', ['eval', 'unused-fixture.zip', '--solver', 'turn-planner',
        '--upper-strategy', upper, '--compare-combats', '1', '--episodes', '1',
        '--device', 'cpu', '--max-steps', '4', '--timeout', '20', '--simulations', '2',
        '--depth', '0', '--planner-nodes', '16', '--planner-worlds', '1',
        '--planner-beam', '2', '--planner-depth', '2', '--output', str(output), '--trace', str(trace)])
    cli.main()
    payload = json.loads(output.read_text())
    assert inspected == [True] and made_contracts == ['native-v1']
    assert payload['state'] == 'complete'
    assert len(payload['rows']) == 1
    assert len(payload['combat_rows']) == 4
    components = {row['candidate']: row for row in payload['combat_rows']}
    assert components['turn-planner']['candidate_combat_action_contract'] == 'native-v1'
    assert all(components[name]['candidate_combat_action_contract'] == 'legacy'
               for name in ('rules', 'policy', 'native'))
    assert len({row['visible_state_sha256'] for row in payload['combat_rows']}) == 1
    assert all(row['fork_decision_step_origin'] == 0 for row in payload['combat_rows'])
    assert payload['effective_strategy']['combat_action_contract'] == 'native-v1'
    assert payload['combat_candidates']['turn-planner']['combat_action_contract'] == 'native-v1'
    assert payload['combat_audit']['decision_step_origin'] == 'fork-local-zero'
    if upper in ('rules', 'demand', 'energy'):
        default = payload['effective_strategy']['default']
        assert default['name'] == f'overworld-{upper}'
        assert default['fallback']['name'] == 'policy'
        assert default['version'] == {'rules': 'ironclad-overworld-v1',
                                      'demand': 'ironclad-overworld-demand-v1',
                                      'energy': 'ironclad-overworld-energy-v1'}[upper]
        if upper == 'energy':
            assert default['scope'] == 'boss-relic-only'
            assert default['base_version'] == 'ironclad-overworld-v1'
    else:
        assert payload['effective_strategy']['default']['name'] == 'policy'
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    native_events = [e for e in events if e['kind'] == 'reference_decision'
                     and e.get('candidate') == 'turn-planner']
    assert native_events and all(e['native_action'] is not None for e in native_events)
    assert json.dumps(root.thick.bc.to_dict(), sort_keys=True) == before_native
    assert all(torch.equal(value, weights[key]) for key, value in model.policy.state_dict().items())
