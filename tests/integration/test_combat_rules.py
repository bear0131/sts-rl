"""Native combat baseline, frozen upper dispatch, and paired fork isolation."""
import json

import numpy as np
import pytest

from env.core import action_space as A
from rl.evaluation.runner import EvaluationConfig, Scenario, evaluate
from rl.evaluation.strategies import Decision, DecisionRequest, PhaseStrategy
from rl.evaluation.baselines.combat_rules import CombatRuleStrategy, visible_combat
from rl.evaluation.baselines.combat_audit import CombatAuditStrategy, summarize_pairs


def request(env):
    return DecisionRequest(env, env._obs(), env._info(), 42, 0)


def test_native_decision_does_not_mutate_or_advance_game(combat_factory):
    env = combat_factory(['STRIKE_RED', 'DEFEND_RED', 'BASH', 'STRIKE_RED', 'DEFEND_RED'])
    before = json.dumps(env.thick.bc.to_dict(), sort_keys=True)
    a = CombatRuleStrategy().decide([request(env)])[0]
    b = CombatRuleStrategy().decide([request(env)])[0]
    assert a == b and env.action_masks()[a.action]
    assert json.dumps(env.thick.bc.to_dict(), sort_keys=True) == before


def test_fixed_native_combat_finishes_with_legal_actions(combat_factory):
    env = combat_factory(['STRIKE_RED']*5, encounter='Cultist')
    branch = env.fork_combat()
    scenario = Scenario('fixture', lambda: branch,
        outcome=lambda e,i,t,tr: dict(result='win' if e.thick.bc.outcome == 1 else 'death', floor=1),
        reset=lambda e,s: (e._obs(), e._info()))
    rows = list(evaluate(CombatRuleStrategy(), scenario,
                        EvaluationConfig(episodes=1, seed=42, max_steps=100, timeout=10)))
    assert rows[0]['result'] == 'win'
    assert rows[0]['rule_decisions'] == rows[0]['steps']


def test_bash_pending_target_and_real_discard_selection(combat_factory):
    env = combat_factory(['HEADBUTT']*5)
    env.thick.bc.apply_custom_state({'hand': [('HEADBUTT',0,1)],
                                    'discard': [('WOUND',0), ('CARNAGE',0)]})
    env._invalidate_backend_cache()
    env.step(A._PLAY)
    assert env._info()['pending'] == ('play',0)
    action = CombatRuleStrategy().decide([request(env)])[0].action
    assert A.segment_of(action) == ('PICK_TARGET',0)
    env.step(action)
    assert env.thick.select_descriptor()['context'] == 'HEADBUTT'
    action = CombatRuleStrategy().decide([request(env)])[0].action
    assert action == A._PICK+1
    env.step(action)
    assert env.thick.select_descriptor() is None


def test_generated_candidates_are_public_and_action_aligned(combat_factory):
    env = combat_factory(['DISCOVERY']*5)
    env.step(A._PLAY)
    if env._info()['pending']:
        env.step(int(np.flatnonzero(env.action_masks())[0]))
    info = env.thick.bc.select_info()
    assert info['context'] == 'DISCOVERY'
    assert len(info['generated_cards']) == 3
    view = visible_combat(request(env))
    assert len(view.piles['generated']) == 3
    selected = CombatRuleStrategy().decide([request(env)])[0].action
    assert env.action_masks()[selected]
    env.step(selected)
    assert env.thick.select_descriptor() is None
    assert env.thick.bc.select_info()['generated_cards'] == []


def test_combat_dispatch_never_calls_frozen_upper(combat_factory):
    env = combat_factory(['STRIKE_RED']*5)
    class Upper:
        name = 'upper-sentinel'
        def decide(self, requests, *, check):
            assert all(r.info['phase'] != 'combat' for r in requests)
            return [Decision(123, self.name) for r in requests]
    strategy = PhaseStrategy(Upper(), {'combat': CombatRuleStrategy()})
    combat = request(env)
    overworld = DecisionRequest(None, {}, {'phase':'map'}, 42, 0)
    result = strategy.decide([overworld, combat, overworld])
    assert result[0].action == result[2].action == 123
    assert env.action_masks()[result[1].action]


def test_paired_fights_share_start_and_leave_reference_unchanged(combat_factory):
    env = combat_factory(['STRIKE_RED']*5, encounter='Cultist')
    before = json.dumps(env.thick.bc.to_dict(), sort_keys=True)
    rows, trace = [], []
    rule = CombatRuleStrategy()
    strategy = CombatAuditStrategy(rule, candidates={'rules':rule,'same':CombatRuleStrategy()},
        limit=1, rows=rows, trace=trace.append, max_steps=100, timeout=10)
    action = strategy.decide([request(env)], check=lambda: None)[0].action
    assert env.action_masks()[action]
    assert json.dumps(env.thick.bc.to_dict(), sort_keys=True) == before
    assert len(rows) == 2
    assert rows[0]['visible_state_sha256'] == rows[1]['visible_state_sha256']
    assert rows[0]['final_hp'] == rows[1]['final_hp']
    assert summarize_pairs(rows)['same']['rules_minus_opponent_hp_loss_mean'] == 0
    assert any(x.get('candidate') == 'same' for x in trace)


def test_trace_retains_native_complete_target_not_just_root_id(combat_factory):
    import slaythespire as sts
    env = combat_factory(['STRIKE_RED']*5)
    trace = []
    class CompleteAction:
        name = 'native-fixture'
        def decide(self, requests, *, check):
            # Same binding contract used by NativeStrategy after search.
            requests[0].env._nosl_cpp_pending = (A._PLAY, 0)
            return [Decision(A._PLAY, self.name)]
    CombatAuditStrategy(CompleteAction(), trace=trace.append).decide([request(env)], check=lambda: None)
    assert trace[0]['action'] == A._PLAY
    assert trace[0]['native_action'] == dict(bits=0, kind='CARD', source=0, target=0)


@pytest.mark.parametrize('contract', ['legacy', 'native-v1'])
def test_audit_trace_keeps_compatible_rule_ranking(combat_factory, contract):
    env = combat_factory(['STRIKE_RED'] * 5, combat_action_contract=contract)
    strategy, trace = CombatRuleStrategy(), []
    expected = strategy.decide([request(env)])
    actual = CombatAuditStrategy(strategy, trace=trace.append).decide(
        [request(env)], check=lambda: None)
    assert actual == expected
    event = json.loads(json.dumps(trace[0]))
    assert event['combat_action_contract'] == contract
    assert event['rule_ranking_status'] == 'available'
    assert event['rule_action'] == expected[0].action == event['ranked'][0]['action']
    assert {row['action'] for row in event['ranked']} == set(np.flatnonzero(env.action_masks()))


@pytest.mark.parametrize('contract', ['joint-v2', 'autoregressive-v2'])
@pytest.mark.parametrize('selection', [False, True], ids=['combat-root', 'selection'])
@pytest.mark.parametrize('solver', ['policy', 'native'])
def test_audit_trace_keeps_neural_decisions_without_legacy_ranking(combat_factory, monkeypatch, contract, selection, solver):
    import slaythespire as sts
    from sb3_contrib import MaskablePPO
    from rl.evaluation.baselines import combat_audit
    from rl.evaluation.strategies import NativeStrategy, PolicyStrategy
    from rl.policies.entity_combat import EntityCombatPolicy
    from rl.search.native import NativeSearchConfig

    env = combat_factory(['PURITY', 'STRIKE_RED', 'DEFEND_RED', 'BASH', 'ANGER'],
                         combat_action_contract=contract, card_language=True)
    if selection:
        env.thick.bc.apply_custom_state({'hand': ['PURITY', 'STRIKE_RED', 'DEFEND_RED']})
        env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0))
        env._invalidate_backend_cache()
        assert env.thick.select_descriptor() is not None
    model = MaskablePPO(EntityCombatPolicy, env, seed=42, n_steps=4, batch_size=4,
                       device='cpu', policy_kwargs=dict(
                           entity_config=dict(width=32, layers=1, heads=4, quantiles=8)))
    strategy = (PolicyStrategy(model) if solver == 'policy' else NativeStrategy(
        model, NativeSearchConfig(simulations=4, fixed_worldline=True, gumbel_scale=0)))
    before = json.dumps(env.thick.bc.to_dict(), sort_keys=True)
    expected = strategy.decide([request(env)])
    expected_native = env._nosl_cpp_pending
    if solver == 'native':
        env.nosl_is_az_cpp_abort()
    def incompatible_ranking(*_args):
        pytest.fail('neural action IDs must not enter the legacy rule scorer')
    monkeypatch.setattr(combat_audit, 'rank_actions', incompatible_ranking)
    trace = []
    actual = CombatAuditStrategy(strategy, trace=trace.append).decide(
        [request(env)], check=lambda: None)
    assert actual[0].action == expected[0].action
    assert actual[0].source == expected[0].source
    assert {key: value for key, value in actual[0].diagnostics.items() if key != 'search_seconds'} == {
        key: value for key, value in expected[0].diagnostics.items() if key != 'search_seconds'}
    assert env._nosl_cpp_pending == expected_native
    assert env.action_masks()[actual[0].action]
    assert json.dumps(env.thick.bc.to_dict(), sort_keys=True) == before
    event = json.loads(json.dumps(trace[0]))
    assert event['action'] == expected[0].action
    assert event['source'] == expected[0].source
    assert event['combat_action_contract'] == contract
    assert event['rule_action'] is None and event['ranked'] == []
    assert event['rule_ranking_status'] == 'unsupported_action_contract'
    if expected_native is not None:
        complete = sts.Action(int(expected_native[1]))
        assert event['native_action'] == dict(bits=int(expected_native[1]),
            kind=complete.action_type.name, source=complete.source_idx, target=complete.target_idx)
    assert event['state']['player']['hp'] == env.thick.bc.to_dict()['player']['hp']
    assert bool(event['state']['select']) == selection


def test_runic_dome_hides_intent_from_rule_state_and_decision(combat_factory):
    from dataclasses import asdict
    env = combat_factory(['STRIKE_RED', 'DEFEND_RED', 'BASH', 'STRIKE_RED', 'DEFEND_RED'],
                         relics=['Runic Dome'])
    views, decisions, raw_moves = [], [], []
    for move in ('JAW_WORM_BELLOW', 'JAW_WORM_CHOMP'):
        env.thick.bc.apply_custom_state({'monsters': [{'move': move}]})
        env._invalidate_backend_cache()
        raw_moves.append(env.thick.bc.to_dict()['monsters'][0]['move'])
        views.append(asdict(visible_combat(request(env))))
        decisions.append(CombatRuleStrategy().decide([request(env)])[0])
    assert raw_moves[0] != raw_moves[1]
    assert views[0] == views[1]
    assert decisions[0] == decisions[1]
