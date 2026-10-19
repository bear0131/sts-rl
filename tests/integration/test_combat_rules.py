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


@pytest.mark.parametrize('hp,expected_hp,expected_outcome', [(20, 14, 0), (5, 0, 1)])
def test_shared_poison_ticks_bypass_block_and_decrement_after_death(
        combat_factory, hp, expected_hp, expected_outcome):
    import slaythespire as sts
    env = combat_factory(['STRIKE_RED'] * 5, 'Cultist')
    battle = env.thick.bc
    battle.apply_custom_state({'monsters': [{
        'max_hp': 20, 'hp': hp, 'block': 80,
        'powers': [('BARRICADE', 1), ('POISON', 6)],
    }]})
    battle.execute(sts.Action(sts.ActionType.END_TURN))
    monster = battle.to_dict()['monsters'][0]
    assert monster['hp'] == expected_hp
    assert dict(monster['powers'])['Poison'] == 5
    assert monster['block'] == (80 if expected_hp else 0)
    assert battle.outcome == expected_outcome


def test_incomplete_monster_restore_cannot_partially_mutate_player(combat_factory):
    battle = combat_factory(['STRIKE_RED'] * 5).thick.bc
    before = battle.to_dict()
    with pytest.raises(ValueError, match='gone disagrees'):
        battle.apply_custom_state({'player': {'hp': 1}, 'monsters': [{'gone': True}]})
    assert battle.to_dict() == before


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
