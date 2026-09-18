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


def test_native_constricted_is_public_and_blockable(combat_factory):
    env = combat_factory(['DEFEND_RED']*5, encounter='Spire Growth')
    env.thick.bc.apply_custom_state({'monsters': [{'move': 'SPIRE_GROWTH_CONSTRICT'}]})
    env._invalidate_backend_cache()
    env.step(A.END_TURN)
    env.thick.bc.apply_custom_state({'hand': [('DEFEND_RED', 0, 1)],
        'monsters': [{'move': 'SPIRE_GROWTH_CONSTRICT'}]})
    env._invalidate_backend_cache()
    view = visible_combat(request(env))
    assert view.player['powers']['Constricted'] == 10
    assert view.incoming == 10
    hp = view.player['hp']
    env.step(A._PLAY)
    if env._info().get('pending'):
        env.step(A._TARGET)
    env.step(A.END_TURN)
    assert visible_combat(request(env)).player['hp'] == hp-5


def test_native_minion_leader_death_wins_with_living_minions(combat_factory):
    env = combat_factory(['STRIKE_RED']*5, encounter='Reptomancer').fork_combat()
    view = visible_combat(request(env))
    slot = next(i for i, m in enumerate(view.monsters) if m and m['id'] == 'Reptomancer')
    assert view.monsters[slot]['powers']['Minion Leader'] == 1
    assert len(view.alive) > 1
    edits = [{} for _ in view.monsters]
    edits[slot] = {'hp': 6}
    env.thick.bc.apply_custom_state({'monsters': edits})
    env._invalidate_backend_cache()
    env.step(A._PLAY)
    action = CombatRuleStrategy().decide([request(env)])[0].action
    assert action == A._TARGET+slot
    env.step(action)
    assert env.thick.bc.outcome == 1
    assert any(m and m['hp'] > 0 for i, m in enumerate(visible_combat(request(env)).monsters) if i != slot)


def test_native_awakened_first_bar_rebirth_then_terminal_second_bar(combat_factory):
    from rl.evaluation.baselines.combat_rules import card_score
    env = combat_factory(['STRIKE_RED']*5, encounter='Awakened One').fork_combat()
    view = visible_combat(request(env))
    slot = next(i for i, m in enumerate(view.monsters) if m and m['id'] == 'AwakenedOne')
    assert view.monsters[slot]['powers']['Curiosity'] == 1
    assert not view.monsters[slot]['powers'].get('Minion Leader')
    # Remove Cultists through real damage so monstersAlive is also correct.
    edits = [{'hp': 40 if i == slot else 1} if m else {} for i, m in enumerate(view.monsters)]
    env.thick.bc.apply_custom_state({'monsters': edits, 'hand': [('CLEAVE', 0, 1)]})
    env._invalidate_backend_cache(); env.step(A._PLAY)
    if env._info().get('pending'):
        env.step(A._TARGET)
    assert len(visible_combat(request(env)).alive) == 1
    edits = [{} for _ in view.monsters]; edits[slot] = {'hp': 6}
    env.thick.bc.apply_custom_state({'monsters': edits, 'hand': [('STRIKE_RED', 0, 1)]})
    env._invalidate_backend_cache()
    view = visible_combat(request(env))
    assert 'combat_lethal' not in card_score(view.hand[0], slot, view)[1]
    env.step(A._PLAY); env.step(A._TARGET+slot)
    assert env.thick.bc.outcome != 1
    env.thick.bc.apply_custom_state({'player': {'block': 1000}})
    env._invalidate_backend_cache(); env.step(A.END_TURN)
    view = visible_combat(request(env))
    assert view.monsters[slot]['powers']['Minion Leader'] == 1
    edits[slot] = {'hp': 6}
    env.thick.bc.apply_custom_state({'monsters': edits, 'hand': [('STRIKE_RED', 0, 1)]})
    env._invalidate_backend_cache()
    view = visible_combat(request(env))
    assert card_score(view.hand[0], slot, view)[1]['combat_lethal'] == 10000
    env.step(A._PLAY); env.step(A._TARGET+slot)
    assert env.thick.bc.outcome == 1
