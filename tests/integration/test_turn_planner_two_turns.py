"""Two-turn native continuations preserve sampled futures and explicit budgets."""
import json

import pytest
import slaythespire as sts

import env.search.turn_planner as planner_module
from env.search.turn_planner import NativeTurnPlanner, TurnPlannerConfig, power_horizon
from rl.evaluation.baselines.turn_planner import TurnPlannerStrategy
from rl.evaluation.strategies import DecisionRequest, strategy_manifest


def _planner(**kwargs):
    return NativeTurnPlanner(TurnPlannerConfig(lookahead_turns=2, **kwargs))


def _headbutt(combat_factory, draw=None):
    battle = combat_factory(['HEADBUTT'] * 5).thick.bc
    battle.apply_custom_state({
        'player': {'hp': 80, 'energy': 3},
        'hand': ['HEADBUTT'], 'draw': ['DEFEND_RED'] * 5 if draw is None else draw,
        'discard': ['WOUND', 'CARNAGE'], 'exhaust': [],
        'monsters': [{'hp': 40, 'move': 'JAW_WORM_CHOMP', 'block': 0}],
    })
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert battle.select_info()['context'] == 'HEADBUTT'
    return battle


def test_headbutt_prefers_actual_next_turn_damage_over_a_wound(combat_factory):
    battle = _headbutt(combat_factory)
    original = battle.to_dict()
    # Both alternatives survive the current enemy phase. Actual next hand
    # includes the chosen card; no beneficial hidden order is manufactured.
    for choice, expected in ((0, 'WOUND'), (1, 'CARNAGE')):
        control = battle.clone()
        control.execute(sts.Action(sts.ActionType.SINGLE_CARD_SELECT, choice))
        control.execute(sts.Action(sts.ActionType.END_TURN))
        assert control.to_dict()['hand'][0][0] == expected
    result = _planner(node_budget=512).choose(battle, state_nonce=888, public_relics=())
    assert sts.Action(result.action_bits).source_idx == 1
    assert result.diagnostics['second_turn_entries'] > 0
    assert result.diagnostics['completed_horizon_leaves'] > 0
    assert result.diagnostics['fallback_leaves'] == 0
    assert battle.to_dict() == original


def test_same_public_root_different_actual_futures_gives_same_two_turn_values(combat_factory):
    # Set all piles together so native unique IDs remain distinct across zones.
    battle = _headbutt(combat_factory, ['DEFEND_RED', 'STRIKE_RED', 'BASH', 'SHRUG_IT_OFF', 'WOUND'])
    roots=[]
    values=[]
    for nonce in (101, 202, 303):
        hidden = battle.clone()
        hidden.nosl_redeterminize(nonce)
        before = hidden.to_dict()
        roots.append(before['draw'])
        result = _planner(node_budget=512).choose(hidden, state_nonce=888, public_relics=())
        values.append((result.action_bits, result.diagnostics['root_values']))
        assert hidden.to_dict() == before
    assert len({json.dumps(x) for x in roots}) > 1
    assert values[0] == values[1] == values[2]


def test_demon_form_grows_once_before_second_enemy_boundary(combat_factory, monkeypatch):
    battle = combat_factory(['DEFEND_RED'] * 10).thick.bc
    battle.apply_custom_state({'player': {'powers': [('DEMON_FORM', 2)]},
                               'hand': ['DEFEND_RED'], 'draw': ['DEFEND_RED'] * 5})
    initial = battle.to_dict(); seen=[]
    original = planner_module.state_value
    def capture(raw, **kwargs):
        seen.append((raw, kwargs))
        return original(raw, **kwargs)
    monkeypatch.setattr(planner_module, 'state_value', capture)
    result = _planner(node_budget=512).choose(battle, state_nonce=9, public_relics=())
    boundaries = [raw for raw, kw in seen if raw['post_enemy_turn_boundary']]
    assert boundaries
    assert all(raw['turn'] == initial['turn'] + 2 for raw in boundaries)
    assert all(dict(raw['player']['powers'])['Strength'] == 2 for raw in boundaries)
    assert {kw['horizon'] for raw, kw in seen} == {power_horizon(initial)}
    assert result.diagnostics['native_transitions'] <= result.diagnostics['effective_total_budget']
    assert battle.to_dict() == initial


@pytest.mark.parametrize('codex', [False, True])
def test_twelfth_card_and_codex_continue_to_the_second_enemy_boundary(combat_factory, codex):
    battle = combat_factory(['POMMEL_STRIKE'] * 25, encounter='Time Eater',
        relics=["Nilry's Codex"] if codex else []).thick.bc
    battle.apply_custom_state({'player': {'energy': 20}, 'hand': ['POMMEL_STRIKE'] * 5,
                               'draw': ['POMMEL_STRIKE'] * 20, 'discard': []})
    for _ in range(11):
        battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    before=battle.to_dict()
    assert ('Time Warp', 11) in before['monsters'][0]['powers']
    result=_planner(node_budget=2048, max_depth=16).choose(battle, state_nonce=17)
    assert result.diagnostics['second_turn_entries'] > 0
    assert result.diagnostics['completed_horizon_leaves'] > 0
    assert result.diagnostics['max_action_depth'] <= 16
    assert result.diagnostics['native_transitions'] <= result.diagnostics['effective_total_budget']
    assert battle.to_dict()==before


def test_tiny_budget_and_depth_report_incomplete_two_turn_search(combat_factory):
    battle = _headbutt(combat_factory)
    result = _planner(node_budget=1, max_depth=1).choose(battle, state_nonce=23)
    d=result.diagnostics
    assert d['effective_node_budget'] == 3 * d['root_actions']
    assert 0 < d['native_transitions'] <= d['effective_total_budget']
    assert d['max_action_depth'] == 1
    assert d['completed_horizon_leaves'] == 0
    assert d['fallback_leaves'] == d['root_actions'] * d['worlds_completed']


def test_existing_codex_choice_is_a_legal_two_turn_root(combat_factory):
    battle = combat_factory(['DEFEND_RED'] * 10, relics=["Nilry's Codex"]).thick.bc
    battle.execute(sts.Action(sts.ActionType.END_TURN))
    before=battle.to_dict()
    assert before['input_state']==2 and before['select']['context']=='CODEX'
    result=_planner(node_budget=512).choose(battle,state_nonce=24)
    assert sts.Action(result.action_bits).action_type == sts.ActionType.SINGLE_CARD_SELECT
    assert result.diagnostics['second_turn_entries'] > 0
    assert result.diagnostics['completed_horizon_leaves'] > 0
    assert battle.to_dict()==before


def test_every_execute_across_both_turns_is_counted(combat_factory, monkeypatch):
    battle=_headbutt(combat_factory); planner=_planner(node_budget=512)
    calls={'ordinary':0,'boundary':0}
    class Counted:
        def __init__(self,inner): self.inner=inner
        def __getattr__(self,name): return getattr(self.inner,name)
        def clone(self): return Counted(self.inner.clone())
        def execute(self,action):
            calls['ordinary']+=1
            return self.inner.execute(action)
        def execute_to_enemy_boundary(self,action):
            calls['boundary']+=1
            return self.inner.execute_to_enemy_boundary(action)
    original=planner.sampler.clone_and_redeterminize
    monkeypatch.setattr(planner.sampler,'clone_and_redeterminize',
                        lambda *a,**kw:Counted(original(*a,**kw)))
    result=planner.choose(battle,state_nonce=25)
    assert calls['ordinary'] > 0 and calls['boundary'] > 0
    assert sum(calls.values())==result.diagnostics['native_transitions']
    assert sum(calls.values())<=result.diagnostics['effective_total_budget']


def test_two_turn_runic_dome_still_never_calls_native_planning(combat_factory, monkeypatch):
    env=combat_factory(['STRIKE_RED'] * 5,relics=['Runic Dome'],combat_action_contract='native-v1')
    strategy=TurnPlannerStrategy(TurnPlannerConfig(lookahead_turns=2))
    monkeypatch.setattr(strategy.planner,'choose',lambda *a,**kw:pytest.fail('Dome reached two-turn planner'))
    request=DecisionRequest(env,env._obs(),env._info(),42,0)
    result=strategy.decide([request])[0]
    assert result.diagnostics['planner_dome_fallback']==1


def test_projected_or_already_armed_private_states_are_not_silently_resumed(combat_factory):
    battle=combat_factory(['DEFEND_RED'] * 10).thick.bc
    battle.execute_to_enemy_boundary(4 << 29)
    before=battle.to_dict()
    with pytest.raises(ValueError,match='live'):
        _planner().choose(battle,state_nonce=26)
    assert battle.to_dict()==before
    pending=combat_factory(['DEFEND_RED'] * 10,relics=["Nilry's Codex"]).thick.bc
    pending.execute_to_enemy_boundary(4 << 29)
    before=pending.to_dict()
    with pytest.raises(ValueError,match='already armed'):
        _planner().choose(pending,state_nonce=27)
    assert pending.to_dict()==before


def test_two_turn_config_manifest_and_cli_preserve_explicit_horizon():
    from rl.evaluation.eval import build_parser
    from rl.evaluation.strategies import compose_strategy
    args=build_parser().parse_args(['unused.zip','--solver','turn-planner','--planner-turns','2'])
    strategy=compose_strategy(object(),args)
    combat=strategy_manifest(strategy)['phases']['combat']
    assert combat['config']['lookahead_turns']==2
    assert combat['horizon']=='two-player-turns-through-second-enemy-phase'
    assert TurnPlannerConfig().lookahead_turns==1
    with pytest.raises(ValueError,match='lookahead_turns'):
        TurnPlannerConfig(lookahead_turns=3)
