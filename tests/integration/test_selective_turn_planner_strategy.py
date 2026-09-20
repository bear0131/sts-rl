"""Conditional search budgets reuse the ordinary planner and public input contract."""
from dataclasses import asdict
import json

import pytest
import slaythespire as sts

from env.search.turn_planner import TurnPlannerConfig
from rl.evaluation.baselines.turn_planner import (
    SelectiveTurnPlannerStrategy, TurnPlannerStrategy,
)
from rl.evaluation.strategies import DecisionRequest, strategy_manifest

pytestmark = pytest.mark.native
END = 4 << 29


def _request(env, step=7):
    env._invalidate_backend_cache()
    return DecisionRequest(env, env._obs(), env._info(), 42, step)


def _strategies(config=None):
    config = config or TurnPlannerConfig(worlds=2, node_budget=128)
    base_trace, selective_trace = [], []
    return (TurnPlannerStrategy(config, trace=base_trace.append),
            SelectiveTurnPlannerStrategy(config, trace=selective_trace.append),
            base_trace, selective_trace)


def _time_eater(combat_factory):
    env = combat_factory(['DEFEND_RED'] * 10, encounter='Time Eater',
                         combat_action_contract='native-v1')
    env.thick.gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 20)
    env.thick.gc.set_deck([('DEFEND_RED', 1)] * 10)
    env.thick.bc = sts.create_battle(env.thick.gc, 'Time Eater')
    battle = env.thick.bc
    for n in (10, 1):
        battle.apply_custom_state({'hand': ['FINESSE'] * n})
        for _ in range(n):
            battle.execute(sts.Action(0))
    battle.apply_custom_state({
        'player': {'hp': 35, 'energy': 1, 'block': 0}, 'hand': ['DEFEND_RED'],
        'draw': [('DEFEND_RED', 1)] * 5, 'discard': ['STRIKE_RED'], 'exhaust': [],
        'monsters': [{'hp': 400, 'move': 'TIME_EATER_RIPPLE'}],
    })
    return env


def _without_timing_and_schedule(diagnostics):
    return {key: value for key, value in diagnostics.items()
            if key not in ('search_seconds', 'selected_search', 'deep_search_decisions')}


def test_ordinary_decision_is_exactly_the_same_search(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5, combat_action_contract='native-v1')
    env.thick.bc.apply_custom_state({
        'hand': ['STRIKE_RED', 'DEFEND_RED'],
        'monsters': [{'hp': 40, 'move': 'JAW_WORM_CHOMP'}],
    })
    base, selective, baseline, selected = _strategies()
    before = env.thick.bc.to_dict()
    first = base.decide([_request(env)])[0]
    second = selective.decide([_request(env)])[0]
    assert first.action == second.action
    assert first.native_action == second.native_action
    assert _without_timing_and_schedule(baseline[0]['diagnostics']) == _without_timing_and_schedule(selected[0]['diagnostics'])
    assert selected[0]['diagnostics']['selected_search'] == {
        'triggers': [], 'config': asdict(base.config),
    }
    assert second.diagnostics['deep_search_decisions'] == 0
    assert env.thick.bc.to_dict() == before


def test_time_warp_trigger_sees_death_beyond_the_first_boundary(combat_factory):
    env = _time_eater(combat_factory)
    base, selective, baseline, selected = _strategies()
    base.decide([_request(env)])
    decision = selective.decide([_request(env)])[0]
    ordinary = baseline[0]['diagnostics']
    deep = selected[0]['diagnostics']
    assert ordinary['root_outcomes'][str(END)]['deaths'] == 0
    assert deep['root_outcomes'][str(END)]['deaths'] == 2
    assert deep['root_outcomes']['0']['deaths'] == 0
    assert decision.native_action == 0
    assert deep['selected_search']['triggers'] == ['time-warp-eleven']
    assert deep['selected_search']['config']['lookahead_turns'] == 2
    assert deep['selected_search']['config']['node_budget'] == 4 * base.config.node_budget
    assert deep['effective_node_budget'] == selective.deep_config.node_budget
    assert deep['second_turn_entries'] > 0
    assert deep['native_transitions'] <= deep['effective_total_budget']
    assert decision.diagnostics['deep_search_decisions'] == 1
    json.dumps(selected, allow_nan=False)


@pytest.mark.parametrize('relics,count,triggered', [
    (('Pocketwatch',), 2, False), (('Pocketwatch',), 3, True),
    (('Pocketwatch',), 4, False), ((), 3, False),
])
def test_pocketwatch_trigger_is_exactly_the_third_public_play(combat_factory, relics, count, triggered):
    env = combat_factory(['STRIKE_RED'] * 10, relics=relics,
                         combat_action_contract='native-v1')
    env.thick.bc.apply_custom_state({'hand': ['FLEX'] * count})
    for _ in range(count):
        env.thick.bc.execute(sts.Action(0))
    env.thick.bc.apply_custom_state({'hand': ['ANGER'], 'draw': ['STRIKE_RED'] * 10,
                                    'monsters': [{'move': 'JAW_WORM_BELLOW'}]})
    _, selective, _, trace = _strategies()
    decision = selective.decide([_request(env)])[0]
    schedule = trace[0]['diagnostics']['selected_search']
    assert schedule['triggers'] == (['pocketwatch-three'] if triggered else [])
    assert schedule['config']['lookahead_turns'] == (2 if triggered else 1)
    assert decision.diagnostics['deep_search_decisions'] == int(triggered)


def test_triggered_search_keeps_immediate_death_below_survival(combat_factory):
    env = _time_eater(combat_factory)
    env.thick.bc.apply_custom_state({
        'player': {'hp': 1, 'powers': [('DEXTERITY', 40)]},
        'hand': ['FLEX', 'DEFEND_RED'],
        'monsters': [{'move': 'TIME_EATER_REVERBERATE'}],
    })
    _, selective, _, trace = _strategies()
    decision = selective.decide([_request(env)])[0]
    outcomes = trace[0]['diagnostics']['root_outcomes']
    assert outcomes['0']['deaths'] == 2
    assert outcomes['1']['deaths'] == 0
    assert decision.native_action == 1


def test_hidden_future_permutations_do_not_change_selected_search(combat_factory):
    env = _time_eater(combat_factory)
    env.thick.bc.apply_custom_state({'draw': ['DEFEND_RED', 'STRIKE_RED', 'BASH',
                                             'POMMEL_STRIKE', 'SHRUG_IT_OFF', 'ANGER']})
    root = env.thick.bc
    _, selective, _, trace = _strategies()
    actions = []
    for hidden_nonce in (11, 22, 33):
        env.thick.bc = root.clone()
        env.thick.bc.nosl_redeterminize(hidden_nonce)
        actions.append(selective.decide([_request(env, step=777)])[0].native_action)
    assert len(set(actions)) == 1
    diagnostics = [_without_timing_and_schedule(row['diagnostics']) for row in trace]
    assert diagnostics[0] == diagnostics[1] == diagnostics[2]
    assert all(row['diagnostics']['selected_search']['triggers'] == ['time-warp-eleven'] for row in trace)


def test_runic_dome_still_bypasses_both_planners(combat_factory, monkeypatch):
    env = combat_factory(['STRIKE_RED'] * 5, relics=['Runic Dome', 'Pocketwatch'],
                         combat_action_contract='native-v1')
    _, selective, _, trace = _strategies()
    def forbidden(*args, **kwargs):
        pytest.fail('hidden-intent Runic Dome state must not enter either planner')
    monkeypatch.setattr(selective.planner, 'choose', forbidden)
    monkeypatch.setattr(selective.deep_planner, 'choose', forbidden)
    decision = selective.decide([_request(env)])[0]
    assert decision.diagnostics['planner_dome_fallback'] == 1
    assert trace[0]['state']['monsters'] == []


def test_manifest_freezes_both_configs_and_rejects_ambiguous_base():
    config = TurnPlannerConfig(worlds=3, node_budget=37, beam_width=5,
                               max_depth=7, master_nonce=12345)
    strategy = SelectiveTurnPlannerStrategy(config)
    manifest = strategy_manifest(strategy)
    selected = manifest['selective_search']
    assert manifest['name'] == 'selective-turn-planner'
    assert selected['ordinary_config'] == asdict(config)
    assert selected['triggered_config'] == dict(asdict(config), lookahead_turns=2, node_budget=148)
    assert strategy.deep_config.master_nonce == config.master_nonce
    with pytest.raises(ValueError, match='one-turn base'):
        SelectiveTurnPlannerStrategy(TurnPlannerConfig(lookahead_turns=2))
    json.dumps(manifest, allow_nan=False)



def _nemesis(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 10, encounter='Nemesis',
                         relics=['Bronze Scales'], combat_action_contract='native-v1')
    env.thick.gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 20)
    env.thick.gc.set_deck([('STRIKE_RED', 0)] * 10)
    env.thick.gc.set_relics(['Bronze Scales'])
    env.thick.bc = sts.create_battle(env.thick.gc, 'Nemesis')
    battle = env.thick.bc
    # Enter Intangible through its real alternating phase, then No Draw via card.
    battle.apply_custom_state({'monsters': [{'move': 'NEMESIS_DEBUFF'}]})
    battle.execute(sts.Action(END))
    battle.apply_custom_state({'hand': ['BATTLE_TRANCE'], 'draw': [],
                               'discard': [], 'exhaust': []})
    battle.execute(sts.Action(0))
    battle.apply_custom_state({
        'player': {'hp': 75, 'energy': 2, 'block': 50, 'powers': [('STRENGTH', 7)]},
        'hand': [('FIEND_FIRE', 1)] + ['STRIKE_RED'] * 5,
        'draw': [], 'discard': [], 'exhaust': [],
        'monsters': [{'hp': 50, 'move': 'NEMESIS_ATTACK'}],
    })
    assert ('Intangible', 1) in battle.to_dict()['monsters'][0]['powers']
    assert ('No Draw', 1) in battle.to_dict()['player']['powers']
    return env


def test_nemesis_trigger_defers_fiend_fire_until_real_intangible_expires(combat_factory):
    env = _nemesis(combat_factory)
    config = TurnPlannerConfig(worlds=2, node_budget=2048)
    base, selective, _, trace = _strategies(config)
    before = env.thick.bc.to_dict()
    assert base.decide([_request(env, step=638)])[0].native_action == 0
    decision = selective.decide([_request(env, step=638)])[0]
    assert decision.native_action == END
    assert env.thick.bc.to_dict() == before
    diagnostics = trace[0]['diagnostics']
    assert diagnostics['selected_search']['triggers'] == ['intangible-nemesis-fiend-fire']
    assert diagnostics['root_outcomes']['0']['wins'] == 0
    assert diagnostics['root_outcomes'][str(END)]['wins'] == 2
    env.thick.bc.execute(sts.Action(decision.native_action))
    assert 'Intangible' not in dict(env.thick.bc.to_dict()['monsters'][0]['powers'])
    second = selective.decide([_request(env, step=639)])[0]
    assert trace[1]['diagnostics']['selected_search']['triggers'] == []
    card = env.thick.bc.to_dict()['hand'][sts.Action(second.native_action).source_idx]
    assert card[0] == 'FIEND_FIRE'
    env.thick.bc.execute(sts.Action(second.native_action))
    assert env.thick.bc.to_dict()['outcome'] == 1


def test_nemesis_trigger_does_not_ban_immediately_winning_fiend_fire(combat_factory):
    env = _nemesis(combat_factory)
    env.thick.bc.apply_custom_state({'monsters': [{'hp': 5}]})
    _, selective, _, trace = _strategies(TurnPlannerConfig(worlds=2, node_budget=2048))
    decision = selective.decide([_request(env, step=638)])[0]
    assert decision.native_action == 0
    assert trace[0]['diagnostics']['root_outcomes']['0']['wins'] == 2
    env.thick.bc.execute(sts.Action(0))
    assert env.thick.bc.to_dict()['outcome'] == 1


def test_nemesis_trigger_keeps_real_defense_above_self_exhausting_death(combat_factory):
    env = _nemesis(combat_factory)
    env.thick.bc.apply_custom_state({
        'player': {'hp': 1, 'block': 0, 'powers': [('DEXTERITY', 50)]},
        'hand': [('FIEND_FIRE', 1), 'DEFEND_RED'],
        'draw': [('DEFEND_RED', 1)] * 5, 'discard': ['STRIKE_RED'],
        'monsters': [{'move': 'NEMESIS_SCYTHE'}],
    })
    _, selective, _, trace = _strategies(TurnPlannerConfig(worlds=2, node_budget=2048))
    decision = selective.decide([_request(env, step=638)])[0]
    assert decision.native_action == 1
    outcomes = trace[0]['diagnostics']['root_outcomes']
    assert outcomes['0']['deaths'] == 2
    assert outcomes['1']['deaths'] == 0
