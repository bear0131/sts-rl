"""Installed engines receive bounded credit only for public future triggers."""
from copy import deepcopy

import pytest
import slaythespire as sts

import env.search.turn_planner as planner_module
from env.search.turn_planner import (
    NativeTurnPlanner, TurnPlannerConfig, public_planner_state,
    state_value, status_engine_opportunities,
)

pytestmark = pytest.mark.native
END = 4 << 29


def _engine(combat_factory, card, *, junk='WOUND'):
    env = combat_factory(['STRIKE_RED'] * 10)
    env.thick.gc.ascension = 20
    env.thick.bc = sts.create_battle(env.thick.gc, 'Jaw Worm')
    battle = env.thick.bc
    battle.apply_custom_state({
        'player': {'hp': 75, 'max_hp': 75, 'energy': 1},
        'monsters': [{'hp': 45, 'block': 0, 'move': 'JAW_WORM_BELLOW'}],
        'hand': [card], 'draw': ['STRIKE_RED'] * 5 + [junk] * 5,
        'discard': [], 'exhaust': [],
    })
    return battle


def _plan(battle):
    return NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=512)).choose(
        battle, state_nonce=731, public_relics=[])


@pytest.mark.parametrize('card', ['EVOLVE', 'FIRE_BREATHING'])
def test_status_engine_breaks_useless_power_end_tie_and_native_trigger_is_real(combat_factory, monkeypatch, card):
    battle = _engine(combat_factory, card)
    before = battle.to_dict()
    original = planner_module.status_engine_opportunities
    monkeypatch.setattr(planner_module, 'status_engine_opportunities', lambda raw, horizon: None)
    old = _plan(battle)
    assert old.action_bits == END
    assert old.diagnostics['root_values']['0'] == old.diagnostics['root_values'][str(END)]
    monkeypatch.setattr(planner_module, 'status_engine_opportunities', original)
    assert _plan(battle).action_bits == 0
    waiting, installed = battle.clone(), battle.clone()
    installed.execute(sts.Action(0))
    waiting.execute(sts.Action(END)); installed.execute(sts.Action(END))
    a, b = waiting.to_dict(), installed.to_dict()
    assert a['player']['hp'] == b['player']['hp'] == 75
    if card == 'EVOLVE':
        assert len(b['hand']) > len(a['hand'])
        assert sum(c[0] == 'STRIKE_RED' for c in b['hand']) > sum(c[0] == 'STRIKE_RED' for c in a['hand'])
    else:
        assert b['monsters'][0]['hp'] < a['monsters'][0]['hp']
    assert battle.to_dict() == before


@pytest.mark.parametrize('card', ['EVOLVE', 'FIRE_BREATHING'])
def test_no_trigger_does_not_invent_engine_value(combat_factory, card):
    battle = _engine(combat_factory, card, junk='STRIKE_RED')
    raw = battle.to_dict()
    opportunities = status_engine_opportunities(raw, 5)
    assert opportunities['evolve'] == opportunities['firebreathing'] == 0
    assert _plan(battle).action_bits == END


def test_curses_trigger_only_firebreathing_and_ethereal_hand_is_not_future_fuel(combat_factory):
    battle = _engine(combat_factory, 'FIRE_BREATHING', junk='INJURY')
    opportunities = status_engine_opportunities(battle.to_dict(), 5)
    assert opportunities['evolve'] == 0 and opportunities['firebreathing'] > 0
    assert _plan(battle).action_bits == 0
    battle.apply_custom_state({'hand': ['EVOLVE', 'DAZED', 'VOID', 'ASCENDERS_BANE', 'CLUMSY'],
                              'draw': ['STRIKE_RED'] * 5, 'discard': [], 'exhaust': ['WOUND'] * 10})
    opportunities = status_engine_opportunities(battle.to_dict(), 5)
    assert opportunities['evolve'] == opportunities['firebreathing'] == 0
    assert _plan(battle).action_bits == END


def test_fixed_root_opportunities_do_not_reward_manufacturing_or_preserving_junk(combat_factory, monkeypatch):
    battle = _engine(combat_factory, 'EVOLVE')
    root = battle.to_dict()
    expected = status_engine_opportunities(root, planner_module.power_horizon(root))
    seen = []
    original = planner_module.state_value
    def capture(raw, **kwargs):
        seen.append(kwargs.get('status_opportunities'))
        return original(raw, **kwargs)
    monkeypatch.setattr(planner_module, 'state_value', capture)
    _plan(battle)
    assert seen and all(value == expected for value in seen)
    battle.execute(sts.Action(0)); battle.execute_to_enemy_boundary(END)
    raw = public_planner_state(battle)
    without = deepcopy(raw); without['draw'] = [c for c in without['draw'] if c[0] != 'WOUND']
    # Exhausting/clearing existing junk improves ordinary pile quality without
    # also erasing the already anchored engine potential.
    assert state_value(without, horizon=3, status_opportunities=expected) > state_value(
        raw, horizon=3, status_opportunities=expected)
    more = deepcopy(raw); more['draw'] += [['WOUND', 0]] * 20
    assert state_value(more, horizon=3, status_opportunities=expected) < state_value(
        raw, horizon=3, status_opportunities=expected)
    permuted = deepcopy(root); permuted['draw'].reverse()
    assert status_engine_opportunities(permuted, 5) == status_engine_opportunities(root, 5)


@pytest.mark.parametrize('card', ['EVOLVE', 'FIRE_BREATHING'])
def test_immediate_defense_and_lethal_override_delayed_engine(combat_factory, card):
    battle = _engine(combat_factory, card)
    battle.apply_custom_state({'player': {'hp': 6, 'block': 0, 'energy': 1},
        'hand': [card, ('DEFEND_RED', 1)], 'monsters': [{'move': 'JAW_WORM_CHOMP'}]})
    result = _plan(battle)
    assert result.action_bits == 1
    assert result.diagnostics['root_outcomes']['0']['deaths'] == 2
    battle.apply_custom_state({'player': {'hp': 75}, 'hand': [card, 'STRIKE_RED'],
                              'monsters': [{'hp': 6}]})
    result = _plan(battle)
    assert result.action_bits == 1
    assert result.diagnostics['root_outcomes']['1']['wins'] == 2


def test_trigger_and_power_stacking_credit_has_finite_caps(combat_factory):
    battle = _engine(combat_factory, 'EVOLVE')
    root = battle.to_dict()
    root['draw'] += [['WOUND', 0]] * 100
    opportunities = status_engine_opportunities(root, 5)
    assert opportunities['evolve'] == opportunities['firebreathing'] == 3
    ended = battle.clone(); ended.execute_to_enemy_boundary(END)
    raw = ended.to_dict()
    strong = deepcopy(raw)
    strong['player']['powers'] += [('Evolve', 100), ('Fire Breathing', 100)]
    credit = state_value(strong, horizon=5, status_opportunities=opportunities) - state_value(
        raw, horizon=5, status_opportunities=opportunities)
    assert credit == pytest.approx(32)
    stronger = deepcopy(strong)
    stronger['player']['powers'] = [('Evolve', 1000), ('Fire Breathing', 1000)]
    assert state_value(stronger, horizon=5, status_opportunities=opportunities) == state_value(
        strong, horizon=5, status_opportunities=opportunities)
