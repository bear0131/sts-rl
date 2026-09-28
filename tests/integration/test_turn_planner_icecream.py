"""Real native banked energy must survive completed-leaf comparison."""
from copy import deepcopy

import pytest
import slaythespire as sts

from env.search.turn_planner import (
    NativeTurnPlanner, TurnPlannerConfig, power_horizon, public_planner_state, state_value,
)

pytestmark = pytest.mark.native
END = 4 << 29


def _safe_battle(combat_factory, *, icecream=True, energy=3):
    battle = combat_factory(['STRIKE_RED'] * 10, encounter='Jaw Worm',
                            relics=['Ice Cream'] if icecream else []).thick.bc
    battle.apply_custom_state({
        'player': {'hp': 75, 'max_hp': 75, 'energy': energy},
        'monsters': [{'hp': 45, 'block': 0, 'move': 'JAW_WORM_BELLOW'}],
        'hand': ['SEEING_RED'], 'draw': ['STRIKE_RED'] * 8, 'discard': [], 'exhaust': [],
    })
    return battle


def _end(battle, *, seeing_red=False, boundary=True):
    branch = battle.clone()
    if seeing_red:
        branch.execute(sts.Action(sts.ActionType.CARD, 0))
    if boundary:
        branch.execute_to_enemy_boundary(END)
    else:
        branch.execute(sts.Action(END))
    return public_planner_state(branch)


def test_seeing_red_banks_real_future_energy_and_wins_the_completed_leaf_tie(combat_factory):
    battle = _safe_battle(combat_factory)
    before = public_planner_state(battle)
    ended, banked = _end(battle), _end(battle, seeing_red=True)
    assert ended['player']['hp'] == banked['player']['hp'] == 75
    assert ended['monsters'] == banked['monsters']
    assert (ended['player']['energy'], banked['player']['energy']) == (3, 4)
    assert _end(battle, boundary=False)['player']['energy'] == 6
    assert _end(battle, seeing_red=True, boundary=False)['player']['energy'] == 7
    horizon = power_horizon(before)
    assert state_value(banked, public_relics=['Ice Cream'], horizon=horizon) > state_value(
        ended, public_relics=['Ice Cream'], horizon=horizon)
    plan = NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=512)).choose(
        battle, state_nonce=8123, public_relics=['Ice Cream'])
    assert plan.action_bits == 0
    assert plan.diagnostics['root_values']['0'] > plan.diagnostics['root_values'][str(END)]
    assert public_planner_state(battle) == before


def test_ordinary_energy_expires_and_does_not_make_seeing_red_worth_playing(combat_factory):
    battle = _safe_battle(combat_factory, icecream=False)
    ended, wasted = _end(battle), _end(battle, seeing_red=True)
    assert ended['player']['energy'] == wasted['player']['energy'] == 0
    assert _end(battle, boundary=False)['player']['energy'] == 3
    assert _end(battle, seeing_red=True, boundary=False)['player']['energy'] == 3
    assert state_value(ended, public_relics=()) == state_value(wasted, public_relics=())
    plan = NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=512)).choose(
        battle, state_nonce=8123, public_relics=())
    assert plan.action_bits == END


def test_energy_credit_is_bounded_and_does_not_generate_beyond_its_useful_horizon(combat_factory):
    battle = _safe_battle(combat_factory, energy=8)
    ordinary = _end(battle)
    extreme = deepcopy(ordinary)
    extreme['player']['energy'] = 999
    empty = deepcopy(ordinary)
    empty['player']['energy'] = 0
    values = [state_value(raw, public_relics=['Ice Cream'], horizon=5)
              for raw in (empty, ordinary, extreme)]
    assert values[1] == values[2] > values[0]
    near = state_value(ordinary, public_relics=['Ice Cream'], horizon=1)
    far = state_value(ordinary, public_relics=['Ice Cream'], horizon=5)
    # Isolate energy credit from the rest of the horizon-dependent value.
    near_credit = near - state_value(empty, public_relics=['Ice Cream'], horizon=1)
    far_credit = far - state_value(empty, public_relics=['Ice Cream'], horizon=5)
    assert 0 < near_credit < far_credit <= 9.0
    plan = NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=512)).choose(
        battle, state_nonce=8123, public_relics=['Ice Cream'])
    assert plan.action_bits == END


def test_known_lethal_keeps_priority_over_banking_energy(combat_factory):
    battle = _safe_battle(combat_factory, energy=1)
    battle.apply_custom_state({'hand': ['SEEING_RED', 'STRIKE_RED'], 'monsters': [{'hp': 6}]})
    plan = NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=512)).choose(
        battle, state_nonce=8123, public_relics=['Ice Cream'])
    assert plan.action_bits == 1  # Immediate Strike, not Seeing Red followed by Strike.
    battle.execute(sts.Action(plan.action_bits))
    won = public_planner_state(battle)
    assert won['outcome'] == 1
    extreme = deepcopy(won)
    extreme['player']['energy'] = 999
    assert state_value(won, public_relics=['Ice Cream']) == state_value(extreme, public_relics=['Ice Cream'])


@pytest.mark.parametrize('relic_id', ['Ice Cream', 'ICE_CREAM', 'IceCream'])
def test_public_relic_ids_are_equivalent_and_no_context_does_not_infer_relic(combat_factory, relic_id):
    raw = _end(_safe_battle(combat_factory))
    empty = deepcopy(raw)
    empty['player']['energy'] = 0
    assert state_value(raw, public_relics=[relic_id]) > state_value(empty, public_relics=[relic_id])
    assert state_value(raw, public_relics=None) == state_value(empty, public_relics=None)
