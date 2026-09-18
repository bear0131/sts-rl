"""Terminal dominance and explicit sampled-death avoidance, without whole runs."""
import json
import math

import pytest
import slaythespire as sts

from env.search.turn_planner import (
    NativeTurnPlanner, TurnPlannerConfig, aggregate_world_values, leaf_value,
    power_horizon, state_value,
)
from rl.evaluation.baselines.turn_planner import TurnPlannerStrategy

pytestmark = pytest.mark.native
END = 4 << 29


def test_world_aggregation_is_conservative_and_finite():
    risky = [(1, 1e100)] + [(-1, -100000)] * 3
    safe = [(0, -1e100)] * 4
    assert aggregate_world_values(safe) > aggregate_world_values(risky)
    assert aggregate_world_values([(1, 10)] * 4) > aggregate_world_values(safe)
    assert aggregate_world_values([(0, 1e308)] * 4) == (0, 0, 1e308)
    assert aggregate_world_values(risky) == aggregate_world_values(list(reversed(risky)))
    for invalid in ([], [(0, math.inf)], [(0, math.nan)], [(2, 1)]):
        with pytest.raises(ValueError):
            aggregate_world_values(invalid)


@pytest.mark.parametrize('scores', [[1, 2, 8, -10], [20.1, -70.3], [-1e6, 1e6, 30]])
def test_all_live_preserves_original_heuristic_mean(scores):
    assert aggregate_world_values([(0, x) for x in scores]) == (0, 0, sum(scores)/len(scores))


def test_high_strength_cannot_outrank_native_victory(combat_factory):
    battle = combat_factory(['STRIKE_RED'] * 5).thick.bc
    battle.apply_custom_state({'player': {'hp': 80, 'energy': 1, 'powers': [('STRENGTH', 500)]},
                               'hand': ['STRIKE_RED', 'DEFEND_RED'],
                               'monsters': [{'hp': 40, 'move': 'JAW_WORM_CHOMP'}]})
    before = battle.to_dict()
    horizon = power_horizon(before)
    won = battle.clone(); won.execute_to_enemy_boundary(0)
    waiting = battle.clone(); waiting.execute_to_enemy_boundary(END)
    assert won.to_dict()['outcome'] == 1 and waiting.to_dict()['outcome'] == 0
    # The original scalar still demonstrates the bug: it is now subordinate.
    assert state_value(waiting.to_dict(), horizon=horizon) > state_value(won.to_dict(), horizon=horizon)
    assert leaf_value(won.to_dict(), horizon=horizon) > leaf_value(waiting.to_dict(), horizon=horizon)
    result = NativeTurnPlanner(TurnPlannerConfig(worlds=4, node_budget=64)).choose(
        battle, state_nonce=5, public_relics=())
    assert result.action_bits == 0
    outcomes = result.diagnostics['root_outcomes']
    assert outcomes['0']['wins'] == 4 and outcomes[str(END)]['wins'] == 0
    assert battle.to_dict() == before
    json.dumps(result.diagnostics, allow_nan=False)


def test_native_world_outcomes_survive_structured_aggregation(combat_factory):
    # One public root, four actual NOSL sampled futures. Havoc wins on the
    # sampled Bludgeon and dies after the other draws; Defend survives all four.
    root = combat_factory(['STRIKE_RED'] * 5).thick.bc
    root.apply_custom_state({'player': {'hp': 7, 'energy': 1},
        'hand': ['HAVOC', 'DEFEND_RED'],
        'draw': ['BLUDGEON', 'STRIKE_RED', 'STRIKE_RED', 'STRIKE_RED'], 'discard': [],
        'monsters': [{'hp': 25, 'move': 'JAW_WORM_CHOMP'}]})
    before = root.to_dict()
    result = NativeTurnPlanner(TurnPlannerConfig(worlds=4, node_budget=64)).choose(
        root, state_nonce=2, public_relics=())
    outcomes = result.diagnostics['root_outcomes']
    assert (outcomes['0']['wins'], outcomes['0']['deaths']) == (1, 3)
    assert (outcomes['1']['wins'], outcomes['1']['deaths'], outcomes['1']['live']) == (0, 0, 4)
    assert result.action_bits == 1
    assert root.to_dict() == before
    json.dumps(result.diagnostics, allow_nan=False)


def test_structured_outcomes_stay_in_trace_not_numeric_decision_metrics(combat_factory):
    from rl.evaluation.strategies import DecisionRequest
    env = combat_factory(['STRIKE_RED'] * 5, combat_action_contract='native-v1')
    trace = []
    strategy = TurnPlannerStrategy(TurnPlannerConfig(worlds=1, node_budget=16), trace=trace.append)
    request = DecisionRequest(env, env._obs(), env._info(), 42, 0)
    decision = strategy.decide([request])[0]
    assert 'root_outcomes' not in decision.diagnostics
    assert all(isinstance(value, (int, float)) for value in decision.diagnostics.values())
    assert isinstance(trace[0]['diagnostics']['root_outcomes'], dict)
    json.dumps(trace, allow_nan=False)


def test_manifest_declares_risk_contract_and_trace_only_structured_diagnostics():
    strategy = TurnPlannerStrategy()
    assert strategy.version == 'ironclad-turn-v8-awakened-phase'
    manifest = strategy.manifest()
    assert manifest['terminal_priority'] == 'per-world-death-live-win'
    assert manifest['world_aggregation'] == 'lexicographic-fewer-deaths-more-wins-mean-heuristic'


def test_unresolved_pending_leaf_is_not_reported_as_surviving_boundary(combat_factory):
    root = combat_factory(['PURITY'] * 5).thick.bc
    root.apply_custom_state({'hand': ['PURITY', 'DEFEND_RED', 'STRIKE_RED']})
    result = NativeTurnPlanner(TurnPlannerConfig(worlds=1, node_budget=1,
        beam_width=1, max_depth=1)).choose(root, state_nonce=3, public_relics=())
    pending = result.diagnostics['root_outcomes']['0']
    assert pending['live'] == pending['unresolved'] == 1
    assert pending['boundary_live'] == pending['wins'] == pending['deaths'] == 0
