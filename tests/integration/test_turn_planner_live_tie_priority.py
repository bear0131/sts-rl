"""Root/world aggregation tests, not evidence of native search transitions.

Native legal roots and NOSL clones are real. Controlled _root scores isolate the
new exact-tie aggregation contract from frontier search and leaf evaluation.
"""
import itertools
import math

import pytest

from env.search import turn_planner as planner_module
from env.search.turn_planner import NativeTurnPlanner, TurnPlannerConfig


pytestmark = pytest.mark.native
END = 4 << 29
ROOTS = (0, 1, END)


def live(value=1.0, depth=1, *, unresolved=False):
    return ((0, value), 0, 0 if unresolved else depth, int(unresolved))


def terminal(rank, value=1.0, depth=1):
    return ((rank, value), depth if rank == 1 else 0, 0, 0)


@pytest.fixture(params=('enemy-one', 'enemy-two', 'next-heuristic', 'next-value'))
def choose_controlled(request, combat_factory, monkeypatch):
    battle = combat_factory(['DEFEND_RED'] * 5).thick.bc
    battle.apply_custom_state({'player': {'energy': 3}, 'hand': ['DEFEND_RED', 'DEFEND_RED'],
        'draw': ['STRIKE_RED', 'BASH'], 'discard': []})
    assert set(planner_module.complete_actions(battle)) == set(ROOTS)
    before = battle.to_dict()
    mode = request.param
    config = TurnPlannerConfig(worlds=2, node_budget=64,
        lookahead_turns=2 if mode == 'enemy-two' else 1,
        leaf_boundary='next-decision' if mode.startswith('next-') else 'enemy-phase')
    planner = NativeTurnPlanner(config)

    def choose(outcomes, *, order=ROOTS):
        assert set(outcomes) == set(order) == set(ROOTS)
        assert all(len(worlds) == config.worlds for worlds in outcomes.values())
        calls = {bits: 0 for bits in ROOTS}

        def controlled_root(sample, bits, quota, check, *args):
            # choose still creates each sampled native branch normally. No
            # execute/model calls are mocked into being real transition proof.
            assert sample is not battle and quota > 0
            score, winning_depth, live_depth, fallback = outcomes[bits][calls[bits]]
            calls[bits] += 1
            return score, dict(native_transitions=0, expanded_nodes=0, fallback_leaves=fallback,
                winning_steps=winning_depth, live_steps=live_depth, second_turn_entries=0,
                completed_horizon_leaves=0, max_action_depth=0, budget_exhausted=False,
                depth_limit_hits=0, model_leaf_evaluations=0, model_leaf_batches=0, model_leaf_seconds=0.0)

        def unused_value_callback(_):
            pytest.fail('aggregation-only fixture must not invoke a model callback')

        monkeypatch.setattr(planner_module, 'complete_actions', lambda _: list(order))
        monkeypatch.setattr(planner, '_root', controlled_root)
        result = planner.choose(battle, state_nonce=19, public_relics=(),
            leaf_evaluator=unused_value_callback if mode == 'next-value' else None)
        assert calls == {bits: config.worlds for bits in ROOTS}
        assert battle.to_dict() == before
        return result

    return choose


def test_one_ulp_utility_improvement_keeps_priority_over_short_live_path(choose_controlled):
    better = math.nextafter(0.5, math.inf)
    result = choose_controlled({0: [live(0.5, depth=1)] * 2,
        1: [live(better, depth=9)] * 2, END: [live(0.25)] * 2})
    assert result.action_bits == 1
    assert result.diagnostics['root_values']['1'] > result.diagnostics['root_values']['0']


@pytest.mark.parametrize('second_win_depth, expected', [(3, 0), (2, 1)])
def test_mixed_win_live_prioritizes_win_depth_before_live_depth(
        choose_controlled, second_win_depth, expected):
    result = choose_controlled({0: [terminal(1, depth=2), live(depth=8)],
        1: [terminal(1, depth=second_win_depth), live(depth=1)],
        END: [live(0.5)] * 2})
    assert result.action_bits == expected


def test_sums_best_live_depths_across_worlds_not_minimum_world(choose_controlled):
    result = choose_controlled({0: [live(depth=1), live(depth=9)],
        1: [live(depth=4), live(depth=4)], END: [live(depth=8)] * 2})
    assert result.action_bits == 1
    outcomes = result.diagnostics['root_outcomes']
    assert outcomes['0']['live_horizon_steps_total'] == 10
    assert outcomes['1']['live_horizon_steps_total'] == 8
    assert outcomes['1']['live_horizon_steps_mean'] == 4


@pytest.mark.parametrize('order', list(itertools.permutations(ROOTS)))
def test_one_fallback_disables_entire_exact_tie_group_for_every_root_order(choose_controlled, order):
    result = choose_controlled({0: [live(depth=4)] * 2,
        1: [live(depth=1)] * 2, END: [live(unresolved=True), live(depth=1)]}, order=order)
    assert result.action_bits == 0  # old native-bit tie, including the fallback root
    fallback = result.diagnostics['root_outcomes'][str(END)]
    assert fallback['live'] == 2 and fallback['unresolved'] == 1 and fallback['boundary_live'] == 1
    assert fallback['live_horizon_steps_total'] == fallback['live_horizon_steps_mean'] == 1


def test_fallback_below_top_utility_tie_does_not_disable_live_tie(choose_controlled):
    result = choose_controlled({0: [live(depth=8)] * 2,
        1: [live(depth=1)] * 2, END: [live(0.5, unresolved=True)] * 2})
    assert result.action_bits == 1


def test_fallback_below_top_winning_depth_tie_does_not_disable_live_tie(choose_controlled):
    result = choose_controlled({0: [terminal(1, depth=2), live(depth=8)],
        1: [terminal(1, depth=2), live(depth=1)],
        END: [terminal(1, depth=3), live(unresolved=True)]})
    assert result.action_bits == 1


@pytest.mark.parametrize('rank, expected', [(1, 1), (-1, 0)])
def test_all_terminal_keeps_existing_win_depth_then_native_bits(choose_controlled, rank, expected):
    result = choose_controlled({0: [terminal(rank, depth=8)] * 2,
        1: [terminal(rank, depth=1)] * 2, END: [terminal(rank, depth=9)] * 2})
    assert result.action_bits == expected
    assert all(r['live'] == r['boundary_live'] == r['live_horizon_steps_total'] == 0
        for r in result.diagnostics['root_outcomes'].values())


def test_all_unresolved_keeps_native_bits_without_fake_zero_depth_advantage(choose_controlled):
    result = choose_controlled({bits: [live(unresolved=True)] * 2 for bits in ROOTS}, order=tuple(reversed(ROOTS)))
    assert result.action_bits == 0
    assert all(r['boundary_live'] == 0 and r['unresolved'] == 2
        for r in result.diagnostics['root_outcomes'].values())
