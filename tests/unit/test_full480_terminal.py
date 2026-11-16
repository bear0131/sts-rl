"""The historical terminal lower bound keeps the exact allocated budget."""
import numpy as np
import pytest
from rl.search.full480.finite_plan import PAIRED_TERMINAL, aggregate, paired_worlds


def test_partial_world_keeps_cut_visits_without_bootstrapping():
    root=dict(contract=PAIRED_TERMINAL,budget=5,action_bits=np.array([1,2,3],np.uint32),
              worlds=paired_worlds(20261105,3,5))
    assert len(root['worlds'])==2
    plan=dict(root=np.zeros(5,np.int32),action_index=np.array([0,1,2,0,1]))
    stats=np.zeros((5,12),np.uint64)
    result=aggregate([root],plan,[1.,np.nan,.4,.2,np.nan],[1,0,3,1,0],stats)[0]
    np.testing.assert_array_equal(result['visits'],[2,2,1])
    np.testing.assert_array_equal(result['terminal_visits'],[2,0,1])
    np.testing.assert_allclose(result['values'],[.6,0.,.4])
    assert result['simulations']==5 and result['depth_cuts']==2


def test_original_budget_is_not_rounded_to_action_count():
    for actions in (1,3,197):
        for budget in (250,1000):
            worlds=paired_worlds(20261105,actions,budget)
            assert len(worlds)==(budget+actions-1)//actions


def test_turn_bootstrap_keeps_deaths_and_depth_exhaustion_distinct():
    root = dict(contract=PAIRED_TERMINAL, value_target='pre-reward-anchor',
                stop_turn=2, budget=4, action_bits=np.array([1, 2], np.uint32))
    plan = dict(root=np.zeros(4, np.int32), action_index=np.array([0, 1, 0, 1]))
    stats = np.zeros((4, 12), np.uint64)
    stats[0, 3] = 1  # Actual victory.
    stats[1, 4] = 1  # Actual death, worth zero.
    stats[2, 6] = 1  # Next-turn state, bootstrapped by the combat value.
    stats[3, 5] = 1  # Atomic-action depth budget, not a completed turn.
    result = aggregate([root], plan, [1.1, 0., .7, .4], [1, 2, 0, 0], stats)[0]
    np.testing.assert_allclose(result['values'], [.9, .2])
    np.testing.assert_array_equal(result['terminal_visits'], [1, 1])
    assert result['turn_boundaries'] == result['depth_cuts'] == 1
    assert result['value_statistic'] == 'turn-bootstrap-v1'


def test_default_anchor_completion_keeps_legacy_cutoff_statistics():
    root = dict(contract=PAIRED_TERMINAL, value_target='pre-reward-anchor',
                budget=3, action_bits=np.array([1], np.uint32))
    plan = dict(root=np.zeros(3, np.int32), action_index=np.zeros(3, np.int32))
    result = aggregate([root], plan, [.9, 0., .6], [1, 2, 0],
                       np.zeros((3, 12), np.uint64))[0]
    np.testing.assert_allclose(result['values'], [.5])
    assert result['turn_boundaries'] == 0 and result['depth_cuts'] == 1
    assert result['value_statistic'] == 'pre-reward-anchor'


def test_chain_value_mixture_weights_chains_not_neural_nodes_and_preserves_returns():
    root = dict(contract=PAIRED_TERMINAL, value_target='pre-reward-anchor',
                budget=4, action_bits=np.array([12, 7], np.uint32), value_mix=.25)
    plan = dict(root=np.zeros(4, np.int32), action_index=np.array([1, 0, 1, 0]))
    returns = np.array([.2, .4, 0., 1.])
    saved = returns.copy()
    stats = np.zeros((4, 12), np.uint64)
    stats[:, 1] = [1, 9, 0, 2]
    kwargs = dict(chain_value_sums=np.array([.8, 1.8, 0., 1.6]),
                  chain_value_counts=np.array([1, 9, 0, 2]))
    result = aggregate([root], plan, returns, [1, 0, 2, 1], stats, **kwargs)[0]
    np.testing.assert_allclose(result['values'], [.65, .175])
    np.testing.assert_array_equal(returns, saved)
    np.testing.assert_array_equal(result['terminal_visits'], [1, 2])
    assert result['chain_value_nodes'] == 12 and result['chain_value_empty'] == 1
    assert result['value_statistic'] == 'chain-value-mix-v1' and result['depth_cuts'] == 1
    ordinary = aggregate([dict(root, value_mix=0.)], plan, returns, [1, 0, 2, 1], stats)[0]
    np.testing.assert_allclose(ordinary['values'], [.7, .1])
    assert ordinary['value_statistic'] == 'pre-reward-anchor'
    with pytest.raises(ValueError, match='recorded'):
        aggregate([root], plan, returns, [1, 0, 2, 1], stats)
    kwargs['chain_value_counts'][1] += 1
    with pytest.raises(ValueError, match='successful neural'):
        aggregate([root], plan, returns, [1, 0, 2, 1], stats, **kwargs)


def test_chain_mixture_keeps_partial_world_action_slots_and_empty_chain_anchor():
    root = dict(contract=PAIRED_TERMINAL, value_target='pre-reward-anchor',
                budget=5, action_bits=np.array([11, 22, 33], np.uint32), value_mix=.25)
    plan = dict(root=np.zeros(5, np.int32), action_index=np.array([2, 0, 1, 2, 0]))
    stats = np.zeros((5, 12), np.uint64)
    stats[:, 1] = [0, 1, 0, 1, 2]
    result = aggregate([root], plan, [.8, .4, 0., .2, 1.], [1, 1, 2, 0, 1], stats,
        chain_value_sums=[0., .8, 0., .6, 1.2], chain_value_counts=[0, 1, 0, 1, 2])[0]
    np.testing.assert_allclose(result['values'], [.7, 0., .55])
    np.testing.assert_array_equal(result['visits'], [2, 1, 2])
    assert result['simulations'] == 5 and result['chain_value_empty'] == 2
