"""The historical terminal lower bound keeps the exact allocated budget."""
import numpy as np
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
