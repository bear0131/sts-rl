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
