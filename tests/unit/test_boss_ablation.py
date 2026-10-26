from copy import deepcopy
import numpy as np
import pytest

from experiments.launchers.boss_ablation import remove_boss_replay


@pytest.mark.parametrize('flags', [[0,1,0,1], [1,1,1,1], [0,0,0,0]])
def test_fork_filters_wrapped_rows_and_preserves_sparse_policy(flags):
    state = dict(size=4,capacity=4,position=2,total_added=6,
        observations={'combat_is_boss':np.array(flags)[:,None], 'token':np.arange(4)[:,None]},
        arrays={'sample_ids':np.array([4,5,2,3],np.uint64), 'value_targets':np.array([.4,.5,.2,.3],np.float32), 'policy_targets':None},
        joint=dict(offsets=np.array([0,1,3,3,4],np.int64), ids=np.array([5,2,4,8],np.uint16),
                   probabilities=np.array([1,.3,.7,1],np.float32)))
    before=deepcopy(state)
    result, removed=remove_boss_replay(state)
    keep=[i for i in [2,3,0,1] if flags[i]==0]
    assert removed==sum(flags) and result['size']==len(keep)
    assert result['total_added']==6 and result['position']==len(keep)%4
    np.testing.assert_array_equal(result['observations']['token'][:,0],keep)
    np.testing.assert_array_equal(state['observations']['combat_is_boss'],before['observations']['combat_is_boss'])
    for j,i in enumerate(keep):
        old=slice(*state['joint']['offsets'][i:i+2]); new=slice(*result['joint']['offsets'][j:j+2])
        for key in ('ids','probabilities'):
            np.testing.assert_array_equal(result['joint'][key][new],state['joint'][key][old])
        assert result['arrays']['value_targets'][j]==state['arrays']['value_targets'][i]


def test_fork_rejects_unknown_boss_flag():
    with pytest.raises(ValueError,match='not binary'):
        remove_boss_replay(dict(size=1,observations={'combat_is_boss':np.array([[.5]])}))


def test_boss_actual_maximum_target_survives_final_label_assignment():
    from rl.algorithms.accumulating_ppo import _fill_nosl_combat_value_targets
    samples=[{},{}]
    assert _fill_nosl_combat_value_targets(samples,1.1,clip_hp=False)==2
    assert [s['value_target'] for s in samples]==[1.1,1.1]
    _fill_nosl_combat_value_targets(samples,1.1)
    assert [s['value_target'] for s in samples]==[1.,1.]
