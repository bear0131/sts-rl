"""Critic targets: analytic boundaries and invariance to room menu length."""
import copy
import numpy as np
import pytest

from env.core import encode
from rl.algorithms.upper_probe import (critic_lambda_returns, room_key,
                                      room_transition_audit, fit_room_calibration)
from rl.algorithms.unified_upper import complete_returns
from rl.algorithms.upper_ppo import macro_advantages


@pytest.fixture
def lambda_row():
    def make(*, floor=1, act=1, value=0., reward=0., terminal=False, env=0, episode=0):
        return dict(env=env, episode=episode, value=value, reward=reward, terminal=terminal,
            observation=dict(act=np.array([act/encode.ACT_SCALE], np.float32),
                             floor=np.array([floor/encode.FLOOR_SCALE], np.float32)))
    return make


def test_constant_endpoints_and_frozen_gae_agree_on_interleaved_games(lambda_row):
    r=lambda_row
    rows=[r(value=1., reward=.1), r(env=1, value=99., reward=.7, terminal=True),
          r(floor=2, value=2., reward=.2), r(floor=3, value=3., reward=.3, terminal=True),
          r(episode=1, reward=.4, terminal=True)]
    np.testing.assert_array_equal(critic_lambda_returns(rows, 1), complete_returns(rows))
    np.testing.assert_allclose(critic_lambda_returns(rows, 0), [2.1,.7,3.2,.3,.4])
    for coefficient in [.5,.8,.9,.95,.97,.99]:
        expected=macro_advantages(rows,np.zeros(len(rows)),1.,coefficient)[1]
        np.testing.assert_allclose(critic_lambda_returns(rows,coefficient),expected,atol=2e-15,rtol=1e-14)


def test_room_decay_only_on_room_boundary_and_across_acts(lambda_row):
    r=lambda_row
    rows=[r(value=1,reward=.1),r(value=2,reward=.2),
          r(floor=2,value=3,reward=.3,terminal=True)]
    np.testing.assert_allclose(critic_lambda_returns(rows,.5,'room'),[1.95,1.85,.3])
    # Same numeric floor in the next act is a distinct visit, never a terminal.
    rows=[r(value=1,reward=.2),r(act=2,value=3,reward=1,terminal=True)]
    np.testing.assert_allclose(critic_lambda_returns(rows,.5,'room'),[2.2,1.])
    np.testing.assert_array_equal(critic_lambda_returns(rows,1,'room'),complete_returns(rows))


def test_extra_same_room_zero_reward_menu_does_not_change_earlier_target(lambda_row):
    r=lambda_row
    rows=[r(value=.1,reward=.2),r(floor=2,value=.9,reward=1,terminal=True)]
    extended=[rows[0],r(value=99.,reward=0),rows[1]]
    baseline=critic_lambda_returns(rows,.5,'room')[0]
    assert critic_lambda_returns(extended,.5,'room')[0] == baseline
    assert critic_lambda_returns(extended,.5)[0] != pytest.approx(critic_lambda_returns(rows,.5)[0])
    assert room_key(r()['observation']) == room_key(r(value=99)['observation'])


def test_target_inputs_unchanged_and_incomplete_or_invalid_rejected(lambda_row):
    rows=[lambda_row(terminal=True,reward=1,value=999)]
    before=copy.deepcopy(rows)
    for scope in ['decision','room']:
        assert critic_lambda_returns(rows,0,scope).tolist()==[1.]
        assert critic_lambda_returns(rows,1,scope).tolist()==[1.]
    for coefficient in [-.1,1.1,float('nan')]:
        with pytest.raises(ValueError):critic_lambda_returns(rows,coefficient)
    with pytest.raises(ValueError):critic_lambda_returns([lambda_row()],.97)
    with pytest.raises(ValueError):critic_lambda_returns(rows,.97,'act')
    for key in ['value','reward','terminal']:
        assert rows[0][key]==before[0][key]
    for key in rows[0]['observation']:
        np.testing.assert_array_equal(rows[0]['observation'][key],before[0]['observation'][key])


def test_room_successor_audit_counts_and_rejects_false_boundary(lambda_row):
    r=lambda_row
    rows=[r(),r(),r(floor=2),r(act=2,floor=2,terminal=True)]
    rows[0]['next_observation']=copy.deepcopy(rows[1]['observation'])
    assert room_transition_audit(rows)==dict(within_room=1,across_room=2,across_act=1,terminals=1)
    rows[0]['next_observation']=r(floor=99)['observation']
    with pytest.raises(ValueError,match='successor'):
        room_transition_audit(rows)


def test_scene_bias_calibration_uses_game_weights_and_preserves_room_menus(lambda_row):
    def scene_row(phase, **kwargs):
        row=lambda_row(**kwargs)
        row['observation']['phase']=np.array([phase])
        return row
    records=[scene_row(1,env=i,value=1.,terminal=True) for i in range(4)]
    records += [scene_row(2,env=i+4,value=0.,terminal=True) for i in range(4)]
    settings=dict(base=.9,span=.07,shrinkage_games=0,noise_floor=.05)
    table=fit_room_calibration(records,settings)
    assert table['calibration_games']==8
    assert table['scenes']['1:1']['coefficient'] > .96
    assert table['scenes']['1:2']['coefficient'] == .9
    first=scene_row(1,value=.1,reward=.2)
    last=scene_row(2,floor=2,value=.9,reward=1.,terminal=True)
    original=critic_lambda_returns([first,last],.9,'calibrated_room',table)[0]
    # Arbitrarily wrong V or different phase on an extra internal menu cannot
    # alter the earlier target, nor re-fit the frozen calibration table.
    before=copy.deepcopy(table)
    extra=scene_row(1,value=99.)
    assert critic_lambda_returns([first,extra,last],.9,'calibrated_room',table)[0]==original
    assert table==before
    with pytest.raises(ValueError,match='frozen calibration'):
        critic_lambda_returns([first,last],.9,'calibrated_room')


def test_scene_calibration_does_not_count_menu_length_as_more_games(lambda_row):
    row=lambda_row(value=1.,terminal=True)
    row['observation']['phase']=np.array([1])
    shorter=[row,dict(row,env=1,value=0.)]
    longer=[dict(row,terminal=False),*shorter]
    settings=dict(base=.9,span=.07,shrinkage_games=32,noise_floor=.05)
    first=fit_room_calibration(shorter,settings)
    second=fit_room_calibration(longer,settings)
    assert first==second
