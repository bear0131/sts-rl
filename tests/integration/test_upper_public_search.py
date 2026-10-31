import copy

import numpy as np
import pytest

from env.core import action_space as A
from env.search.upper_public import build_tree, backup, SearchBudget

pytestmark = [pytest.mark.native, pytest.mark.timeout(90)]


def fingerprint(env):
    return (copy.deepcopy(env.thick.obs_canonical()),
            env._array_mapping_digest(env._obs()),
            copy.deepcopy(env._episode_action_journal),
            copy.deepcopy(env.np_random.bit_generator.state))


@pytest.mark.parametrize('phase', ['shop', 'event', 'rest', 'combat_reward'])
def test_public_branches_preserve_actual_trajectory_and_hidden_card_rng(upper_search_env_factory, phase):
    env = upper_search_env_factory(phase)
    before = fingerprint(env)
    control = env.fork_run()
    hidden = env.fork_run()
    hidden.thick.gc._dbg_set_card_rng(12345)
    a = build_tree(env)
    b = build_tree(hidden)
    assert a['reason'] == b['reason'] == 'ok'
    assert a['transitions'] == b['transitions'] <= 64
    assert len(a['nodes']) == len(b['nodes'])
    for x, y in zip(a['nodes'], b['nodes']):
        for key in ('action', 'reward', 'terminal', 'children'):
            assert x[key] == y[key]
        if x['observation'] is not None:
            for key in x['observation']:
                np.testing.assert_array_equal(x['observation'][key], y['observation'][key])
    assert fingerprint(env) == before
    action = a['nodes'][a['nodes'][0]['children'][0]]['action']
    observed, *rest = env.step(action)
    expected, *other = control.step(action)
    for key in expected:
        np.testing.assert_array_equal(observed[key], expected[key])
    assert rest[:3] == other[:3]
    assert fingerprint(env) == fingerprint(control)


def test_shop_remove_grid_and_callback_are_independent(upper_search_env_factory):
    env = upper_search_env_factory('shop')
    before = fingerprint(env)
    branch = env.fork_run()
    branch.step(A.seg_base('SHOP_REMOVE'))
    assert branch._phase() == 'card_select'
    tree = build_tree(branch)
    assert tree['reason'] == 'ok'
    first = tree['nodes'][tree['nodes'][0]['children'][0]]
    branch.step(first['action'])
    assert branch._phase() == 'shop'
    branch.step(A.seg_base('PROCEED'))
    assert branch._phase() == 'map'
    assert fingerprint(env) == before


def test_whole_level_budget_and_timeout_discard(upper_search_env_factory):
    env = upper_search_env_factory('shop')
    shallow = build_tree(env, SearchBudget(depth=1))
    width = shallow['transitions']
    tree = build_tree(env, SearchBudget(transitions=width))
    assert tree['reason'] == 'ok' and tree['transitions'] == width
    assert all(not node['children'] for node in tree['nodes'][1:])
    assert build_tree(env, SearchBudget(transitions=1))['reason'] == 'root_budget'
    cut = build_tree(env, SearchBudget(seconds=1e-12))
    assert cut['reason'] == 'timeout' and cut['nodes'] == []
    with pytest.raises(ValueError, match='discarded'):
        backup(cut, {})


def test_complete_fork_does_not_enable_combat_only_api_on_upper(upper_search_env_factory):
    env = upper_search_env_factory('shop')
    with pytest.raises(RuntimeError, match='combat'):
        env.fork_combat()
    with pytest.raises(RuntimeError, match='combat'):
        env.fork_run(worldline=12)
