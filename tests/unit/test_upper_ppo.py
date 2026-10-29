from types import SimpleNamespace

import numpy as np
import pytest

from env.core import encode, vocab as V
from rl.algorithms.upper_ppo import capture_step, macro_advantages, strategic_reward


def test_macro_discount_is_per_upper_action_and_episode_isolated():
    records = [dict(env=0, episode=0, terminal=False, reward=.2, value=.4),
               dict(env=1, episode=0, terminal=True, reward=1., value=.5),
               dict(env=0, episode=0, terminal=True, reward=1., value=.6),
               dict(env=0, episode=1, terminal=True, reward=0., value=.1)]
    advantages, returns = macro_advantages(records, [.6, 0., 0., 0.], .9, .8)
    np.testing.assert_allclose(advantages, [.2 + .9 * .6 - .4 + .9 * .8 * .4, .5, .4, -.1], atol=1e-7)
    assert returns[0] == pytest.approx(.2 + .9 * .6 + .9 * .8 * .4)


def test_reward_native_act3_victory_without_act_increment():
    stats = dict(act=3, floor=50, outcome='PLAYER_VICTORY')
    assert strategic_reward(3, 50, stats, done=True, censored=False) == 1
    assert strategic_reward(3, 50, stats, done=True, censored=True) == 0
    assert strategic_reward(3, 50, dict(stats, outcome='PLAYER_LOSS'), done=True, censored=False) == 0


def test_pending_combat_survives_rollout_boundary_and_counts_once():
    def obs(phase, floor):
        return dict(phase=np.array([[V.PHASE_VOCAB_INDEX[phase]]]),
                    act=np.array([[1 / encode.ACT_SCALE]]), floor=np.array([[floor / encode.FLOOR_SCALE]]))
    model = SimpleNamespace(_upper_completed=[], _upper_censored=0)
    map_obs, fight = obs('map', 0), obs('combat', 1)
    kwargs = dict(actions=np.array([2]), masks=np.ones((1, 198), bool), values=np.array([.2]), log_probs=np.array([-.7]))
    stats = dict(act=1, floor=1, outcome='UNDECIDED')
    capture_step(model, map_obs, **kwargs, new_obs=fight, dones=[False], infos=[{'run_stats': stats}])
    assert not model._upper_completed
    # A collection boundary discards neither the entering action nor its accumulated reward.
    import copy
    resumed = SimpleNamespace(_upper_pending=copy.deepcopy(model._upper_pending),
        _upper_episode_ids=model._upper_episode_ids.copy(), _upper_completed=[], _upper_censored=0)
    for _ in range(5):
        capture_step(resumed, fight, **kwargs, new_obs=fight, dones=[False], infos=[{'run_stats': stats}])
    capture_step(resumed, fight, **kwargs, new_obs=obs('combat_reward', 1), dones=[False], infos=[{'run_stats': stats}])
    assert resumed._upper_pending == [None]
    assert len(resumed._upper_completed) == 1
    row = resumed._upper_completed[0]
    assert row['action'] == 2 and row['log_prob'] == -.7 and row['reward'] == pytest.approx(.002)
    assert row['next_observation']['phase'][0] == V.PHASE_VOCAB_INDEX['combat_reward']


def test_censored_macro_not_a_defeat_label():
    model = SimpleNamespace(_upper_pending=[dict(reward=.2)], _upper_episode_ids=[0], _upper_completed=[], _upper_censored=0)
    obs = dict(phase=np.array([[V.PHASE_VOCAB_INDEX['combat']]]), act=np.array([[.25]]), floor=np.array([[.5]]))
    capture_step(model, obs, [0], np.ones((1, 198)), [0], [0], obs, [True],
                 [dict(run_stats=dict(act=1, floor=28, outcome='UNDECIDED'), **{'TimeLimit.truncated': True})])
    assert model._upper_pending == [None] and model._upper_completed == [] and model._upper_censored == 1


def test_small_playout_budget_counts_natural_games_and_retains_prior_count():
    model = SimpleNamespace(_upper_pending=[None], _upper_episode_ids=[0], _upper_completed=[],
                            _upper_censored=0, nosl_is_az_combat_strategy='small-playout',
                            small_playout_games_local=123)
    obs = dict(phase=np.array([[V.PHASE_VOCAB_INDEX['combat']]]), act=np.array([[.25]]), floor=np.array([[.5]]))
    def step(done, **flags):
        capture_step(model, obs, [0], np.ones((1, 198)), [0], [0], obs, [done],
                     [dict(run_stats=dict(act=1, floor=28, outcome='PLAYER_LOSS'), **flags)])
    step(False)  # A combat boundary is not a full game.
    step(True, **{'TimeLimit.truncated': True})
    step(True, dead_screen=True)
    step(True, startup_stagger_hold=True)
    assert model.small_playout_games_local == 123
    step(True)
    assert model.small_playout_games_local == 124


def test_lambda_one_returns_real_rewards_and_retains_only_truncation_bootstrap():
    # Interleaved environments, one true terminal and one collection boundary.
    records = [dict(env=0, episode=0, terminal=False, reward=.02, value=.8),
               dict(env=1, episode=0, terminal=False, reward=.1, value=.9),
               dict(env=0, episode=0, terminal=True, reward=.3, value=.2),
               dict(env=1, episode=0, terminal=False, reward=.2, value=.4),
               dict(env=0, episode=1, terminal=True, reward=0., value=.7)]
    advantages, returns = macro_advantages(records, [.2, .4, 99., .6, 99.], 1., 1.)
    np.testing.assert_allclose(returns, [.32, .9, .3, .8, 0.], atol=1e-7)
    np.testing.assert_allclose(advantages, [-.48, 0., .1, .4, -.7], atol=1e-7)
