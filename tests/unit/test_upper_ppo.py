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
    advantages, returns = macro_advantages(records, [float("nan"), -123., 99., .6, 99.], 1., 1.)
    np.testing.assert_allclose(returns, [.32, .9, .3, .8, 0.], atol=1e-7)
    np.testing.assert_allclose(advantages, [-.48, 0., .1, .4, -.7], atol=1e-7)

@pytest.mark.parametrize('micro', [2, 4, 16])
@pytest.mark.parametrize('multi_step', [False, True, 'tail'])
def test_upper_execution_preserves_global_sample_weights_and_targets(micro, multi_step):
    import copy
    from contextlib import nullcontext
    import torch
    from rl.algorithms.upper_ppo import update_upper
    from rl.policies.entity_joint import UpperConfig

    predicted_rows = []
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([.2, -.1]))
            self.core = SimpleNamespace(joint_upper=True,
                upper_config=UpperConfig(epochs=1, micro_batch_size=micro, gae_lambda=1., entropy_coef=0., value_coef=1.),
                upper_parameters=lambda: iter([self.weight]))
            self.upper_optimizer = torch.optim.Adam([self.weight], lr=1e-4)
        def set_training_mode(self, training):
            self.train(training)
        def predict_values(self, observations):
            assert not torch.is_autocast_enabled('cpu')
            predicted_rows.append(len(observations['x']))
            return observations['x'].float() @ self.weight
        def evaluate_actions(self, observations, actions, action_masks):
            values = observations['x'].float() @ self.weight
            zero = values * 0.
            return values, zero, zero

    phases = ['map', 'shop', 'map', 'event', 'map', 'map', 'map']
    x = np.array([[1., 0.], [0., 3.], [2., 1.], [-1., 2.], [1., 1.], [.5, -.5], [3., -2.]], dtype=np.float32)
    targets = np.array([.4, 1.2, -.1, .8, .3, -.5, .7], dtype=np.float32)
    records = [dict(env=i, episode=0, terminal=True, reward=float(targets[i]), value=0.,
                    observation={'phase': np.array([V.PHASE_VOCAB_INDEX[p]]), 'x': x[i]},
                    action=0, mask=np.array([True]), log_prob=0.) for i, p in enumerate(phases)]
    policy = Policy()
    if multi_step:
        for i, record in enumerate(records):
            record['env'] = i % 2
            record['value'] = float(torch.as_tensor(x[i]) @ policy.weight.detach())
            record['terminal'] = i + 2 >= len(records)
            if not record['terminal']:
                record['next_observation'] = records[i + 2]['observation']
        targets = np.array([sum(targets[i::2]) for i in range(len(records))], dtype=np.float32)
        if multi_step == 'tail':
            for env in range(2):
                last = max(i for i, r in enumerate(records) if r['env'] == env)
                records[last]['terminal'] = False
                records[last]['next_observation'] = records[last]['observation']
                targets[env::2] += records[last]['value']
    original = copy.deepcopy(records)
    expected_gradient = 2 * torch.as_tensor(x).T @ (torch.as_tensor(x) @ policy.weight.detach() - torch.as_tensor(targets)) / len(records)
    seen = []
    model = SimpleNamespace(policy=policy, device=torch.device('cpu'), _upper_completed=records,
        _upper_pending=[], autocast_context=nullcontext, amp_enabled=False,
        logger=SimpleNamespace(record=lambda *args: None))
    result = update_upper(model, synchronize=lambda epoch: seen.append(policy.weight.grad.detach().clone()))
    torch.testing.assert_close(seen[0], expected_gradient, rtol=2e-6, atol=2e-7)
    assert result['samples'] == 7 and result['updates'] == 1
    assert sum(predicted_rows) == result['bootstrap_samples'] == (2 if multi_step == 'tail' else 0)
    assert result['bootstrap_precision_bits'] == 32 and result['gae_precision_bits'] == 64
    assert model._upper_completed == []
    # The caller's chronological records are neither reordered nor rewritten.
    for before, after in zip(original, records):
        assert before['env'] == after['env'] and before['reward'] == after['reward']
        np.testing.assert_array_equal(before['observation']['x'], after['observation']['x'])



def test_tail_selection_and_fp64_reward_sum_ignore_intermediate_value_noise():
    from rl.algorithms.upper_ppo import bootstrap_indices
    records = [dict(env=0, episode=0, terminal=False, reward=1e-6,
                    value=(i % 5) * .1234567) for i in range(10000)]
    bootstrap = np.full(len(records), np.nan)
    bootstrap[-1] = .6
    assert bootstrap_indices(records) == [9999]
    advantages, returns = macro_advantages(records, bootstrap, 1., 1.)
    assert advantages.dtype == returns.dtype == np.float64
    np.testing.assert_allclose(returns, .6 + np.arange(10000, 0, -1) * 1e-6,
                               rtol=0, atol=2e-12)
    records[-1]['terminal'] = True
    assert bootstrap_indices(records) == []
    _, returns = macro_advantages(records, bootstrap, 1., 1.)
    np.testing.assert_allclose(returns, np.arange(10000, 0, -1) * 1e-6,
                               rtol=0, atol=2e-12)


def test_tail_selection_isolated_across_envs_episodes_and_true_ends():
    from rl.algorithms.upper_ppo import bootstrap_indices
    records = [dict(env=0, episode=0, terminal=False),
               dict(env=1, episode=0, terminal=False),
               dict(env=0, episode=0, terminal=True),
               dict(env=0, episode=1, terminal=False),
               dict(env=1, episode=0, terminal=False)]
    assert bootstrap_indices(records) == [3, 4]


def test_full_precision_disables_nested_autocast_and_restores_on_error():
    import torch
    from rl.policies.precision import full_precision
    with torch.autocast('cpu', dtype=torch.bfloat16):
        assert torch.is_autocast_enabled('cpu')
        with pytest.raises(RuntimeError, match='intentional'):
            with full_precision('cpu'):
                assert not torch.is_autocast_enabled('cpu')
                x = torch.ones(2, 2)
                assert (x @ x).dtype == torch.float32
                raise RuntimeError('intentional')
        assert torch.is_autocast_enabled('cpu')
