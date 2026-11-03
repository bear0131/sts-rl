"""Censoring discards whole games, not just their unfinished last decision."""
from types import SimpleNamespace
import numpy as np
from rl.training.episode_rollout import EpisodeRolloutConfig, exclude_samples
import pytest


def test_censor_excludes_all_previous_fights_and_strategic_decisions_of_that_game():
    # Earlier completed fights from the censored game cannot leak into CE/PPO.
    retained_upper={'env':0,'episode':8,'reward':1.}
    retained_lower={'env':0,'recommended_action':12}
    model=SimpleNamespace(_upper_completed=[{'env':1,'episode':8,'reward':.5},retained_upper,
                                            {'env':1,'episode':8,'reward':.7}],
        _upper_pending=[None,{'reward':.2}],_upper_censored=0,
        _small_playout_samples=[{'env':1,'recommended_action':2},retained_lower,{'env':1,'recommended_action':3}])
    exclude_samples(model,SimpleNamespace(episode_rollout_censored=np.array([False,True])))
    assert model._upper_completed==[retained_upper]
    assert model._small_playout_samples==[retained_lower]
    assert model._upper_pending==[None,None]
    assert model._cohort_excluded_samples=={'excluded_upper_samples':3,'excluded_lower_samples':2}


def test_budget_contract_rejects_unknown_censor_policy():
    with pytest.raises(ValueError,match='budget'):
        EpisodeRolloutConfig(240,20261101,censored_games='count-as-death')


@pytest.fixture
def growth_model():
    config = dict(games_per_rank=240, seed=20261103, max_steps=6000,
                  max_seconds=415., censored_games='error')
    return SimpleNamespace(policy=SimpleNamespace(core=SimpleNamespace(joint_upper=True,
                upper_config=SimpleNamespace(architecture='upper-latent-actions-v1')),
                upper_auxiliary_config=None),
            nosl_is_az_inner_only=True, nosl_is_az_combat_strategy='small-playout',
            nosl_is_az_replay_enabled=False,
            env=SimpleNamespace(num_envs=360, begin_episode_rollout=lambda *a, **k: None),
            episode_rollout_config=config, episode_rollout_world_size=4,
            episode_rollouts_completed=192,
            episode_rollout_stats=dict(rollout=192, games=240, completed_games=240,
                                      censored_games=0, unfinished_games=0), num_timesteps=123)


@pytest.mark.parametrize('censor_policy', ['error', 'exclude'])
def test_growth_keeps_next_global_seed_across_all_ranks(growth_model, censor_policy):
    import copy
    from rl.training.episode_rollout import configure, begin
    growth_model.episode_rollout_config['censored_games'] = censor_policy
    starts = []
    for rank in range(4):
        model = copy.deepcopy(growth_model)
        configure(model, dict(model.episode_rollout_config, games_per_rank=360),
                  rank=rank, world_size=4, grow_environments=True)
        start = begin(model, model.env)['first_seed']
        starts.append(start)
        assert start == 20261103 + 192 * 960 + rank * 360
        # Ordinary resume preserves the offset and advances a full 1440 games.
        model.episode_rollouts_completed += 1
        configure(model, model.episode_rollout_config, rank=rank, world_size=4)
        assert begin(model, model.env)['first_seed'] == start + 1440
    assert starts == list(range(starts[0], starts[0] + 1440, 360))


@pytest.mark.parametrize('case', ['unfinished', 'censored', 'fewer', 'ordinary', 'world', 'budget', 'target', 'no_source'])
def test_growth_rejects_incompatible_boundary(growth_model, case):
    from rl.training.episode_rollout import configure
    model = growth_model
    config = dict(model.episode_rollout_config, games_per_rank=360)
    world, grow = 4, True
    if case == 'unfinished': model.episode_rollout_stats['unfinished_games'] = 1
    if case == 'censored': model.episode_rollout_stats['censored_games'] = 1
    if case == 'fewer': config['games_per_rank'] = model.env.num_envs = 120
    if case == 'ordinary': grow = False
    if case == 'world': world = 3
    if case == 'budget': config['max_steps'] += 1
    if case == 'target': model.policy.core.upper_config.architecture = 'legacy'
    if case == 'no_source': model.episode_rollout_config = None
    with pytest.raises(ValueError):
        configure(model, config, world_size=world, grow_environments=grow)
