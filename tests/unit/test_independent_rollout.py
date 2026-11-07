"""Per-game random streams and explicit diagnostic support contracts."""
import numpy as np
import pytest

from rl.algorithms.independent_rollout import GameRandomStreams, IndependentRolloutConfig


def test_random_streams_ignore_scheduling_and_restore_each_game():
    a, b = [GameRandomStreams(20261104, 20581743, 3) for _ in range(2)]
    expected = {}
    for i in (0, 1, 2):
        expected[i] = [(a.action(i, [0., .3, .7]), a.nonce(i)) for _ in range(20)]
    actual = {i: [] for i in range(3)}
    for _ in range(20):
        for i in (2, 0, 1):
            actual[i].append((b.action(i, [0., .3, .7]), b.nonce(i)))
    assert actual == expected
    saved = b.state_dict()
    c = GameRandomStreams(123, 9, 3)
    c.load_state_dict(saved)
    assert c.nonce(2) == b.nonce(2)
    assert c.action(0, [.6, .4]) == b.action(0, [.6, .4])
    # Search scheduling never consumes an upper-policy draw.
    for _ in range(50):
        c.nonce(0)
    assert c.action(0, [.6, .4]) == b.action(0, [.6, .4])


def test_categorical_sampling_respects_probabilities_and_zero_support():
    streams = GameRandomStreams(42, 19, 1)
    counts = np.bincount([streams.action(0, [0., .2, .8]) for _ in range(20000)], minlength=3)
    assert counts[0] == 0
    np.testing.assert_allclose(counts[1:]/20000, [.2, .8], atol=.015)
    for invalid in ([0., 0.], [-1., 2.], [np.nan, 1.]):
        with pytest.raises(ValueError):
            streams.action(0, invalid)


@pytest.mark.parametrize('values', [dict(threads=0), dict(upper_batch=0),
    dict(upper_wait_seconds=float('nan')), dict(mode='production'),
    dict(max_combat_turns=0), dict(max_combat_turns=True)])
def test_diagnostic_rejects_unsupported_configuration(values):
    with pytest.raises(ValueError):
        IndependentRolloutConfig(**values)


def test_scheduler_migration_requires_complete_fork_and_resume_keeps_contract():
    from types import SimpleNamespace
    from rl.algorithms.independent_rollout import configure
    model = SimpleNamespace(num_timesteps=100, episode_rollouts_completed=3,
        episode_rollout_config=dict(games_per_rank=2, censored_games='error'),
        episode_rollout_stats=dict(rollout=3, completed_games=2, censored_games=0, unfinished_games=0))
    assert configure(model, None) is None
    with pytest.raises(ValueError, match='explicit fork'):
        configure(model, {})
    model.episode_rollout_stats['unfinished_games'] = 1
    with pytest.raises(ValueError, match='explicit fork'):
        configure(model, {}, fork=True)
    model.episode_rollout_stats['unfinished_games'] = 0
    config = configure(model, {}, fork=True)
    assert configure(model, None) == config
    with pytest.raises(ValueError, match='explicit fork'):
        configure(model, dict(config, seed=44))
    assert model.independent_rollout_config == config
    assert model.independent_rollout_migration['source_round'] == 3
    with pytest.raises(ValueError, match='diagnostic'):
        configure(model, dict(config, mode='barrier'), fork=True)


def test_observation_reading_callbacks_keep_full_transport_including_subclasses():
    from stable_baselines3.common.callbacks import CallbackList
    from rl.algorithms.independent_rollout import _callback_uses_observations
    from rl.training.distributed_sb3 import NoopCallback, RolloutProgressCallback
    from rl.tasks.wholerun import StSMetricsCallback

    class ReadingCallback(NoopCallback):
        def _on_step(self):
            assert 'hand_ids' in self.model._last_obs
            return True

    class CustomList(CallbackList):
        pass

    ordinary = [NoopCallback(), RolloutProgressCallback(0, 10), StSMetricsCallback()]
    assert not _callback_uses_observations(CallbackList(ordinary))
    assert _callback_uses_observations(ReadingCallback())
    assert _callback_uses_observations(CallbackList([*ordinary, CallbackList([ReadingCallback()])]))
    assert _callback_uses_observations(CustomList(ordinary))


def test_resume_can_resize_workers_only_at_complete_boundary():
    from types import SimpleNamespace
    from rl.algorithms.independent_rollout import configure

    model = SimpleNamespace(num_timesteps=0, episode_rollouts_completed=3,
        episode_rollout_config=dict(games_per_rank=2, censored_games='error'),
        episode_rollout_stats=dict(rollout=3, completed_games=2, censored_games=0, unfinished_games=0))
    old = configure(model, dict(threads=16, max_combat_turns=50))
    model.num_timesteps = 100
    selected = dict(old, threads=24)
    for field, value in [('rollout', 2), ('completed_games', 1),
                         ('censored_games', 1), ('unfinished_games', 1)]:
        original = model.episode_rollout_stats[field]
        model.episode_rollout_stats[field] = value
        with pytest.raises(ValueError, match='complete natural boundary'):
            configure(model, selected)
        assert model.independent_rollout_config == old
        model.episode_rollout_stats[field] = original
    for change in [dict(seed=44), dict(upper_batch=128),
                   dict(upper_wait_seconds=.1), dict(max_combat_turns=60)]:
        with pytest.raises(ValueError, match='explicit fork'):
            configure(model, dict(selected, **change))
    assert configure(model, selected) == selected
    assert configure(model, None) == selected
    assert model.independent_rollout_migration['previous'] == old
    assert configure(model, old) == old
    model.episode_rollout_stats.update(completed_games=1, censored_games=1)
    assert configure(model, selected) == selected
    # Excluded games are only an existing capped-cohort contract, not permission
    # to resume arbitrary interrupted/uncapped collections.
    uncapped = {k: v for k, v in selected.items() if k != 'max_combat_turns'}
    model.independent_rollout_config = uncapped
    with pytest.raises(ValueError, match='complete natural boundary'):
        configure(model, dict(uncapped, threads=16))


def test_enable_combat_turn_limit_at_completed_boundary_and_keep_it_on_resume():
    from types import SimpleNamespace
    from rl.algorithms.independent_rollout import configure

    model = SimpleNamespace(num_timesteps=0, episode_rollouts_completed=3,
        episode_rollout_config=dict(games_per_rank=2, censored_games='error'),
        episode_rollout_stats=dict(rollout=3, completed_games=2, censored_games=0, unfinished_games=0))
    old = configure(model, {})
    model.num_timesteps = 100
    guarded = dict(old, max_combat_turns=50)
    with pytest.raises(ValueError):
        configure(model, dict(guarded, seed=123))
    model.episode_rollout_stats['unfinished_games'] = 1
    with pytest.raises(ValueError):
        configure(model, guarded)
    model.episode_rollout_stats['unfinished_games'] = 0
    assert configure(model, guarded) == configure(model, None) == guarded
    with pytest.raises(ValueError):
        configure(model, old)
