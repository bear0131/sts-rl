from rl.training.distributed_sb3 import (
    _DISTRIBUTED_GAME_METRIC_KEYS,
    _add_game_metrics_to_wandb_payload,
    _finalize_distributed_game_scalars,
    _prepare_distributed_game_scalars,
)


def test_game_metrics_are_weighted_by_their_actual_rank_windows():
    left = {
        "game/episodes_window": 40.0,
        "game/episodes_total": 60.0,
        "game/startup_episodes_excluded_total": 5.0,
        **{key: 0.25 for key in _DISTRIBUTED_GAME_METRIC_KEYS},
    }
    right = {
        "game/episodes_window": 100.0,
        "game/episodes_total": 120.0,
        "game/startup_episodes_excluded_total": 7.0,
        **{key: 0.5 for key in _DISTRIBUTED_GAME_METRIC_KEYS},
    }
    left = _prepare_distributed_game_scalars(left)
    right = _prepare_distributed_game_scalars(right)
    averaged = {key: (left[key] + right[key]) / 2.0 for key in left}

    _finalize_distributed_game_scalars(averaged, world_size=2)

    assert averaged["game/win_rate"] == (40 * 0.25 + 100 * 0.5) / 140
    assert averaged["game/episodes_window"] == 140
    assert averaged["game/episodes_total"] == 180
    assert averaged["game/startup_episodes_excluded_total"] == 12


def test_wandb_omits_game_rates_until_a_complete_fresh_episode_exists():
    record = {
        "game/episodes_window": 0.0,
        "game/episodes_total": 0.0,
        "game/startup_episodes_excluded_total": 384.0,
        **{key: 0.0 for key in _DISTRIBUTED_GAME_METRIC_KEYS},
    }
    payload = {}

    _add_game_metrics_to_wandb_payload(payload, record)

    assert payload == {
        "game/episodes_window": 0.0,
        "game/episodes_total": 0.0,
        "game/startup_episodes_excluded_total": 384.0,
    }
