from rl.training.distributed_sb3 import (
    _DISTRIBUTED_GAME_METRIC_KEYS,
    _add_game_metrics_to_wandb_payload,
    _finalize_distributed_game_scalars,
    _prepare_distributed_game_scalars,
    logger_scalars,
)


def test_chest_windows_are_independent_of_episodes_and_rank_sizes():
    from types import SimpleNamespace
    from stable_baselines3.common.logger import Logger
    from rl.tasks.wholerun import StSMetricsCallback

    scalars = []
    for events in ([{'chest_opened': True}, {'chest_relic_taken': True}],
                   [{'chest_opened': False}] * 3 + [{'chest_opened': True},
                    {'chest_relic_taken': False}, {}, {}]):
        callback = StSMetricsCallback(window=100)
        callback.model = SimpleNamespace(logger=Logger(None, []))
        callback.locals = {'infos': events}
        callback._on_step()
        callback._on_rollout_end()
        scalars.append(_prepare_distributed_game_scalars(logger_scalars(callback.model)))
    aggregate = {k: sum(s[k] for s in scalars) / 2 for k in scalars[0]}
    _finalize_distributed_game_scalars(aggregate, world_size=2)
    payload = {}
    _add_game_metrics_to_wandb_payload(payload, aggregate)
    assert payload['game/chest_open_rate'] == 2 / 5
    assert payload['game/chest_relic_take_rate'] == 1 / 2
    assert payload['game/chest_opportunities_window'] == 5
    assert payload['game/chest_relic_opportunities_window'] == 2
    assert 'game/win_rate' not in payload


def test_no_chest_offers_does_not_report_zero_probability():
    values = _prepare_distributed_game_scalars({})
    _finalize_distributed_game_scalars(values, world_size=4)
    payload = {}
    _add_game_metrics_to_wandb_payload(payload, values)
    assert 'game/chest_open_rate' not in payload
    assert 'game/chest_relic_take_rate' not in payload


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
    left["game/deck_size_mean"] = 18.0
    right["game/deck_size_mean"] = 30.0
    left = _prepare_distributed_game_scalars(left)
    right = _prepare_distributed_game_scalars(right)
    averaged = {key: (left[key] + right[key]) / 2.0 for key in left}

    _finalize_distributed_game_scalars(averaged, world_size=2)

    assert averaged["game/win_rate"] == (40 * 0.25 + 100 * 0.5) / 140
    assert averaged["game/deck_size_mean"] == (40 * 18.0 + 100 * 30.0) / 140
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


def test_boss_windows_keep_encounter_denominators_and_omit_run_clear_rates():
    counts = []
    for wins, total in [(3, 4), (8, 20)]:
        values = _prepare_distributed_game_scalars({
            'game/episodes_window': total, 'game/win_rate': wins / total,
            'boss/CHAMP/episodes_window': total, 'boss/CHAMP/wins_window': wins,
        })
        counts.append(values)
    reduced = {key: sum(row[key] for row in counts) / 2 for key in counts[0]}
    _finalize_distributed_game_scalars(reduced, world_size=2)
    reduced['episode_scope'] = 'boss-opening'
    payload = {}
    _add_game_metrics_to_wandb_payload(payload, reduced)
    assert payload['boss/CHAMP/episodes_window'] == 24
    assert payload['boss/CHAMP/win_rate'] == 11 / 24
    assert payload['boss/win_rate'] == 11 / 24
    assert not any(key.startswith(('game/act_', 'game/floor_', 'game/layer_')) for key in payload)


def test_boss_callback_counts_reach_the_production_logger_and_payload():
    from types import SimpleNamespace
    from stable_baselines3.common.logger import Logger
    from rl.tasks.wholerun import StSMetricsCallback

    callback = StSMetricsCallback(window=100)
    callback.model = SimpleNamespace(logger=Logger(None, []))
    for boss, outcome in [('CHAMP', 'win'), ('CHAMP', 'loss'),
                          ('AUTOMATON', 'win'), ('COLLECTOR', 'escaped')]:
        callback.locals = {'infos': [{'episode': {'r': 0, 'l': 10},
            'run_stats': {'episode_scope': 'boss-opening', 'boss': boss,
                'combat_outcome': outcome, 'act': 2, 'floor': 33, 'deck_size': 20,
                'outcome': 'PLAYER_VICTORY' if outcome == 'win' else 'PLAYER_LOSS'}}]}
        callback._on_step()
    callback._on_rollout_end()
    scalars = _prepare_distributed_game_scalars(logger_scalars(callback.model))
    _finalize_distributed_game_scalars(scalars, world_size=1)
    scalars['episode_scope'] = 'boss-opening'
    payload = {}
    _add_game_metrics_to_wandb_payload(payload, scalars)
    assert payload['boss/CHAMP/episodes_window'] == 2
    assert payload['boss/CHAMP/win_rate'] == .5
    assert payload['boss/AUTOMATON/win_rate'] == 1
    assert payload['boss/COLLECTOR/episodes_window'] == 0
    assert payload['boss/censored_episodes_total'] == 1
    assert payload['boss/win_rate'] == 2 / 3
