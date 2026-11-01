"""Fixed cohorts of fresh whole games for the existing joint collector.

One environment owns one game. Finished slots wait; no replacement games or
partial-episode labels enter the update. The explicit exclusion mode removes
all samples of a budget-censored game; it never calls that game a loss or
silently replaces its seed. Historical error mode remains available.
"""
from dataclasses import asdict, dataclass
import math
import time

import numpy as np


@dataclass(frozen=True)
class EpisodeRolloutConfig:
    games_per_rank: int
    seed: int
    max_steps: int = 6000
    max_seconds: float = 1800.
    censored_games: str = 'error'

    def __post_init__(self):
        if (any(type(x) is not int for x in (self.games_per_rank, self.seed, self.max_steps))
                or min(self.games_per_rank, self.max_steps) < 1 or self.seed < 0
                or not math.isfinite(self.max_seconds) or self.max_seconds <= 0
                or self.censored_games not in ('error', 'exclude')):
            raise ValueError('invalid full-game rollout budget')


def configure(model, config, *, rank=0, world_size=1):
    if config is None:
        return
    cfg = EpisodeRolloutConfig(**config)
    if (not getattr(model.policy.core, 'joint_upper', False)
            or not model.nosl_is_az_inner_only
            or model.nosl_is_az_combat_strategy != 'small-playout'
            or not hasattr(model.env, 'begin_episode_rollout')
            or model.env.num_envs != cfg.games_per_rank
            or model.policy.upper_auxiliary_config is not None
            or model.nosl_is_az_replay_enabled):
        raise ValueError('whole-game cohorts require native joint small-playout, aux/replay off, one env per game')
    if not 0 <= rank < world_size:
        raise ValueError('invalid cohort rank')
    old = getattr(model, 'episode_rollout_config', None)
    if old is not None:
        old = asdict(EpisodeRolloutConfig(**old))
    if old is not None and old != asdict(cfg):
        raise ValueError('ordinary resume cannot change the whole-game cohort contract')
    if old is not None and getattr(model, 'episode_rollout_world_size', world_size) != world_size:
        raise ValueError('ordinary resume cannot change cohort world size/seed allocation')
    model.episode_rollout_config = asdict(cfg)
    model.episode_rollout_rank = rank
    model.episode_rollout_world_size = world_size


def begin(model, env):
    config = getattr(model, 'episode_rollout_config', None)
    if config is None:
        return None
    cfg = EpisodeRolloutConfig(**config)
    if any(getattr(model, '_upper_pending', None) or ()):
        raise RuntimeError('whole-game rollout cannot discard pending upper decisions')
    if any(getattr(model, '_nosl_is_az_pending_fights', None) or ()):
        raise RuntimeError('whole-game rollout cannot discard pending combat labels')
    iteration = int(getattr(model, 'episode_rollouts_completed', 0))
    first_seed = (cfg.seed + iteration * model.episode_rollout_world_size * cfg.games_per_rank
                  + model.episode_rollout_rank * cfg.games_per_rank)
    model._last_obs = env.begin_episode_rollout(first_seed, censored_games=cfg.censored_games)
    # VecMonitor wraps the resident vector in production. All prior games
    # finished naturally; reset its bookkeeping without resetting games twice.
    if hasattr(env, 'episode_returns'):
        env.episode_returns.fill(0.)
        env.episode_lengths.fill(0)
    model._last_episode_starts = np.ones(env.num_envs, dtype=bool)
    model._upper_pending = [None] * env.num_envs
    model._upper_episode_ids = [iteration] * env.num_envs
    return dict(config=cfg, started=time.monotonic(), first_seed=first_seed,
                iteration=iteration, initial_timesteps=model.num_timesteps, wins=0, floor_sum=0.)


def record_terminals(session, dones, infos):
    for done, info in zip(dones, infos):
        if done and not info.get('TimeLimit.truncated') and not info.get('episode_cohort_censored'):
            stats = info['run_stats']
            session['wins'] += int(stats['act'] > 3 or stats['outcome'] == 'PLAYER_VICTORY')
            session['floor_sum'] += stats['floor']


def check_budget(session, steps, env):
    cfg = session['config']
    if steps >= cfg.max_steps or time.monotonic() - session['started'] >= cfg.max_seconds:
        if cfg.censored_games == 'exclude':
            env.exclude_episode_rollout_slots(np.flatnonzero(env.episode_rollout_active).tolist(),
                reason='action-budget' if steps >= cfg.max_steps else 'wall-clock-budget')
            return False
        raise TimeoutError('whole-game rollout budget exhausted; unfinished games are not training losses')
    return True


def exclude_samples(model, env):
    """Remove entire games, including samples collected before the limit."""
    excluded = set(np.flatnonzero(env.episode_rollout_censored).tolist())
    upper_before, lower_before = len(model._upper_completed), len(model._small_playout_samples)
    pending = sum(model._upper_pending[i] is not None for i in excluded)
    if excluded:
        model._upper_completed = [r for r in model._upper_completed if r['env'] not in excluded]
        # Every online sample is tagged at collection; missing provenance is an error.
        model._small_playout_samples = [r for r in model._small_playout_samples if r['env'] not in excluded]
        for i in excluded:
            model._upper_pending[i] = None
            fights = getattr(model, '_nosl_is_az_pending_fights', None)
            if fights is not None:
                fights[i] = []
    model._cohort_excluded_samples = dict(
        excluded_upper_samples=upper_before-len(model._upper_completed)+pending+model._upper_censored,
        excluded_lower_samples=lower_before-len(model._small_playout_samples))
    model._upper_censored = 0


def finish(model, env, session, steps):
    if env.episode_rollout_active.any():
        raise RuntimeError('whole-game cohort has unfinished games')
    if any(model._upper_pending) or model._upper_censored:
        raise RuntimeError('whole-game cohort has pending/censored strategic decisions')
    from rl.algorithms.upper_ppo import bootstrap_indices
    if bootstrap_indices(model._upper_completed):
        raise RuntimeError('complete games must not use a tail bootstrap')
    stats = dict(games=env.num_envs, completed_games=int(env.episode_rollout_finished.sum()),
                 censored_games=int(env.episode_rollout_censored.sum()),
                 unfinished_games=0, vector_steps=steps,
                 transitions=int(model.num_timesteps - session['initial_timesteps']),
                 first_seed=session['first_seed'], last_seed=session['first_seed'] + env.num_envs - 1,
                 rollout=session['iteration'] + 1, seconds=time.monotonic() - session['started'],
                 wins=session['wins'], floor_sum=session['floor_sum'],
                 max_completed_actions=int(env.episode_rollout_action_counts[env.episode_rollout_finished].max(initial=0)),
                 **model._cohort_excluded_samples)
    if stats['completed_games'] + stats['censored_games'] != stats['games']:
        raise RuntimeError('whole-game cohort game count mismatch')
    model.episode_rollouts_completed = session['iteration'] + 1
    model.episode_rollout_stats = stats
    model.episode_rollout_censored_slots = [dict(slot=int(i),seed=session['first_seed']+int(i),
        actions=int(env.episode_rollout_action_counts[i]),reason=env.episode_rollout_censor_reasons[i])
        for i in np.flatnonzero(env.episode_rollout_censored)]
    for name, value in stats.items():
        model.logger.record('episode_rollout/' + name, value)
    return stats
