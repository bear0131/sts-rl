"""Independent whole-game collection with explicit checkpointed scheduling.

Reuse native environments, compact search, strategic credit and cohort checks.
The barrier control uses the same new random streams and sample construction.
Neither mode updates weights or changes search/terminal/return definitions.
"""
from dataclasses import asdict, dataclass
from contextlib import contextmanager
import time

import numpy as np
import torch

from env.core import native_actions as C, vocab as V
from env.search.small_playout import SmallPlayoutStream
from rl.algorithms.upper_ppo import UPPER_KEYS, capture_step
from rl.search.rollout_inference import UpperPolicyInference
from rl.training import episode_rollout, reward_monitor, campfire_monitor
from rl.training.small_playout import configure_collection


@dataclass(frozen=True)
class IndependentRolloutConfig:
    mode: str = 'independent'
    threads: int = 16
    upper_batch: int = 64
    upper_wait_seconds: float = .002
    seed: int = 20261104

    def __post_init__(self):
        if (self.mode not in ('barrier', 'independent')
                or any(type(x) is not int or x < 1 for x in (self.threads, self.upper_batch))
                or type(self.seed) is not int or self.seed < 0
                or not np.isfinite(self.upper_wait_seconds) or self.upper_wait_seconds < 0):
            raise ValueError('invalid independent rollout diagnostic configuration')


class GameRandomStreams:
    """Separate policy/search streams keyed by the globally unique game seed."""
    version = 'independent-games-pcg64-v1'

    def __init__(self, seed, first_game_seed, count):
        self.upper = [np.random.default_rng(np.random.SeedSequence([seed, first_game_seed+i, 0]))
                      for i in range(count)]
        self.search = [np.random.default_rng(np.random.SeedSequence([seed, first_game_seed+i, 1]))
                       for i in range(count)]

    def nonce(self, index):
        return int(self.search[index].bit_generator.random_raw())

    def action(self, index, probabilities):
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
                or probabilities.sum() <= 0):
            raise ValueError('invalid policy probabilities')
        cumulative = np.cumsum(probabilities)
        # Inverse-CDF categorical sampling; independent of completion order.
        return int(np.searchsorted(cumulative, self.upper[index].random()*cumulative[-1], side='right'))

    def state_dict(self):
        return dict(version=self.version,
                    upper=[r.bit_generator.state for r in self.upper],
                    search=[r.bit_generator.state for r in self.search])

    def load_state_dict(self, state):
        if (state['version'] != self.version or len(state['upper']) != len(self.upper)
                or len(state['search']) != len(self.search)):
            raise ValueError('incompatible per-game RNG state')
        for kind in ('upper', 'search'):
            for rng, saved in zip(getattr(self, kind), state[kind], strict=True):
                rng.bit_generator.state = saved


def configure(model, config, *, fork=False):
    """Opt in at a natural cohort boundary; plain resume uses its saved contract."""
    old = getattr(model, 'independent_rollout_config', None)
    selected = old if config is None else asdict(IndependentRolloutConfig(**config))
    if selected is None:
        return None
    if selected['mode'] != 'independent':
        raise ValueError('barrier mode is a frozen diagnostic control only')
    cohort = getattr(model, 'episode_rollout_config', None)
    if cohort is None or cohort['censored_games'] != 'error':
        raise ValueError('independent training requires complete natural cohorts with error budgets')
    if selected != old and model.num_timesteps:
        stats = getattr(model, 'episode_rollout_stats', {})
        if (not fork or stats.get('rollout') != getattr(model, 'episode_rollouts_completed', None)
                or stats.get('completed_games') != cohort['games_per_rank']
                or stats.get('censored_games') != 0 or stats.get('unfinished_games') != 0):
            raise ValueError('scheduler migration requires an explicit fork at a complete natural boundary')
        model.independent_rollout_migration = dict(
            source_round=stats['rollout'], source_global_step=model.num_timesteps,
            previous=old, random_stream=GameRandomStreams.version)
    model.independent_rollout_config = selected
    return selected


def _callback_uses_observations(callback):
    from stable_baselines3.common.callbacks import CallbackList
    if type(callback) is CallbackList:
        return any(_callback_uses_observations(child) for child in callback.callbacks)
    # A subclass may add an observation consumer. Require its own declaration,
    # never inherit the optimization permission from a known parent callback.
    return type(callback).__dict__.get('_uses_rollout_observations', True) is not False


@contextmanager
def _strategic_observations(model, vec, callback):
    if vec.native_root_observations is None or _callback_uses_observations(callback):
        yield vec.step_subset
        return
    keys = UPPER_KEYS.intersection(model._last_obs)
    try:
        with vec.strategic_observations(keys) as step:
            model._last_obs = {key: value for key, value in model._last_obs.items() if key in keys}
            yield step
    finally:
        # Full observation hashes and checkpoint input contracts are unchanged.
        model._last_obs = vec._obs_from_buf()


def collect(model, env, callback, rollout_buffer, n_rollout_steps, use_masking=True, *, config):
    """Collect real policy/search samples with weights frozen for the whole cohort.

    An explicit synchronous control isolates scheduling from the new RNG
    contract. Per-game sample order is canonicalized before returning.
    Legacy unused combat-value diagnostic rows are not materialized in either
    mode. Complete cohorts are the only supported checkpoint boundary.
    """
    from stable_baselines3.common.vec_env import VecMonitor
    from rl.training.native_vec_env import NativeSearchVecEnv
    cfg = IndependentRolloutConfig(**config)
    if (not isinstance(env, VecMonitor) or type(env.venv) is not NativeSearchVecEnv
            or not use_masking or not getattr(model, 'episode_rollout_config', None)
            or model.nosl_is_az_combat_strategy != 'small-playout'
            or not model.nosl_is_az_inner_only or model.nosl_is_az_replay_enabled
            or getattr(model, '_outer_return', False) or getattr(model, '_frozen_upper', False)
            or model.policy.upper_auxiliary_config is not None
            or getattr(model, 'amp_mode', None) not in (None, 'off', 'fp32')
            or getattr(model, 'nosl_is_az_upper_strategy', 'model') != 'model'):
        raise ValueError('independent diagnostic requires FP32 native complete-game joint small-playout')
    vec = env.venv
    vec.strategic_observation_rows_omitted = 0
    model.policy.set_training_mode(False)
    rollout_buffer.reset()
    model._upper_completed, model._upper_censored = [], 0
    model._nosl_is_az_samples = []
    model._nosl_is_az_rollout_upper_counts = {}
    model._nosl_is_az_rollout_hp_order = {}
    reward_monitor.begin_collection(model)
    campfire_monitor.begin_collection(model)
    session = episode_rollout.begin(model, env)
    configure_collection(model, env)
    callback.on_rollout_start()
    randoms = GameRandomStreams(cfg.seed, session['first_seed'], vec.num_envs)
    combat = V.PHASE_VOCAB_INDEX['combat']
    end_action = C.policy_offset(model.combat_action_contract)
    ready = set(range(vec.num_envs))
    upper_waiting, actions, masks, estimates = {}, {}, {}, {}
    upper_policy = {}
    stats = dict(root_policy_seconds=0., env_step_seconds=0., search_wait_seconds=0.,
                 upper_batches=0, upper_rows=0, transition_batches=0, simulator_transitions=0,
                 selection_prefix_decisions=0, forced_end_turn_roots=0)
    model._independent_rollout_progress = stats

    with (vec.omit_unused_combat_value_observations(),
          _strategic_observations(model, vec, callback) as step_subset,
          UpperPolicyInference(model.policy.core, model.device) as upper_inference,
          SmallPlayoutStream(vec.small_playout_batch, threads=cfg.threads) as stream):
        while vec.episode_rollout_active.any():
            maximum = int(vec.episode_rollout_action_counts.max(initial=0))
            if (maximum >= session['config'].max_steps
                    or time.monotonic()-session['started'] >= session['config'].max_seconds):
                raise TimeoutError('independent diagnostic budget exhausted; no incomplete-game result')
            calls = {}
            for i in sorted(ready):
                mask = np.asarray(vec.envs[i].get_wrapper_attr('action_masks')(), dtype=bool)
                masks[i] = mask
                estimates[i] = (0., 0.)
                if int(model._last_obs['phase'][i, 0]) == combat:
                    if mask[end_action] and mask.sum() == 1:
                        actions[i] = end_action
                        stats['forced_end_turn_roots'] += 1
                    else:
                        calls[i] = ((), dict(state_nonce=randoms.nonce(i)))
                else:
                    upper_waiting[i] = time.monotonic()
            ready.clear()
            stream.submit(calls)
            if upper_waiting and (len(upper_waiting) >= cfg.upper_batch or not stream.pending
                    or time.monotonic() - min(upper_waiting.values()) >= cfg.upper_wait_seconds):
                started = time.perf_counter()
                rows = sorted(upper_waiting)[:cfg.upper_batch]
                obs = {k: v[rows] for k, v in model._last_obs.items()}
                action_masks = np.stack([masks[i] for i in rows])
                with torch.inference_mode():
                    logits, values = upper_inference(obs)
                    dist = model.policy.action_dist.proba_distribution(action_logits=logits.float())
                    dist.apply_masking(action_masks)
                    ps = dist.distribution.probs.detach().cpu().numpy()
                    chosen = [randoms.action(i, ps[j]) for j, i in enumerate(rows)]
                    logps = dist.log_prob(torch.as_tensor(chosen, device=model.device)).float().cpu().numpy()
                    values = values.float().flatten().cpu().numpy()
                    reward_policy = reward_monitor.policy_probabilities(dist, obs)
                    campfire_policy = campfire_monitor.policy_probabilities(dist, obs)
                reward_rows = {int(j): p for j, p in zip(*reward_policy)}
                campfire_rows = {int(j): p for j, p in zip(*campfire_policy)}
                for j, i in enumerate(rows):
                    del upper_waiting[i]
                    actions[i] = chosen[j]
                    estimates[i] = (float(values[j]), float(logps[j]))
                    upper_policy[i] = (reward_rows.get(j), campfire_rows.get(j))
                stats['root_policy_seconds'] += time.perf_counter() - started
                stats['upper_batches'] += 1
                stats['upper_rows'] += len(rows)
            waiting = not actions or (cfg.mode == 'barrier'
                                      and len(actions) < int(vec.episode_rollout_active.sum()))
            started = time.perf_counter()
            decisions = stream.receive(timeout=.001 if waiting else 0.)
            stats['search_wait_seconds'] += time.perf_counter() - started
            actions.update({i: d['action'] for i, d in decisions.items()})
            if not actions or (cfg.mode == 'barrier' and len(actions) < int(vec.episode_rollout_active.sum())):
                continue
            started = time.perf_counter()
            indices = sorted(actions)
            ordered = {i: actions[i] for i in indices}
            # Strategic records never consume the combat policy's encoded
            # inputs. Within a fight only phase/act/floor accrue its reward;
            # upper decisions retain every field used by the existing record.
            old_phase = model._last_obs['phase'][indices]
            old_keys = UPPER_KEYS if np.any(old_phase != combat) else {'phase', 'act', 'floor'}
            old = {k: old_phase if k == 'phase' else v[indices]
                   for k, v in model._last_obs.items() if k in old_keys}
            action_rows = np.array([actions.pop(i) for i in indices], dtype=np.int64)
            mask_rows = np.stack([masks.pop(i) for i in indices])
            value_rows, logp_rows = np.array([estimates.pop(i) for i in indices]).T
            observations, rewards, dones, infos = step_subset(ordered)
            # VecMonitor has a full-vector API; preserve its per-slot accounting.
            env.episode_returns[indices] += rewards
            env.episode_lengths[indices] += 1
            for offset, i in enumerate(indices):
                if dones[offset]:
                    infos[offset]['episode'] = dict(r=float(env.episode_returns[i]),
                        l=int(env.episode_lengths[i]), t=round(time.time()-env.t_start, 6))
                    env.episode_returns[i] = 0.
                    env.episode_lengths[i] = 0
                    env.episode_count += 1
            capture_step(model, old, action_rows, mask_rows, value_rows, logp_rows,
                         observations, dones, infos, indices=indices)
            reward_rows, campfire_rows = [], []
            for offset, i in enumerate(indices):
                rp, cp = upper_policy.pop(i, (None, None))
                if rp is not None:
                    reward_rows.append((offset, rp))
                if cp is not None:
                    campfire_rows.append((offset, cp))
            reward_monitor.record_step(model, old, action_rows, mask_rows,
                ([r[0] for r in reward_rows], np.asarray([r[1] for r in reward_rows], dtype=np.float32)), infos)
            campfire_monitor.record_step(model, action_rows, mask_rows,
                ([r[0] for r in campfire_rows], np.asarray([r[1] for r in campfire_rows], dtype=np.float32)), infos)
            episode_rollout.record_terminals(session, dones, infos)
            prefix = sum(info.get('simulator_transition') is False for info in infos)
            stats['selection_prefix_decisions'] += prefix
            stats['simulator_transitions'] += len(indices)-prefix
            stats['transition_batches'] += 1
            model.num_timesteps += len(indices)
            model._update_info_buffer(infos, dones)
            for k in observations:
                model._last_obs[k][indices] = observations[k]
            model._last_episode_starts[indices] = dones
            ready.update(i for i in indices if vec.episode_rollout_active[i])
            stats['env_step_seconds'] += time.perf_counter()-started
            callback.update_locals(locals())
            if not callback.on_step():
                return False
        if stream.pending or upper_waiting or actions:
            raise RuntimeError('completed cohort retained pending work')
        stats.update(upper_inference.statistics())
        stats['combat_observation_rows_omitted'] = vec.strategic_observation_rows_omitted
    model._small_playout_stats = vec.small_playout_stats()
    model._small_playout_samples = sorted(vec.take_small_playout_samples(), key=lambda r: r['env'])
    model._upper_completed.sort(key=lambda r: (r['env'], r['episode']))
    episode_rollout.exclude_samples(model, env)
    episode_rollout.finish(model, env, session, int(vec.episode_rollout_action_counts.max(initial=0)))
    model._independent_rollout_rng = randoms.state_dict()
    stats['collection_seconds'] = time.monotonic()-session['started']
    model._independent_rollout_diagnostic = dict(random_stream=randoms.version, **stats,
        mode=cfg.mode, note='Unused legacy combat-value diagnostic rows omitted; complete natural cohorts only.')
    model._nosl_is_az_rollout_timings = dict(stats)
    model._nosl_is_az_rollout_upper_counts = dict(forced_end_turn_roots=stats['forced_end_turn_roots'])
    for k, value in stats.items():
        model.logger.record('nosl_is_az/'+k, value)
    for k, value in model._small_playout_stats.items():
        model.logger.record('small_playout/'+k, value)
    callback.on_rollout_end()
    return True
