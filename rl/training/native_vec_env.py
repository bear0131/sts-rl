"""Resident Python environments with a GIL-free C++ search thread pool."""
from copy import deepcopy
import time
import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from env.search.native_batch import NativeSearchBatch
import slaythespire as sts


_INFO_ATOMIC_TYPES = frozenset((type(None), bool, int, float, str, bytes))


def _copy_step_infos(infos):
    """Copy Gym info containers without visiting every immutable scalar twice.

    Keep deepcopy's shared-reference/cycle semantics and delegate arrays,
    subclasses and custom objects to their existing copy protocol.
    """
    memo = {}
    atomic = _INFO_ATOMIC_TYPES

    def clone(value):
        kind = type(value)
        if kind in atomic:
            return value
        identity = id(value)
        if identity in memo:
            return memo[identity]
        if kind is dict and all(type(key) in atomic for key in value):
            result = {}
            memo[identity] = result
            for key, item in value.items():
                result[key] = item if type(item) in atomic else clone(item)
            return result
        if kind is list:
            result = []
            memo[identity] = result
            result.extend(item if type(item) in atomic else clone(item) for item in value)
            return result
        return deepcopy(value, memo)

    return clone(infos)


class NativeSearchVecEnv(DummyVecEnv):
    def __init__(self, env_fns, *, threads=4):
        super().__init__(env_fns)
        self._small_step_plan = None
        self._prefetched_steps = {}
        self._small_overlap_stats = dict(prefetched_steps=0, step_call_seconds_sum=0.)
        try:
            self.native_search_batch = NativeSearchBatch(self.envs, threads, direct_observations=True)
            self.native_root_observations = None
            if (hasattr(sts, 'pack_neural_roots') and all(self._batchable(env) for env in self.envs)
                    and all(env.unwrapped.observation_space == self.envs[0].unwrapped.observation_space
                            for env in self.envs)):
                from env.core.native_observation import NativeRootObservationBatch
                self.native_root_observations = NativeRootObservationBatch(self.envs[0].unwrapped)
        except BaseException:
            if hasattr(self, 'native_search_batch'):
                self.native_search_batch.close()
            super().close()
            raise

    @staticmethod
    def _batchable(env):
        # Observation-transforming/stateful user wrappers keep DummyVecEnv's
        # path. Only the actual training stack can carry the internal placeholder.
        from env.core.run_env import SlayTheSpireEnv
        from rl.training.environments import CheckpointObservationCompat, TrainingStateWrapper
        from rl.training.reward import ShapedRewardWrapper
        allowed = (CheckpointObservationCompat, TrainingStateWrapper,
                   ShapedRewardWrapper, gym.wrappers.TimeLimit)
        current = env
        while isinstance(current, gym.Wrapper):
            if type(current) not in allowed:
                return False
            current = current.env
        return (type(current) is SlayTheSpireEnv and current._autoregressive_actions
                and not current._combat_language)

    def step_wait(self):
        cohort = getattr(self, 'episode_rollout_active', None)
        if self.native_root_observations is None and cohort is None:
            return super().step_wait()
        return self._step_indices(range(self.num_envs))

    def step_subset(self, actions):
        """Advance only ready logical slots, preserving all other slot state.

        Used by the independent-game diagnostic; caller owns search/step
        exclusion per slot. Returned rows follow the mapping's insertion order.
        """
        indices = list(actions)
        if (not indices or any(type(i) is not int or not 0 <= i < self.num_envs for i in indices)
                or any(not self.action_space.contains(a) for a in actions.values())):
            raise ValueError('invalid ready-slot actions')
        if self._small_step_plan is not None or self._prefetched_steps:
            raise RuntimeError('cannot partially step a pending vector batch')
        cohort = getattr(self, 'episode_rollout_active', None)
        if cohort is not None and any(not cohort[i] for i in indices):
            raise ValueError('cannot advance a completed cohort slot')
        self.actions = np.zeros(self.num_envs, dtype=np.int64)
        for i, action in actions.items():
            self.actions[i] = action
        return self._step_indices(indices, subset=True)

    def _step_indices(self, indices, *, subset=False):
        cohort = getattr(self, 'episode_rollout_active', None)
        prefetched, self._prefetched_steps = self._prefetched_steps, {}
        self._small_step_plan = None
        pending = []
        for index in indices:
            env = self.envs[index]
            if cohort is not None and not cohort[index]:
                self.buf_rews[index] = 0.
                self.buf_dones[index] = False
                self.buf_infos[index] = dict(episode_cohort_hold=True, simulator_transition=False)
                continue
            base = env.unwrapped
            ready = prefetched.pop(index, None)
            if ready is None:
                transition = self._step_one(index, self.actions[index])
            else:
                action, transition = ready
                if action != int(self.actions[index]) or transition is None:
                    raise RuntimeError('prefetched environment action/result disagrees with step')
            obs, self.buf_rews[index], terminated, truncated, self.buf_infos[index], deferred = transition
            self.buf_dones[index] = terminated or truncated
            self.buf_infos[index]['TimeLimit.truncated'] = truncated and not terminated
            if cohort is not None and ((truncated and not terminated) or self.buf_infos[index].get('dead_screen')):
                if truncated and not self.buf_infos[index].get('dead_screen') and self.episode_rollout_censor_policy=='exclude':
                    self.exclude_episode_rollout_slots([index], reason='environment-time-limit')
                    self.buf_rews[index] = 0.
                    self.buf_dones[index] = False
                    self.buf_infos[index].update(episode_cohort_hold=True, episode_cohort_censored=True)
                    if deferred:
                        full=base._obs()
                        obs={key:full[key] for key in self.keys}
                    self._save_obs(index,obs)
                    continue
                self._step_failure = dict(slot=index, action=int(self.actions[index]),
                                          reason='censored-cohort', truncated=bool(truncated),
                                          dead_screen=bool(self.buf_infos[index].get('dead_screen')))
                raise RuntimeError(f'censored whole-game cohort slot {index}; refusing partial-game training')
            if truncated and cohort is None:
                recorder = getattr(self, 'failure_scene_recorder', None)
                if recorder is not None:
                    recorder(dict(slot=index, action=int(self.actions[index]), reason='truncated'))
            if self.buf_dones[index]:
                if deferred:
                    # TimeLimit may truncate outside the base environment.
                    # Materialize the real final observation before auto-reset.
                    full = base._obs()
                    obs = {key: full[key] for key in self.keys}
                self.buf_infos[index]['terminal_observation'] = obs
                if cohort is None:
                    obs, self.reset_infos[index] = env.reset()
                else:
                    cohort[index] = False
                    self.episode_rollout_finished[index] = True
            elif deferred:
                pending.append(index)
                continue
            self._save_obs(index, obs)
        if prefetched:
            raise RuntimeError('prefetched transition belongs to an inactive environment')
        if pending:
            observations = self.native_root_observations.encode([self.envs[i].unwrapped for i in pending])
            pending_indices = np.asarray(pending, dtype=np.intp)
            # Respect the checkpoint's visible keys and leave reset/overworld
            # rows intact. The codec already returned one array per field.
            for key in self.keys:
                self.buf_obs[key][pending_indices] = observations[key]
        if subset:
            rows = np.asarray(indices, dtype=np.intp)
            return ({key: value[rows].copy() for key, value in self.buf_obs.items()},
                    self.buf_rews[rows].copy(), self.buf_dones[rows].copy(),
                    _copy_step_infos([self.buf_infos[i] for i in indices]))
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), _copy_step_infos(self.buf_infos))

    def _step_one(self, index, action):
        base = self.envs[index].unwrapped
        base._combat_observation_pending = False
        base._defer_combat_observation = self.native_root_observations is not None
        try:
            transition = self.envs[index].step(action)
            if getattr(self, 'episode_rollout_active', None) is not None:
                self.episode_rollout_action_counts[index] += 1
        except BaseException:
            self._step_failure = dict(slot=index, action=int(action), reason='step-exception')
            raise
        finally:
            base._defer_combat_observation = False
        deferred = base._combat_observation_pending
        base._combat_observation_pending = False
        return (*transition, deferred)

    def prepare_small_playout_steps(self, actions, ordinary_indices):
        if self._small_step_plan is not None or self._prefetched_steps:
            raise RuntimeError('previous prefetched step was not consumed')
        if (self.native_root_observations is None
                or not getattr(sts, 'small_playout_parallel_steps', False)):
            return False
        actions = np.asarray(actions).copy()
        indices = tuple(int(i) for i in ordinary_indices)
        if actions.shape != (self.num_envs,) or len(set(indices)) != len(indices):
            raise ValueError('invalid small-playout step plan')
        if any(not 0 <= i < self.num_envs for i in indices):
            raise ValueError('small-playout step slot outside vector')
        self._small_step_plan = (actions, indices)
        return True

    def _prefetch_step(self, index, action):
        cohort = getattr(self, 'episode_rollout_active', None)
        if index in self._prefetched_steps or (cohort is not None and not cohort[index]):
            raise RuntimeError('duplicate or inactive prefetched environment step')
        self._prefetched_steps[index] = (int(action), None)
        started = time.perf_counter()
        self._prefetched_steps[index] = (int(action), self._step_one(index, action))
        self._small_overlap_stats['prefetched_steps'] += 1
        # Callback wall durations can overlap while native code releases the
        # GIL. This sum is diagnostic, never additive with collector wall time.
        self._small_overlap_stats['step_call_seconds_sum'] += time.perf_counter() - started

    def _prefetch_ordinary_steps(self):
        actions, indices = self._small_step_plan
        for index in indices:
            self._prefetch_step(index, actions[index])

    def reset(self):
        if getattr(self, '_small_step_plan', None) is not None or getattr(self, '_prefetched_steps', None):
            raise RuntimeError('cannot reset before consuming a prefetched step')
        return super().reset()

    def begin_episode_rollout(self, first_seed, *, censored_games='error'):
        if censored_games not in ('error', 'exclude'):
            raise ValueError('invalid censored-game policy')
        active = getattr(self, 'episode_rollout_active', None)
        if active is not None and active.any():
            raise RuntimeError('cannot reset an unfinished whole-game cohort')
        self.episode_rollout_active = np.ones(self.num_envs, dtype=bool)
        self.episode_rollout_finished = np.zeros(self.num_envs, dtype=bool)
        self.episode_rollout_censored = np.zeros(self.num_envs, dtype=bool)
        self.episode_rollout_action_counts = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_rollout_censor_policy = censored_games
        self.episode_rollout_censor_reasons = {}
        self.seed(int(first_seed))
        return self.reset()

    def exclude_episode_rollout_slots(self, indices, *, reason):
        if self.episode_rollout_censor_policy != 'exclude':
            raise RuntimeError('censored-game exclusion was not configured')
        for index in indices:
            if not self.episode_rollout_active[index]:
                raise ValueError('only active games can be censored')
            recorder = getattr(self, 'failure_scene_recorder', None)
            if recorder is not None:
                recorder(dict(slot=int(index), reason=reason, censored_game_excluded=True))
            self.episode_rollout_active[index] = False
            self.episode_rollout_censored[index] = True
            self.episode_rollout_censor_reasons[int(index)] = reason

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        active = getattr(self, 'episode_rollout_active', None)
        if method_name == 'action_masks' and active is not None:
            rows = []
            for index in self._get_indices(indices):
                if active[index]:
                    rows.extend(super().env_method(method_name, *method_args, indices=index, **method_kwargs))
                else:
                    # The collector never sends this placeholder to the simulator.
                    mask = np.zeros(self.action_space.n, dtype=bool)
                    mask[0] = True
                    rows.append(mask)
            return rows
        return super().env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def close(self):
        self.__dict__.pop('small_playout_batch', None)
        self.native_search_batch.close()
        super().close()

    def env_method_batch(self, method_name, calls):
        if method_name == 'small_playout_decision':
            enabled = self._small_step_plan is not None
            return self.small_playout_batch.decisions(calls,
                pre_step=self._prefetch_step if enabled else None,
                start_steps=self._prefetch_ordinary_steps if enabled else None)
        if method_name == 'rulebase_turn_decision':
            return self.native_search_batch.rulebase_turn_decisions(calls)
        return {index: self.env_method(method_name, *args, indices=index, **kwargs)[0]
                for index, (args, kwargs) in calls.items()}

    def configure_small_playout(self, pair):
        from env.search.small_playout import SmallPlayoutBatch
        if hasattr(self, 'small_playout_batch') and self.small_playout_batch.samples:
            raise RuntimeError('previous compact training samples were not consumed')
        if not hasattr(self, 'small_playout_batch') or self.small_playout_batch.pair['sha256'] != pair['sha256']:
            self.small_playout_batch = SmallPlayoutBatch(self.native_search_batch.envs,
                                                       self.native_search_batch.pool, pair)
        self.small_playout_batch.stats = {}
        self._small_overlap_stats = dict(prefetched_steps=0, step_call_seconds_sum=0.)

    def small_playout_stats(self):
        return {**self.small_playout_batch.scalar_stats(),
                **{f'overlap/{key}': float(value) for key, value in self._small_overlap_stats.items()}}

    def take_small_playout_samples(self):
        samples, self.small_playout_batch.samples = self.small_playout_batch.samples, []
        return samples
