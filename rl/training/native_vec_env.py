"""Resident Python environments with a GIL-free C++ search thread pool."""
from collections import OrderedDict
from copy import deepcopy
from contextlib import contextmanager
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
        self.native_threads = threads
        self._small_step_plan = None
        self._prefetched_steps = {}
        self._small_overlap_stats = dict(prefetched_steps=0, step_call_seconds_sum=0.)
        self._step_observation_buffers = None
        self._root_destination_cache = OrderedDict()
        self._strategic_observation_keys = None
        self._deferred_observation_rows = set()
        self.strategic_observation_rows_omitted = 0
        self.step_seconds = dict(wrappers=0., encoding=0.)
        try:
            self.native_search_batch = NativeSearchBatch(self.envs, threads, direct_observations=True)
            self.native_root_observations = None
            if (hasattr(sts, 'pack_neural_roots') and all(self._batchable(env) for env in self.envs)
                    and all(env.unwrapped.observation_space == self.envs[0].unwrapped.observation_space
                            for env in self.envs)):
                from env.core.native_observation import NativeRootObservationBatch
                self.native_root_observations = NativeRootObservationBatch(self.envs[0].unwrapped)
                # The base encoder sees its full schema, including fields an
                # older checkpoint hides. Visible fields write directly into
                # their existing vector row; public vector outputs still copy.
                self._step_observation_buffers = [
                    {key: self.buf_obs[key][index] if key in self.buf_obs
                     else np.empty(space.shape, dtype=space.dtype)
                     for key, space in env.unwrapped.observation_space.spaces.items()}
                    for index, env in enumerate(self.envs)]
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
        from rl.training.reward import ShapedRewardWrapper, StrategicRewardWrapper
        allowed = (CheckpointObservationCompat, TrainingStateWrapper,
                   ShapedRewardWrapper, StrategicRewardWrapper, gym.wrappers.TimeLimit)
        current = env
        while isinstance(current, gym.Wrapper):
            if type(current) not in allowed:
                return False
            current = current.env
        return (type(current) is SlayTheSpireEnv and current._autoregressive_actions
                and not current._combat_language)

    def step_wait(self):
        if self._strategic_observation_keys is not None:
            raise RuntimeError('complete vector steps cannot interrupt strategic observation collection')
        cohort = getattr(self, 'episode_rollout_active', None)
        if self.native_root_observations is None and cohort is None:
            return super().step_wait()
        return self._step_indices(range(self.num_envs))

    def _step_for_transport(self, actions):
        """Borrow results only until the worker's synchronous write/send ends.

        Public step()/step_subset() retain owned arrays and infos. The private
        transport owner never advances a slot until serialization has finished.
        """
        if self.native_root_observations is None:
            return self.step(actions)
        if self._strategic_observation_keys is not None:
            raise RuntimeError('cannot transport partial observations')
        self.step_async(actions)
        return self._step_indices(range(self.num_envs), borrow=True)

    def step_subset(self, actions):
        """Advance only ready logical slots, preserving all other slot state.

        Used by the independent-game diagnostic; caller owns search/step
        exclusion per slot. Returned rows follow the mapping's insertion order.
        """
        if self._strategic_observation_keys is not None:
            raise RuntimeError('complete subset steps cannot interrupt strategic observation collection')
        return self._step_subset(actions)

    def _step_subset(self, actions, *, observation_keys=None, borrow=False):
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
        return self._step_indices(indices, subset=True, observation_keys=observation_keys, borrow=borrow)

    @contextmanager
    def strategic_observations(self, keys):
        """Private projection for a complete-game collector with no obs callback.

        Inside combat only phase/act/floor are valid; other strategic fields
        become current at the next upper decision. Terminal info stays complete.
        Restore every deferred full row on exit, including callback cancellation.
        """
        if (self.native_root_observations is None or self._strategic_observation_keys is not None
                or getattr(self, 'episode_rollout_censor_policy', None) != 'error'
                or not {'phase', 'act', 'floor'} <= set(keys) <= set(self.keys)
                or any(base._emit_nosl_value_observations or base.outer_return_context
                       for base in (env.unwrapped for env in self.envs))):
            raise ValueError('strategic observation projection requires native complete small-policy collection')
        self._strategic_observation_keys = tuple(key for key in self.keys if key in keys)
        self.strategic_observation_rows_omitted = 0
        try:
            yield self._step_strategic_subset
        finally:
            try:
                for index in sorted(self._deferred_observation_rows):
                    # The qualified wrapper stack only selects visible keys;
                    # it never changes their values. Copy into the owned row.
                    observation = self.envs[index].unwrapped._obs()
                    self._save_obs(index, {key: observation[key] for key in self.keys})
                self._deferred_observation_rows.clear()
            finally:
                self._strategic_observation_keys = None

    def _step_strategic_subset(self, actions):
        if self._strategic_observation_keys is None:
            raise RuntimeError('strategic observation scope is not active')
        return self._step_subset(actions, observation_keys=self._strategic_observation_keys)

    @contextmanager
    def omit_unused_combat_value_observations(self):
        """Independent small-policy collection has no combat-value consumer.

        Keep ordinary observations and terminal labels, but avoid generating
        legacy NOSL auxiliary observations. Restore even on callback failure;
        ordinary NOSL and strategic outer-return callers keep their contract.
        """
        bases = [env.unwrapped for env in self.envs]
        if any(base.outer_return_context for base in bases):
            raise ValueError('outer-return collection requires combat value observations')
        previous = [base._emit_nosl_value_observations for base in bases]
        try:
            for base in bases:
                base._emit_nosl_value_observations = False
            yield
        finally:
            for base, enabled in zip(bases, previous, strict=True):
                base._emit_nosl_value_observations = enabled

    def _step_indices(self, indices, *, subset=False, observation_keys=None, borrow=False):
        started = time.perf_counter()
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
            if observation_keys is not None:
                self._deferred_observation_rows.add(index)
            ready = prefetched.pop(index, None)
            if ready is None:
                transition = self._step_one(index, self.actions[index])
            else:
                action, transition = ready
                if action != int(self.actions[index]) or transition is None:
                    raise RuntimeError('prefetched environment action/result disagrees with step')
            obs, self.buf_rews[index], terminated, truncated, self.buf_infos[index], deferred, buffered = transition
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
            if not buffered:
                self._save_obs(index, obs)
            self._deferred_observation_rows.discard(index)
        if prefetched:
            raise RuntimeError('prefetched transition belongs to an inactive environment')
        now = time.perf_counter()
        self.step_seconds['wrappers'] += now-started
        started = now
        if pending:
            encoder = (self.native_root_observations.encode if observation_keys is None
                       else self.native_root_observations.encode_progress)
            destination = None
            # A contiguous subset can use the final vector rows directly.
            # Legacy checkpoints missing codec fields retain the owned-output
            # path; noncontiguous rows still use the original scatter below.
            if (observation_keys is None
                    and pending == list(range(pending[0], pending[-1] + 1))
                    and self.buf_obs.keys() >= self.native_root_observations.prototypes.keys()):
                span = (pending[0], pending[-1] + 1)
                destination = self._root_destination_cache.get(span)
                if destination is None:
                    destination = {key: self.buf_obs[key][span[0]:span[1]]
                                   for key in self.native_root_observations.prototypes}
                    self._root_destination_cache[span] = destination
                    if len(self._root_destination_cache) > 16:
                        self._root_destination_cache.popitem(last=False)
                self._root_destination_cache.move_to_end(span)
            observations = encoder([self.envs[i].unwrapped for i in pending],
                **({'destination': destination} if destination is not None else {}))
            if observation_keys is not None:
                self.strategic_observation_rows_omitted += len(pending)
            pending_indices = np.asarray(pending, dtype=np.intp)
            # Respect the checkpoint's visible keys and leave reset/overworld
            # rows intact. The codec already returned one array per field.
            if destination is None:
                if observation_keys is None and hasattr(sts, 'copy_observation_rows'):
                    sts.copy_observation_rows(self.buf_obs, {k:observations[k] for k in self.keys},
                        pending_indices, np.arange(len(pending), dtype=np.int64), 1)
                else:
                    for key in (self.keys if observation_keys is None else observations):
                        self.buf_obs[key][pending_indices] = observations[key]
        self.step_seconds['encoding'] += time.perf_counter()-started
        if subset:
            rows = np.asarray(indices, dtype=np.intp)
            if borrow:
                return (self.buf_obs, self.buf_rews[rows], self.buf_dones[rows],
                        [self.buf_infos[i] for i in indices])
            keys = (self.keys if observation_keys is None else
                    ('phase', 'act', 'floor') if len(pending) == len(indices) else observation_keys)
            # Advanced indexing copies each selected row; no second copy is needed.
            return ({key: self.buf_obs[key][rows] for key in keys},
                    self.buf_rews[rows].copy(), self.buf_dones[rows].copy(),
                    _copy_step_infos([self.buf_infos[i] for i in indices]))
        if borrow:
            return self.buf_obs, self.buf_rews, self.buf_dones, self.buf_infos
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), _copy_step_infos(self.buf_infos))

    def _step_one(self, index, action):
        base = self.envs[index].unwrapped
        base._combat_observation_pending = False
        base._defer_combat_observation = self.native_root_observations is not None
        base._step_observation_buffer = (None if self._step_observation_buffers is None
                                        else self._step_observation_buffers[index])
        base._step_observation_buffer_used = False
        try:
            transition = self.envs[index].step(action)
            buffered = base._step_observation_buffer_used
            if buffered and (transition[2] or transition[3]):
                # A wrapper such as TimeLimit can terminate after the base has
                # written a live row. Keep this final observation across reset
                # and the subsequent writes to the same vector slot.
                transition = ({key: value.copy() for key, value in transition[0].items()},
                              *transition[1:])
                buffered = False
            if getattr(self, 'episode_rollout_active', None) is not None:
                self.episode_rollout_action_counts[index] += 1
        except BaseException:
            self._step_failure = dict(slot=index, action=int(action), reason='step-exception')
            raise
        finally:
            base._defer_combat_observation = False
            base._step_observation_buffer = None
            base._step_observation_buffer_used = False
        deferred = base._combat_observation_pending
        base._combat_observation_pending = False
        return (*transition, deferred, buffered)

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
        if getattr(self, '_strategic_observation_keys', None) is not None:
            raise RuntimeError('cannot reset during strategic observation collection')
        if getattr(self, '_small_step_plan', None) is not None or getattr(self, '_prefetched_steps', None):
            raise RuntimeError('cannot reset before consuming a prefetched step')
        return super().reset()

    def restore_training_states(self, states):
        """Restore simulator and its resident observation bank at one boundary.

        Generic VecEnv.env_method returns restored observations without writing
        DummyVecEnv's bank. The native search exporter and borrowed PPO consumer
        both read that bank, so it must publish the same verified observations.
        """
        if (len(states) != self.num_envs or self._strategic_observation_keys is not None
                or self._small_step_plan is not None or self._prefetched_steps):
            raise ValueError('native environment restore requires a drained matching vector')
        restored = [self.env_method('restore_training_state', state, indices=i)[0]
                    for i, state in enumerate(states)]
        for i, (observation, info) in enumerate(restored):
            self._save_obs(i, observation)
            self.reset_infos[i] = info
        self.buf_rews.fill(0.)
        self.buf_dones.fill(False)
        self.buf_infos = [{} for _ in states]
        return self._obs_from_buf()

    def begin_episode_rollout(self, first_seed, *, censored_games='error'):
        if self._strategic_observation_keys is not None:
            raise RuntimeError('cannot reset during strategic observation collection')
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
            self._exclude_episode_rollout_slot(index, reason)

    def exclude_overlong_combat(self, index, *, max_turns):
        # The engine counts turns from zero; the public game screen starts at one.
        turn = self.envs[index].unwrapped.thick.bc.turn + 1
        if turn <= max_turns:
            return False
        self._exclude_episode_rollout_slot(index, f'combat-turn-limit:{turn}>{max_turns}')
        return True

    def _exclude_episode_rollout_slot(self, index, reason):
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
        self._root_destination_cache.clear()
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

    def configure_small_playout(self, pair, *, batch_factory=None):
        from env.search.small_playout import SmallPlayoutBatch
        if hasattr(self, 'small_playout_batch') and self.small_playout_batch.samples:
            raise RuntimeError('previous compact training samples were not consumed')
        if not hasattr(self, 'small_playout_batch') or self.small_playout_batch.pair['sha256'] != pair['sha256']:
            factory = batch_factory or SmallPlayoutBatch
            self.small_playout_batch = factory(self.native_search_batch.envs,
                                                       self.native_search_batch.pool, pair)
        self.small_playout_batch.stats = {}
        self._small_overlap_stats = dict(prefetched_steps=0, step_call_seconds_sum=0.)

    def small_playout_stats(self):
        return {**self.small_playout_batch.scalar_stats(),
                **{f'overlap/{key}': float(value) for key, value in self._small_overlap_stats.items()}}

    def take_small_playout_samples(self):
        samples, self.small_playout_batch.samples = self.small_playout_batch.samples, []
        return samples
