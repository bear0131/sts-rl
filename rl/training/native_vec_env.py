"""Resident Python environments with a GIL-free C++ search thread pool."""
from copy import deepcopy
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
        pending = []
        for index, env in enumerate(self.envs):
            if cohort is not None and not cohort[index]:
                self.buf_rews[index] = 0.
                self.buf_dones[index] = False
                self.buf_infos[index] = dict(episode_cohort_hold=True, simulator_transition=False)
                continue
            base = env.unwrapped
            base._combat_observation_pending = False
            base._defer_combat_observation = self.native_root_observations is not None
            try:
                obs, self.buf_rews[index], terminated, truncated, self.buf_infos[index] = env.step(self.actions[index])
            finally:
                base._defer_combat_observation = False
            deferred = base._combat_observation_pending
            base._combat_observation_pending = False
            self.buf_dones[index] = terminated or truncated
            self.buf_infos[index]['TimeLimit.truncated'] = truncated and not terminated
            if cohort is not None and (truncated or self.buf_infos[index].get('dead_screen')):
                raise RuntimeError(f'censored whole-game cohort slot {index}; refusing partial-game training')
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
        if pending:
            observations = self.native_root_observations.encode([self.envs[i].unwrapped for i in pending])
            indices = np.asarray(pending, dtype=np.intp)
            # Respect the checkpoint's visible keys and leave reset/overworld
            # rows intact. The codec already returned one array per field.
            for key in self.keys:
                self.buf_obs[key][indices] = observations[key]
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), _copy_step_infos(self.buf_infos))

    def begin_episode_rollout(self, first_seed):
        active = getattr(self, 'episode_rollout_active', None)
        if active is not None and active.any():
            raise RuntimeError('cannot reset an unfinished whole-game cohort')
        self.episode_rollout_active = np.ones(self.num_envs, dtype=bool)
        self.episode_rollout_finished = np.zeros(self.num_envs, dtype=bool)
        self.seed(int(first_seed))
        return self.reset()

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
            return self.small_playout_batch.decisions(calls)
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

    def small_playout_stats(self):
        return self.small_playout_batch.scalar_stats()

    def take_small_playout_samples(self):
        samples, self.small_playout_batch.samples = self.small_playout_batch.samples, []
        return samples
