"""Reusable whole-run PPO storage, with SB3's original sample ordering."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib.common.maskable.buffers import (
    MaskableDictRolloutBuffer, MaskableDictRolloutBufferSamples,
)
from stable_baselines3.common.buffers import BaseBuffer


class SharedRolloutBuffer(MaskableDictRolloutBuffer):
    """Keep time/environment axes; gather the same env-major shuffled indices.

    The usual buffer flattens by copying its entire observation dictionary at
    update time, then reallocates it on reset. Gathering both indices directly
    preserves its sample/RNG contract while reusing the original allocation.
    Public arrays remain time-major even after get(); only this pure-PPO path
    uses this buffer. Samples and add() destinations still own their data.
    """

    def reset(self):
        self.normalization_generation = getattr(self, 'normalization_generation', 0) + 1
        if not hasattr(self, 'observations'):
            super().reset()
        else:
            arrays = [*self.observations.values(), *[getattr(self, key) for key in
                ('actions', 'rewards', 'episode_starts', 'values', 'log_probs',
                 'advantages', 'returns')], self.action_masks]
            parallel = (sum(value.nbytes for value in arrays) >= 256 * 1024 * 1024
                        and all(value.flags.c_contiguous for value in arrays))
            if parallel:
                jobs = []
                for value in arrays:
                    flat = value.reshape(-1)
                    chunk = max(1, 32 * 1024 * 1024 // value.itemsize)
                    fill = int(value is self.action_masks)
                    jobs.extend((flat[start:start + chunk], fill)
                                for start in range(0, len(flat), chunk))
                with ThreadPoolExecutor(max_workers=8) as pool:
                    list(pool.map(lambda item: item[0].fill(item[1]), jobs))
            else:
                for value in arrays:
                    value.fill(int(value is self.action_masks))
        self.generator_ready = False
        self._prefilled = None
        BaseBuffer.reset(self)

    def _write_observation(self, obs):
        for key in self.observations:
            source = np.asarray(obs[key])
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                source = source.reshape((self.n_envs,) + self.obs_shape[key])
            # Assignment already copies; no intermediate np.array is needed.
            self.observations[key][self.pos] = source

    def prefill_observation(self, obs):
        """Copy the pre-action observation while environment workers advance.

        The collector may use this only when intervening callbacks explicitly
        promise not to inspect or mutate rollout observations. add() still owns
        all bookkeeping; reset discards a prefill from an aborted collection.
        """
        self._write_observation(obs)
        self._prefilled = (self.pos, obs)

    def add(self, obs, action, reward, episode_start, value, log_prob, *, action_masks=None):
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape(self.n_envs, self.mask_dims)
        pending = self._prefilled
        if pending is None or pending[0] != self.pos or pending[1] is not obs:
            self._write_observation(obs)
        self._prefilled = None
        self.actions[self.pos] = action.reshape(self.n_envs, self.action_dim)
        self.rewards[self.pos] = reward
        self.episode_starts[self.pos] = episode_start
        self.values[self.pos] = value.cpu().numpy().reshape(-1)
        self.log_probs[self.pos] = log_prob.cpu().numpy().reshape(-1)
        self.pos += 1
        self.full = self.pos == self.buffer_size

    def get(self, batch_size=None):
        assert self.full, 'rollout must be complete before updating'
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        self.generator_ready = True
        if batch_size is None:
            batch_size = len(indices)
        # Cache only this update's immutable dataset, never checkpoint or
        # cross-collection storage. Advanced-indexed samples still own data.
        arrays = [*self.observations.values(), self.actions, self.values,
                  self.log_probs, self.advantages, self.returns, self.action_masks]
        size = sum(array.nbytes for array in arrays)
        storage = None
        if (self.device.type == 'cuda'
                and size <= torch.cuda.get_device_properties(self.device).total_memory // 4):
            storage = {}
            try:
                for array in arrays:
                    storage[id(array)] = torch.as_tensor(array, device=self.device)
            except torch.OutOfMemoryError:
                # No sample/optimizer update has started. Retain the same
                # permutation and use the ordinary host-backed gather below.
                storage = None
            if storage is None:
                torch.cuda.empty_cache()
        for start in range(0, len(indices), batch_size):
            yield self._get_samples(indices[start:start + batch_size], device_storage=storage)

    def _get_samples(self, batch_inds, env=None, *, device_storage=None):
        # SB3 swap_and_flatten enumerates environment first, then vector time.
        if device_storage is not None:
            batch_inds = torch.as_tensor(batch_inds, device=self.device)
        rows = (batch_inds % self.buffer_size, batch_inds // self.buffer_size)

        def take(array):
            if device_storage is not None:
                return device_storage[id(array)][rows]
            # Advanced indexing owns this batch; to_torch need not copy again.
            return self.to_torch(array[rows], copy=False)

        return MaskableDictRolloutBufferSamples(
            observations={key: take(value) for key, value in self.observations.items()},
            actions=take(self.actions), old_values=take(self.values).flatten(),
            old_log_prob=take(self.log_probs).flatten(),
            advantages=take(self.advantages).flatten(), returns=take(self.returns).flatten(),
            action_masks=take(self.action_masks).reshape(-1, self.mask_dims),
        )
