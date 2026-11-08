"""Native step batching within workers, shared observations across processes.

Control, masks, terminal infos and recovery use the existing grouped protocol.
Only current observations use shared storage; public returns always own copies.
"""
from multiprocessing import shared_memory
import os
import selectors

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper

from rl.training import cached_mask_vec_env as cv
from rl.training.native_vec_env import NativeSearchVecEnv


class _NativeStepPipe:

    def __init__(self, remote, envs):
        self.remote = remote
        self.envs = envs
        self.native = None
        self.memory = None
        self.closed = False

    def send(self, x):
        self.remote.send(x)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.native is not None:
                self.native.close()
        finally:
            if self.memory is not None:
                self.memory.close()
            self.remote.close()

    def write(self, obs):
        for key, value in obs.items():
            self.arrays[key][self.start:self.end] = value

    def recv(self):
        while True:
            cmd, data = self.remote.recv()
            if cmd == 'native_shared':
                name, layout, self.start, self.end = data
                self.memory = shared_memory.SharedMemory(name=name)
                self.arrays = {
                    key: np.ndarray(shape, dtype=dtype, buffer=self.memory.buf, offset=offset)
                    for key, shape, dtype, offset in layout}
                self.native = NativeSearchVecEnv([lambda env=env: env for env in self.envs], threads=1)
                self.remote.send(None)
            elif cmd == 'step' and self.native is not None:
                obs, rewards, dones, infos = self.native._step_for_transport(np.asarray(data))
                self.write(obs)
                self.remote.send([(None, rewards[i], dones[i], info, self.native.reset_infos[i])
                                  for i, info in enumerate(infos)])
            elif cmd == 'reset' and self.native is not None:
                self.native._seeds = [r[0] for r in data]
                self.native._options = [r[1] for r in data]
                obs = self.native.reset()
                self.write(obs)
                self.remote.send([(None, i) for i in self.native.reset_infos])
            else:
                return (cmd, data)

def _native_group_worker(remote, parent_remote, wrappers, group_index):
    envs = []

    def make(wrapper):

        def create():
            env = wrapper.var()
            envs.append(env)
            return env
        return CloudpickleWrapper(create)
    pipe = _NativeStepPipe(remote, envs)
    try:
        cv._group_worker(pipe, parent_remote, [make(w) for w in wrappers], group_index)
    finally:
        pipe.close()

class NativeSubprocVecEnv(cv.GroupedCachedMaskSubprocVecEnv):
    """Same logical ordering/control RPCs as grouped VecEnv, native step batches."""
    _worker_target = staticmethod(_native_group_worker)

    def __init__(self, *args, **kwargs):
        self.memory = None
        try:
            super().__init__(*args, **kwargs)
            self._create_shared_observations()
        except BaseException:
            if hasattr(self, 'processes'):
                self.close()
            raise

    def _create_shared_observations(self):
        layout = []
        offset = 0
        for key, space in self.observation_space.spaces.items():
            shape = (self.num_envs, *space.shape)
            offset = (offset + 63) // 64 * 64
            layout.append((key, shape, space.dtype.str, offset))
            offset += int(np.prod(shape)) * space.dtype.itemsize
        self.memory = shared_memory.SharedMemory(create=True, size=offset)
        self.arrays = {
            key: np.ndarray(shape, dtype=dtype, buffer=self.memory.buf, offset=offset)
            for key, shape, dtype, offset in layout}
        for remote, group in zip(self.remotes, self._groups, strict=True):
            remote.send(('native_shared', (self.memory.name, layout, group[0], group[-1] + 1)))
        self._receive_groups({r: i for i, r in enumerate(self.remotes)})

    def observations(self):
        # SB3 retains the previous observation until after step_wait. Returning
        # views would silently overwrite its training sample on the next step.
        return {k: v.copy() for k, v in self.arrays.items()}

    def _receive_groups(self, pending):
        if os.name != 'posix':
            return super()._receive_groups(pending)
        # Register each pipe/sentinel once for this request batch. Rebuilding a
        # selector after each early worker response was quadratic in workers.
        # Windows retains multiprocessing's named-pipe-aware implementation.
        results = {}
        with selectors.PollSelector() as selector:
            for remote in pending:
                selector.register(remote, selectors.EVENT_READ)
            for sentinel in self._worker_sentinels:
                selector.register(sentinel, selectors.EVENT_READ)
            while pending:
                ready = [key.fileobj for key, events in selector.select()]
                for item in ready:
                    if item in self._worker_sentinels:
                        self._raise_worker_failure(self._worker_sentinels[item])
                for remote in ready:
                    group_index = pending.pop(remote)
                    selector.unregister(remote)
                    try:
                        payload, _timing = cv._unwrap_timed_payload(remote.recv())
                        results[group_index] = cv._unpack_leaf_responses(payload)
                    except (EOFError, BrokenPipeError, OSError) as exc:
                        self._raise_worker_failure(group_index, exc)
        return results

    def results(self):
        grouped = self._receive_groups({r: i for i, r in enumerate(self.remotes)})
        return [grouped[i][j] for i, g in enumerate(self._groups) for j, _ in enumerate(g)]

    def step_wait(self):
        try:
            results = self.results()
        finally:
            self.waiting = False
        _, rewards, dones, infos, self.reset_infos = zip(*results, strict=True)
        for i, (done, info) in enumerate(zip(dones, infos, strict=True)):
            self._cached_action_masks[i] = self._info_mask(self.reset_infos[i] if done else info)
        return (self.observations(), np.asarray(rewards), np.asarray(dones), infos)

    def reset(self):
        for remote, group in zip(self.remotes, self._groups, strict=True):
            remote.send(('reset', [(self._seeds[i], self._options[i]) for i in group]))
        _, infos = zip(*self.results(), strict=True)
        self.reset_infos = list(infos)
        self._reset_seeds()
        self._reset_options()
        self._cached_action_masks = [self._info_mask(i) for i in infos]
        return self.observations()

    def close(self):
        try:
            super().close()
        finally:
            if self.memory is not None:
                self.memory.close()
                self.memory.unlink()
                self.memory = None
