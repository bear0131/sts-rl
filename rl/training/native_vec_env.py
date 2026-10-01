"""Resident Python environments with a GIL-free C++ search thread pool."""
from stable_baselines3.common.vec_env import DummyVecEnv

from env.search.native_batch import NativeSearchBatch


class NativeSearchVecEnv(DummyVecEnv):
    def __init__(self, env_fns, *, threads=4):
        super().__init__(env_fns)
        try:
            self.native_search_batch = NativeSearchBatch(self.envs, threads)
        except BaseException:
            super().close()
            raise

    def close(self):
        self.native_search_batch.close()
        super().close()
